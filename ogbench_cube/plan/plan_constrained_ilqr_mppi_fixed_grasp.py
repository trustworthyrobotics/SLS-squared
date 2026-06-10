#!/usr/bin/env python3
"""Plan in OGBench space using MPPI-warmstarted Conformal SLS MPC with a fixed post-oracle grasp command."""

import os
import sys
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

mpl_config_dir = Path(__file__).resolve().parent / ".mplconfig"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import mujoco
from tqdm.auto import tqdm
import pyrallis

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import config, lax
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

ERROR_MODEL_ROOT = Path(__file__).resolve().parents[2] / "error_calib" / "error_model"
if str(ERROR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ERROR_MODEL_ROOT))

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.mppi_planner import MPPIPlanner

import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie
from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset
from error_model import MGNLLPredictor

@dataclass
class PlanSLSMoppiCubeConfig:
    """Configuration for warmstarted conformal SLS MPC on OGBench cubes."""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("ogbench_cube/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("ogbench_cube/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("ogbench_cube/eval/fixed_error_covariance.pt"))
    state_region_path: Optional[Path] = field(default=Path("ogbench_cube/eval/latent_ellipsoid"))
    state_abs_limit: Optional[float] = field(default=None)
    enable_obstacle: bool = field(default=False)
    obstacle_model_path: Path = field(
        default=Path("obs_height_ogbench/obs_net_height_embd_12_strtn/linear/model.pt")
    )
    obstacle_margin: float = field(default=0.0)
    mppi_obstacle_margin: Optional[float] = field(default=None)
    obstacle_penalty_weight: float = field(default=1000.0)
    dataset_path: Path = field(default=Path("ogbench_cube/data/test_data/ogbench_cube_test.h5"))
    action_stats_dataset_path: Optional[Path] = field(default=None)
    out_dir: Path = field(default=Path("ogbench_cube/plan/sls_mppi_conformal_fixed_grasp"))
    device: str = field(default="auto")
    horizon: int = field(default=16)
    mppi_horizon: Optional[int] = field(default=None)
    max_mpc_steps: int = field(default=120)
    max_oracle_steps: int = field(default=80)
    video_fps: int = field(default=20)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    visualize_success_colors: bool = field(default=False)
    terminate_on_ogbench_success: bool = field(default=True)
    
    mppi_samples: int = 512
    mppi_update_iter: int = 5
    mppi_reward_weight: float = 30.0
    mppi_noise_level: float = 0.25
    mppi_beta_filter: float = 0.6
    mppi_q_stage: float = 100.0
    mppi_q_stage_pos: Optional[float] = field(default=None)
    mppi_q_stage_vel: Optional[float] = field(default=None)
    mppi_q_terminal: float = 100.0
    mppi_state_box_penalty: float = 0.0
    mppi_state_region_penalty: Optional[float] = field(default=None)
    mppi_r_control: float = 0.01
    mppi_r_control_u4: float = 0.01
    mppi_goal_epsilon: Optional[float] = field(default=None)
    mppi_recede_min_horizon: int = field(default=1)
    bootstrap_mppi_steps: int = 5
    
    grasp_contact_threshold: float = 0.5
    grasp_alignment_threshold: float = 0.03
    gripper_height_threshold: float = 0.09
    q_stage: float = field(default=10.0)
    q_stage_pos: Optional[float] = field(default=None)
    q_stage_vel: Optional[float] = field(default=None)
    q_terminal: float = field(default=100.0)
    r_control: float = field(default=0.01)
    r_control_u4: float = field(default=0.01)
    fixed_grasp_raw_u4: Optional[float] = field(default=None)
    fixed_grasp_normalized_u4: Optional[float] = field(default=None)

# --- Layer Weight Ingestion ---
class JAXObstacleMLP(eqx.Module):
    linear_layers: list
    layer_norm_scales: list
    layer_norm_biases: list
    feature_mean: jax.Array
    feature_std: jax.Array
    threshold: jax.Array
    input_dim: int
    activation: str = eqx.field(static=True)
    classifier: str = eqx.field(static=True)
    spectral_norm: bool = eqx.field(static=True)

    def __call__(self, state):
        z = state[: self.input_dim]
        x = (z - self.feature_mean) / self.feature_std
        for i, linear in enumerate(self.linear_layers[:-1]):
            x = linear(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = x * self.layer_norm_scales[i] + self.layer_norm_biases[i]
            x = jnp.tanh(x) if self.activation == "tanh" else jax.nn.gelu(x)
        return self.linear_layers[-1](x).squeeze(-1)


class JAXEllipsoidConstraint(eqx.Module):
    center: jax.Array
    unit_precision: jax.Array
    input_dim: int

    def score(self, state: jax.Array) -> jax.Array:
        x = jnp.asarray(state[: self.input_dim], dtype=jnp.float64) - self.center
        return x @ self.unit_precision @ x

    def violation(self, state: jax.Array) -> jax.Array:
        return self.score(state) - 1.0

def resolve_obstacle_activation(artifact: dict, artifact_path: Path) -> str:
    activation = artifact.get("activation") or artifact.get("cache_config", {}).get("activation")
    if activation is not None:
        return str(activation).lower()
    return "tanh" if int(artifact["hidden_dim"]) == 6 or "obs_net_small" in str(artifact_path) else "gelu"

def build_jax_obstacle_from_artifact(artifact_path: Path, key: jax.Array) -> JAXObstacleMLP:
    artifact_path = artifact_path.expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Obstacle model artifact not found: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    state_dict = artifact["state_dict"]
    input_dim = int(artifact["input_dim"])
    hidden_dim = int(artifact["hidden_dim"])
    depth = int(artifact["depth"])
    dropout = float(artifact["dropout"])
    activation = resolve_obstacle_activation(artifact, artifact_path)
    classifier = str(artifact.get("classifier", artifact.get("cache_config", {}).get("classifier", "mlp")))
    spectral_norm = bool(artifact.get("spectral_norm", artifact.get("cache_config", {}).get("spectral_norm", False)))

    linear_layers = []
    layer_norm_scales = []
    layer_norm_biases = []
    if classifier == "linear":
        linear = eqx.nn.Linear(input_dim, 1, key=key)
        linear = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            linear,
            (
                jnp.asarray(state_dict["linear.weight"].detach().cpu().numpy()),
                jnp.asarray(state_dict["linear.bias"].detach().cpu().numpy()),
            ),
        )
        linear_layers.append(linear)
    elif classifier == "mlp":
        keys = jax.random.split(key, depth)
        module_idx = 0
        current_dim = input_dim
        for i in range(depth - 1):
            linear = eqx.nn.Linear(current_dim, hidden_dim, key=keys[i])
            linear = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                linear,
                (
                    jnp.asarray(state_dict[f"net.{module_idx}.weight"].detach().cpu().numpy()),
                    jnp.asarray(state_dict[f"net.{module_idx}.bias"].detach().cpu().numpy()),
                ),
            )
            linear_layers.append(linear)

            ln_idx = module_idx + 1
            layer_norm_scales.append(jnp.asarray(state_dict[f"net.{ln_idx}.weight"].detach().cpu().numpy()))
            layer_norm_biases.append(jnp.asarray(state_dict[f"net.{ln_idx}.bias"].detach().cpu().numpy()))
            module_idx += 4 if dropout > 0.0 else 3
            current_dim = hidden_dim

        output_linear = eqx.nn.Linear(current_dim, 1, key=keys[-1])
        output_linear = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            output_linear,
            (
                jnp.asarray(state_dict[f"net.{module_idx}.weight"].detach().cpu().numpy()),
                jnp.asarray(state_dict[f"net.{module_idx}.bias"].detach().cpu().numpy()),
            ),
        )
        linear_layers.append(output_linear)
    else:
        raise ValueError(f"{artifact_path} contains unsupported classifier={classifier!r}.")

    return JAXObstacleMLP(
        linear_layers=linear_layers,
        layer_norm_scales=layer_norm_scales,
        layer_norm_biases=layer_norm_biases,
        feature_mean=jnp.asarray(artifact["feature_mean"], dtype=jnp.float64),
        feature_std=jnp.maximum(jnp.asarray(artifact["feature_std"], dtype=jnp.float64), 1e-6),
        threshold=jnp.asarray(float(artifact["conformal_safe_score_threshold"]), dtype=jnp.float64),
        input_dim=input_dim,
        activation=activation,
        classifier=classifier,
        spectral_norm=spectral_norm,
    )


def _resolve_state_region_artifact_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    candidates = (
        path / "latent_ellipsoid.pt",
        path / "latent_ellipsoid.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find a latent ellipsoid artifact under {path}. "
        "Expected a file path or a directory containing latent_ellipsoid.pt or latent_ellipsoid.npz."
    )


def build_jax_state_region_from_artifact(path: Path, state_dim: int) -> JAXEllipsoidConstraint:
    artifact_path = _resolve_state_region_artifact_path(path)
    if artifact_path.suffix == ".pt":
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        center = artifact.get("markov_center")
        unit_precision = artifact.get("markov_unit_precision")
        threshold = artifact.get("markov_threshold")
    elif artifact_path.suffix == ".npz":
        artifact = np.load(artifact_path)
        center = artifact["markov_center"]
        unit_precision = artifact["markov_unit_precision"] if "markov_unit_precision" in artifact else None
        threshold = artifact["markov_threshold"] if "markov_threshold" in artifact else None
        if unit_precision is None:
            precision = artifact["markov_precision"]
            if threshold is None:
                raise KeyError(f"{artifact_path} is missing both markov_unit_precision and markov_threshold.")
            unit_precision = precision / float(np.asarray(threshold))
    else:
        raise ValueError(f"Unsupported latent ellipsoid artifact format: {artifact_path.suffix}")

    if center is None or unit_precision is None:
        raise KeyError(
            f"{artifact_path} must contain markov_center and markov_unit_precision "
            "(or markov_precision together with markov_threshold)."
        )
    center = jnp.asarray(_as_numpy(center, dtype=np.float64), dtype=jnp.float64)
    unit_precision = jnp.asarray(_as_numpy(unit_precision, dtype=np.float64), dtype=jnp.float64)
    if center.shape != (state_dim,):
        raise ValueError(f"State-region center shape {center.shape} does not match planner state_dim={state_dim}.")
    if unit_precision.shape != (state_dim, state_dim):
        raise ValueError(
            f"State-region unit_precision shape {unit_precision.shape} does not match planner state_dim={state_dim}."
        )
    return JAXEllipsoidConstraint(center=center, unit_precision=unit_precision, input_dim=state_dim)

def build_equinox_mlp_from_pytorch(pt_model: torch.nn.Module, key: jax.Array) -> eqx.Module:
    pt_linears = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    layers = []
    keys = jax.random.split(key, len(pt_linears))
    for i, pt_layer in enumerate(pt_linears):
        out_f, in_f = pt_layer.weight.shape
        eqx_linear = eqx.nn.Linear(in_f, out_f, key=keys[i])
        w, b = jnp.array(pt_layer.weight.detach().cpu().numpy()), jnp.array(pt_layer.bias.detach().cpu().numpy())
        layers.append(eqx.tree_at(lambda l: (l.weight, l.bias), eqx_linear, (w, b)))
        if i < len(pt_linears) - 1: layers.append(jax.nn.gelu)
    
    class JAXMLP(eqx.Module):
        layers: list
        def __call__(self, x):
            for layer in self.layers: x = layer(x)
            return x
    return JAXMLP(layers)

def make_jax_disturbance(eqx_error_model, q_learned, state_dim, diagonal):
    def _mgnll_forward(raw):
        if diagonal: return jnp.diag(jnp.exp(raw) + 1e-4)
        L = jnp.zeros((state_dim, state_dim))
        L = L.at[jnp.tril_indices(state_dim)].set(raw)
        return L.at[jnp.arange(state_dim), jnp.arange(state_dim)].set(jnp.exp(jnp.diag(L)) + 1e-4)
    return lambda X, U: q_learned * jax.vmap(_mgnll_forward)(jax.vmap(eqx_error_model)(jnp.concatenate([X, U], axis=-1)))

def load_calibrated_cholesky(path: Path) -> np.ndarray:
    payload = torch.load(path.expanduser(), map_location="cpu")
    if "calibrated_cholesky" in payload:
        matrix = payload["calibrated_cholesky"]
    elif "cholesky" in payload and "q_fixed" in payload:
        matrix = payload["cholesky"] * payload["q_fixed"]
    else:
        raise KeyError(f"{path} must contain either 'calibrated_cholesky' or both 'cholesky' and 'q_fixed'.")
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float64)

def make_constant_jax_disturbance(calibrated_cholesky: np.ndarray, state_dim: int):
    calibrated_cholesky = jnp.asarray(calibrated_cholesky, dtype=jnp.float64)
    if calibrated_cholesky.shape != (state_dim, state_dim):
        raise ValueError(f"Expected calibrated Cholesky shape {(state_dim, state_dim)}, got {calibrated_cholesky.shape}.")

    def jax_disturbance(X_prefix, U_prefix):
        return jnp.broadcast_to(calibrated_cholesky, (X_prefix.shape[0], state_dim, state_dim))

    return jax_disturbance

def make_mppi_rollout_and_eval(
    dynamics_fn,
    W_mppi_stage,
    W_mppi_terminal,
    goal_state,
    *,
    obstacle_model: JAXObstacleMLP | None = None,
    state_region: JAXEllipsoidConstraint | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
    state_region_penalty_weight: float = 0.0,
    state_box_min: jnp.ndarray | None = None,
    state_box_max: jnp.ndarray | None = None,
    state_box_penalty_weight: float = 0.0,
    r_control: float = 0.01,
    r_control_u4: float | None = None,
    action_ref: jnp.ndarray | None = None,
):
    if action_ref is None:
        raise ValueError("action_ref must be provided for MPPI action regularization.")
    action_ref = jnp.asarray(action_ref, dtype=jnp.float64)
    action_weights = make_action_weights(
        int(action_ref.shape[0]),
        r_control,
        r_control if r_control_u4 is None else r_control_u4,
    )

    def rollout(state_cur, act_seqs, reach_config=None):
        def step_fn(s, u):
            nxt = dynamics_fn(s, u)
            return nxt, nxt
        return jax.vmap(lambda actions: lax.scan(step_fn, state_cur, actions)[1])(act_seqs), {}
        
    def eval_fn(states, acts, reach_config=None, aux=None, *args, **kwargs):
        delta = states - goal_state[None, None, :]
        stage_costs = jnp.sum(W_mppi_stage[None, None, :] * delta**2, axis=-1)
        terminal_costs = jnp.sum(W_mppi_terminal[None, :] * delta[:, -1, :] ** 2, axis=-1)
        action_delta = acts - action_ref[None, None, :]
        action_costs = jnp.sum(action_weights[None, None, :] * action_delta ** 2, axis=-1)
        if state_region is not None and state_region_penalty_weight > 0.0:
            flat_states = states.reshape((-1, states.shape[-1]))
            state_region_violation = jax.vmap(
                lambda z: jax.nn.softplus(state_region.violation(z))
            )(flat_states).reshape(states.shape[:-1])
            state_region_costs = state_region_penalty_weight * state_region_violation**2
        else:
            state_region_costs = jnp.zeros_like(stage_costs)
        if state_box_min is not None and state_box_max is not None and state_box_penalty_weight > 0.0:
            lower_violation = jnp.maximum(state_box_min[None, None, :] - states, 0.0)
            upper_violation = jnp.maximum(states - state_box_max[None, None, :], 0.0)
            state_box_costs = state_box_penalty_weight * jnp.sum(lower_violation**2 + upper_violation**2, axis=-1)
        else:
            state_box_costs = jnp.zeros_like(stage_costs)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            flat_states = states.reshape((-1, states.shape[-1]))
            obstacle_violation = jax.vmap(
                lambda z: jax.nn.softplus(obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z))
            )(flat_states).reshape(states.shape[:-1])
            obstacle_costs = obstacle_penalty_weight * obstacle_violation**2
        else:
            obstacle_costs = jnp.zeros_like(stage_costs)
        total_cost = jnp.sum(stage_costs + action_costs + state_region_costs + state_box_costs + obstacle_costs, axis=-1) + terminal_costs
        return {"rewards": -total_cost}
    return rollout, eval_fn

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    return lambda x, u, t: jnp.concatenate([u - u_max, u_min - u], axis=0)

def make_state_box_constraints(x_min, x_max):
    x_min, x_max = jnp.asarray(x_min), jnp.asarray(x_max)
    return lambda x, u, t: jnp.concatenate([x - x_max, x_min - x], axis=0)

def combine_constraints(*constraints):
    return lambda x, u, t: jnp.concatenate([constraint(x, u, t) for constraint in constraints], axis=0)

def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    def constraint(x, u, t):
        return jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])
    return constraint


def make_state_region_constraint(state_region: JAXEllipsoidConstraint):
    def constraint(x, u, t):
        return jnp.asarray([state_region.violation(x)], dtype=jnp.float64)
    return constraint

def make_tracking_cost(
    action_weights: jnp.ndarray,
    horizon: int,
    W_stage: jnp.ndarray,
    W_terminal: jnp.ndarray,
    goal_state: jnp.ndarray,
    action_ref: jnp.ndarray,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
):
    action_ref = jnp.asarray(action_ref, dtype=jnp.float64)
    action_weights = jnp.asarray(action_weights, dtype=jnp.float64)
    def cost(W_ignored, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W_stage, W_terminal)
        active_ref = jnp.where(is_not_terminal, reference[t], goal_state)
        state_error = z - active_ref
        action_error = u - action_ref
        total_cost = jnp.sum(active_W * (state_error ** 2)) + jnp.sum(action_weights * action_error ** 2)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            obstacle_violation = jax.nn.softplus(
                obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z)
            )
            total_cost = total_cost + obstacle_penalty_weight * obstacle_violation**2
        return total_cost
    return cost

def make_action_weights(action_dim: int, r_control: float, r_control_u4: float, grip_idx: int = 4) -> jnp.ndarray:
    weights = jnp.ones((int(action_dim),), dtype=jnp.float64) * float(r_control)
    if 0 <= int(grip_idx) < int(action_dim):
        weights = weights.at[int(grip_idx)].set(float(r_control_u4))
    return weights

def planned_action_dim(full_action_dim: int, grip_idx: int = 4) -> int:
    if not 0 <= int(grip_idx) < int(full_action_dim):
        raise ValueError(f"grip_idx={grip_idx} must lie in [0, {full_action_dim - 1}].")
    return int(full_action_dim) - 1

def augment_action_jax(action: jax.Array, fixed_grasp_u4: float, grip_idx: int = 4) -> jax.Array:
    action = jnp.asarray(action, dtype=jnp.float64)
    prefix = action[..., :grip_idx]
    suffix = action[..., grip_idx:]
    fixed = jnp.full(action.shape[:-1] + (1,), float(fixed_grasp_u4), dtype=action.dtype)
    return jnp.concatenate([prefix, fixed, suffix], axis=-1)

def augment_action_np(action: np.ndarray, fixed_grasp_u4: float, grip_idx: int = 4) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    prefix = action[..., :grip_idx]
    suffix = action[..., grip_idx:]
    fixed = np.full(action.shape[:-1] + (1,), float(fixed_grasp_u4), dtype=action.dtype)
    return np.concatenate([prefix, fixed, suffix], axis=-1)

def resolve_fixed_grasp_action(
    cfg: PlanSLSMoppiCubeConfig,
    *,
    oracle_action_raw: np.ndarray | None,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    grip_idx: int = 4,
) -> tuple[float, float]:
    mean = float(np.asarray(action_mean, dtype=np.float64).reshape(-1)[grip_idx])
    std = float(np.asarray(action_std, dtype=np.float64).reshape(-1)[grip_idx])
    if cfg.fixed_grasp_normalized_u4 is not None:
        grasp_norm = float(cfg.fixed_grasp_normalized_u4)
        grasp_raw = grasp_norm * std + mean
        return grasp_norm, grasp_raw
    if cfg.fixed_grasp_raw_u4 is not None:
        grasp_raw = float(cfg.fixed_grasp_raw_u4)
        grasp_norm = (grasp_raw - mean) / std
        return grasp_norm, grasp_raw
    if oracle_action_raw is None:
        return 0.0, mean
    grasp_raw = float(np.asarray(oracle_action_raw, dtype=np.float64).reshape(-1)[grip_idx])
    grasp_norm = (grasp_raw - mean) / std
    return grasp_norm, grasp_raw

def make_action_reference(action_dim: int, u_min: jnp.ndarray, u_max: jnp.ndarray, grip_idx: int = 4) -> jnp.ndarray:
    action_ref = jnp.zeros((action_dim,), dtype=jnp.float64)
    if not 0 <= grip_idx < action_dim:
        return action_ref
    grip_midpoint = 0.5 * (u_min[grip_idx] + u_max[grip_idx])
    return action_ref.at[grip_idx].set(grip_midpoint)

def wrap_reduced_action_dynamics(base_dynamics, fixed_grasp_u4: float, grip_idx: int = 4):
    return lambda x, u, t=0.0, parameter=1.0: base_dynamics(x, augment_action_jax(u, fixed_grasp_u4, grip_idx), t, parameter)

def wrap_reduced_action_disturbance(base_disturbance, fixed_grasp_u4: float, grip_idx: int = 4):
    return lambda X_p, U_p: base_disturbance(X_p, augment_action_jax(U_p, fixed_grasp_u4, grip_idx))

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)

def action_plan_is_valid(
    u0: np.ndarray,
    X_pred: np.ndarray,
    U_pred: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    *,
    bound_tol: float = 1e-4,
) -> bool:
    u0_np = np.asarray(u0, dtype=np.float64)
    X_np = np.asarray(X_pred, dtype=np.float64)
    U_np = np.asarray(U_pred, dtype=np.float64)
    u_min_np = np.asarray(u_min, dtype=np.float64).reshape(1, -1)
    u_max_np = np.asarray(u_max, dtype=np.float64).reshape(1, -1)
    return bool(
        np.all(np.isfinite(u0_np))
        and np.all(np.isfinite(X_np))
        and np.all(np.isfinite(U_np))
        and np.all(U_np >= (u_min_np - bound_tol))
        and np.all(U_np <= (u_max_np + bound_tol))
    )

def append_state_log(
    log_path: Path,
    *,
    step: int,
    state: np.ndarray,
    latent_err: float,
    status: str,
    timings: dict[str, float] | None = None,
    metrics: dict[str, float | int | str | bool] | None = None,
) -> None:
    state_np = np.asarray(state, dtype=np.float64).reshape(-1)
    payload = {
        "step": int(step),
        "latent_err": float(latent_err),
        "status": str(status),
        "state": state_np.tolist(),
    }
    if timings is not None:
        payload["timings"] = {key: float(value) for key, value in timings.items()}
    if metrics is not None:
        payload["metrics"] = metrics
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


@torch.no_grad()
def predict_next_state_torch(
    torch_dynamics_net: torch.nn.Module,
    state_np: np.ndarray,
    action_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    state_t = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(device)
    action_t = torch.from_numpy(np.asarray(action_np, dtype=np.float32)).to(device)
    pred = torch_dynamics_net(torch.cat((state_t, action_t), dim=-1).unsqueeze(0))[0]
    return pred.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def compute_error_ellipsoid_membership(
    *,
    error_np: np.ndarray,
    state_np: np.ndarray,
    action_np: np.ndarray,
    use_constant_covariance: bool,
    calibrated_cholesky: np.ndarray | None,
    error_model: torch.nn.Module | None,
    q_learned: float,
    device: torch.device,
) -> tuple[bool | None, float | None]:
    def _raw_to_cholesky(raw: torch.Tensor, state_dim: int) -> torch.Tensor:
        if raw.ndim == 2:
            return raw
        if raw.ndim != 1:
            raise ValueError(f"Expected raw error-model output to have ndim 1 or 2, got shape {tuple(raw.shape)}.")
        if raw.shape[0] == state_dim:
            return torch.diag(torch.exp(raw) + 1e-4)
        tril_count = state_dim * (state_dim + 1) // 2
        if raw.shape[0] != tril_count:
            raise ValueError(
                f"Cannot interpret raw error-model output of shape {tuple(raw.shape)} for state_dim={state_dim}."
            )
        L = torch.zeros((state_dim, state_dim), dtype=raw.dtype, device=raw.device)
        tril_idx = torch.tril_indices(state_dim, state_dim, device=raw.device)
        L[tril_idx[0], tril_idx[1]] = raw
        diag_idx = torch.arange(state_dim, device=raw.device)
        L[diag_idx, diag_idx] = torch.exp(L[diag_idx, diag_idx]) + 1e-4
        return L

    error_vec = np.asarray(error_np, dtype=np.float64).reshape(-1)
    if use_constant_covariance:
        if calibrated_cholesky is None:
            return None, None
        whitened = np.linalg.solve(calibrated_cholesky, error_vec)
        score = float(np.linalg.norm(whitened))
        return score <= 1.0, score
    if error_model is None:
        return None, None
    state_t = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(device)
    action_t = torch.from_numpy(np.asarray(action_np, dtype=np.float32)).to(device)
    inp = torch.cat((state_t, action_t), dim=-1).unsqueeze(0)
    L = _raw_to_cholesky(error_model(inp)[0], error_vec.shape[0])
    err_t = torch.from_numpy(error_vec.astype(np.float32)).to(device).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(L, err_t, upper=False).squeeze(-1)
    score = float(torch.linalg.vector_norm(whitened, ord=2).item() / max(float(q_learned), 1e-12))
    return score <= 1.0, score


def state_region_membership(
    state_np: np.ndarray,
    state_region: JAXEllipsoidConstraint | None,
) -> tuple[bool | None, float | None]:
    if state_region is None:
        return None, None
    score = float(state_region.score(jnp.asarray(state_np, dtype=jnp.float64)))
    return score <= 1.0, score


def obstacle_safety_status(
    state_np: np.ndarray,
    obstacle_model: JAXObstacleMLP | None,
    margin: float,
) -> tuple[bool | None, float | None, float | None]:
    if obstacle_model is None:
        return None, None, None
    score = float(obstacle_model(jnp.asarray(state_np, dtype=jnp.float64)))
    required = float(obstacle_model.threshold) + float(margin)
    return score > required, score, required


def true_gripper_height_safety_status(info: dict, threshold: float) -> tuple[bool, float, float]:
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
    gripper_height = float(effector_pos[2])
    return gripper_height < float(threshold), gripper_height, float(threshold)


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def compute_cube_task_metrics(
    info: dict,
    *,
    contact_thresh: float,
    align_thresh: float,
) -> dict[str, float | bool | list[float]]:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float64)
    target_pos = np.asarray(info["privileged/target_block_pos"], dtype=np.float64)
    block_yaw = float(np.asarray(info[f"privileged/block_{target_block}_yaw"], dtype=np.float64).reshape(-1)[0])
    target_yaw = float(np.asarray(info["privileged/target_block_yaw"], dtype=np.float64).reshape(-1)[0])
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
    gripper_height = float(effector_pos[2])
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float64).reshape(-1)[0])
    pos_error_vec = block_pos - target_pos
    pos_error_norm = float(np.linalg.norm(pos_error_vec))
    yaw_error = wrap_to_pi(block_yaw - target_yaw)
    effector_block_dist = float(np.linalg.norm(block_pos - effector_pos))
    grasped = bool(gripper_contact >= float(contact_thresh) and effector_block_dist <= float(align_thresh))
    return {
        "block_pos": block_pos.tolist(),
        "target_block_pos": target_pos.tolist(),
        "block_yaw": block_yaw,
        "target_block_yaw": target_yaw,
        "block_pos_error_vec": pos_error_vec.tolist(),
        "block_pos_error_norm": pos_error_norm,
        "block_yaw_error": yaw_error,
        "block_pose_error_l2": float(np.sqrt(pos_error_norm**2 + yaw_error**2)),
        "effector_pos": effector_pos.tolist(),
        "gripper_height": gripper_height,
        "effector_block_distance": effector_block_dist,
        "gripper_contact": gripper_contact,
        "grasped": grasped,
        "reached_ogbench_success": bool(ogbench_success(info)),
    }

def resolve_action_stats_dataset_path(cfg: PlanSLSMoppiCubeConfig) -> Path:
    return cfg.action_stats_dataset_path if cfg.action_stats_dataset_path is not None else cfg.dataset_path

def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array

def _pick_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Expected one of keys {names}, got {sorted(mapping.keys())}.")

def _pick_optional_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None

def _pick_optional_or(mapping: dict, names: tuple[str, ...], default):
    value = _pick_optional_key(mapping, names)
    return default if value is None else value

def _infer_pair_count(payload) -> Optional[int]:
    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and "pair_count" in metadata:
            return int(metadata["pair_count"])
        if "pair_count" in payload:
            return int(payload["pair_count"])
        for key in ("pairs", "episodes", "endpoint_pairs"):
            if key in payload and isinstance(payload[key], (list, tuple)):
                return len(payload[key])
        if "start" in payload and ("goal" in payload or "end" in payload or "target" in payload):
            start = payload["start"]
            if isinstance(start, dict):
                for item in start.values():
                    if isinstance(item, (torch.Tensor, np.ndarray)) and item.ndim > 0:
                        return int(item.shape[0])
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return None

def _select_pair_value(value, episode_idx: int, pair_count: Optional[int]):
    if isinstance(value, dict):
        return {key: _select_pair_value(item, episode_idx, pair_count) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and pair_count is not None and len(value) == pair_count:
        return value[episode_idx]
    if isinstance(value, torch.Tensor) and pair_count is not None and value.ndim > 0 and int(value.shape[0]) == pair_count:
        return value[episode_idx]
    if isinstance(value, np.ndarray) and pair_count is not None and value.ndim > 0 and int(value.shape[0]) == pair_count:
        return value[episode_idx]
    return value

def _select_endpoint_pair(payload, episode_idx: int):
    pair_count = _infer_pair_count(payload)
    if pair_count is not None and not 0 <= episode_idx < pair_count:
        raise ValueError(f"episode_idx must be in [0, {pair_count - 1}], got {episode_idx}.")
    if isinstance(payload, (list, tuple)):
        return payload[episode_idx], pair_count
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported endpoint payload type: {type(payload)!r}.")
    for key in ("pairs", "episodes", "endpoint_pairs"):
        if key in payload:
            pairs = payload[key]
            if not isinstance(pairs, (list, tuple)):
                raise TypeError(f"Endpoint payload key '{key}' must be a list/tuple, got {type(pairs)!r}.")
            return pairs[episode_idx], len(pairs)
    if "start" in payload and ("goal" in payload or "end" in payload or "target" in payload):
        return {
            "start": _select_pair_value(payload["start"], episode_idx, pair_count),
            "goal": _select_pair_value(_pick_key(payload, ("goal", "end", "target")), episode_idx, pair_count),
        }, pair_count
    raise KeyError(
        "Endpoint .pt payload must be a list of pairs, contain a 'pairs'/'episodes' list, "
        "or contain top-level 'start' and 'goal'/'end'/'target' entries."
    )

def _load_endpoint_pair(path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, np.ndarray | int | str], int]:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    pair_count = _infer_pair_count(payload)
    rng = np.random.default_rng(seed)
    if episode_idx is None:
        episode_idx = 0 if pair_count is None else int(rng.integers(pair_count))
    pair, pair_count = _select_endpoint_pair(payload, int(episode_idx))
    if not isinstance(pair, dict):
        raise TypeError(f"Selected endpoint pair must be a dict, got {type(pair)!r}.")

    start = _pick_key(pair, ("start", "initial", "source"))
    goal = _pick_key(pair, ("goal", "end", "target"))
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise TypeError("Endpoint pair 'start' and 'goal'/'end' entries must both be dicts.")

    start_qpos_raw = _pick_optional_key(start, ("qpos", "q_pos"))
    goal_qpos_raw = _pick_optional_key(goal, ("qpos", "q_pos"))
    start_block_pos = _as_numpy(_pick_key(start, ("task_target", "block_pos", "object_pos", "pos")), dtype=np.float32)
    goal_block_pos = _as_numpy(_pick_key(goal, ("task_target", "block_pos", "object_pos", "pos")), dtype=np.float32)
    start_block_yaw = float(_as_numpy(_pick_key(start, ("yaw", "block_yaw", "object_yaw"))).reshape(-1)[0])
    goal_block_yaw = float(_as_numpy(_pick_key(goal, ("yaw", "block_yaw", "object_yaw"))).reshape(-1)[0])
    start_qpos = None if start_qpos_raw is None else _as_numpy(start_qpos_raw, dtype=np.float32)
    goal_qpos = None if goal_qpos_raw is None else _as_numpy(goal_qpos_raw, dtype=np.float32)
    start_qvel_raw = _pick_optional_key(start, ("qvel", "q_vel"))
    goal_qvel_raw = _pick_optional_key(goal, ("qvel", "q_vel"))
    start_pixels_raw = _pick_optional_key(start, ("pixels", "image", "rgb"))
    goal_pixels_raw = _pick_optional_key(goal, ("pixels", "image", "rgb"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    episode = {
        "needs_qpos_synthesis": start_qpos is None or goal_qpos is None,
        "block_pos_init": start_block_pos,
        "block_yaw_init": start_block_yaw,
        "block_pos_goal": goal_block_pos,
        "block_yaw_goal": goal_block_yaw,
        "qpos_init": start_qpos,
        "qvel_init": None if start_qpos is None else (np.zeros_like(start_qpos, dtype=np.float32) if start_qvel_raw is None else _as_numpy(start_qvel_raw, dtype=np.float32)),
        "qpos_goal": goal_qpos,
        "qvel_goal": None if goal_qpos is None else (np.zeros_like(goal_qpos, dtype=np.float32) if goal_qvel_raw is None else _as_numpy(goal_qvel_raw, dtype=np.float32)),
        "target_block_pos_init": _as_numpy(_pick_optional_or(start, ("target_block_pos", "privileged/target_block_pos", "target_pos", "goal_pos"), goal_block_pos), dtype=np.float32),
        "target_block_yaw_init": float(_as_numpy(_pick_optional_or(start, ("target_block_yaw", "privileged/target_block_yaw", "target_yaw", "goal_yaw"), goal_block_yaw)).reshape(-1)[0]),
        "target_block_pos_goal": _as_numpy(_pick_optional_or(goal, ("target_block_pos", "privileged/target_block_pos", "target_pos", "goal_pos"), goal_block_pos), dtype=np.float32),
        "target_block_yaw_goal": float(_as_numpy(_pick_optional_or(goal, ("target_block_yaw", "privileged/target_block_yaw", "target_yaw", "goal_yaw"), goal_block_yaw)).reshape(-1)[0]),
        "start_pixels": None if start_pixels_raw is None else _as_numpy(start_pixels_raw, dtype=np.uint8),
        "goal_pixels": None if goal_pixels_raw is None else _as_numpy(goal_pixels_raw, dtype=np.uint8),
        "episode_seed": int(metadata.get("episode_seed", seed)) if isinstance(metadata, dict) else int(seed),
        "env_name": str(metadata.get("env_name", "cube-single-v0")) if isinstance(metadata, dict) else "cube-single-v0",
        "camera": str(metadata.get("camera", "front_pixels")) if isinstance(metadata, dict) else "front_pixels",
    }
    print(
        f"Loaded endpoint pair {episode_idx}"
        + (f"/{pair_count}" if pair_count is not None else "")
        + f" from {path}"
    )
    return episode, int(episode_idx)

def synthesize_qpos_qvel_from_block_pose(env: gymnasium.Env, pos: np.ndarray, yaw: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    joint_qpos = unwrapped._data.joint("object_joint_0").qpos
    joint_qpos[:3] = np.asarray(pos, dtype=np.float64)
    joint_qpos[3:] = np.asarray(lie.SO3.from_z_radians(float(yaw)).wxyz, dtype=np.float64)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    return (
        np.asarray(unwrapped._data.qpos, dtype=np.float32).copy(),
        np.zeros_like(np.asarray(unwrapped._data.qvel, dtype=np.float32)),
    )

def load_planning_episode(path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, np.ndarray | int | str], int]:
    path = path.expanduser()
    if path.suffix.lower() == ".pt":
        return _load_endpoint_pair(path, episode_idx, seed)
    with h5py.File(path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        selected_idx = episode_idx if episode_idx is not None else int(np.random.choice(np.flatnonzero(ep_len >= 2)))
        rows = np.arange(int(h5["ep_offset"][selected_idx]), int(h5["ep_offset"][selected_idx]) + int(h5["ep_len"][selected_idx]))
        episode = {
            "qpos_init": np.asarray(h5["qpos"][rows[0]], dtype=np.float32),
            "qvel_init": np.asarray(h5["qvel"][rows[0]], dtype=np.float32),
            "qpos_goal": np.asarray(h5["qpos"][rows[-1]], dtype=np.float32),
            "qvel_goal": np.asarray(h5["qvel"][rows[-1]], dtype=np.float32),
            "start_pixels": np.asarray(h5["pixels"][rows[0]], dtype=np.uint8),
            "goal_pixels": np.asarray(h5["pixels"][rows[-1]], dtype=np.uint8),
            "target_block_pos_init": np.asarray(h5["target_block_pos"][rows[0]], dtype=np.float32),
            "target_block_yaw_init": float(h5["target_block_yaw"][rows[0], 0]),
            "target_block_pos_goal": np.asarray(h5["target_block_pos"][rows[-1]], dtype=np.float32),
            "target_block_yaw_goal": float(h5["target_block_yaw"][rows[-1], 0]),
            "episode_seed": int(h5["episode_seed"][selected_idx]) if "episode_seed" in h5 else int(seed),
            "env_name": str(h5.attrs.get("env_name", "cube-single-v0")),
            "camera": str(h5.attrs.get("camera", "front_pixels")),
        }
    return episode, int(selected_idx)

def cube_is_grasped(info, contact_thresh, align_thresh) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_thresh and block_alignment <= align_thresh)

def ogbench_success(info: dict) -> bool:
    success = info.get("success", False)
    if isinstance(success, dict):
        return all(bool(value) for value in success.values())
    return bool(np.asarray(success).item())

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch[_=](\d+).*\.ckpt$")
    candidates = []
    for path in model_dir.glob("*.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No valid checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

# --- Target Cube Sync & Hiding Utilities ---
def hide_target_cube(env: gymnasium.Env) -> None:
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0

def restore_target_pose(env: gymnasium.Env, target_block_pos: np.ndarray, target_block_yaw: float) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )
    hide_target_cube(env)

def render_without_target_cube(env: gymnasium.Env, camera: str) -> np.ndarray:
    hide_target_cube(env)
    return np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)

def reset_env_to_state(env: gymnasium.Env, seed: int, qpos: np.ndarray, qvel: np.ndarray, target_block_pos: np.ndarray, target_block_yaw: float, camera: str) -> tuple[np.ndarray, dict]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    unwrapped._data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    restore_target_pose(env, target_block_pos=target_block_pos, target_block_yaw=target_block_yaw)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    frame = render_without_target_cube(env, camera)
    info = unwrapped.get_step_info()
    return frame, info

# --- Vision Frame Encoding Utilities ---
@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel_np: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(pixel_np.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - pixel_mean.to(tensor.device)) / pixel_std.to(tensor.device)
    output = model.encoder(tensor.to(device), interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - pixel_mean.to(tensor.device)) / pixel_std.to(tensor.device)
    
    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)

def main():
    cfg = pyrallis.parse(config_class=PlanSLSMoppiCubeConfig)
    if cfg.mppi_state_region_penalty is None:
        cfg.mppi_state_region_penalty = float(cfg.mppi_state_box_penalty)
    if cfg.mppi_horizon is None:
        cfg.mppi_horizon = int(cfg.horizon)
    if int(cfg.mppi_horizon) < int(cfg.horizon):
        raise ValueError(f"mppi_horizon={cfg.mppi_horizon} must be >= horizon={cfg.horizon}.")
    if cfg.mppi_obstacle_margin is None:
        cfg.mppi_obstacle_margin = float(cfg.obstacle_margin)
    if cfg.mppi_q_stage_pos is None:
        cfg.mppi_q_stage_pos = float(cfg.mppi_q_stage)
    if cfg.mppi_q_stage_vel is None:
        cfg.mppi_q_stage_vel = 1.0
    cfg.mppi_recede_min_horizon = max(int(cfg.horizon), int(cfg.mppi_recede_min_horizon))
    if cfg.q_stage_pos is None:
        cfg.q_stage_pos = float(cfg.q_stage)
    if cfg.q_stage_vel is None:
        cfg.q_stage_vel = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_root = cfg.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json") as f: config_dict = json.load(f)
    
    checkpoint_path = latest_object_checkpoint(model_dir)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).eval()
    
    state_dim = int(config_dict.get("markov_state_dim", 48))
    action_dim = int(config_dict.get("action_dim", 5))
    control_dim = planned_action_dim(action_dim)
    img_size = config_dict.get("img_size", 224)

    init_key = jax.random.PRNGKey(cfg.seed)
    k1, k2, k3 = jax.random.split(init_key, 3)
    eqx_dyn = build_equinox_mlp_from_pytorch(model.predictor.net, k1)
    full_dynamics = lambda x, u, t=0.0, parameter=1.0: eqx_dyn(jnp.concatenate([x, u], axis=-1))
    obstacle_model = None
    obstacle_constraint = None
    state_region = None
    state_region_constraint = None
    state_box_constraint = None
    state_box_constraint_count = 0
    error_model_torch = None
    calibrated_cholesky = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, k3)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(f"Obstacle classifier input_dim={obstacle_model.input_dim} exceeds planner state_dim={state_dim}.")
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(
            f"Using conformal obstacle classifier from {cfg.obstacle_model_path} "
            f"with threshold {float(obstacle_model.threshold):.6g}, margin {cfg.obstacle_margin:.6g}, "
            f"classifier={obstacle_model.classifier}, spectral_norm={obstacle_model.spectral_norm}"
        )
    else:
        print("Obstacle avoidance disabled.")
    if cfg.state_region_path is not None:
        state_region = build_jax_state_region_from_artifact(cfg.state_region_path, state_dim)
        state_region_constraint = make_state_region_constraint(state_region)
        print(f"Using conformal Markov state region from {cfg.state_region_path}")
    else:
        print("Conformal latent state-region constraint disabled.")
    state_box_min = None
    state_box_max = None
    if cfg.state_abs_limit is not None:
        state_box_min = -float(cfg.state_abs_limit) * jnp.ones(state_dim, dtype=jnp.float64)
        state_box_max = float(cfg.state_abs_limit) * jnp.ones(state_dim, dtype=jnp.float64)
        state_box_constraint = make_state_box_constraints(state_box_min, state_box_max)
        state_box_constraint_count = 2 * state_dim
        print(f"Using Markov state box constraints with abs limit {float(cfg.state_abs_limit):.6g}.")

    if cfg.use_constant_covariance:
        calibrated_cholesky = load_calibrated_cholesky(cfg.constant_covariance_path)
        disturbance = make_constant_jax_disturbance(calibrated_cholesky, state_dim)
        print(f"Using fixed calibrated covariance disturbance from {cfg.constant_covariance_path}")
    else:
        error_model_torch = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
        disturbance = make_jax_disturbance(build_equinox_mlp_from_pytorch(error_model_torch.net, k2), cfg.q_learned, state_dim, error_model_torch.diagonal)

    episode, episode_idx = load_planning_episode(cfg.dataset_path, cfg.episode_idx, cfg.seed)
    out_dir = out_root / f"{int(time.time())}_mppi_sls_cube_episode_{episode_idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    qpos_init = episode["qpos_init"]
    qvel_init = episode["qvel_init"]
    qpos_goal = episode["qpos_goal"]
    qvel_goal = episode["qvel_goal"]
    target_block_pos_init = episode["target_block_pos_init"]
    target_block_yaw_init = float(episode["target_block_yaw_init"])
    target_block_pos_goal = episode["target_block_pos_goal"]
    target_block_yaw_goal = float(episode["target_block_yaw_goal"])
    start_pixels = episode.get("start_pixels")
    goal_pixels = episode.get("goal_pixels")
    episode_seed = int(episode["episode_seed"])
    env_name = str(episode["env_name"])
    camera = str(episode["camera"])

    env = gymnasium.make(
        env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=cfg.visualize_success_colors,
        width=256,
        height=256,
    )
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0)

    if bool(episode.get("needs_qpos_synthesis", False)):
        qpos_init, qvel_init = synthesize_qpos_qvel_from_block_pose(
            env,
            np.asarray(episode["block_pos_init"], dtype=np.float32),
            float(episode["block_yaw_init"]),
            episode_seed,
        )
        qpos_goal, qvel_goal = synthesize_qpos_qvel_from_block_pose(
            env,
            np.asarray(episode["block_pos_goal"], dtype=np.float32),
            float(episode["block_yaw_goal"]),
            episode_seed,
        )

    goal_frame, _ = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_goal,
        qvel=qvel_goal,
        target_block_pos=target_block_pos_goal,
        target_block_yaw=target_block_yaw_goal,
        camera=camera,
    )

    current_frame, current_info = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_init,
        qvel=qvel_init,
        target_block_pos=target_block_pos_init,
        target_block_yaw=target_block_yaw_init,
        camera=camera,
    )
    start_image = np.asarray(start_pixels, dtype=np.uint8).copy() if start_pixels is not None else current_frame.copy()
    goal_image = np.asarray(goal_pixels, dtype=np.uint8).copy() if goal_pixels is not None else goal_frame.copy()

    save_rgb_image(out_dir / "start_image.png", start_image)
    save_rgb_image(out_dir / "goal_image.png", goal_image)

    pixel_mean, pixel_std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    goal_emb = encode_single_frame(model, goal_image, device, img_size, pixel_mean, pixel_std)
    goal_state = torch.cat([goal_emb, torch.zeros_like(goal_emb)], dim=-1).cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    grasped = False
    last_oracle_action_raw = None

    oracle.reset(None, current_info)
    
    # Run analytical planner tracking stages until containment validation triggers
    for _ in range(cfg.max_oracle_steps):
        if cube_is_grasped(current_info, cfg.grasp_contact_threshold, cfg.grasp_alignment_threshold): grasped = True; break
        oracle_action = np.asarray(oracle.select_action(None, current_info), dtype=np.float32)
        last_oracle_action_raw = oracle_action.copy()
        current_info = env.step(oracle_action)[4]
        rollout_frames.append(render_without_target_cube(env, camera))

    if not grasped: return env.close()

    action_stats_dataset_path = resolve_action_stats_dataset_path(cfg)
    train_stats = LeWMOGBenchCubeDataset(
        str(action_stats_dataset_path),
        markov_deriv=1,
        num_preds=1,
        frameskip=1,
        img_size=img_size,
        action_dim=action_dim
    )
    action_mean = train_stats.action_mean.astype(np.float32)
    action_std = train_stats.action_std.astype(np.float32)
    print(f"Using action statistics from {action_stats_dataset_path}")
    fixed_grasp_norm, fixed_grasp_raw = resolve_fixed_grasp_action(
        cfg,
        oracle_action_raw=last_oracle_action_raw,
        action_mean=action_mean,
        action_std=action_std,
    )
    dynamics = wrap_reduced_action_dynamics(full_dynamics, fixed_grasp_norm)
    disturbance = wrap_reduced_action_disturbance(disturbance, fixed_grasp_norm)
    print(
        "Using fixed post-oracle grasp action: "
        f"u4_norm={fixed_grasp_norm:.6g}, u4_raw={fixed_grasp_raw:.6g}"
    )

    W_mppi_stage = jnp.ones((state_dim,)) * float(cfg.mppi_q_stage_vel)
    W_mppi_stage = W_mppi_stage.at[: state_dim // 2].set(float(cfg.mppi_q_stage_pos))
    W_mppi_terminal = jnp.ones((state_dim,)) * cfg.mppi_q_terminal
    W_mppi_terminal = W_mppi_terminal.at[state_dim // 2:].set(1.0)
    W_stage_scaled = jnp.ones((state_dim,)) * float(cfg.q_stage_vel)
    W_stage_scaled = W_stage_scaled.at[: state_dim // 2].set(float(cfg.q_stage_pos))
    W_terminal_scaled = jnp.ones((state_dim,)) * cfg.q_terminal
    W_terminal_scaled = W_terminal_scaled.at[state_dim // 2:].set(1.0)

    u_min, u_max = -2.0 * jnp.ones(control_dim), 2.0 * jnp.ones(control_dim)
    action_ref = jnp.zeros((control_dim,), dtype=jnp.float64)
    mppi_roll, mppi_ev = make_mppi_rollout_and_eval(
        dynamics,
        W_mppi_stage,
        W_mppi_terminal,
        jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        state_region=state_region,
        obstacle_margin=cfg.mppi_obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
        state_region_penalty_weight=(
            float(cfg.mppi_state_region_penalty) if state_region is not None else 0.0
        ),
        state_box_min=state_box_min,
        state_box_max=state_box_max,
        state_box_penalty_weight=float(cfg.mppi_state_box_penalty),
        r_control=cfg.mppi_r_control,
        r_control_u4=cfg.mppi_r_control_u4,
        action_ref=action_ref,
    )
    
    def build_mppi_runner(active_horizon: int):
        planner = MPPIPlanner(
            config={"planning": {"action_dim": control_dim, "n_sample": cfg.mppi_samples, "horizon": int(active_horizon), "n_update_iter": cfg.mppi_update_iter, "use_last": True, "reject_bad": False, "mppi": {"reward_weight": cfg.mppi_reward_weight, "noise_level": cfg.mppi_noise_level, "noise_decay": 1.0, "beta_filter": cfg.mppi_beta_filter}}},
            model_rollout_fn=mppi_roll,
            evaluate_traj_fn=mppi_ev,
            action_lower_lim=u_min,
            action_upper_lim=u_max,
        )

        @eqx.filter_jit
        def run_mppi_opt(key_arg, state_arg, actions_arg):
            return planner.trajectory_optimization(key_arg, state_arg, actions_arg, skip=False)

        return run_mppi_opt

    action_weights = make_action_weights(control_dim, cfg.r_control, cfg.r_control_u4)
    cost = make_tracking_cost(
        action_weights=action_weights, 
        horizon=cfg.horizon, 
        W_stage=W_stage_scaled, 
        W_terminal=W_terminal_scaled,
        goal_state=jnp.asarray(goal_state),
        action_ref=action_ref,
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
    )

    constraints_all = combine_constraints(
        make_control_box_constraints(u_min, u_max),
        *(() if state_region_constraint is None else (state_region_constraint,)),
        *(() if state_box_constraint is None else (state_box_constraint,)),
        *(() if obstacle_constraint is None else (obstacle_constraint,)),
    )

    u_init = jnp.zeros((cfg.horizon, control_dim))

    controller = GenericMPC(
        SLSConfig(max_sls_iterations=1, enable_fastsls=False, initialize_nominal=True, warm_start=True, rti=True),
        SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=5e-2, step_tol=1e-4, line_search=True),
        ADMMConfig(eps_abs=5e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=300, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=control_dim, N=cfg.horizon, W=W_stage_scaled, u_ref=action_ref, dt=1.0/20.0),
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)),
        cost=cost,
        num_constraints=(
            2 * control_dim
            + (1 if state_region_constraint is not None else 0)
            + state_box_constraint_count
            + (1 if obstacle_constraint is not None else 0)
        ),
        disturbance=disturbance, shift=1, X_in=jnp.zeros((cfg.horizon + 1, state_dim)), U_in=u_init
    )

    current_emb = encode_single_frame(model, rollout_frames[-1], device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    state_log_path = out_dir / "latent_state_log.jsonl"
    prev_U = jnp.zeros((cfg.mppi_horizon, control_dim))
    active_mppi_horizon = int(cfg.mppi_horizon)
    run_mppi_opt = build_mppi_runner(active_mppi_horizon)
    jax_seed = jax.random.PRNGKey(cfg.seed)

    if obstacle_model is not None:
        handoff_safe, handoff_height, height_threshold = true_gripper_height_safety_status(
            current_info, cfg.gripper_height_threshold
        )
        handoff_score = float(obstacle_model(jnp.asarray(current_state)))
        goal_score = float(obstacle_model(jnp.asarray(goal_state)))
        required_score = float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        print(
            "True gripper-height safety checks will be applied to executed states after oracle grasp. "
            f"Handoff_height={handoff_height:.6g}, required_height<{height_threshold:.6g}, "
            f"handoff_safe={handoff_safe}. "
            "Classifier proxy scores for reference: "
            f"handoff_score={handoff_score:.6g}, goal_score={goal_score:.6g}, required_score>{required_score:.6g}"
        )
    if state_region is not None:
        start_region_score = float(state_region.score(jnp.asarray(current_state)))
        goal_region_score = float(state_region.score(jnp.asarray(goal_state)))
        if start_region_score > 1.0 or goal_region_score > 1.0:
            print(
                "Terminating: start and goal must both lie inside the conformal Markov state region. "
                f"Required score <= 1; start_score={start_region_score:.6g}, goal_score={goal_region_score:.6g}."
            )
            env.close()
            return
        print(
            "State-region sanity check passed: "
            f"start_score={start_region_score:.6g}, goal_score={goal_region_score:.6g}, required_score<=1"
        )

    prev_u0 = np.zeros(control_dim, dtype=np.float32)
    executed_actions_plan_norm: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_raw: list[np.ndarray] = []
    executed_states: list[np.ndarray] = [current_state.copy()]
    executed_embeddings: list[np.ndarray] = [current_emb.detach().cpu().numpy().astype(np.float64).copy()]
    initial_task_metrics = compute_cube_task_metrics(
        current_info,
        contact_thresh=cfg.grasp_contact_threshold,
        align_thresh=cfg.grasp_alignment_threshold,
    )
    task_errors: list[dict[str, float | bool | list[float]]] = [initial_task_metrics]
    latent_goal_distances: list[float] = [float(np.linalg.norm(current_state - goal_state))]
    reached_success_flags: list[bool] = [bool(initial_task_metrics["reached_ogbench_success"])]
    initial_obstacle_free, initial_obstacle_score, initial_obstacle_required = true_gripper_height_safety_status(
        current_info, cfg.gripper_height_threshold
    )
    (
        initial_classifier_obstacle_free,
        initial_classifier_obstacle_score,
        initial_classifier_obstacle_required,
    ) = obstacle_safety_status(current_state, obstacle_model, cfg.obstacle_margin)
    initial_state_region_ok, initial_state_region_score = state_region_membership(current_state, state_region)
    step_records = [
        {
            "step": 0,
            "phase": "handoff_state",
            "markov_state": current_state.astype(np.float64).tolist(),
            "embedding": executed_embeddings[0].tolist(),
            "latent_goal_error": float(latent_goal_distances[0]),
            "task_metrics": initial_task_metrics,
            "state_in_latent_ellipsoid": initial_state_region_ok,
            "state_latent_ellipsoid_score": initial_state_region_score,
            "obstacle_free": initial_obstacle_free,
            "obstacle_check_source": "true_gripper_height",
            "gripper_height": initial_obstacle_score,
            "gripper_height_threshold": initial_obstacle_required,
            "classifier_obstacle_free": initial_classifier_obstacle_free,
            "classifier_obstacle_score": initial_classifier_obstacle_score,
            "classifier_obstacle_required_score": initial_classifier_obstacle_required,
            "one_step_prediction_error": None,
            "one_step_error_in_disturbance_ellipsoid": None,
            "one_step_error_disturbance_score": None,
            "solver_status": "handoff_state",
            "timings_sec": {
                "vit_encode": 0.0,
                "mppi_run": 0.0,
                "sls_solve": 0.0,
                "total": 0.0,
            },
        }
    ]
    stop_reason = "max_mpc_steps"
    append_state_log(
        state_log_path,
        step=-1,
        state=current_state,
        latent_err=float(np.linalg.norm(current_state - goal_state)),
        status="handoff_state",
        metrics={
            "obs_free": initial_obstacle_free,
            "gripper_height": initial_obstacle_score,
            "gripper_height_threshold": initial_obstacle_required,
            "obstacle_check_source": "true_gripper_height",
            "classifier_obstacle_free": initial_classifier_obstacle_free
            if initial_classifier_obstacle_free is not None
            else "n/a",
        },
    )

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="Refined MPPI + SLS tracking loops")
    for step_idx in mpc_pbar:
        jax_seed, subkey = jax.random.split(jax_seed)
        init_act_seq = jnp.concatenate([prev_U[1:active_mppi_horizon], prev_U[active_mppi_horizon - 1 : active_mppi_horizon]], axis=0)
        pre_step_state = current_state.copy()
        
        mppi_ok = False
        mppi_time_s = 0.0
        X_ws = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
        U_ws = init_act_seq[: cfg.horizon]
        try:
            mppi_start = time.perf_counter()
            mppi_res = run_mppi_opt(subkey, jnp.asarray(current_state), init_act_seq)
            mppi_time_s = time.perf_counter() - mppi_start
            X_mppi = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
            U_mppi = jnp.asarray(mppi_res["act_seq"])
            if np.all(np.isfinite(np.asarray(X_mppi))) and np.all(np.isfinite(np.asarray(U_mppi))):
                X_ws = X_mppi[: cfg.horizon + 1]
                U_ws = U_mppi[: cfg.horizon]
                mppi_ok = True
        except Exception:
            pass
        
        controller.X_in, controller.U_in = X_ws, U_ws

        sls_solve_time_s = 0.0
        try:
            sls_start = time.perf_counter()
            u0, X_pred, U_pred, *_ = controller.run(x0=current_state, reference=X_ws, parameter=1.0/20.0)
            sls_solve_time_s = time.perf_counter() - sls_start
            status = "sls_refined" if mppi_ok else "sls_mpc"
        except Exception:
            u0, X_pred, U_pred = None, None, None
            status = "exception_fallback"

        if u0 is None or X_pred is None or U_pred is None:
            u0, X_pred, U_pred = None, None, None
        elif not action_plan_is_valid(u0, X_pred, U_pred, u_min, u_max):
            u0, X_pred, U_pred = None, None, None
            status = "invalid_sls_fallback"

        if u0 is None:
            if mppi_ok:
                u0 = np.asarray(U_ws[0], dtype=np.float32)
                prev_u0 = u0
                prev_U = prev_U.at[:active_mppi_horizon].set(jnp.asarray(U_mppi[:active_mppi_horizon]))
                status = f"{status}|mppi_fallback"
            else:
                u0 = prev_u0
                status = f"{status}|frozen"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)
            prev_U = prev_U.at[: cfg.horizon].set(jnp.asarray(U_pred))

        u0_plan_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
        u0_norm = augment_action_np(u0_plan_norm, fixed_grasp_norm)
        u_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        predicted_next_state = predict_next_state_torch(model.predictor.net, pre_step_state, u0_norm, device)
        current_info = env.step(u_raw)[4]
        executed_actions_plan_norm.append(u0_plan_norm.astype(np.float32))
        executed_actions_norm.append(u0_norm.astype(np.float32))
        executed_actions_raw.append(u_raw.astype(np.float32))
        reached_ogbench_success = ogbench_success(current_info)
        
        frame = render_without_target_cube(env, camera)
        rollout_frames.append(frame)

        encode_start = time.perf_counter()
        next_emb = encode_single_frame(model, frame, device, img_size, pixel_mean, pixel_std)
        encode_time_s = time.perf_counter() - encode_start
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb
        executed_states.append(current_state.copy())
        executed_embeddings.append(current_emb.detach().cpu().numpy().astype(np.float64).copy())

        lat_err = float(np.linalg.norm(current_state - goal_state))
        task_metric = compute_cube_task_metrics(
            current_info,
            contact_thresh=cfg.grasp_contact_threshold,
            align_thresh=cfg.grasp_alignment_threshold,
        )
        task_errors.append(task_metric)
        reached_success_flags.append(bool(task_metric["reached_ogbench_success"]))
        one_step_error = current_state - predicted_next_state
        err_in_ellipsoid, err_ellipsoid_score = compute_error_ellipsoid_membership(
            error_np=one_step_error,
            state_np=pre_step_state,
            action_np=u0_norm,
            use_constant_covariance=cfg.use_constant_covariance,
            calibrated_cholesky=calibrated_cholesky,
            error_model=error_model_torch.net if error_model_torch is not None else None,
            q_learned=cfg.q_learned,
            device=device,
        )
        obstacle_free, obstacle_score, obstacle_required = true_gripper_height_safety_status(
            current_info, cfg.gripper_height_threshold
        )
        classifier_obstacle_free, classifier_obstacle_score, classifier_obstacle_required = obstacle_safety_status(
            current_state,
            obstacle_model,
            cfg.obstacle_margin,
        )
        state_region_ok, state_region_score = state_region_membership(current_state, state_region)
        latent_goal_distances.append(lat_err)
        if reached_ogbench_success:
            status = "ogbench_success"
        if (
            cfg.mppi_goal_epsilon is not None
            and float(cfg.mppi_goal_epsilon) > 0.0
            and lat_err <= float(cfg.mppi_goal_epsilon)
            and active_mppi_horizon > cfg.mppi_recede_min_horizon
        ):
            active_mppi_horizon -= 1
            run_mppi_opt = build_mppi_runner(active_mppi_horizon)
        step_compute_time_s = encode_time_s + mppi_time_s + sls_solve_time_s
        step_records.append(
            {
                "step": int(step_idx + 1),
                "phase": "post_step",
                "markov_state": current_state.astype(np.float64).tolist(),
                "embedding": executed_embeddings[-1].tolist(),
                "latent_goal_error": float(lat_err),
                "task_metrics": task_metric,
                "state_in_latent_ellipsoid": state_region_ok,
                "state_latent_ellipsoid_score": state_region_score,
                "obstacle_free": obstacle_free,
                "obstacle_check_source": "true_gripper_height",
                "gripper_height": obstacle_score,
                "gripper_height_threshold": obstacle_required,
                "classifier_obstacle_free": classifier_obstacle_free,
                "classifier_obstacle_score": classifier_obstacle_score,
                "classifier_obstacle_required_score": classifier_obstacle_required,
                "one_step_prediction_error": one_step_error.astype(np.float64).tolist(),
                "one_step_error_in_disturbance_ellipsoid": err_in_ellipsoid,
                "one_step_error_disturbance_score": err_ellipsoid_score,
                "solver_status": status,
                "timings_sec": {
                    "vit_encode": float(encode_time_s),
                    "mppi_run": float(mppi_time_s),
                    "sls_solve": float(sls_solve_time_s),
                    "total": float(step_compute_time_s),
                },
            }
        )
        append_state_log(
            state_log_path,
            step=step_idx,
            state=current_state,
            latent_err=lat_err,
            status=f"{status}|mppi_h={active_mppi_horizon}",
            timings={
                "encode_time_s": encode_time_s,
                "mppi_time_s": mppi_time_s,
                "sls_solve_time_s": sls_solve_time_s,
                "step_compute_time_s": step_compute_time_s,
            },
            metrics={
                "mppi_horizon": int(active_mppi_horizon),
                "obs_free": obstacle_free if obstacle_free is not None else "n/a",
                "gripper_height": obstacle_score,
                "gripper_height_threshold": obstacle_required,
                "status": status,
                "reached_ogbench_success": bool(reached_ogbench_success),
            },
        )
        mpc_pbar.set_postfix(
            step_compute_time_s=f"{step_compute_time_s:.4f}",
            obs_free=obstacle_free,
            gripper_z=f"{obstacle_score:.4f}",
        )
        if cfg.terminate_on_ogbench_success and reached_ogbench_success:
            stop_reason = "ogbench_success"
            break
        if lat_err <= 0.05:
            stop_reason = "latent_goal_reached"
            break

    post_step_records = [record for record in step_records if record["phase"] == "post_step"]
    disturbance_checks = [record["one_step_error_in_disturbance_ellipsoid"] for record in post_step_records]
    valid_disturbance_checks = [bool(flag) for flag in disturbance_checks if flag is not None]
    obstacle_checks = [record["obstacle_free"] for record in step_records]
    valid_obstacle_checks = [bool(flag) for flag in obstacle_checks if flag is not None]
    state_region_checks = [record["state_in_latent_ellipsoid"] for record in step_records]
    valid_state_region_checks = [bool(flag) for flag in state_region_checks if flag is not None]
    timing_totals = {
        "vit_encode": float(sum(record["timings_sec"]["vit_encode"] for record in post_step_records)),
        "mppi_run": float(sum(record["timings_sec"]["mppi_run"] for record in post_step_records)),
        "sls_solve": float(sum(record["timings_sec"]["sls_solve"] for record in post_step_records)),
        "total": float(sum(record["timings_sec"]["total"] for record in post_step_records)),
    }

    imageio.mimwrite(out_dir / "cube_mppi_sls.mp4", rollout_frames, fps=cfg.video_fps)
    np.savez(
        out_dir / "executed_actions.npz",
        executed_actions_plan_norm=np.asarray(executed_actions_plan_norm, dtype=np.float32),
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    np.savez(
        out_dir / "executed_states.npz",
        markov_states=np.asarray(executed_states, dtype=np.float64),
        embeddings=np.asarray(executed_embeddings, dtype=np.float64),
        latent_goal_distances=np.asarray(latent_goal_distances, dtype=np.float64),
        block_pos_error_norm=np.asarray([float(metrics["block_pos_error_norm"]) for metrics in task_errors], dtype=np.float64),
        block_yaw_error=np.asarray([float(metrics["block_yaw_error"]) for metrics in task_errors], dtype=np.float64),
        block_pose_error_l2=np.asarray([float(metrics["block_pose_error_l2"]) for metrics in task_errors], dtype=np.float64),
        effector_block_distance=np.asarray([float(metrics["effector_block_distance"]) for metrics in task_errors], dtype=np.float64),
        gripper_height=np.asarray([float(metrics["gripper_height"]) for metrics in task_errors], dtype=np.float64),
        gripper_contact=np.asarray([float(metrics["gripper_contact"]) for metrics in task_errors], dtype=np.float64),
        ogbench_success=np.asarray(reached_success_flags, dtype=bool),
    )
    summary_payload = {
        "metadata": {
            "run_dir": str(out_dir),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint_path),
            "dataset_path": str(cfg.dataset_path.expanduser().resolve()),
            "episode_idx": int(episode_idx),
            "episode_seed": int(episode_seed),
            "env_name": env_name,
            "camera": camera,
            "stop_reason": stop_reason,
            "goal_reached": bool(stop_reason in {"ogbench_success", "latent_goal_reached"}),
            "reached_ogbench_success": bool(any(reached_success_flags)),
            "obstacle_check_source": "true_gripper_height",
            "gripper_height_threshold": float(cfg.gripper_height_threshold),
            "trajectory_safe_by_true_gripper_height": None if not valid_obstacle_checks else bool(all(valid_obstacle_checks)),
            "trajectory_safe_by_classifier": None
            if obstacle_model is None
            else bool(
                all(
                    bool(record["classifier_obstacle_free"])
                    for record in step_records
                    if record.get("classifier_obstacle_free") is not None
                )
            ),
            "disturbance_ellipsoid_coverage": {
                "covered_steps": int(sum(valid_disturbance_checks)),
                "checked_steps": int(len(valid_disturbance_checks)),
                "fraction": None if not valid_disturbance_checks else float(sum(valid_disturbance_checks) / len(valid_disturbance_checks)),
            },
            "state_latent_ellipsoid_coverage": {
                "covered_steps": int(sum(valid_state_region_checks)),
                "checked_steps": int(len(valid_state_region_checks)),
                "fraction": None if not valid_state_region_checks else float(sum(valid_state_region_checks) / len(valid_state_region_checks)),
            },
            "executed_steps": int(len(executed_actions_norm)),
            "num_logged_states": int(len(executed_states)),
            "timing_totals_sec": timing_totals,
            "fixed_grasp": {
                "u4_norm": float(fixed_grasp_norm),
                "u4_raw": float(fixed_grasp_raw),
            },
            "video_path": str(out_dir / "cube_mppi_sls.mp4"),
            "artifacts": {
                "latent_state_log_path": str(state_log_path),
                "executed_actions_path": str(out_dir / "executed_actions.npz"),
                "executed_states_path": str(out_dir / "executed_states.npz"),
                "trajectory_summary_path": str(out_dir / "trajectory_summary.json"),
            },
        },
        "start_goal": {
            "qpos_init": np.asarray(qpos_init, dtype=np.float64).tolist(),
            "qvel_init": np.asarray(qvel_init, dtype=np.float64).tolist(),
            "qpos_goal": np.asarray(qpos_goal, dtype=np.float64).tolist(),
            "qvel_goal": np.asarray(qvel_goal, dtype=np.float64).tolist(),
            "target_block_pos_init": np.asarray(target_block_pos_init, dtype=np.float64).tolist(),
            "target_block_yaw_init": float(target_block_yaw_init),
            "target_block_pos_goal": np.asarray(target_block_pos_goal, dtype=np.float64).tolist(),
            "target_block_yaw_goal": float(target_block_yaw_goal),
            "goal_state": np.asarray(goal_state, dtype=np.float64).tolist(),
        },
        "step_records": step_records,
    }
    with (out_dir / "trajectory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    env.close()

if __name__ == "__main__":
    main()
