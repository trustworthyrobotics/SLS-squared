#!/usr/bin/env python3
"""Plan in Rope pixel space using MPPI-warmstarted Conformal SLS MPC over a PyTorch world model."""

import os
import sys
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm
import pyrallis

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import config
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

ERROR_MODEL_ROOT = Path(__file__).resolve().parents[2] / "error_calib" / "error_model"
if str(ERROR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ERROR_MODEL_ROOT))

# gpu_sls core modules
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints
from gpu_sls.mppi_planner import MPPIPlanner

# Rope training framework imports
from rope.train.mlpdyn_train import (
    preprocess_pixels,
)
from rope.shared.lab_env import LabEnv, TaskState
from error_model import MGNLLPredictor

DEFAULT_DATASET_PATH = Path("rope/data/expert_data/rope_random_cubic_spline.h5")
DEFAULT_ACTION_STATS_CANDIDATES = (
    DEFAULT_DATASET_PATH,
    Path("rope/data/test_data_noshadow/rope_random_cubic_spline.h5"),
)

@dataclass
class PlanSLSRopeConfig:
    """Configuration for MPPI-warmstarted Conformal SLS MPC Rope Planning"""
    q_learned: float = field(default=0.0, metadata={"help": "Conformal quantile for the disturbance bound."})
    model_dir: Path = field(default=Path("rope/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("rope/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("rope/eval/fixed_error_covariance.pt"))
    enable_obstacle: bool = field(default=False)
    obstacle_model_path: Path = field(default=Path("rope/models/obs_net/da270d7d1050f110/model.pt"))
    obstacle_margin: float = field(default=0.0)
    obstacle_penalty_weight: float = field(default=1000.0)
    use_latent_ellipsoid_constraint: bool = field(default=False)
    latent_ellipsoid_path: Path = field(default=Path("rope/eval/latent_ellipsoid_noshadow/latent_ellipsoid.pt"))
    latent_ellipsoid_margin: float = field(default=0.0)
    action_stats_dataset_path: Optional[Path] = field(default=None)
    dataset_path: Path = field(default=DEFAULT_DATASET_PATH)
    out_dir: Path = field(default=Path("rope/plan/sls_mppi_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=20)
    max_mpc_steps: int = field(default=150)
    video_fps: int = field(default=30)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    q_stage: float = field(default=0.005)
    q_terminal: float = field(default=5.0)
    r_control: float = field(default=0.01)
    mppi_horizon: Optional[int] = field(default=None)
    mppi_samples: int = field(default=2048)
    mppi_update_iter: int = field(default=6)
    mppi_reward_weight: float = field(default=25.0)
    mppi_stage_weight: float = field(default=0.005)
    mppi_terminal_weight: float = field(default=5.0)
    mppi_r_control: float = field(default=0.01)
    mppi_noise_level: float = field(default=0.2)
    mppi_beta_filter: float = field(default=0.65)
    mppi_state_box_penalty: float = field(default=0.0)
    mppi_ellipsoid_penalty_weight: float = field(default=0.0)

# --- JAX / PyTorch Bridge Engines ---

class JAXObstacleMLP(eqx.Module):
    linear_layers: list
    layer_norm_scales: list
    layer_norm_biases: list
    feature_mean: jax.Array
    feature_std: jax.Array
    threshold: jax.Array
    input_dim: int

    def __call__(self, state):
        z = state[: self.input_dim]
        x = (z - self.feature_mean) / self.feature_std
        for i, linear in enumerate(self.linear_layers[:-1]):
            x = linear(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = x * self.layer_norm_scales[i] + self.layer_norm_biases[i]
            x = jax.nn.gelu(x)
        return self.linear_layers[-1](x).squeeze(-1)

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

    linear_layers = []
    layer_norm_scales = []
    layer_norm_biases = []
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

    return JAXObstacleMLP(
        linear_layers=linear_layers,
        layer_norm_scales=layer_norm_scales,
        layer_norm_biases=layer_norm_biases,
        feature_mean=jnp.asarray(artifact["feature_mean"], dtype=jnp.float64),
        feature_std=jnp.maximum(jnp.asarray(artifact["feature_std"], dtype=jnp.float64), 1e-6),
        threshold=jnp.asarray(float(artifact["conformal_safe_score_threshold"]), dtype=jnp.float64),
        input_dim=input_dim,
    )

def build_jax_dynamics(torch_dynamics_net: torch.nn.Module, device: torch.device, state_dim: int, action_dim: int):
    def _fwd_fn(x_np, u_np):
        with torch.no_grad():
            x_t = torch.from_numpy(np.array(x_np)).float().to(device)
            u_t = torch.from_numpy(np.array(u_np)).float().to(device)
            inp = torch.cat((x_t, u_t), dim=-1)
            out = torch_dynamics_net(inp.unsqueeze(0)).squeeze(0) if inp.ndim == 1 else torch_dynamics_net(inp)
            return np.asarray(out.cpu().numpy(), dtype=np.float64)

    def _vjp_fn(x_np, u_np, g_np):
        x_t = torch.from_numpy(np.array(x_np)).float().to(device).requires_grad_(True)
        u_t = torch.from_numpy(np.array(u_np)).float().to(device).requires_grad_(True)
        inp = torch.cat((x_t, u_t), dim=-1)
        out = torch_dynamics_net(inp.unsqueeze(0)).squeeze(0) if inp.ndim == 1 else torch_dynamics_net(inp)
        g_t = torch.from_numpy(np.asarray(g_np)).float().to(device)
        out.backward(g_t)
        return np.asarray(x_t.grad.cpu().numpy(), dtype=np.float64), np.asarray(u_t.grad.cpu().numpy(), dtype=np.float64)

    @jax.custom_vjp
    def jax_dynamics(x, u, t, parameter):
        result_shape = jax.ShapeDtypeStruct((state_dim,), jnp.float64)
        return jax.pure_callback(_fwd_fn, result_shape, x, u, vmap_method="sequential")

    def jax_dynamics_fwd(x, u, t, parameter):
        y = jax_dynamics(x, u, t, parameter)
        return y, (x, u)

    def jax_dynamics_bwd(res, g):
        x, u = res
        jac_x_shape = jax.ShapeDtypeStruct((state_dim,), jnp.float64)
        jac_u_shape = jax.ShapeDtypeStruct((action_dim,), jnp.float64)
        vjp_x, vjp_u = jax.pure_callback(_vjp_fn, (jac_x_shape, jac_u_shape), x, u, g, vmap_method="sequential")
        return vjp_x, vjp_u, None, None

    jax_dynamics.defvjp(jax_dynamics_fwd, jax_dynamics_bwd)
    return jax_dynamics

def build_jax_disturbance(error_model: torch.nn.Module, q_learned: float, device: torch.device, state_dim: int, action_dim: int):
    def _dist_fn(X_prefix_np, U_prefix_np):
        with torch.no_grad():
            X_t = torch.from_numpy(np.array(X_prefix_np)).float().to(device)
            U_t = torch.from_numpy(np.array(U_prefix_np)).float().to(device)
            if X_t.ndim == 1:
                X_t = X_t.unsqueeze(0)
                U_t = U_t.unsqueeze(0)
            model_input = torch.cat([X_t, U_t], dim=-1)
            L = error_model(model_input) 
            return np.asarray((q_learned * L).cpu().numpy(), dtype=np.float64)

    def jax_disturbance(X_prefix, U_prefix):
        T = X_prefix.shape[0]
        result_shape = jax.ShapeDtypeStruct((T, state_dim, state_dim), jnp.float64)
        return jax.pure_callback(_dist_fn, result_shape, X_prefix, U_prefix, vmap_method="sequential")
        
    return jax_disturbance

def make_constant_jax_disturbance(calibrated_cholesky: np.ndarray, state_dim: int):
    calibrated_cholesky = jnp.asarray(calibrated_cholesky, dtype=jnp.float64)
    if calibrated_cholesky.shape != (state_dim, state_dim):
        raise ValueError(
            f"Expected calibrated Cholesky shape {(state_dim, state_dim)}, got {calibrated_cholesky.shape}."
        )

    def jax_disturbance(X_prefix, U_prefix):
        seq_len = X_prefix.shape[0]
        return jnp.broadcast_to(calibrated_cholesky, (seq_len, state_dim, state_dim))

    return jax_disturbance

def load_calibrated_cholesky(path: Path) -> np.ndarray:
    payload = torch.load(path.expanduser(), map_location="cpu")
    if "calibrated_cholesky" in payload:
        matrix = payload["calibrated_cholesky"]
    elif "cholesky" in payload and "q_fixed" in payload:
        matrix = payload["cholesky"] * payload["q_fixed"]
    else:
        raise KeyError(
            f"{path} must contain either 'calibrated_cholesky' or both 'cholesky' and 'q_fixed'."
        )
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float64)

def load_action_stats_from_dataset(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Action-statistics dataset not found: {dataset_path}")
    with h5py.File(dataset_path, "r") as h5:
        if "action" not in h5:
            raise KeyError(f"{dataset_path} does not contain an 'action' dataset.")
        if int(h5["action"].shape[-1]) != int(action_dim):
            raise ValueError(
                f"Expected action_dim={action_dim} in {dataset_path}, got {h5['action'].shape[-1]}."
            )
        actions = np.asarray(h5["action"][:], dtype=np.float32)
    finite_actions = actions[~np.isnan(actions).any(axis=1)]
    if finite_actions.shape[0] == 0:
        raise ValueError(f"No finite actions found in {dataset_path}.")
    action_mean = finite_actions.mean(axis=0).astype(np.float64)
    action_std = np.maximum(finite_actions.std(axis=0).astype(np.float64), 1e-6)
    return action_mean, action_std

def resolve_action_stats_dataset_path(cfg: PlanSLSRopeConfig) -> Path:
    if cfg.action_stats_dataset_path is not None:
        return cfg.action_stats_dataset_path
    if cfg.dataset_path.suffix.lower() in {".h5", ".hdf5"}:
        return cfg.dataset_path
    for candidate in DEFAULT_ACTION_STATS_CANDIDATES:
        if candidate.expanduser().is_file():
            return candidate
    return DEFAULT_DATASET_PATH

def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array

def _pick_endpoint_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Expected one of keys {names}, got {sorted(mapping.keys())}.")

def _pick_optional_endpoint_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None

def _select_pair_value(value, episode_idx: int, pair_count: Optional[int]):
    if isinstance(value, dict):
        return {key: _select_pair_value(item, episode_idx, pair_count) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if pair_count is not None and len(value) == int(pair_count):
            return value[episode_idx]
        return value
    if isinstance(value, torch.Tensor):
        if pair_count is not None and value.ndim > 0 and int(value.shape[0]) == int(pair_count):
            return value[episode_idx]
        return value
    if isinstance(value, np.ndarray):
        if pair_count is not None and value.ndim > 0 and int(value.shape[0]) == int(pair_count):
            return value[episode_idx]
        return value
    return value

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
        if "start" in payload and ("goal" in payload or "end" in payload):
            start = payload["start"]
            if isinstance(start, dict):
                for item in start.values():
                    if isinstance(item, torch.Tensor) and item.ndim > 0:
                        return int(item.shape[0])
                    if isinstance(item, np.ndarray) and item.ndim > 0:
                        return int(item.shape[0])
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return None

def _select_endpoint_pair(payload, episode_idx: int):
    pair_count = _infer_pair_count(payload)
    if pair_count is not None:
        if episode_idx < 0 or episode_idx >= pair_count:
            raise ValueError(f"episode_idx must be in [0, {pair_count - 1}], got {episode_idx}.")

    if isinstance(payload, (list, tuple)):
        return payload[episode_idx], pair_count

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported endpoint pair payload type: {type(payload)!r}")

    for key in ("pairs", "episodes", "endpoint_pairs"):
        if key in payload:
            pairs = payload[key]
            if not isinstance(pairs, (list, tuple)):
                raise TypeError(f"Endpoint payload key '{key}' must be a list/tuple, got {type(pairs)!r}.")
            return pairs[episode_idx], len(pairs)

    if "start" in payload and ("goal" in payload or "end" in payload):
        return {
            "start": _select_pair_value(payload["start"], episode_idx, pair_count),
            "goal": _select_pair_value(_pick_endpoint_key(payload, ("goal", "end")), episode_idx, pair_count),
        }, pair_count

    raise KeyError(
        "Endpoint .pt payload must be a list of pairs, contain a 'pairs'/'episodes' list, "
        "or contain top-level 'start' and 'goal'/'end' entries."
    )

def _load_endpoint_pair_episode(dataset_path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, np.ndarray], int]:
    payload = torch.load(dataset_path.expanduser(), map_location="cpu", weights_only=False)
    pair_count = _infer_pair_count(payload)
    rng = np.random.default_rng(seed)
    if episode_idx is None:
        if pair_count is None:
            episode_idx = 0
        else:
            episode_idx = int(rng.integers(pair_count))

    pair, pair_count = _select_endpoint_pair(payload, int(episode_idx))
    if not isinstance(pair, dict):
        raise TypeError(f"Selected endpoint pair must be a dict, got {type(pair)!r}.")

    start = _pick_endpoint_key(pair, ("start", "initial", "source"))
    goal = _pick_endpoint_key(pair, ("goal", "end", "target"))
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise TypeError("Endpoint pair 'start' and 'goal'/'end' entries must both be dicts.")

    pixels_np = np.stack(
        [
            _as_numpy(_pick_endpoint_key(start, ("pixels", "pixel", "image", "rgb")), dtype=np.uint8),
            _as_numpy(_pick_endpoint_key(goal, ("pixels", "pixel", "image", "rgb")), dtype=np.uint8),
        ],
        axis=0,
    )
    task_target_np = np.stack(
        [
            _as_numpy(_pick_endpoint_key(start, ("task_target", "task_state", "target")), dtype=np.float32),
            _as_numpy(_pick_endpoint_key(goal, ("task_target", "task_state", "target")), dtype=np.float32),
        ],
        axis=0,
    )
    start_qpos = _as_numpy(_pick_endpoint_key(start, ("qpos", "q_pos")), dtype=np.float32)
    goal_qpos = _as_numpy(_pick_endpoint_key(goal, ("qpos", "q_pos")), dtype=np.float32)
    qpos_np = np.stack([start_qpos, goal_qpos], axis=0)
    start_qvel_raw = _pick_optional_endpoint_key(start, ("qvel", "q_vel"))
    goal_qvel_raw = _pick_optional_endpoint_key(goal, ("qvel", "q_vel"))
    start_qvel = np.zeros_like(start_qpos, dtype=np.float32) if start_qvel_raw is None else _as_numpy(start_qvel_raw, dtype=np.float32)
    goal_qvel = np.zeros_like(goal_qpos, dtype=np.float32) if goal_qvel_raw is None else _as_numpy(goal_qvel_raw, dtype=np.float32)
    qvel_np = np.stack(
        [
            start_qvel,
            goal_qvel,
        ],
        axis=0,
    )
    control_np = np.stack(
        [
            _as_numpy(_pick_endpoint_key(start, ("control", "ctrl")), dtype=np.float32),
            _as_numpy(_pick_endpoint_key(goal, ("control", "ctrl")), dtype=np.float32),
        ],
        axis=0,
    )
    time_np = np.asarray([[0.0], [1.0 / 30.0]], dtype=np.float32)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    episode = {
        "pixels": pixels_np,
        "task_target": task_target_np,
        "qpos": qpos_np,
        "qvel": qvel_np,
        "control": control_np,
        "time": time_np,
        "camera_name": str(metadata.get("camera", "video_cam")) if isinstance(metadata, dict) else "video_cam",
        "control_decimation": int(metadata.get("control_decimation", 25)) if isinstance(metadata, dict) else 25,
        "disable_shadows": bool(metadata.get("disable_shadows", True)) if isinstance(metadata, dict) else True,
        "control_timestep": float(metadata.get("control_timestep", 1.0 / 30.0)) if isinstance(metadata, dict) else 1.0 / 30.0,
        "is_endpoint_pair": True,
    }
    print(
        f"Loaded endpoint pair {episode_idx}"
        + (f"/{pair_count}" if pair_count is not None else "")
        + f" from {dataset_path}"
    )
    return episode, int(episode_idx)

def _load_hdf5_episode(dataset_path: Path, episode_idx: Optional[int], horizon: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        valid_episodes = np.flatnonzero(ep_len >= horizon)
        if valid_episodes.size == 0:
            raise ValueError(f"No episodes in {dataset_path} have length >= horizon={horizon}.")
        if episode_idx is None:
            episode_idx = int(rng.choice(valid_episodes))
        elif episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"episode_idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        elif ep_len[episode_idx] < horizon:
            raise ValueError(f"episode_idx {episode_idx} has length {ep_len[episode_idx]}, below horizon={horizon}.")

        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        episode = {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "task_target": np.asarray(h5["task_target"][rows], dtype=np.float32),
            "qpos": np.asarray(h5["qpos"][rows], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][rows], dtype=np.float32),
            "control": np.asarray(h5["control"][rows], dtype=np.float32),
            "time": np.asarray(h5["time"][rows], dtype=np.float32) if "time" in h5 else np.zeros((len(rows), 1), dtype=np.float32),
            "camera_name": str(h5.attrs.get("camera", "video_cam")),
            "control_decimation": int(h5.attrs.get("control_decimation", 25)),
            "disable_shadows": bool(h5.attrs.get("disable_shadows", True)),
            "control_timestep": float(h5.attrs.get("control_timestep", 1.0 / 30.0)),
            "is_endpoint_pair": False,
        }
    return episode, int(episode_idx)

def load_planning_episode(dataset_path: Path, episode_idx: Optional[int], horizon: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Planning dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".pt":
        return _load_endpoint_pair_episode(dataset_path, episode_idx, seed)
    return _load_hdf5_episode(dataset_path, episode_idx, horizon, seed)

# --- Cost, Context & Utilities ---

def make_goal_tracking_cost(
    r_control: float,
    horizon: int,
    W_terminal: jnp.ndarray,
    goal_state: jnp.ndarray,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
):
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W, W_terminal)
        dz = z - goal_state
        total_cost = jnp.sum(active_W * dz**2) + r_control * jnp.sum(u**2)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            obstacle_violation = jax.nn.softplus(
                obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z)
            )
            total_cost = total_cost + obstacle_penalty_weight * obstacle_violation**2
        return total_cost
    return cost

def make_mppi_rollout_and_eval(
    torch_dynamics_net: torch.nn.Module,
    device: torch.device,
    state_dim: int,
    action_dim: int,
    W_stage: jnp.ndarray,
    W_terminal: jnp.ndarray,
    goal_state: jnp.ndarray,
    *,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
    box_min: jnp.ndarray | None = None,
    box_max: jnp.ndarray | None = None,
    box_penalty_weight: float = 0.0,
    ellipsoid_unit_precision: jnp.ndarray | None = None,
    ellipsoid_margin: float = 0.0,
    ellipsoid_penalty_weight: float = 0.0,
    r_control: float = 0.01,
):
    W_stage = jnp.asarray(W_stage, dtype=jnp.float64)
    W_terminal = jnp.asarray(W_terminal, dtype=jnp.float64)
    goal_state = jnp.asarray(goal_state, dtype=jnp.float64)
    if box_min is not None:
        box_min = jnp.asarray(box_min, dtype=jnp.float64)
    if box_max is not None:
        box_max = jnp.asarray(box_max, dtype=jnp.float64)
    if ellipsoid_unit_precision is not None:
        ellipsoid_unit_precision = jnp.asarray(ellipsoid_unit_precision, dtype=jnp.float64)

    def _torch_rollout_fn(state_cur_np, act_seqs_np):
        with torch.no_grad():
            state_cur_t = torch.from_numpy(np.asarray(state_cur_np)).float().to(device)
            act_seqs_t = torch.from_numpy(np.asarray(act_seqs_np)).float().to(device)
            n_sample, horizon, _ = act_seqs_t.shape
            states = state_cur_t.unsqueeze(0).expand(n_sample, -1)
            rolled_states = []
            for t in range(horizon):
                model_input = torch.cat([states, act_seqs_t[:, t, :]], dim=-1)
                states = torch_dynamics_net(model_input)
                rolled_states.append(states)
            return np.asarray(torch.stack(rolled_states, dim=1).cpu().numpy(), dtype=np.float64)

    def mppi_rollout_fn(state_cur, act_seqs, reach_config=None):
        result_shape = jax.ShapeDtypeStruct(
            (act_seqs.shape[0], act_seqs.shape[1], state_dim),
            jnp.float64,
        )
        states = jax.pure_callback(
            _torch_rollout_fn,
            result_shape,
            state_cur,
            act_seqs,
            vmap_method="sequential",
        )
        return states, {}

    def mppi_eval_fn(state_seqs, act_seqs, reach_config=None, aux=None, *args, **kwargs):
        delta = state_seqs - goal_state[None, None, :]
        stage_costs = jnp.sum(W_stage[None, None, :] * delta**2, axis=-1)
        terminal_costs = jnp.sum(W_terminal[None, :] * delta[:, -1, :] ** 2, axis=-1)
        action_costs = r_control * jnp.sum(act_seqs**2, axis=-1)

        if box_min is not None and box_max is not None and box_penalty_weight > 0.0:
            lower_violation = jnp.maximum(box_min[None, None, :] - state_seqs, 0.0)
            upper_violation = jnp.maximum(state_seqs - box_max[None, None, :], 0.0)
            box_costs = box_penalty_weight * jnp.sum(lower_violation**2 + upper_violation**2, axis=-1)
        else:
            box_costs = jnp.zeros_like(stage_costs)

        if ellipsoid_unit_precision is not None and ellipsoid_penalty_weight > 0.0:
            ellipsoid_scores = jnp.einsum("bti,ij,btj->bt", state_seqs, ellipsoid_unit_precision, state_seqs)
            ellipsoid_violation = jnp.maximum(ellipsoid_scores - 1.0 + float(ellipsoid_margin), 0.0)
            ellipsoid_costs = ellipsoid_penalty_weight * ellipsoid_violation**2
        else:
            ellipsoid_costs = jnp.zeros_like(stage_costs)

        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            flat_states = state_seqs.reshape((-1, state_seqs.shape[-1]))
            obstacle_violation = jax.vmap(
                lambda z: jax.nn.softplus(obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z))
            )(flat_states).reshape(state_seqs.shape[:-1])
            obstacle_costs = obstacle_penalty_weight * obstacle_violation**2
        else:
            obstacle_costs = jnp.zeros_like(stage_costs)

        total_cost = (
            jnp.sum(stage_costs + action_costs + box_costs + ellipsoid_costs + obstacle_costs, axis=-1)
            + terminal_costs
        )
        return {"rewards": -total_cost}

    return mppi_rollout_fn, mppi_eval_fn

def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    def constraint(x, u, t):
        return jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])
    return constraint

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    def constraints(x, u, t):
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
    return constraints

def load_markov_ellipsoid_unit_precision(path: Path, state_dim: int) -> np.ndarray:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Latent ellipsoid artifact not found: {path}")
    if path.suffix == ".npz":
        payload = np.load(path)
        if "markov_unit_precision" in payload:
            matrix = payload["markov_unit_precision"]
        elif "unit_precision" in payload:
            matrix = payload["unit_precision"]
        else:
            raise KeyError(f"{path} must contain 'markov_unit_precision' or 'unit_precision'.")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "markov_unit_precision" in payload:
            matrix = payload["markov_unit_precision"]
        elif "unit_precision" in payload:
            matrix = payload["unit_precision"]
        else:
            raise KeyError(f"{path} must contain 'markov_unit_precision' or 'unit_precision'.")
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (state_dim, state_dim):
        raise ValueError(f"Expected ellipsoid matrix shape {(state_dim, state_dim)}, got {matrix.shape}.")
    return matrix

def make_markov_ellipsoid_constraint(unit_precision: np.ndarray, margin: float = 0.0):
    unit_precision = jnp.asarray(unit_precision, dtype=jnp.float64)
    def constraint(x, u, t):
        score = x @ unit_precision @ x
        return jnp.asarray([score - 1.0 + float(margin)])
    return constraint

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None: candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel_np: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(pixel_np.copy()).permute(2, 0, 1).contiguous()
    tensor = preprocess_pixels(tensor.unsqueeze(0), img_size).to(device)
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0)
    output = model.encoder(tensor, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).contiguous()
    tensor = preprocess_pixels(tensor, img_size).to(device)
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0)
    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32]
        if chunk.ndim == 5:
            chunk = chunk.squeeze(0)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float64)

def render_rgb_frame(renderer: mujoco.Renderer, env: LabEnv, camera_id: int, *, disable_shadows: bool) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    if disable_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()

def reset_env_to_state(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control: np.ndarray,
    task_target: np.ndarray,
    camera_id: int,
    elapsed_time: float,
    disable_shadows: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.reset(TaskState.from_array(task_target))
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)
    return frame, {
        "task_target": env.task_controller.desired_state.as_array().astype(np.float32),
        "time": np.asarray([elapsed_time], dtype=np.float32),
    }

def step_env_with_action(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    action: np.ndarray,
    control_decimation: int,
    camera_id: int,
    elapsed_time: float,
    disable_shadows: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.apply_task_delta(np.asarray(action, dtype=np.float64))
    env.step(int(control_decimation))
    frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)
    return frame, {
        "task_target": env.task_controller.desired_state.as_array().astype(np.float32),
        "time": np.asarray([elapsed_time], dtype=np.float32),
    }


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


def state_ellipsoid_membership(state_np: np.ndarray, unit_precision: np.ndarray | None, margin: float) -> tuple[bool | None, float | None]:
    if unit_precision is None:
        return None, None
    score = float(np.asarray(state_np, dtype=np.float64) @ unit_precision @ np.asarray(state_np, dtype=np.float64))
    return score <= 1.0 + float(margin), score


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

def main():
    cfg = pyrallis.parse(config_class=PlanSLSRopeConfig)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    init_key = jax.random.PRNGKey(cfg.seed)

    # 1. Load Configurations & Model Parameters
    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json", "r") as f: config_dict = json.load(f)
    
    checkpoint_path = latest_object_checkpoint(model_dir)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).eval()

    state_dim = config_dict.get("markov_state_dim", 36)
    action_dim = config_dict.get("action_dim", 5) # Rope standard is 5 dimensions (x, y, z gripper displacements)
    img_size = config_dict.get("img_size", 224)

    dynamics = build_jax_dynamics(model.predictor.net, device, state_dim, action_dim)
    obstacle_model = None
    obstacle_constraint = None
    error_model_torch = None
    calibrated_cholesky = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, init_key)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(
                f"Obstacle classifier input_dim={obstacle_model.input_dim} exceeds planner state_dim={state_dim}."
            )
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(
            f"Using conformal obstacle classifier from {cfg.obstacle_model_path} "
            f"with threshold {float(obstacle_model.threshold):.6g} and margin {cfg.obstacle_margin:.6g}"
        )
    else:
        print("Obstacle avoidance disabled.")
    if cfg.use_constant_covariance:
        calibrated_cholesky = load_calibrated_cholesky(cfg.constant_covariance_path)
        disturbance = make_constant_jax_disturbance(calibrated_cholesky, state_dim)
        print(f"Using fixed calibrated covariance disturbance from {cfg.constant_covariance_path}")
    else:
        error_model_torch = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
        disturbance = build_jax_disturbance(error_model_torch, cfg.q_learned, device, state_dim, action_dim)

    # 2. Extract action normalization parameters.
    action_stats_dataset_path = resolve_action_stats_dataset_path(cfg)
    action_mean, action_std = load_action_stats_from_dataset(action_stats_dataset_path, action_dim)
    print(f"Using action statistics from {action_stats_dataset_path}")

    episode, episode_idx = load_planning_episode(cfg.dataset_path, cfg.episode_idx, cfg.horizon, cfg.seed)
    pixels_np = episode["pixels"]
    task_target_np = episode["task_target"]
    qpos_np = episode["qpos"]
    qvel_np = episode["qvel"]
    control_np = episode["control"]
    time_np = episode["time"]
    camera_name = episode["camera_name"]
    control_decimation = episode["control_decimation"]
    disable_shadows = episode["disable_shadows"]
    control_timestep = episode["control_timestep"]

    run_dir = out_dir / f"{int(time.time())}_mppi_sls_rope_episode_{episode_idx:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Reference Extraction Sequence
    true_latents = encode_frames(model, pixels_np, device, img_size)
    
    # Emulate the Markov-state strategy safely: [z_t, delta_z_t].
    start_z = true_latents[0]
    start_state = torch.cat([start_z, torch.zeros_like(start_z)], dim=-1).cpu().numpy().astype(np.float64)
    
    goal_z = true_latents[-1]
    if episode.get("is_endpoint_pair", False):
        goal_delta = torch.zeros_like(goal_z)
    else:
        goal_delta = goal_z - true_latents[-2] if true_latents.shape[0] > 1 else torch.zeros_like(goal_z)
    goal_state = torch.cat([goal_z, goal_delta], dim=-1).cpu().numpy().astype(np.float64)

    if obstacle_model is not None:
        start_score = float(obstacle_model(jnp.asarray(start_state)))
        goal_score = float(obstacle_model(jnp.asarray(goal_state)))
        required_score = float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        if start_score <= required_score or goal_score <= required_score:
            print(
                "Terminating: start and goal must both be outside the conformal obstacle set. "
                f"Required score > {required_score:.6g}; "
                f"start_score={start_score:.6g}, goal_score={goal_score:.6g}."
            )
            sys.exit(1)
        print(
            "Obstacle sanity check passed: "
            f"start_score={start_score:.6g}, goal_score={goal_score:.6g}, "
            f"required_score>{required_score:.6g}"
        )

    imageio.imwrite(run_dir / "start_rope.png", pixels_np[0])
    imageio.imwrite(run_dir / "goal_rope.png", pixels_np[-1])

    # 4. iLQR-style nominal objective: every stage and terminal state targets the final goal.
    W_state = jnp.ones((state_dim,)) * cfg.q_stage
    W_term = jnp.ones((state_dim,)) * cfg.q_terminal
    W_mppi_stage = jnp.ones((state_dim,)) * cfg.mppi_stage_weight
    W_mppi_stage = W_mppi_stage.at[state_dim // 2:].set(1.0)
    W_mppi_term = jnp.ones((state_dim,)) * cfg.mppi_terminal_weight
    W_mppi_term = W_mppi_term.at[state_dim // 2:].set(1.0)

    cost = make_goal_tracking_cost(
        r_control=cfg.r_control,
        horizon=cfg.horizon,
        W_terminal=W_term,
        goal_state=jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
    )

    # 5. Solver Parameter Building Footprint
    sls_cfg = SLSConfig(max_sls_iterations=1, sls_primal_tol=1e-2, enable_fastsls=True, initialize_nominal=True, warm_start=False, rti=True)
    sqp_cfg = SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=True)
    admm_cfg = ADMMConfig(eps_abs=5e-2, eps_rel=1e-4, rho_max=1e5, max_iterations=1200, rho_update_frequency=20, initial_rho=1.0)
    
    mpc_dt = 1.0 / 30.0 # Match standard frame processing loop frequency
    mpc_cfg = MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_state, u_ref=jnp.zeros(action_dim), dt=mpc_dt)

    u_min, u_max = -3.5 * jnp.ones(action_dim), 3.5 * jnp.ones(action_dim)
    x_min, x_max = -2.5 * jnp.ones(state_dim), 2.5 * jnp.ones(state_dim)
    # Keep the finite-difference half of the Markov state tightly bounded.
    x_min = x_min.at[x_min.shape[0]//2:].set(-0.5)
    x_max = x_max.at[x_max.shape[0]//2:].set(0.5)
    ellipsoid_unit_precision = None
    if cfg.use_latent_ellipsoid_constraint:
        ellipsoid_unit_precision = load_markov_ellipsoid_unit_precision(cfg.latent_ellipsoid_path, state_dim)
        state_constraint = make_markov_ellipsoid_constraint(
            ellipsoid_unit_precision,
            margin=cfg.latent_ellipsoid_margin,
        )
        state_constraint_count = 1
        print(
            f"Using calibrated Markov ellipsoid state constraint from {cfg.latent_ellipsoid_path} "
            f"with margin {cfg.latent_ellipsoid_margin:.6g}"
        )
        
        # Ellipsoid sanity check: ensure start and goal are within the ellipsoid
        unit_precision_np = np.asarray(ellipsoid_unit_precision, dtype=np.float64)
        start_ellipsoid_score = float(start_state @ unit_precision_np @ start_state)
        goal_ellipsoid_score = float(goal_state @ unit_precision_np @ goal_state)
        ellipsoid_margin_score = 1.0 + float(cfg.latent_ellipsoid_margin)
        if start_ellipsoid_score > ellipsoid_margin_score or goal_ellipsoid_score > ellipsoid_margin_score:
            print(
                "Terminating: start and goal must both be inside the conformal ellipsoid. "
                f"Required score <= {ellipsoid_margin_score:.6g}; "
                f"start_score={start_ellipsoid_score:.6g}, goal_score={goal_ellipsoid_score:.6g}."
            )
            sys.exit(1)
        print(
            "Ellipsoid sanity check passed: "
            f"start_score={start_ellipsoid_score:.6g}, goal_score={goal_ellipsoid_score:.6g}, "
            f"max_allowed_score<={ellipsoid_margin_score:.6g}"
        )
    else:
        state_constraint = make_state_box_constraints(x_min, x_max)
        state_constraint_count = 2 * state_dim
        print("Using hand-tuned Markov state box constraints.")
    if obstacle_constraint is not None:
        constraints_all = combine_constraints(
            state_constraint,
            obstacle_constraint,
            make_control_box_constraints(u_min, u_max),
        )
    else:
        constraints_all = combine_constraints(
            state_constraint,
            make_control_box_constraints(u_min, u_max),
        )

    mppi_horizon = int(cfg.mppi_horizon) if cfg.mppi_horizon is not None else int(cfg.horizon)
    if mppi_horizon < cfg.horizon:
        raise ValueError(f"mppi_horizon must be >= horizon. Got mppi_horizon={mppi_horizon}, horizon={cfg.horizon}.")
    print(f"Using MPPI horizon {mppi_horizon}; SLS tracking horizon {cfg.horizon}.")

    mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(
        model.predictor.net,
        device,
        state_dim,
        action_dim,
        W_mppi_stage,
        W_mppi_term,
        jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
        box_min=x_min,
        box_max=x_max,
        box_penalty_weight=cfg.mppi_state_box_penalty,
        ellipsoid_unit_precision=ellipsoid_unit_precision,
        ellipsoid_margin=cfg.latent_ellipsoid_margin,
        ellipsoid_penalty_weight=cfg.mppi_ellipsoid_penalty_weight,
        r_control=cfg.mppi_r_control,
    )
    mppi_planner = MPPIPlanner(
        config={
            "planning": {
                "action_dim": action_dim,
                "n_sample": cfg.mppi_samples,
                "horizon": mppi_horizon,
                "n_update_iter": cfg.mppi_update_iter,
                "use_last": True,
                "reject_bad": False,
                "mppi": {
                    "reward_weight": cfg.mppi_reward_weight,
                    "noise_level": cfg.mppi_noise_level,
                    "noise_decay": 1.0,
                    "beta_filter": cfg.mppi_beta_filter,
                },
            }
        },
        model_rollout_fn=mppi_rollout,
        evaluate_traj_fn=mppi_eval,
        action_lower_lim=u_min,
        action_upper_lim=u_max,
    )
    jit_mppi_trajopt = jax.jit(lambda key, state, act_seq: mppi_planner.trajectory_optimization(key, state, act_seq, skip=False))

    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg, config=mpc_cfg, dynamics=dynamics, constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + state_constraint_count + (1 if obstacle_constraint is not None else 0),
        disturbance=disturbance, shift=1, X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64), U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64)
    )

    # 6. Receding Horizon Closed-Loop Execution
    env = LabEnv()
    camera_id = env.model.camera(camera_name).id
    with mujoco.Renderer(env.model, height=int(pixels_np.shape[1]), width=int(pixels_np.shape[2])) as renderer:
        current_frame, current_info = reset_env_to_state(
            env,
            renderer,
            qpos=qpos_np[0],
            qvel=qvel_np[0],
            control=control_np[0],
            task_target=task_target_np[0],
            camera_id=camera_id,
            elapsed_time=float(time_np[0, 0]),
            disable_shadows=disable_shadows,
        )
        current_emb = encode_single_frame(model, current_frame, device, img_size)
        current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)

        rollout_frames = [current_frame.copy()]
        executed_actions_norm: list[np.ndarray] = []
        executed_actions_raw: list[np.ndarray] = []
        executed_states: list[np.ndarray] = [current_state.copy()]
        executed_embeddings: list[np.ndarray] = [current_emb.detach().cpu().numpy().astype(np.float64).copy()]
        executed_task_targets: list[np.ndarray] = [np.asarray(current_info["task_target"], dtype=np.float64).copy()]
        executed_times: list[float] = [float(current_info["time"][0])]
        latent_goal_distances: list[float] = [float(np.linalg.norm(current_state - goal_state))]
        task_target_distances: list[float] = [
            float(np.linalg.norm(np.asarray(current_info["task_target"], dtype=np.float64) - task_target_np[-1].astype(np.float64)))
        ]
        initial_obstacle_free, initial_obstacle_score, initial_obstacle_required = obstacle_safety_status(
            current_state, obstacle_model, cfg.obstacle_margin
        )
        initial_state_in_ellipsoid, initial_state_ellipsoid_score = state_ellipsoid_membership(
            current_state, ellipsoid_unit_precision, cfg.latent_ellipsoid_margin
        )
        step_records = [
            {
                "step": 0,
                "phase": "initial",
                "markov_state": current_state.astype(np.float64).tolist(),
                "embedding": executed_embeddings[0].tolist(),
                "task_target": executed_task_targets[0].tolist(),
                "elapsed_time": float(executed_times[0]),
                "latent_goal_error": float(latent_goal_distances[0]),
                "task_error": float(task_target_distances[0]),
                "state_in_latent_ellipsoid": initial_state_in_ellipsoid,
                "state_latent_ellipsoid_score": initial_state_ellipsoid_score,
                "obstacle_free": initial_obstacle_free,
                "obstacle_score": initial_obstacle_score,
                "obstacle_required_score": initial_obstacle_required,
                "one_step_prediction_error": None,
                "one_step_error_in_disturbance_ellipsoid": None,
                "one_step_error_disturbance_score": None,
                "solver_status": "initial_state",
                "timings_sec": {
                    "vit_encode": 0.0,
                    "mppi_run": 0.0,
                    "sls_solve": 0.0,
                    "total": 0.0,
                },
            }
        ]
        X_ref = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
        prev_u0 = np.zeros(action_dim, dtype=np.float32)
        prev_mppi_U = jnp.zeros((mppi_horizon, action_dim), dtype=jnp.float64)
        jax_seed_key = jax.random.PRNGKey(cfg.seed)
        stop_reason = "max_mpc_steps"

        pbar = tqdm(range(cfg.max_mpc_steps), desc="Rope MPPI + Conformal SLS execution loop")
        for step_idx in pbar:
            jax_seed_key, subkey = jax.random.split(jax_seed_key)
            init_act_seq = jnp.concatenate([prev_mppi_U[1:], prev_mppi_U[-1:]], axis=0)
            X_warmstart = X_ref
            U_warmstart = jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
            mppi_ok = False
            mppi_time_sec = 0.0
            sls_time_sec = 0.0
            encode_time_sec = 0.0
            pre_step_state = current_state.copy()

            try:
                mppi_start = time.perf_counter()
                mppi_res = jit_mppi_trajopt(subkey, jnp.asarray(current_state), init_act_seq)
                mppi_time_sec = time.perf_counter() - mppi_start
                X_mppi = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
                U_mppi = jnp.asarray(mppi_res["act_seq"])
                if np.all(np.isfinite(np.asarray(X_mppi))) and np.all(np.isfinite(np.asarray(U_mppi))):
                    X_warmstart = X_mppi[: cfg.horizon + 1]
                    U_warmstart = U_mppi[: cfg.horizon]
                    prev_mppi_U = U_mppi
                    mppi_ok = True
            except Exception:
                pass

            controller.X_in = X_warmstart
            controller.U_in = U_warmstart

            try:
                sls_start = time.perf_counter()
                u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_warmstart, parameter=mpc_dt)
                sls_time_sec = time.perf_counter() - sls_start
                solver_status = "sls_refined" if mppi_ok else "sls_mpc"
            except Exception:
                if mppi_ok:
                    u0, X_pred, U_pred = U_warmstart[0], X_warmstart, U_warmstart
                    solver_status = "mppi_fallback"
                else:
                    u0, X_pred, U_pred = None, None, None
                    solver_status = "exception_fallback"

            if u0 is None or X_pred is None or U_pred is None:
                u0, X_pred, U_pred = None, None, None
                solver_status = "exception_fallback"
            elif not (
                np.all(np.isfinite(np.asarray(u0)))
                and np.all(np.isfinite(np.asarray(X_pred)))
                and np.all(np.isfinite(np.asarray(U_pred)))
            ):
                if mppi_ok:
                    u0, X_pred, U_pred = U_warmstart[0], X_warmstart, U_warmstart
                    solver_status = "mppi_fallback"
                else:
                    u0, X_pred, U_pred = None, None, None
                    solver_status = "nonfinite_fallback"

            if u0 is None:
                u0 = prev_u0
                solver_status = "frozen_fallback"
            else:
                prev_u0 = np.asarray(u0, dtype=np.float32)
                if mppi_ok:
                    refined_U = jnp.asarray(U_pred)
                    if refined_U.shape[0] == cfg.horizon:
                        prev_mppi_U = prev_mppi_U.at[: cfg.horizon].set(refined_U)

            u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
            u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
            predicted_next_state = predict_next_state_torch(model.predictor.net, pre_step_state, u0_norm, device)

            current_frame, current_info = step_env_with_action(
                env,
                renderer,
                action=u0_raw,
                control_decimation=control_decimation,
                camera_id=camera_id,
                elapsed_time=(step_idx + 1) * control_timestep,
                disable_shadows=disable_shadows,
            )
            rollout_frames.append(current_frame.copy())
            executed_actions_norm.append(u0_norm.astype(np.float32))
            executed_actions_raw.append(u0_raw.astype(np.float32))

            encode_start = time.perf_counter()
            next_emb = encode_single_frame(model, current_frame, device, img_size)
            encode_time_sec = time.perf_counter() - encode_start
            current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
            current_emb = next_emb
            current_task_target = np.asarray(current_info["task_target"], dtype=np.float64).copy()
            current_time = float(current_info["time"][0])
            executed_states.append(current_state.copy())
            executed_embeddings.append(current_emb.detach().cpu().numpy().astype(np.float64).copy())
            executed_task_targets.append(current_task_target)
            executed_times.append(current_time)

            latent_err = float(np.linalg.norm(current_state - goal_state))
            task_err = float(np.linalg.norm(current_task_target - task_target_np[-1].astype(np.float64)))
            one_step_error = current_state - predicted_next_state
            err_in_ellipsoid, err_ellipsoid_score = compute_error_ellipsoid_membership(
                error_np=one_step_error,
                state_np=pre_step_state,
                action_np=u0_norm,
                use_constant_covariance=cfg.use_constant_covariance,
                calibrated_cholesky=calibrated_cholesky,
                error_model=error_model_torch,
                q_learned=cfg.q_learned,
                device=device,
            )
            obstacle_free, obstacle_score, obstacle_required = obstacle_safety_status(
                current_state, obstacle_model, cfg.obstacle_margin
            )
            state_in_ellipsoid, ellipsoid_score = state_ellipsoid_membership(
                current_state, ellipsoid_unit_precision, cfg.latent_ellipsoid_margin
            )
            latent_goal_distances.append(latent_err)
            task_target_distances.append(task_err)
            step_records.append(
                {
                    "step": int(step_idx + 1),
                    "phase": "post_step",
                    "markov_state": current_state.astype(np.float64).tolist(),
                    "embedding": executed_embeddings[-1].tolist(),
                    "task_target": current_task_target.tolist(),
                    "elapsed_time": current_time,
                    "latent_goal_error": float(latent_err),
                    "task_error": float(task_err),
                    "state_in_latent_ellipsoid": state_in_ellipsoid,
                    "state_latent_ellipsoid_score": ellipsoid_score,
                    "obstacle_free": obstacle_free,
                    "obstacle_score": obstacle_score,
                    "obstacle_required_score": obstacle_required,
                    "one_step_prediction_error": one_step_error.astype(np.float64).tolist(),
                    "one_step_error_in_disturbance_ellipsoid": err_in_ellipsoid,
                    "one_step_error_disturbance_score": err_ellipsoid_score,
                    "solver_status": solver_status,
                    "timings_sec": {
                        "vit_encode": float(encode_time_sec),
                        "mppi_run": float(mppi_time_sec),
                        "sls_solve": float(sls_time_sec),
                        "total": float(encode_time_sec + mppi_time_sec + sls_time_sec),
                    },
                }
            )
            pbar.set_postfix(
                latent_error=f"{latent_err:.4f}",
                task_error=f"{task_err:.4f}",
                obs_free=obstacle_free if obstacle_free is not None else "n/a",
                ellip_in=state_in_ellipsoid if state_in_ellipsoid is not None else "n/a",
                status=solver_status,
            )

            if latent_err <= 0.2 or task_err <= 0.05:
                stop_reason = "goal_reached"
                break
        else:
            stop_reason = "max_mpc_steps"

    post_step_records = [record for record in step_records if record["phase"] == "post_step"]
    disturbance_checks = [record["one_step_error_in_disturbance_ellipsoid"] for record in post_step_records]
    valid_disturbance_checks = [bool(flag) for flag in disturbance_checks if flag is not None]
    obstacle_checks = [record["obstacle_free"] for record in step_records]
    valid_obstacle_checks = [bool(flag) for flag in obstacle_checks if flag is not None]
    state_ellipsoid_checks = [record["state_in_latent_ellipsoid"] for record in step_records]
    valid_state_ellipsoid_checks = [bool(flag) for flag in state_ellipsoid_checks if flag is not None]
    timing_totals = {
        "vit_encode": float(sum(record["timings_sec"]["vit_encode"] for record in post_step_records)),
        "mppi_run": float(sum(record["timings_sec"]["mppi_run"] for record in post_step_records)),
        "sls_solve": float(sum(record["timings_sec"]["sls_solve"] for record in post_step_records)),
        "total": float(sum(record["timings_sec"]["total"] for record in post_step_records)),
    }

    imageio.mimwrite(run_dir / "mppi_sls_rope.mp4", rollout_frames, fps=cfg.video_fps, quality=8, macro_block_size=1)
    np.savez(
        run_dir / "executed_actions.npz",
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    np.savez(
        run_dir / "executed_states.npz",
        markov_states=np.asarray(executed_states, dtype=np.float64),
        embeddings=np.asarray(executed_embeddings, dtype=np.float64),
        task_targets=np.asarray(executed_task_targets, dtype=np.float64),
        elapsed_time=np.asarray(executed_times, dtype=np.float64),
        latent_goal_distances=np.asarray(latent_goal_distances, dtype=np.float64),
        task_target_distances=np.asarray(task_target_distances, dtype=np.float64),
    )
    summary_payload = {
        "metadata": {
            "run_dir": str(run_dir),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint_path),
            "dataset_path": str(cfg.dataset_path.expanduser().resolve()),
            "episode_idx": int(episode_idx),
            "seed": int(cfg.seed),
            "stop_reason": stop_reason,
            "goal_reached": bool(stop_reason == "goal_reached"),
            "trajectory_safe_by_classifier": None if not valid_obstacle_checks else bool(all(valid_obstacle_checks)),
            "disturbance_ellipsoid_coverage": {
                "covered_steps": int(sum(valid_disturbance_checks)),
                "checked_steps": int(len(valid_disturbance_checks)),
                "fraction": None if not valid_disturbance_checks else float(sum(valid_disturbance_checks) / len(valid_disturbance_checks)),
            },
            "state_latent_ellipsoid_coverage": {
                "covered_steps": int(sum(valid_state_ellipsoid_checks)),
                "checked_steps": int(len(valid_state_ellipsoid_checks)),
                "fraction": None if not valid_state_ellipsoid_checks else float(sum(valid_state_ellipsoid_checks) / len(valid_state_ellipsoid_checks)),
            },
            "executed_steps": int(len(executed_actions_norm)),
            "num_logged_states": int(len(executed_states)),
            "timing_totals_sec": timing_totals,
            "camera_name": str(camera_name),
            "control_decimation": int(control_decimation),
            "control_timestep": float(control_timestep),
            "video_path": str(run_dir / "mppi_sls_rope.mp4"),
            "artifacts": {
                "executed_actions_path": str(run_dir / "executed_actions.npz"),
                "executed_states_path": str(run_dir / "executed_states.npz"),
                "trajectory_summary_path": str(run_dir / "trajectory_summary.json"),
            },
        },
        "start_goal": {
            "start_task_target": np.asarray(task_target_np[0], dtype=np.float64).tolist(),
            "goal_task_target": np.asarray(task_target_np[-1], dtype=np.float64).tolist(),
            "start_qpos": np.asarray(qpos_np[0], dtype=np.float64).tolist(),
            "goal_qpos": np.asarray(qpos_np[-1], dtype=np.float64).tolist(),
            "start_state": np.asarray(start_state, dtype=np.float64).tolist(),
            "goal_state": np.asarray(goal_state, dtype=np.float64).tolist(),
        },
        "step_records": step_records,
    }
    with (run_dir / "trajectory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    print(f"Rope MPPI + SLS MPC planning sequence logged cleanly inside: {run_dir}")

if __name__ == "__main__":
    main()
