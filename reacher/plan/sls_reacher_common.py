#!/usr/bin/env python3
"""Shared Reacher Conformal SLS MPC utilities."""

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform != "darwin" else "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

ERROR_MODEL_ROOT = Path(__file__).resolve().parents[2] / "error_calib" / "error_model"
if str(ERROR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ERROR_MODEL_ROOT))

import equinox as eqx
import h5py
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
import torch
from jax import config, lax
from tqdm.auto import tqdm

from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.mppi_planner import MPPIPlanner
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.mlpdyn_train import LeWMReacherDataset
from reacher.train.reacher_policy_train import DmControlGymEnv, flatten_observation

config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)


DEFAULT_START_QPOS_A = (0.146451935172081, -0.7491843104362488)
DEFAULT_GOAL_QPOS_A = (2.4196903705596924, -0.9535070657730103)
DEFAULT_START_QPOS_B = (0.1, -2.0)
DEFAULT_GOAL_QPOS_B = (2.42, -0.95)


@dataclass
class PlanSLSReacherConfig:
    """Configuration for Reacher Conformal SLS MPC."""

    q_learned: float = 0.0
    model_dir: Path = Path("reacher/models/mlpdyn_ft_6")
    error_model_ckpt: Path = Path("reacher/models/error_model/best-error-model.ckpt")
    use_constant_covariance: bool = True
    constant_covariance_path: Path = Path("reacher/eval/fixed_error_covariance.pt")
    dataset_path: Path = Path("reacher/data/test_data_50hz/reacher_test.h5")
    action_stats_dataset_path: Optional[Path] = Path("reacher/data/expert_data_50hz/reacher_expert.h5")
    out_dir: Path = Path("reacher/plan/sls_mpc_conformal")
    device: str = "auto"
    episode_idx: Optional[int] = None
    start_goal_path: Optional[Path] = None
    seed: int = 42
    horizon: int = 20
    max_mpc_steps: int = 100
    video_fps: int = 60
    frame_batch_size: int = 32
    swap_start_goal: bool = True
    start_qpos: Optional[list[float]] = field(default=None)
    goal_qpos: Optional[list[float]] = field(default=None)
    goal_threshold: float = 0.05
    latent_goal_threshold: float = 0.05
    q_stage: float = 0.005
    q_terminal: float = 5.0
    q_stage_delta: float = 1.0
    q_terminal_delta: float = 1.0
    r_control: float = 0.01
    r_control_per_action: Optional[list[float]] = field(default=None)
    u_abs_limit: float = 500.0
    state_abs_limit: float = 5.0
    state_delta_abs_limit: float = 1.0
    enable_obstacle: bool = False
    obstacle_model_path: Path = Path("reacher/models/obs_net/model.pt")
    obstacle_margin: float = 0.0
    obstacle_penalty_weight: float = 1000.0
    use_latent_ellipsoid_constraint: bool = False
    latent_ellipsoid_path: Path = Path("reacher/eval/latent_ellipsoid/latent_ellipsoid.pt")
    latent_ellipsoid_margin: float = 0.0
    mppi_horizon: Optional[int] = None
    mppi_samples: int = 512
    mppi_update_iter: int = 5
    mppi_reward_weight: float = 20.0
    mppi_noise_level: float = 0.15
    mppi_beta_filter: float = 0.7
    mppi_stage_weight: float = 0.005
    mppi_terminal_weight: float = 5.0
    mppi_delta_weight: float = 1.0
    mppi_r_control: float = 0.01
    mppi_r_control_per_action: Optional[list[float]] = field(default=None)
    mppi_state_box_penalty: float = 0.0
    mppi_ellipsoid_penalty_weight: float = 0.0


class JAXObstacleMLP(eqx.Module):
    linear_layers: list
    layer_norm_scales: list
    layer_norm_biases: list
    feature_mean: jax.Array
    feature_std: jax.Array
    threshold: jax.Array
    input_dim: int

    def __call__(self, state):
        x = (state[: self.input_dim] - self.feature_mean) / self.feature_std
        for i, linear in enumerate(self.linear_layers[:-1]):
            x = linear(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = x * self.layer_norm_scales[i] + self.layer_norm_biases[i]
            x = jax.nn.gelu(x)
        return self.linear_layers[-1](x).squeeze(-1)


def build_jax_obstacle_from_artifact(path: Path, key: jax.Array) -> JAXObstacleMLP:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"]
    input_dim = int(payload["input_dim"])
    hidden_dim = int(payload["hidden_dim"])
    depth = int(payload["depth"])
    dropout = float(payload.get("dropout", 0.0))
    keys = jax.random.split(key, depth)
    layers = []
    ln_scales = []
    ln_biases = []
    module_idx = 0
    in_dim = input_dim
    for i in range(depth - 1):
        linear = eqx.nn.Linear(in_dim, hidden_dim, key=keys[i])
        linear = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            linear,
            (
                jnp.asarray(state_dict[f"net.{module_idx}.weight"].detach().cpu().numpy()),
                jnp.asarray(state_dict[f"net.{module_idx}.bias"].detach().cpu().numpy()),
            ),
        )
        layers.append(linear)
        ln_scales.append(jnp.asarray(state_dict[f"net.{module_idx + 1}.weight"].detach().cpu().numpy()))
        ln_biases.append(jnp.asarray(state_dict[f"net.{module_idx + 1}.bias"].detach().cpu().numpy()))
        module_idx += 4 if dropout > 0.0 else 3
        in_dim = hidden_dim
    linear = eqx.nn.Linear(in_dim, 1, key=keys[-1])
    linear = eqx.tree_at(
        lambda layer: (layer.weight, layer.bias),
        linear,
        (
            jnp.asarray(state_dict[f"net.{module_idx}.weight"].detach().cpu().numpy()),
            jnp.asarray(state_dict[f"net.{module_idx}.bias"].detach().cpu().numpy()),
        ),
    )
    layers.append(linear)
    return JAXObstacleMLP(
        linear_layers=layers,
        layer_norm_scales=ln_scales,
        layer_norm_biases=ln_biases,
        feature_mean=jnp.asarray(payload["feature_mean"], dtype=jnp.float64),
        feature_std=jnp.maximum(jnp.asarray(payload["feature_std"], dtype=jnp.float64), 1e-6),
        threshold=jnp.asarray(float(payload["conformal_safe_score_threshold"]), dtype=jnp.float64),
        input_dim=input_dim,
    )


def build_equinox_mlp_from_pytorch(pt_model: torch.nn.Module, key: jax.Array) -> eqx.Module:
    pt_linears = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    keys = jax.random.split(key, len(pt_linears))
    layers = []
    for i, pt_layer in enumerate(pt_linears):
        out_f, in_f = pt_layer.weight.shape
        linear = eqx.nn.Linear(in_f, out_f, key=keys[i])
        linear = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            linear,
            (
                jnp.asarray(pt_layer.weight.detach().cpu().numpy()),
                jnp.asarray(pt_layer.bias.detach().cpu().numpy()),
            ),
        )
        layers.append(linear)
        if i < len(pt_linears) - 1:
            layers.append(jax.nn.gelu)

    class JAXMLP(eqx.Module):
        layers: list

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return JAXMLP(layers)


def make_jax_dynamics(eqx_dyn):
    return lambda x, u, t=0.0, parameter=1.0: eqx_dyn(jnp.concatenate([x, u], axis=-1))


def make_constant_jax_disturbance(cholesky: np.ndarray, state_dim: int):
    cholesky_j = jnp.asarray(cholesky, dtype=jnp.float64)
    if cholesky_j.shape != (state_dim, state_dim):
        raise ValueError(f"Expected calibrated Cholesky shape {(state_dim, state_dim)}, got {cholesky_j.shape}.")
    return lambda X_prefix, U_prefix: jnp.broadcast_to(cholesky_j, (X_prefix.shape[0], state_dim, state_dim))


def load_calibrated_cholesky(path: Path) -> np.ndarray:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    if "calibrated_cholesky" in payload:
        matrix = payload["calibrated_cholesky"]
    elif "cholesky" in payload and "q_fixed" in payload:
        matrix = payload["cholesky"] * payload["q_fixed"]
    else:
        raise KeyError(f"{path} must contain 'calibrated_cholesky' or both 'cholesky' and 'q_fixed'.")
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float64)


def make_jax_disturbance(eqx_error_model, q_learned: float, state_dim: int, diagonal: bool):
    def _mgnll_forward(raw):
        if diagonal:
            return jnp.diag(jnp.exp(raw) + 1e-4)
        L = jnp.zeros((state_dim, state_dim), dtype=jnp.float64)
        L = L.at[jnp.tril_indices(state_dim)].set(raw)
        diag = jnp.arange(state_dim)
        return L.at[diag, diag].set(jnp.exp(L[diag, diag]) + 1e-4)

    return lambda X, U: float(q_learned) * jax.vmap(_mgnll_forward)(
        jax.vmap(eqx_error_model)(jnp.concatenate([X, U], axis=-1))
    )


def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    return lambda x, u, t: jnp.concatenate([u - u_max, u_min - u], axis=0)


def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    return lambda x, u, t: jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])


def make_markov_ellipsoid_constraint(unit_precision: np.ndarray, margin: float):
    P = jnp.asarray(unit_precision, dtype=jnp.float64)
    return lambda x, u, t: jnp.asarray([x @ P @ x - 1.0 + float(margin)])


def load_markov_ellipsoid_unit_precision(path: Path, state_dim: int) -> np.ndarray:
    path = path.expanduser()
    if path.is_dir():
        for name in ("latent_ellipsoid.pt", "latent_ellipsoid.npz"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
    if path.suffix == ".npz":
        payload = np.load(path)
        matrix = payload["markov_unit_precision"] if "markov_unit_precision" in payload else payload["unit_precision"]
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        matrix = payload["markov_unit_precision"] if "markov_unit_precision" in payload else payload["unit_precision"]
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (state_dim, state_dim):
        raise ValueError(f"Expected ellipsoid matrix shape {(state_dim, state_dim)}, got {matrix.shape}.")
    return matrix


def latest_object_checkpoint(model_dir: Path) -> Path:
    patterns = (
        re.compile(r".*_epoch_(\d+)_object\.ckpt$"),
        re.compile(r".*_epoch[=_](\d+).*\.ckpt$"),
    )
    candidates = []
    for path in model_dir.glob("*.ckpt"):
        for pattern in patterns:
            match = pattern.match(path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path))
                break
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def require_device(device_arg: str) -> torch.device:
    if device_arg in {"auto", "gpu"}:
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def configure_dm_control_timing(env: DmControlGymEnv, *, physics_timestep: float, time_limit: float) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def make_render_env(seed: int, time_limit: float, width: int, height: int, physics_freq_hz: float) -> DmControlGymEnv:
    extended_time_limit = float(time_limit) * 5.0
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name="hard",
        seed=seed,
        time_limit=extended_time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / physics_freq_hz, time_limit=extended_time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    return env


def reset_env_to_state(env: DmControlGymEnv, seed: int, qpos: np.ndarray, qvel: np.ndarray, height: int, width: int) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def preprocess_pixels(pixels: np.ndarray, *, img_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (tensor.to(device) - mean) / std


@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel: np.ndarray, *, device: torch.device, img_size: int) -> torch.Tensor:
    batch = preprocess_pixels(pixel, img_size=img_size, device=device)
    output = model.encoder(batch, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]


def make_markov_state(embedding: torch.Tensor, previous_embedding: torch.Tensor | None = None) -> torch.Tensor:
    delta = torch.zeros_like(embedding) if previous_embedding is None else embedding - previous_embedding
    return torch.cat((embedding, delta), dim=-1)


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def load_action_stats(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = LeWMReacherDataset(
        dataset_path,
        history_size=1,
        num_preds=1,
        frameskip=1,
        img_size=224,
        action_dim=action_dim,
    )
    return dataset.action_mean.astype(np.float32), dataset.action_std.astype(np.float32)


def load_episode(dataset_path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, object], int]:
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        valid = np.flatnonzero(ep_len >= 2)
        if valid.size == 0:
            raise ValueError("Need at least one episode with 2 or more frames.")
        if episode_idx is None:
            episode_idx = int(rng.choice(valid))
        if not 0 <= int(episode_idx) < ep_len.shape[0]:
            raise ValueError(f"episode_idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        episode = {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "qpos": np.asarray(h5["qpos"][rows], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][rows], dtype=np.float32),
            "observation": np.asarray(h5["observation"][rows], dtype=np.float32),
            "episode_seed": int(h5["episode_seed"][episode_idx]) if "episode_seed" in h5 else int(seed),
            "physics_freq_hz": float(h5.attrs.get("physics_freq_hz", 100.0)),
            "time_limit": float(h5.attrs.get("time_limit", 10.0)),
            "height": int(h5["pixels"].shape[1]),
            "width": int(h5["pixels"].shape[2]),
        }
    return episode, int(episode_idx)


def _load_start_goal_pairs(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "pairs" in payload:
            pairs = payload["pairs"]
            metadata = payload.get("metadata", {})
        else:
            pairs = payload
            metadata = {}
    else:
        pairs = payload
        metadata = {}
    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError(f"Expected a nonempty list of start/goal pairs in {path}.")
    return pairs, metadata if isinstance(metadata, dict) else {}


def resolve_start_goal_qpos(
    cfg: PlanSLSReacherConfig,
    qpos: np.ndarray,
    *,
    pair_episode_idx: Optional[int],
) -> tuple[np.ndarray, np.ndarray, str, Optional[int]]:
    if cfg.start_goal_path is not None:
        pairs, _metadata = _load_start_goal_pairs(cfg.start_goal_path)
        rng = np.random.default_rng(cfg.seed)
        selected_idx = int(pair_episode_idx) if pair_episode_idx is not None else int(rng.integers(len(pairs)))
        if not 0 <= selected_idx < len(pairs):
            raise ValueError(f"start_goal episode_idx must be in [0, {len(pairs) - 1}], got {selected_idx}.")
        pair = pairs[selected_idx]
        if not isinstance(pair, dict) or "start" not in pair or "goal" not in pair:
            raise ValueError(f"Each start/goal pair in {cfg.start_goal_path} must have 'start' and 'goal' entries.")
        start = pair["start"]
        goal = pair["goal"]
        if not isinstance(start, dict) or not isinstance(goal, dict) or "qpos" not in start or "qpos" not in goal:
            raise ValueError(f"Each pair in {cfg.start_goal_path} must contain start.qpos and goal.qpos.")
        start_qpos = np.asarray(start["qpos"], dtype=np.float32).reshape(-1)
        goal_qpos = np.asarray(goal["qpos"], dtype=np.float32).reshape(-1)
        if start_qpos.shape != (qpos.shape[1],) or goal_qpos.shape != (qpos.shape[1],):
            raise ValueError(f"start/goal qpos in {cfg.start_goal_path} must have shape {(qpos.shape[1],)}.")
        return start_qpos, goal_qpos, "start_goal_pt", selected_idx
    if (cfg.start_qpos is None) != (cfg.goal_qpos is None):
        raise ValueError("start_qpos and goal_qpos must both be provided or both omitted.")
    if cfg.start_qpos is not None:
        start = np.asarray(cfg.start_qpos, dtype=np.float32)
        goal = np.asarray(cfg.goal_qpos, dtype=np.float32)
        if start.shape != (qpos.shape[1],) or goal.shape != (qpos.shape[1],):
            raise ValueError(f"start_qpos/goal_qpos must have shape {(qpos.shape[1],)}.")
        return start, goal, "config_qpos", None
    start_idx = -1 if cfg.swap_start_goal else 0
    goal_idx = 0 if cfg.swap_start_goal else -1
    return qpos[start_idx].astype(np.float32).copy(), qpos[goal_idx].astype(np.float32).copy(), "dataset_episode", None


def build_goal_observation(env: DmControlGymEnv, goal_qpos: np.ndarray, goal_qvel: np.ndarray, obs_dim: int) -> np.ndarray:
    if obs_dim == 8:
        return np.concatenate((goal_qpos, goal_qvel, goal_qpos, np.zeros_like(goal_qpos)), axis=0).astype(np.float32)
    if obs_dim == 6:
        return flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    raise ValueError(f"Unsupported observation dimension: {obs_dim}.")


def build_observation_from_env(env: DmControlGymEnv, goal_obs: np.ndarray) -> np.ndarray:
    obs_dim = int(goal_obs.shape[0])
    if obs_dim == 6:
        return flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    if obs_dim == 8:
        qpos = np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy()
        qvel = np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy()
        goal_qpos = goal_obs[4:6].astype(np.float32).copy()
        return np.concatenate((qpos, qvel, goal_qpos, goal_qpos - qpos), axis=0).astype(np.float32)
    raise ValueError(f"Unsupported observation dimension: {obs_dim}.")


def goal_distance(current_obs: np.ndarray, goal_obs: np.ndarray) -> float:
    if current_obs.shape[0] == 8:
        return float(np.linalg.norm(current_obs[:2] - current_obs[4:6]))
    return float(np.linalg.norm(current_obs - goal_obs))


def qpos_goal_distance(env: DmControlGymEnv, goal_qpos: np.ndarray) -> float:
    current_qpos = np.asarray(env._env.physics.data.qpos[: goal_qpos.shape[0]], dtype=np.float32)
    return float(np.linalg.norm(current_qpos - goal_qpos))


def current_qpos_qvel(env: DmControlGymEnv, qpos_dim: int, qvel_dim: int) -> tuple[np.ndarray, np.ndarray]:
    physics = env._env.physics
    qpos = np.asarray(physics.data.qpos[:qpos_dim], dtype=np.float32).copy()
    qvel = np.asarray(physics.data.qvel[:qvel_dim], dtype=np.float32).copy()
    return qpos, qvel


@torch.no_grad()
def predict_next_state_torch(
    torch_dynamics_net: torch.nn.Module,
    state_np: np.ndarray,
    action_np: np.ndarray,
    *,
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


def resolve_control_weights(
    action_dim: int,
    *,
    scalar_weight: float,
    per_action_weight: Optional[list[float]],
    field_name: str,
) -> jnp.ndarray:
    if per_action_weight is None:
        return jnp.ones((action_dim,), dtype=jnp.float64) * float(scalar_weight)
    weights = np.asarray(per_action_weight, dtype=np.float64).reshape(-1)
    if weights.shape != (action_dim,):
        raise ValueError(f"{field_name} must have shape ({action_dim},), got {weights.shape}.")
    return jnp.asarray(weights, dtype=jnp.float64)


def make_tracking_cost(
    r_control_weights: jnp.ndarray,
    horizon: int,
    W_stage: jnp.ndarray,
    W_terminal: jnp.ndarray,
    goal_state: jnp.ndarray,
    obstacle_model: JAXObstacleMLP | None,
    obstacle_margin: float,
    obstacle_penalty_weight: float,
):
    r_control_weights = jnp.asarray(r_control_weights, dtype=jnp.float64)

    def cost(W_ignored, reference, z, u, t):
        active_W = jnp.where(t < horizon, W_stage, W_terminal)
        dz = z - goal_state
        total = jnp.sum(active_W * dz**2) + jnp.sum(r_control_weights * u**2)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            violation = jax.nn.softplus(obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z))
            total = total + float(obstacle_penalty_weight) * violation**2
        return total

    return cost


def make_mppi_rollout_and_eval(
    dynamics,
    W_stage,
    W_terminal,
    goal_state,
    *,
    obstacle_model: JAXObstacleMLP | None,
    obstacle_margin: float,
    obstacle_penalty_weight: float,
    box_min: jnp.ndarray | None,
    box_max: jnp.ndarray | None,
    box_penalty_weight: float,
    ellipsoid_unit_precision: jnp.ndarray | None,
    ellipsoid_margin: float,
    ellipsoid_penalty_weight: float,
    r_control_weights: jnp.ndarray,
):
    r_control_weights = jnp.asarray(r_control_weights, dtype=jnp.float64)

    def rollout(state_cur, act_seqs, reach_config=None):
        def step_fn(state, action):
            nxt = dynamics(state, action, 0.0, 1.0)
            return nxt, nxt

        return jax.vmap(lambda actions: lax.scan(step_fn, state_cur, actions)[1])(act_seqs), {}

    def eval_fn(states, acts, reach_config=None, aux=None, *args, **kwargs):
        delta = states - goal_state[None, None, :]
        stage = jnp.sum(W_stage[None, None, :] * delta**2, axis=-1)
        terminal = jnp.sum(W_terminal[None, :] * delta[:, -1, :] ** 2, axis=-1)
        action_cost = jnp.sum(r_control_weights[None, None, :] * acts**2, axis=-1)
        box_cost = jnp.zeros_like(stage)
        if box_min is not None and box_max is not None and box_penalty_weight > 0.0:
            lower = jnp.maximum(box_min[None, None, :] - states, 0.0)
            upper = jnp.maximum(states - box_max[None, None, :], 0.0)
            box_cost = float(box_penalty_weight) * jnp.sum(lower**2 + upper**2, axis=-1)
        ellipsoid_cost = jnp.zeros_like(stage)
        if ellipsoid_unit_precision is not None and ellipsoid_penalty_weight > 0.0:
            scores = jnp.einsum("bti,ij,btj->bt", states, ellipsoid_unit_precision, states)
            violation = jnp.maximum(scores - 1.0 + float(ellipsoid_margin), 0.0)
            ellipsoid_cost = float(ellipsoid_penalty_weight) * violation**2
        obstacle_cost = jnp.zeros_like(stage)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            flat = states.reshape((-1, states.shape[-1]))
            violation = jax.vmap(
                lambda z: jax.nn.softplus(obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z))
            )(flat).reshape(states.shape[:-1])
            obstacle_cost = float(obstacle_penalty_weight) * violation**2
        total = jnp.sum(stage + action_cost + box_cost + ellipsoid_cost + obstacle_cost, axis=-1) + terminal
        return {"rewards": -total}

    return rollout, eval_fn


def save_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    mp4 = out_dir / "rollout.mp4"
    try:
        imageio.mimwrite(mp4, frames, fps=fps, quality=8, macro_block_size=1)
        return mp4
    except Exception:
        gif = out_dir / "rollout.gif"
        imageio.mimwrite(gif, frames, fps=fps)
        return gif


def run_planner(*, use_mppi: bool) -> None:
    cfg = pyrallis.parse(config_class=PlanSLSReacherConfig)
    device = require_device(cfg.device)
    model_dir = cfg.model_dir.expanduser().resolve()
    dataset_path = cfg.dataset_path.expanduser().resolve()
    out_root = cfg.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)
    checkpoint = latest_object_checkpoint(model_dir).resolve()
    model = torch.load(checkpoint, map_location=device, weights_only=False).to(device).eval()
    model.requires_grad_(False)

    state_dim = int(model_cfg.get("markov_state_dim", 2 * int(model_cfg.get("embed_dim", 5))))
    embed_dim = int(model_cfg.get("embed_dim", state_dim // 2))
    action_dim = int(model_cfg.get("action_dim", 2))
    img_size = int(model_cfg.get("img_size", 224))

    key_dyn, key_err, key_obs = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
    dynamics = make_jax_dynamics(build_equinox_mlp_from_pytorch(model.predictor.net, key_dyn))
    error_model_torch = None
    calibrated_cholesky_np = None
    if cfg.use_constant_covariance:
        calibrated_cholesky_np = load_calibrated_cholesky(cfg.constant_covariance_path)
        disturbance = make_constant_jax_disturbance(calibrated_cholesky_np, state_dim)
        print(f"Using fixed calibrated covariance from {cfg.constant_covariance_path}")
    else:
        from error_model import MGNLLPredictor

        error_model_torch = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
        disturbance = make_jax_disturbance(
            build_equinox_mlp_from_pytorch(error_model_torch.net, key_err),
            cfg.q_learned,
            state_dim,
            bool(error_model_torch.diagonal),
        )
        print(f"Using learned error model from {cfg.error_model_ckpt}")

    obstacle_model = None
    obstacle_constraint = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, key_obs)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(f"Obstacle input_dim={obstacle_model.input_dim} exceeds state_dim={state_dim}.")
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(f"Using obstacle classifier from {cfg.obstacle_model_path}")
    else:
        print("Obstacle avoidance disabled.")

    if cfg.action_stats_dataset_path is not None:
        action_stats_path = cfg.action_stats_dataset_path.expanduser().resolve()
    else:
        configured_stats_path = Path(str(model_cfg.get("dataset_path", dataset_path))).expanduser().resolve()
        action_stats_path = configured_stats_path if configured_stats_path.is_file() else dataset_path
    action_mean, action_std = load_action_stats(action_stats_path.expanduser().resolve(), action_dim)
    dataset_episode_idx_arg = None if cfg.start_goal_path is not None else cfg.episode_idx
    episode, dataset_episode_idx = load_episode(dataset_path, dataset_episode_idx_arg, cfg.seed)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    obs_np = np.asarray(episode["observation"], dtype=np.float32)
    episode_seed = int(episode["episode_seed"])
    height = int(episode["height"])
    width = int(episode["width"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    zero_qvel = np.zeros_like(qvel_np[0], dtype=np.float32)
    start_qpos, goal_qpos, start_goal_source, start_goal_episode_idx = resolve_start_goal_qpos(
        cfg,
        qpos_np,
        pair_episode_idx=cfg.episode_idx,
    )

    run_index = start_goal_episode_idx if start_goal_episode_idx is not None else dataset_episode_idx
    run_name = f"{int(time.time())}_{'mppi_' if use_mppi else ''}sls_reacher_{start_goal_source}_{run_index:05d}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_render_env(episode_seed, time_limit, width, height, physics_freq_hz)
    start_frame = reset_env_to_state(env, episode_seed, start_qpos, zero_qvel, height, width)
    start_emb = encode_single_frame(model, start_frame, device=device, img_size=img_size)
    goal_frame = reset_env_to_state(env, episode_seed, goal_qpos, zero_qvel, height, width)
    goal_emb = encode_single_frame(model, goal_frame, device=device, img_size=img_size)
    start_state = make_markov_state(start_emb).detach().cpu().numpy().astype(np.float64)
    goal_state = make_markov_state(goal_emb).detach().cpu().numpy().astype(np.float64)
    imageio.imwrite(run_dir / "start_image.png", start_frame)
    imageio.imwrite(run_dir / "goal_image.png", goal_frame)

    goal_obs = build_goal_observation(env, goal_qpos, zero_qvel, int(obs_np.shape[1]))
    current_frame = reset_env_to_state(env, episode_seed, start_qpos, zero_qvel, height, width)
    current_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size)
    current_state = make_markov_state(current_emb).detach().cpu().numpy().astype(np.float64)
    current_obs = build_observation_from_env(env, goal_obs)
    sls_r_control_weights = resolve_control_weights(
        action_dim,
        scalar_weight=cfg.r_control,
        per_action_weight=cfg.r_control_per_action,
        field_name="r_control_per_action",
    )
    mppi_r_control_weights = resolve_control_weights(
        action_dim,
        scalar_weight=cfg.mppi_r_control,
        per_action_weight=cfg.mppi_r_control_per_action,
        field_name="mppi_r_control_per_action",
    )

    W_stage = jnp.ones((state_dim,), dtype=jnp.float64) * float(cfg.q_stage)
    W_stage = W_stage.at[embed_dim:].set(float(cfg.q_stage_delta))
    W_terminal = jnp.ones((state_dim,), dtype=jnp.float64) * float(cfg.q_terminal)
    W_terminal = W_terminal.at[embed_dim:].set(float(cfg.q_terminal_delta))
    cost = make_tracking_cost(
        sls_r_control_weights,
        cfg.horizon,
        W_stage,
        W_terminal,
        jnp.asarray(goal_state),
        obstacle_model,
        cfg.obstacle_margin,
        cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0,
    )

    u_min = -float(cfg.u_abs_limit) * jnp.ones(action_dim, dtype=jnp.float64)
    u_max = float(cfg.u_abs_limit) * jnp.ones(action_dim, dtype=jnp.float64)
    x_min = -float(cfg.state_abs_limit) * jnp.ones(state_dim, dtype=jnp.float64)
    x_max = float(cfg.state_abs_limit) * jnp.ones(state_dim, dtype=jnp.float64)
    x_min = x_min.at[embed_dim:].set(-float(cfg.state_delta_abs_limit))
    x_max = x_max.at[embed_dim:].set(float(cfg.state_delta_abs_limit))

    ellipsoid_unit_precision = None
    if cfg.use_latent_ellipsoid_constraint:
        ellipsoid_unit_precision = load_markov_ellipsoid_unit_precision(cfg.latent_ellipsoid_path, state_dim)
        state_constraint = make_markov_ellipsoid_constraint(ellipsoid_unit_precision, cfg.latent_ellipsoid_margin)
        state_constraint_count = 1
        for name, state in (("start", start_state), ("goal", goal_state)):
            score = float(state @ ellipsoid_unit_precision @ state)
            if score > 1.0 + float(cfg.latent_ellipsoid_margin):
                raise ValueError(f"{name} state is outside the latent ellipsoid: score={score:.6g}.")
        print(f"Using latent ellipsoid constraint from {cfg.latent_ellipsoid_path}")
    else:
        state_constraint = make_state_box_constraints(x_min, x_max)
        state_constraint_count = 2 * state_dim
        print("Using Markov state box constraint.")

    constraints = (
        combine_constraints(state_constraint, obstacle_constraint, make_control_box_constraints(u_min, u_max))
        if obstacle_constraint is not None
        else combine_constraints(state_constraint, make_control_box_constraints(u_min, u_max))
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        warm_start=use_mppi,
        rti=True,
    )
    sqp_cfg = SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False)
    admm_cfg = ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=400, rho_update_frequency=20, initial_rho=1.0)
    mpc_dt = 1.0 / physics_freq_hz
    mpc_cfg = MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage, u_ref=jnp.zeros(action_dim), dt=mpc_dt)
    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=mpc_cfg,
        dynamics=dynamics,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3)),
        cost=cost,
        num_constraints=2 * action_dim + state_constraint_count + (1 if obstacle_constraint is not None else 0),
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64),
    )

    mppi_horizon = int(cfg.mppi_horizon or cfg.horizon)
    prev_mppi_U = jnp.zeros((mppi_horizon, action_dim), dtype=jnp.float64)
    jit_mppi = None
    if use_mppi:
        if mppi_horizon < cfg.horizon:
            raise ValueError(f"mppi_horizon={mppi_horizon} must be >= horizon={cfg.horizon}.")
        W_mppi_stage = jnp.ones((state_dim,), dtype=jnp.float64) * float(cfg.mppi_stage_weight)
        W_mppi_stage = W_mppi_stage.at[embed_dim:].set(float(cfg.mppi_delta_weight))
        W_mppi_terminal = jnp.ones((state_dim,), dtype=jnp.float64) * float(cfg.mppi_terminal_weight)
        W_mppi_terminal = W_mppi_terminal.at[embed_dim:].set(float(cfg.mppi_delta_weight))
        mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(
            dynamics,
            W_mppi_stage,
            W_mppi_terminal,
            jnp.asarray(goal_state),
            obstacle_model=obstacle_model,
            obstacle_margin=cfg.obstacle_margin,
            obstacle_penalty_weight=cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0,
            box_min=x_min,
            box_max=x_max,
            box_penalty_weight=cfg.mppi_state_box_penalty,
            ellipsoid_unit_precision=None if ellipsoid_unit_precision is None else jnp.asarray(ellipsoid_unit_precision),
            ellipsoid_margin=cfg.latent_ellipsoid_margin,
            ellipsoid_penalty_weight=cfg.mppi_ellipsoid_penalty_weight,
            r_control_weights=mppi_r_control_weights,
        )
        mppi = MPPIPlanner(
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
        jit_mppi = jax.jit(lambda key, state, acts: mppi.trajectory_optimization(key, state, acts, skip=False))

    rollout_frames = [current_frame.copy()]
    executed_actions_norm = []
    executed_actions_raw = []
    current_qpos_np, current_qvel_np = current_qpos_qvel(env, qpos_np.shape[1], qvel_np.shape[1])
    executed_states = [current_state.copy()]
    executed_embeddings = [current_emb.detach().cpu().numpy().astype(np.float64).copy()]
    executed_qpos = [current_qpos_np.astype(np.float64).copy()]
    executed_qvel = [current_qvel_np.astype(np.float64).copy()]
    latent_goal_distances = [float(np.linalg.norm(current_state - goal_state))]
    observation_goal_distances = [goal_distance(current_obs, goal_obs)]
    qpos_goal_distances = [float(np.linalg.norm(current_qpos_np - goal_qpos))]
    solver_statuses = []
    nominal_states = []
    nominal_actions = []
    initial_state_in_ellipsoid, initial_state_ellipsoid_score = state_ellipsoid_membership(
        current_state, ellipsoid_unit_precision, cfg.latent_ellipsoid_margin
    )
    initial_obstacle_free, initial_obstacle_score, initial_obstacle_required = obstacle_safety_status(
        current_state,
        obstacle_model,
        cfg.obstacle_margin,
    )
    step_records = [
        {
            "step": 0,
            "phase": "initial",
            "markov_state": current_state.astype(np.float64).tolist(),
            "embedding": executed_embeddings[0].tolist(),
            "qpos": current_qpos_np.astype(np.float64).tolist(),
            "qvel": current_qvel_np.astype(np.float64).tolist(),
            "latent_goal_error": float(latent_goal_distances[0]),
            "observation_goal_error": float(observation_goal_distances[0]),
            "qpos_goal_error": float(qpos_goal_distances[0]),
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
    X_ref_goal = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
    prev_u0 = np.zeros(action_dim, dtype=np.float32)
    rng_key = jax.random.PRNGKey(cfg.seed)
    stop_reason = "max_mpc_steps"

    pbar = tqdm(range(cfg.max_mpc_steps), desc=("MPPI + SLS Reacher" if use_mppi else "SLS Reacher"))
    for step_idx in pbar:
        X_warmstart = X_ref_goal
        U_warmstart = jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
        mppi_ok = False
        mppi_time_sec = 0.0
        sls_time_sec = 0.0
        encode_time_sec = 0.0
        pre_step_state = current_state.copy()
        if use_mppi and jit_mppi is not None:
            rng_key, subkey = jax.random.split(rng_key)
            try:
                init_actions = jnp.concatenate([prev_mppi_U[1:], prev_mppi_U[-1:]], axis=0)
                mppi_start = time.perf_counter()
                mppi_res = jit_mppi(subkey, jnp.asarray(current_state), init_actions)
                mppi_time_sec = time.perf_counter() - mppi_start
                U_mppi = jnp.asarray(mppi_res["act_seq"])
                X_mppi = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
                if np.all(np.isfinite(np.asarray(U_mppi))) and np.all(np.isfinite(np.asarray(X_mppi))):
                    prev_mppi_U = U_mppi
                    X_warmstart = X_mppi[: cfg.horizon + 1]
                    U_warmstart = U_mppi[: cfg.horizon]
                    mppi_ok = True
            except Exception:
                mppi_ok = False
        controller.X_in = X_warmstart
        controller.U_in = U_warmstart
        try:
            sls_start = time.perf_counter()
            u0, X_pred, U_pred, *_ = controller.run(x0=current_state, reference=X_warmstart, parameter=mpc_dt)
            sls_time_sec = time.perf_counter() - sls_start
            status = "sls_refined" if mppi_ok else "sls_mpc"
        except Exception:
            if mppi_ok:
                u0, X_pred, U_pred = U_warmstart[0], X_warmstart, U_warmstart
                status = "mppi_fallback"
            else:
                u0, X_pred, U_pred = None, None, None
                status = "exception_fallback"
        if u0 is None or X_pred is None or U_pred is None or not np.all(np.isfinite(np.asarray(u0))):
            u0 = prev_u0
            X_pred = X_warmstart
            U_pred = U_warmstart
            status = "frozen_fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)
            if use_mppi and np.asarray(U_pred).shape[0] == cfg.horizon:
                prev_mppi_U = prev_mppi_U.at[: cfg.horizon].set(jnp.asarray(U_pred))

        nominal_states.append(np.asarray(X_pred, dtype=np.float64))
        nominal_actions.append(np.asarray(U_pred, dtype=np.float64))
        solver_statuses.append(status)
        u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        predicted_next_state = predict_next_state_torch(model.predictor.net, pre_step_state, u0_norm, device=device)
        executed_actions_norm.append(u0_norm.astype(np.float32))
        executed_actions_raw.append(u0_raw.astype(np.float32))

        obs, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        encode_start = time.perf_counter()
        next_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size)
        encode_time_sec = time.perf_counter() - encode_start
        current_state = make_markov_state(next_emb, current_emb).detach().cpu().numpy().astype(np.float64)
        current_emb = next_emb
        rollout_frames.append(current_frame.copy())
        current_qpos_np, current_qvel_np = current_qpos_qvel(env, qpos_np.shape[1], qvel_np.shape[1])
        executed_states.append(current_state.copy())
        executed_embeddings.append(current_emb.detach().cpu().numpy().astype(np.float64).copy())
        executed_qpos.append(current_qpos_np.astype(np.float64).copy())
        executed_qvel.append(current_qvel_np.astype(np.float64).copy())
        latent_dist = float(np.linalg.norm(current_state - goal_state))
        obs_dist = goal_distance(current_obs, goal_obs)
        qpos_dist = float(np.linalg.norm(current_qpos_np - goal_qpos))
        one_step_error = current_state - predicted_next_state
        err_in_ellipsoid, err_ellipsoid_score = compute_error_ellipsoid_membership(
            error_np=one_step_error,
            state_np=pre_step_state,
            action_np=u0_norm,
            use_constant_covariance=cfg.use_constant_covariance,
            calibrated_cholesky=calibrated_cholesky_np,
            error_model=error_model_torch.net if error_model_torch is not None else None,
            q_learned=cfg.q_learned,
            device=device,
        )
        state_in_ellipsoid, state_ellipsoid_score = state_ellipsoid_membership(
            current_state, ellipsoid_unit_precision, cfg.latent_ellipsoid_margin
        )
        obstacle_free, obstacle_score, obstacle_required = obstacle_safety_status(
            current_state,
            obstacle_model,
            cfg.obstacle_margin,
        )
        latent_goal_distances.append(latent_dist)
        observation_goal_distances.append(obs_dist)
        qpos_goal_distances.append(qpos_dist)
        step_records.append(
            {
                "step": int(step_idx + 1),
                "phase": "post_step",
                "markov_state": current_state.astype(np.float64).tolist(),
                "embedding": executed_embeddings[-1].tolist(),
                "qpos": current_qpos_np.astype(np.float64).tolist(),
                "qvel": current_qvel_np.astype(np.float64).tolist(),
                "latent_goal_error": float(latent_dist),
                "observation_goal_error": float(obs_dist),
                "qpos_goal_error": float(qpos_dist),
                "state_in_latent_ellipsoid": state_in_ellipsoid,
                "state_latent_ellipsoid_score": state_ellipsoid_score,
                "obstacle_free": obstacle_free,
                "obstacle_score": obstacle_score,
                "obstacle_required_score": obstacle_required,
                "one_step_prediction_error": one_step_error.astype(np.float64).tolist(),
                "one_step_error_in_disturbance_ellipsoid": err_in_ellipsoid,
                "one_step_error_disturbance_score": err_ellipsoid_score,
                "solver_status": status,
                "timings_sec": {
                    "vit_encode": float(encode_time_sec),
                    "mppi_run": float(mppi_time_sec),
                    "sls_solve": float(sls_time_sec),
                    "total": float(encode_time_sec + mppi_time_sec + sls_time_sec),
                },
            }
        )
        pbar.set_postfix(latent=f"{latent_dist:.3f}", obs=f"{obs_dist:.3f}", qpos=f"{qpos_dist:.3f}", status=status)
        if qpos_dist <= 0.1:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

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

    np.savez(
        run_dir / "executed_actions.npz",
        actions_norm=np.stack(executed_actions_norm, axis=0) if executed_actions_norm else np.empty((0, action_dim)),
        actions_raw=np.stack(executed_actions_raw, axis=0) if executed_actions_raw else np.empty((0, action_dim)),
    )
    np.savez(
        run_dir / "executed_states.npz",
        markov_states=np.stack(executed_states, axis=0) if executed_states else np.empty((0, state_dim)),
        embeddings=np.stack(executed_embeddings, axis=0) if executed_embeddings else np.empty((0, embed_dim)),
        qpos=np.stack(executed_qpos, axis=0) if executed_qpos else np.empty((0, qpos_np.shape[1])),
        qvel=np.stack(executed_qvel, axis=0) if executed_qvel else np.empty((0, qvel_np.shape[1])),
    )

    video_path = save_video(rollout_frames, run_dir, cfg.video_fps)
    summary_payload = {
        "metadata": {
            "run_name": run_name,
            "use_mppi": bool(use_mppi),
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint),
            "episode_idx": int(run_index),
            "dataset_episode_idx": int(dataset_episode_idx),
            "start_goal_episode_idx": None if start_goal_episode_idx is None else int(start_goal_episode_idx),
            "episode_seed": int(episode_seed),
            "start_goal_source": start_goal_source,
            "start_goal_path": None if cfg.start_goal_path is None else str(cfg.start_goal_path),
            "stop_reason": stop_reason,
            "goal_reached": bool(stop_reason == "goal_reached"),
            "trajectory_safe_by_classifier": None if not valid_obstacle_checks else bool(all(valid_obstacle_checks)),
            "disturbance_ellipsoid_coverage": {
                "covered_steps": int(sum(valid_disturbance_checks)),
                "checked_steps": int(len(valid_disturbance_checks)),
                "fraction": None
                if not valid_disturbance_checks
                else float(sum(valid_disturbance_checks) / len(valid_disturbance_checks)),
            },
            "state_latent_ellipsoid_coverage": {
                "covered_steps": int(sum(valid_state_ellipsoid_checks)),
                "checked_steps": int(len(valid_state_ellipsoid_checks)),
                "fraction": None
                if not valid_state_ellipsoid_checks
                else float(sum(valid_state_ellipsoid_checks) / len(valid_state_ellipsoid_checks)),
            },
            "executed_steps": int(len(executed_actions_norm)),
            "num_logged_states": int(len(executed_states)),
            "video_path": str(video_path),
            "horizon": int(cfg.horizon),
            "max_mpc_steps": int(cfg.max_mpc_steps),
            "fastsls_enable": True,
            "timing_totals_sec": timing_totals,
            "artifacts": {
                "executed_actions_path": str(run_dir / "executed_actions.npz"),
                "executed_states_path": str(run_dir / "executed_states.npz"),
                "trajectory_summary_path": str(run_dir / "trajectory_summary.json"),
                "rollout_payload_path": str(run_dir / "sls_rollout.pt"),
            },
        },
        "start_goal": {
            "start_qpos": start_qpos.astype(np.float64).tolist(),
            "goal_qpos": goal_qpos.astype(np.float64).tolist(),
            "goal_obs": np.asarray(goal_obs, dtype=np.float64).tolist(),
            "start_state": start_state.astype(np.float64).tolist(),
            "goal_state": goal_state.astype(np.float64).tolist(),
        },
        "step_records": step_records,
    }
    with (run_dir / "trajectory_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    payload = {
        "metadata": summary_payload["metadata"],
        "planner_data": {
            "action_mean": action_mean,
            "action_std": action_std,
            "start_qpos": start_qpos,
            "goal_qpos": goal_qpos,
            "start_state": start_state,
            "goal_state": goal_state,
            "goal_obs": goal_obs,
        },
        "nominal_rollouts": {
            "state_plans": np.stack(nominal_states, axis=0) if nominal_states else np.empty((0, cfg.horizon + 1, state_dim)),
            "action_plans": np.stack(nominal_actions, axis=0) if nominal_actions else np.empty((0, cfg.horizon, action_dim)),
            "solver_statuses": solver_statuses,
        },
        "executed_rollout": {
            "actions_norm": np.stack(executed_actions_norm, axis=0) if executed_actions_norm else np.empty((0, action_dim)),
            "actions_raw": np.stack(executed_actions_raw, axis=0) if executed_actions_raw else np.empty((0, action_dim)),
            "states": np.stack(executed_states, axis=0) if executed_states else np.empty((0, state_dim)),
            "embeddings": np.stack(executed_embeddings, axis=0) if executed_embeddings else np.empty((0, embed_dim)),
            "qpos": np.stack(executed_qpos, axis=0) if executed_qpos else np.empty((0, qpos_np.shape[1])),
            "qvel": np.stack(executed_qvel, axis=0) if executed_qvel else np.empty((0, qvel_np.shape[1])),
            "latent_goal_distances": np.asarray(latent_goal_distances, dtype=np.float64),
            "observation_goal_distances": np.asarray(observation_goal_distances, dtype=np.float64),
            "qpos_goal_distances": np.asarray(qpos_goal_distances, dtype=np.float64),
        },
        "trajectory_summary": summary_payload,
    }
    torch.save(payload, run_dir / "sls_rollout.pt")
    env.close()
    print(f"Saved Reacher {'MPPI + ' if use_mppi else ''}SLS MPC run to {run_dir}")
