#!/usr/bin/env python3
"""Plan in Reacher pixel space with nominal iLQR MPC over a Markov-state MLP world model."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm.auto import tqdm
import json

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.reacher_policy_train import DmControlGymEnv, flatten_observation

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_noisy.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_embd_5"
DEFAULT_OUT_DIR = "reacher/plan/ilqr_mpc_mlpdyn"

DEVICE = "cuda"
HORIZON = 15
MAX_MPC_STEPS = 50
Q_TERMINAL = 5.0
Q_STAGE = 0.05
R_CONTROL = 0.1
VIDEO_FPS = 60
EPISODE_IDX = 326
# DEFAULT_START_QPOS = np.array([3.1258087158203125, -2.279094696044922], dtype=np.float32)
# DEFAULT_GOAL_QPOS = np.array([0.370098, -2.092896], dtype=np.float32)
# DEFAULT_GOAL_QPOS = np.array([0.9, -2.2], dtype=np.float32)
# DEFAULT_START_QPOS = np.array([2.42, -0.5], dtype=np.float32)
DEFAULT_START_QPOS = None
DEFAULT_GOAL_QPOS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--episode-idx", type=int, default=EPISODE_IDX)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--swap-start-goal", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_training_dataset_paths(dataset_config: object, default_dataset_path: Path) -> list[Path]:
    if dataset_config is None:
        raw_paths = [default_dataset_path]
    elif isinstance(dataset_config, (str, Path)):
        raw_paths = [dataset_config]
    elif isinstance(dataset_config, list):
        if not dataset_config:
            raise ValueError("Training config dataset_path is an empty list.")
        raw_paths = dataset_config
    else:
        raise TypeError(f"Unsupported dataset_path config type: {type(dataset_config).__name__}.")

    dataset_paths = [Path(str(path)).expanduser().resolve() for path in raw_paths]
    missing_paths = [path for path in dataset_paths if not path.is_file()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Training dataset file not found: {missing_str}")
    return dataset_paths


def load_action_stats(dataset_paths: list[Path], action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    total_count = 0
    action_sum = np.zeros((action_dim,), dtype=np.float64)
    action_sq_sum = np.zeros((action_dim,), dtype=np.float64)

    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as h5:
            if int(h5["action"].shape[-1]) != action_dim:
                raise ValueError(f"Expected action_dim={action_dim}, got {h5['action'].shape[-1]} in {dataset_path}.")
            actions = np.asarray(h5["action"][:], dtype=np.float32)
        finite_actions = actions[~np.isnan(actions).any(axis=1)]
        if finite_actions.size == 0:
            continue
        total_count += int(finite_actions.shape[0])
        action_sum += finite_actions.sum(axis=0, dtype=np.float64)
        action_sq_sum += np.square(finite_actions, dtype=np.float64).sum(axis=0, dtype=np.float64)

    if total_count == 0:
        raise ValueError("No finite actions found across the configured training datasets.")

    action_mean = action_sum / total_count
    action_var = np.maximum(action_sq_sum / total_count - np.square(action_mean), 0.0)
    action_std = np.maximum(np.sqrt(action_var), 1e-6)
    return action_mean.astype(np.float32, copy=False), action_std.astype(np.float32, copy=False)


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def configure_dm_control_timing(env: DmControlGymEnv, *, physics_timestep: float, time_limit: float) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def save_rollout_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    mp4_path = out_dir / "rollout.mp4"
    gif_path = out_dir / "rollout.gif"
    try:
        imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
        return mp4_path
    except Exception:
        imageio.mimwrite(gif_path, frames, fps=fps)
        return gif_path


def save_torch_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def preprocess_pixels(
    pixels: np.ndarray | torch.Tensor,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    if isinstance(pixels, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    else:
        tensor = pixels
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return pixel_mean, pixel_std


@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    frame_batch_size: int,
) -> torch.Tensor:
    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    return torch.cat(latents, dim=0)


@torch.no_grad()
def encode_single_frame(
    model: torch.nn.Module,
    pixel: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    batch = preprocess_pixels(pixel, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std).to(device)
    output = model.encoder(batch, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]


def make_markov_state(embedding: torch.Tensor, previous_embedding: torch.Tensor | None = None) -> torch.Tensor:
    if previous_embedding is None:
        delta = torch.zeros_like(embedding)
    else:
        delta = embedding - previous_embedding
    return torch.cat((embedding, delta), dim=-1)


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def load_dataset_episode(
    dataset_path: Path,
    episode_idx: int,
) -> dict[str, np.ndarray | int | float]:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        return {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "action": np.asarray(h5["action"][rows], dtype=np.float32),
            "observation": np.asarray(h5["observation"][rows], dtype=np.float32),
            "qpos": np.asarray(h5["qpos"][rows], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][rows], dtype=np.float32),
            "episode_seed": int(h5["episode_seed"][episode_idx]),
            "physics_freq_hz": float(h5.attrs.get("physics_freq_hz", 100.0)),
            "time_limit": float(h5.attrs.get("time_limit", 10.0)),
            "height": int(h5["pixels"].shape[1]),
            "width": int(h5["pixels"].shape[2]),
        }


def resolve_start_goal_qpos(
    *,
    dataset_qpos: np.ndarray,
    swap_start_goal: bool,
    default_start_qpos: np.ndarray | None,
    default_goal_qpos: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None, str]:
    if (default_start_qpos is None) != (default_goal_qpos is None):
        raise ValueError("DEFAULT_START_QPOS and DEFAULT_GOAL_QPOS must both be set or both be None.")

    if default_start_qpos is not None and default_goal_qpos is not None:
        expected_shape = (int(dataset_qpos.shape[1]),)
        start_qpos = np.asarray(default_start_qpos, dtype=np.float32)
        goal_qpos = np.asarray(default_goal_qpos, dtype=np.float32)
        if start_qpos.shape != expected_shape:
            raise ValueError(f"DEFAULT_START_QPOS must have shape {expected_shape}, got {start_qpos.shape}.")
        if goal_qpos.shape != expected_shape:
            raise ValueError(f"DEFAULT_GOAL_QPOS must have shape {expected_shape}, got {goal_qpos.shape}.")
        return start_qpos.copy(), goal_qpos.copy(), None, None, "fixed_qpos"

    start_idx = -1 if swap_start_goal else 0
    goal_idx = 0 if swap_start_goal else -1
    return (
        np.asarray(dataset_qpos[start_idx], dtype=np.float32).copy(),
        np.asarray(dataset_qpos[goal_idx], dtype=np.float32).copy(),
        int(start_idx),
        int(goal_idx),
        "dataset_episode",
    )


def make_render_env(
    *,
    seed: int,
    time_limit: float,
    width: int,
    height: int,
    physics_freq_hz: float,
) -> DmControlGymEnv:
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name="hard",
        seed=seed,
        time_limit=time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / physics_freq_hz, time_limit=time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    return env


def reset_env_to_state(
    env: DmControlGymEnv,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def build_observation_from_env(
    env: DmControlGymEnv,
    *,
    obs_dim: int,
    goal_obs: np.ndarray | None = None,
) -> np.ndarray:
    if obs_dim == 6:
        return flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    if obs_dim == 8:
        if goal_obs is None or goal_obs.shape[0] != 8:
            raise ValueError("Need an 8D goal observation to reconstruct random-goal planner observations.")
        physics = env._env.physics
        qpos = np.asarray(physics.data.qpos[:2], dtype=np.float32).copy()
        qvel = np.asarray(physics.data.qvel[:2], dtype=np.float32).copy()
        goal_qpos = np.asarray(goal_obs[4:6], dtype=np.float32).copy()
        goal_delta = goal_qpos - qpos
        return np.concatenate((qpos, qvel, goal_qpos, goal_delta), axis=0).astype(np.float32)
    raise ValueError(f"Unsupported observation dimension: {obs_dim}")


def build_goal_observation(
    env: DmControlGymEnv,
    *,
    goal_qpos: np.ndarray,
    goal_qvel: np.ndarray,
    obs_dim: int,
) -> np.ndarray:
    if obs_dim == 8:
        goal_delta = np.zeros_like(goal_qpos, dtype=np.float32)
        return np.concatenate((goal_qpos, goal_qvel, goal_qpos, goal_delta), axis=0).astype(np.float32)
    if obs_dim == 6:
        placeholder = np.zeros((8,), dtype=np.float32)
        placeholder[4 : 4 + goal_qpos.shape[0]] = goal_qpos
        return build_observation_from_env(env, obs_dim=obs_dim, goal_obs=placeholder)
    raise ValueError(f"Unsupported observation dimension: {obs_dim}")


def goal_reached(current_obs: np.ndarray, goal_obs: np.ndarray, threshold: float = 0.05) -> tuple[bool, float]:
    if current_obs.shape != goal_obs.shape:
        raise ValueError(f"Observation shape mismatch: {current_obs.shape} vs {goal_obs.shape}")
    if current_obs.shape[0] == 8:
        qpos = current_obs[:2]
        goal_qpos = current_obs[4:6]
        goal_distance = float(np.linalg.norm(qpos - goal_qpos))
        return goal_distance <= threshold, goal_distance
    obs_err = float(np.linalg.norm(current_obs - goal_obs))
    return obs_err <= threshold, obs_err


def compute_observation_goal_distance(current_obs: np.ndarray, goal_obs: np.ndarray) -> float:
    if current_obs.shape != goal_obs.shape:
        raise ValueError(f"Observation shape mismatch: {current_obs.shape} vs {goal_obs.shape}")
    if current_obs.shape[0] == 8:
        qpos = current_obs[:2]
        goal_qpos = current_obs[4:6]
        return float(np.linalg.norm(qpos - goal_qpos))
    return float(np.linalg.norm(current_obs - goal_obs))


class MarkovDynamicsTorch:
    def __init__(self, model: torch.nn.Module, state_dim: int, action_dim: int, device: torch.device) -> None:
        predictor = model.predictor
        if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError(
                "This planner expects a one-step Markov MLP dynamics model with "
                "history_size=1, action_history_size=1, and num_preds=1."
            )
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("This planner assumes an identity action encoder.")
        if int(predictor.embed_dim) != state_dim:
            raise ValueError(f"Predictor state dim mismatch: expected {state_dim}, got {predictor.embed_dim}.")
        if int(predictor.action_dim) != action_dim:
            raise ValueError(f"Predictor action dim mismatch: expected {action_dim}, got {predictor.action_dim}.")

        self.net = predictor.net.to(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, u), dim=-1))


class ILQRMPCSolver:
    def __init__(
        self,
        dynamics: MarkovDynamicsTorch,
        *,
        state_cost_dim: int,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        max_iters: int,
        tol: float,
        regularization: float,
        device: torch.device,
    ) -> None:
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim
        self.state_cost_dim = int(state_cost_dim)
        self.action_dim = dynamics.action_dim
        self.horizon = int(horizon)
        self.q_terminal = float(q_terminal)
        self.q_stage = float(q_stage)
        self.r_control = float(r_control)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.regularization = float(regularization)
        self.device = device
        if self.state_cost_dim <= 0 or self.state_cost_dim > self.state_dim:
            raise ValueError(
                f"state_cost_dim must be in [1, {self.state_dim}], got {self.state_cost_dim}."
            )
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)
        self.eye_x = torch.eye(self.state_dim, dtype=torch.float32, device=device)
        self.eye_x_cost = torch.zeros((self.state_dim, self.state_dim), dtype=torch.float32, device=device)
        self.eye_x_cost[: self.state_cost_dim, : self.state_cost_dim] = torch.eye(
            self.state_cost_dim, dtype=torch.float32, device=device
        )
        self.eye_u = torch.eye(self.action_dim, dtype=torch.float32, device=device)
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

    def _state_error(self, x: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        err = torch.zeros_like(x)
        err[: self.state_cost_dim] = x[: self.state_cost_dim] - x_goal[: self.state_cost_dim]
        return err

    def _make_initial_action_guess(self) -> torch.Tensor:
        if self.horizon <= 1:
            return self.prev_u_guess.clone()
        guess = torch.empty_like(self.prev_u_guess)
        guess[:-1] = self.prev_u_guess[1:]
        guess[-1] = self.prev_u_guess[-1]
        return guess

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x_traj = torch.empty((self.horizon + 1, self.state_dim), dtype=x0.dtype, device=self.device)
        x_traj[0] = x0
        x_curr = x0
        for step in range(self.horizon):
            x_curr = self.dynamics.step(x_curr, u_seq[step])
            x_traj[step + 1] = x_curr
        return x_traj

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = self._state_error(x_traj[step], x_goal)
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = self._state_error(x_traj[self.horizon], x_goal)
        cost = cost + self.q_terminal * torch.dot(terminal_err, terminal_err)
        return cost

    def _linearize_dynamics(
        self,
        x_traj: torch.Tensor,
        u_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_list = []
        b_list = []

        def dyn_cat(inp: torch.Tensor) -> torch.Tensor:
            x = inp[: self.state_dim]
            u = inp[self.state_dim :]
            return self.dynamics.step(x, u)

        for step in range(self.horizon):
            xu = torch.cat((x_traj[step], u_seq[step]), dim=0).detach().requires_grad_(True)
            jac = torch.autograd.functional.jacobian(dyn_cat, xu, vectorize=True)
            a_list.append(jac[:, : self.state_dim].detach())
            b_list.append(jac[:, self.state_dim :].detach())

        return torch.stack(a_list, dim=0), torch.stack(b_list, dim=0)

    def solve(self, x0_np: np.ndarray, x_goal_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_goal = torch.tensor(x_goal_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess()

        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_cost(x_traj, u_seq, x_goal).item())
        iterations = 0
        reg = self.regularization

        for iteration in range(self.max_iters):
            iterations = iteration + 1
            a_seq, b_seq = self._linearize_dynamics(x_traj, u_seq)
            k_seq = torch.empty((self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            kk_seq = torch.empty((self.horizon, self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)

            terminal_err = self._state_error(x_traj[self.horizon], x_goal)
            v_x = 2.0 * self.q_terminal * terminal_err
            v_xx = 2.0 * self.q_terminal * self.eye_x_cost
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = self._state_error(x_traj[step], x_goal)
                u = u_seq[step]
                a = a_seq[step]
                b = b_seq[step]

                l_x = 2.0 * self.q_stage * x_err
                l_u = 2.0 * self.r_control * u
                l_xx = 2.0 * self.q_stage * self.eye_x_cost
                l_uu = 2.0 * self.r_control * self.eye_u

                q_x = l_x + a.T @ v_x
                q_u = l_u + b.T @ v_x
                q_xx = l_xx + a.T @ v_xx @ a
                q_ux = b.T @ v_xx @ a
                q_uu = l_uu + b.T @ v_xx @ b + reg * self.eye_u
                q_uu = 0.5 * (q_uu + q_uu.T)

                try:
                    q_uu_inv = torch.linalg.inv(q_uu)
                except RuntimeError:
                    backward_ok = False
                    break

                k = -q_uu_inv @ q_u
                kk = -q_uu_inv @ q_ux
                k_seq[step] = k
                kk_seq[step] = kk

                v_x = q_x + kk.T @ q_uu @ k + kk.T @ q_u + q_ux.T @ k
                v_xx = q_xx + kk.T @ q_uu @ kk + kk.T @ q_ux + q_ux.T @ kk
                v_xx = 0.5 * (v_xx + v_xx.T)

            if not backward_ok:
                reg = min(reg * 10.0, 1e6)
                continue

            accepted = False
            candidate_best = None
            for alpha in self.line_search_alphas:
                x_new = torch.empty_like(x_traj)
                u_new = torch.empty_like(u_seq)
                x_new[0] = x0
                for step in range(self.horizon):
                    dx = x_new[step] - x_traj[step]
                    u_new[step] = u_seq[step] + alpha * k_seq[step] + kk_seq[step] @ dx
                    x_new[step + 1] = self.dynamics.step(x_new[step], u_new[step])
                new_cost = float(self._trajectory_cost(x_new, u_new, x_goal).item())
                if np.isfinite(new_cost) and new_cost < current_cost:
                    candidate_best = (x_new, u_new, new_cost, alpha)
                    accepted = True
                    break

            if not accepted:
                reg = min(reg * 10.0, 1e6)
                if reg >= 1e6:
                    break
                continue

            x_traj, u_seq, new_cost, alpha = candidate_best
            max_du = float(torch.max(torch.abs(alpha * k_seq)).item())
            cost_improvement = current_cost - new_cost
            current_cost = new_cost
            reg = max(self.regularization, reg * 0.5)

            if cost_improvement <= self.tol or max_du <= self.tol:
                break

        self.prev_u_guess = u_seq.detach().clone()

        maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0
        return (
            x_traj.detach().cpu().numpy().astype(np.float64),
            u_seq.detach().cpu().numpy().astype(np.float64),
            solve_time,
            iterations,
            current_cost,
        )


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    model = load_model(checkpoint_path, device)

    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for the finetuned MLP model, got {history_size}.")

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))

    train_dataset_paths = resolve_training_dataset_paths(config.get("dataset_path"), dataset_path)
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    action_mean, action_std = load_action_stats(train_dataset_paths, action_dim)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 2)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one test trajectory with 2 or more frames.")

    rng = np.random.default_rng(args.seed)
    if args.episode_idx is None:
        episode_idx = int(rng.choice(valid_episodes))
    else:
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"--episode-idx {episode_idx} must have at least 2 frames, got {ep_len[episode_idx]}.")

    episode = load_dataset_episode(dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    start_qpos, goal_qpos, start_idx, goal_idx, start_goal_source = resolve_start_goal_qpos(
        dataset_qpos=qpos_np,
        swap_start_goal=args.swap_start_goal,
        default_start_qpos=DEFAULT_START_QPOS,
        default_goal_qpos=DEFAULT_GOAL_QPOS,
    )
    print(f"start_qpos: {np.array2string(start_qpos, precision=6)}")
    print(f"goal_qpos: {np.array2string(goal_qpos, precision=6)}")
    run_name = (
        f"{int(time.time())}_episode_custom"
        if start_goal_source == "fixed_qpos"
        else f"{int(time.time())}_episode_{episode_idx:05d}"
    )
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    zero_qvel = np.zeros_like(qvel_np[0], dtype=np.float32)

    env = make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    start_frame = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=zero_qvel,
        height=height,
        width=width,
    )
    goal_frame = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=goal_qpos,
        qvel=zero_qvel,
        height=height,
        width=width,
    )
    start_emb = encode_single_frame(
        model,
        start_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    goal_emb = encode_single_frame(
        model,
        goal_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    start_state = make_markov_state(start_emb)
    goal_state = make_markov_state(goal_emb)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

    save_rgb_image(out_dir / "start_image.png", start_frame)
    save_rgb_image(out_dir / "goal_image.png", goal_frame)

    render_start = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=zero_qvel,
        height=height,
        width=width,
    )
    dynamics = MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    mpc_solver = ILQRMPCSolver(
        dynamics,
        state_cost_dim=embed_dim,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        device=device,
    )

    current_frame = render_start
    current_emb = encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_state = make_markov_state(current_emb)
    initial_current_state = current_state.detach().cpu().clone()
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_obs = build_goal_observation(env, goal_qpos=goal_qpos, goal_qvel=zero_qvel, obs_dim=int(obs_np.shape[1]))
    obs_dim = int(goal_obs.shape[0])
    current_obs = build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)

    rollout_frames = [current_frame.copy()]
    executed_qpos = [np.asarray(env._env.physics.data.qpos[: qpos_np.shape[1]], dtype=np.float32).copy()]
    executed_qvel = [np.asarray(env._env.physics.data.qvel[: qvel_np.shape[1]], dtype=np.float32).copy()]
    executed_embeddings = [current_emb.detach().cpu().numpy().astype(np.float32)]
    executed_states = [current_state.detach().cpu().numpy().astype(np.float32)]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    nominal_state_rollouts: list[np.ndarray] = []
    nominal_action_rollouts: list[np.ndarray] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
    embedding_goal_distances = [float(torch.linalg.vector_norm(current_emb - goal_emb).item())]
    observation_goal_distances = [compute_observation_goal_distance(current_obs, goal_obs)]
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    stop_reason = "max_mpc_steps"

    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        x_plan, u_plan, solve_time, n_iters, plan_cost = mpc_solver.solve(current_state_np, goal_state_np)
        nominal_state_rollouts.append(x_plan.copy())
        nominal_action_rollouts.append(u_plan.copy())
        solve_times_ms.append(solve_time * 1000.0)
        ilqr_iterations.append(int(n_iters))
        ilqr_costs.append(float(plan_cost))

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        _, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = make_markov_state(next_emb, current_emb)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        executed_qpos.append(np.asarray(env._env.physics.data.qpos[: qpos_np.shape[1]], dtype=np.float32).copy())
        executed_qvel.append(np.asarray(env._env.physics.data.qvel[: qvel_np.shape[1]], dtype=np.float32).copy())
        executed_embeddings.append(current_emb.detach().cpu().numpy().astype(np.float32))
        executed_states.append(current_state.detach().cpu().numpy().astype(np.float32))
        latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
        embedding_goal_distance = float(torch.linalg.vector_norm(current_emb - goal_emb).item())
        obs_goal_distance = compute_observation_goal_distance(current_obs, goal_obs)
        latent_goal_distances.append(latent_goal_distance)
        embedding_goal_distances.append(embedding_goal_distance)
        observation_goal_distances.append(obs_goal_distance)

        pbar.set_postfix(
            solve_ms=f"{solve_times_ms[-1]:.1f}",
            iters=f"{ilqr_iterations[-1]}",
            latent_goal=f"{latent_goal_distance:.3f}",
            obs_goal=f"{obs_goal_distance:.3f}",
        )

        reached_goal, _ = goal_reached(current_obs, goal_obs)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    final_qpos = np.asarray(env._env.physics.data.qpos[: qpos_np.shape[1]], dtype=np.float32)
    final_qvel = np.asarray(env._env.physics.data.qvel[: qvel_np.shape[1]], dtype=np.float32)
    final_obs = build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)
    video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    env.close()

    executed_actions_norm_np = (
        np.stack(executed_actions_norm, axis=0)
        if executed_actions_norm
        else np.empty((0, action_dim), dtype=np.float32)
    )
    executed_actions_raw_np = (
        np.stack(executed_actions_raw, axis=0)
        if executed_actions_raw
        else np.empty((0, action_dim), dtype=np.float32)
    )
    executed_qpos_np = (
        np.stack(executed_qpos, axis=0)
        if executed_qpos
        else np.empty((0, qpos_np.shape[1]), dtype=np.float32)
    )
    executed_qvel_np = (
        np.stack(executed_qvel, axis=0)
        if executed_qvel
        else np.empty((0, qvel_np.shape[1]), dtype=np.float32)
    )
    executed_embeddings_np = (
        np.stack(executed_embeddings, axis=0)
        if executed_embeddings
        else np.empty((0, embed_dim), dtype=np.float32)
    )
    executed_states_np = (
        np.stack(executed_states, axis=0)
        if executed_states
        else np.empty((0, markov_state_dim), dtype=np.float32)
    )
    nominal_state_plans_np = (
        np.stack(nominal_state_rollouts, axis=0)
        if nominal_state_rollouts
        else np.empty((0, args.horizon + 1, markov_state_dim), dtype=np.float64)
    )
    nominal_action_plans_np = (
        np.stack(nominal_action_rollouts, axis=0)
        if nominal_action_rollouts
        else np.empty((0, args.horizon, action_dim), dtype=np.float64)
    )
    plan_steps_np = np.arange(nominal_action_plans_np.shape[0], dtype=np.int64)

    rollout_payload = {
        "metadata": {
            "run_name": run_name,
            "out_dir": str(out_dir),
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint_path": str(checkpoint_path),
            "episode_idx": int(episode_idx),
            "episode_seed": int(episode_seed),
            "episode_length": int(pixels_np.shape[0]),
            "physics_freq_hz": float(physics_freq_hz),
            "time_limit": float(time_limit),
            "height": int(height),
            "width": int(width),
            "device": str(device),
            "img_size": int(img_size),
            "action_dim": int(action_dim),
            "embed_dim": int(embed_dim),
            "markov_state_dim": int(markov_state_dim),
            "obs_dim": int(obs_dim),
            "frame_batch_size": int(args.frame_batch_size),
            "horizon": int(args.horizon),
            "max_mpc_steps": int(args.max_mpc_steps),
            "video_fps": int(args.video_fps),
            "q_terminal": float(args.q_terminal),
            "q_stage": float(args.q_stage),
            "r_control": float(args.r_control),
            "ilqr_max_iters": int(args.ilqr_max_iters),
            "ilqr_tol": float(args.ilqr_tol),
            "ilqr_regularization": float(args.ilqr_regularization),
            "seed": None if args.seed is None else int(args.seed),
            "swap_start_goal": bool(args.swap_start_goal),
            "start_idx": None if start_idx is None else int(start_idx),
            "goal_idx": None if goal_idx is None else int(goal_idx),
            "start_goal_source": start_goal_source,
            "planned_steps": int(len(nominal_state_rollouts)),
            "executed_steps": int(len(executed_actions_raw)),
            "stop_reason": stop_reason,
            "video_path": video_path,
            "artifacts": {
                "executed_actions_path": str(out_dir / "executed_actions.npz"),
                "executed_states_path": str(out_dir / "executed_states.npz"),
                "mpc_plans_path": str(out_dir / "mpc_plans.npz"),
                "rollout_payload_path": str(out_dir / "nominal_rollout.pt"),
            },
        },
        "episode_data": {
            "pixels": pixels_np,
            "qpos": qpos_np,
            "qvel": qvel_np,
            "observation": obs_np,
        },
        "planner_data": {
            "action_mean": action_mean,
            "action_std": action_std,
            "start_qpos": start_qpos,
            "goal_qpos": goal_qpos,
            "start_qvel": zero_qvel,
            "goal_qvel": zero_qvel,
            "goal_obs": goal_obs,
            "start_embedding": start_emb.detach().cpu(),
            "goal_embedding": goal_emb.detach().cpu(),
            "start_state": start_state.detach().cpu(),
            "goal_state": goal_state.detach().cpu(),
            "initial_current_state": initial_current_state,
            "final_qpos": final_qpos,
            "final_qvel": final_qvel,
            "final_obs": final_obs,
        },
        "nominal_rollouts": {
            "state_plans": nominal_state_plans_np,
            "action_plans": nominal_action_plans_np,
            "solve_times_ms": np.asarray(solve_times_ms, dtype=np.float64),
            "ilqr_iterations": np.asarray(ilqr_iterations, dtype=np.int64),
            "ilqr_costs": np.asarray(ilqr_costs, dtype=np.float64),
        },
        "executed_rollout": {
            "frames": np.stack(rollout_frames, axis=0),
            "qpos": executed_qpos_np,
            "qvel": executed_qvel_np,
            "embeddings": executed_embeddings_np,
            "states": executed_states_np,
            "actions_raw": executed_actions_raw_np,
            "actions_norm": executed_actions_norm_np,
            "latent_goal_distances": np.asarray(latent_goal_distances, dtype=np.float64),
            "embedding_goal_distances": np.asarray(embedding_goal_distances, dtype=np.float64),
            "observation_goal_distances": np.asarray(observation_goal_distances, dtype=np.float64),
        },
    }
    np.savez(
        out_dir / "executed_actions.npz",
        actions_norm=executed_actions_norm_np,
        actions_raw=executed_actions_raw_np,
    )
    np.savez(
        out_dir / "executed_states.npz",
        markov_states=executed_states_np,
        embeddings=executed_embeddings_np,
        qpos=executed_qpos_np,
        qvel=executed_qvel_np,
    )
    np.savez_compressed(
        out_dir / "mpc_plans.npz",
        plan_steps=plan_steps_np,
        nominal_centers=nominal_state_plans_np,
        nominal_actions=nominal_action_plans_np,
        executed_markov_states=executed_states_np,
        goal_state=goal_state_np,
        start_state=start_state.detach().cpu().numpy().astype(np.float64),
        state_dim=np.asarray(markov_state_dim, dtype=np.int64),
        horizon=np.asarray(args.horizon, dtype=np.int64),
        episode_idx=np.asarray(episode_idx, dtype=np.int64),
        solve_times_ms=np.asarray(solve_times_ms, dtype=np.float64),
        ilqr_iterations=np.asarray(ilqr_iterations, dtype=np.int64),
        ilqr_costs=np.asarray(ilqr_costs, dtype=np.float64),
    )
    save_torch_payload(out_dir / "nominal_rollout.pt", rollout_payload)
    save_json(
        out_dir / "executed_qpos.json",
        {
            "start_qpos": start_qpos.tolist(),
            "goal_qpos": goal_qpos.tolist(),
            "final_qpos": final_qpos.tolist(),
            "qpos_over_time": executed_qpos_np.tolist(),
        },
    )

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
