#!/usr/bin/env python3
"""Plan in PushT pixel space with nominal iLQR MPC over a Markov-state MLP world model."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from pusht.shared.pusht_env import (
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_no_target_env,
    make_pusht_env,
    reset_pusht_env_to_state,
)
from pusht.train.mlpdyn_train import build_markov_state, required_markov_history

DEFAULT_DATASET_PATH = "pusht/data/pusht_diffusion_eval.h5"
DEFAULT_MODEL_DIR = "pusht/models/mlpdyn_embd_48"
DEFAULT_OUT_DIR = "pusht/plan/ilqr_mpc_mlpdyn"

DEVICE = "auto"
HORIZON = 15
MAX_MPC_STEPS = 250
Q_TERMINAL = 10.0
Q_STAGE = 0.005
R_CONTROL = 0.001
VIDEO_FPS = 10
EPISODE_IDX = 8 
CONTROL_MIN_NORM = -1.5
CONTROL_MAX_NORM = 1.5
ENV_ACTION_SCALE = 100.0
PUSHT_WALL_MIN = 5.0
PUSHT_WALL_MAX = 506.0
PUSHT_WALL_RADIUS = 2.0
PUSHT_AGENT_RADIUS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
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
    parser.add_argument("--ilqr-max-iters", type=int, default=50)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
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


def resolve_dataset_paths(value: object, fallback: Path) -> list[Path]:
    if value is None:
        raw_paths: list[object] = [fallback]
    elif isinstance(value, (str, Path)):
        parsed_value = value
        if isinstance(value, str):
            try:
                literal = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                literal = None
            if isinstance(literal, list):
                parsed_value = literal
        raw_paths = parsed_value if isinstance(parsed_value, list) else [parsed_value]
    elif isinstance(value, list):
        raw_paths = value
    else:
        raise TypeError(f"Unsupported dataset path config type: {type(value).__name__}.")

    resolved_paths = [Path(path).expanduser().resolve() for path in raw_paths]
    missing_paths = [path for path in resolved_paths if not path.is_file()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Dataset file(s) not found: {missing_str}")
    return resolved_paths


def load_action_stats(dataset_paths: list[Path], action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    finite_action_blocks: list[np.ndarray] = []
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as h5:
            if int(h5["action"].shape[-1]) != action_dim:
                raise ValueError(
                    f"Expected action_dim={action_dim} for {dataset_path}, got {h5['action'].shape[-1]}."
                )
            finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
            finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
            if finite_actions.size:
                finite_action_blocks.append(finite_actions)
    if not finite_action_blocks:
        raise ValueError("No finite actions found across the configured training datasets.")
    finite_actions = np.concatenate(finite_action_blocks, axis=0)
    action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)
    return action_mean, action_std


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve().parent
    return args.model_dir.expanduser().resolve()


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
    return (tensor - pixel_mean) / pixel_std


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


def make_markov_state(history: list[torch.Tensor], markov_deriv: int) -> torch.Tensor:
    context_len = required_markov_history(markov_deriv)
    if not history:
        raise ValueError("At least one embedding is required to build the Markov state.")
    history_tensor = torch.stack(history[-context_len:], dim=0)
    if history_tensor.shape[0] < context_len:
        pad = history_tensor[:1].repeat(context_len - history_tensor.shape[0], 1)
        history_tensor = torch.cat((pad, history_tensor), dim=0)
    return build_markov_state(history_tensor, markov_deriv)


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def raw_to_env_action(
    raw_action: np.ndarray,
    agent_pos: np.ndarray,
    *,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> np.ndarray:
    action = (agent_pos.astype(np.float32) + ENV_ACTION_SCALE * raw_action.astype(np.float32)).astype(np.float32)
    if action_low is not None and action_high is not None:
        action = np.clip(action, action_low.astype(np.float32), action_high.astype(np.float32))
    return action


def pusht_agent_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    # Keep the blue pusher fully inside the inner edge of the rendered gray walls.
    min_coord = PUSHT_WALL_MIN + PUSHT_WALL_RADIUS + PUSHT_AGENT_RADIUS
    max_coord = PUSHT_WALL_MAX - PUSHT_WALL_RADIUS - PUSHT_AGENT_RADIUS
    low = np.full((2,), min_coord, dtype=np.float32)
    high = np.full((2,), max_coord, dtype=np.float32)
    return low, high


def angle_diff(angle: float, target: float) -> float:
    return float((angle - target + np.pi) % (2.0 * np.pi) - np.pi)


def block_pose_distance(current_block_pose: np.ndarray, goal_block_pose: np.ndarray) -> float:
    pose_err = current_block_pose.astype(np.float64) - goal_block_pose.astype(np.float64)
    pose_err[2] = angle_diff(float(current_block_pose[2]), float(goal_block_pose[2]))
    return float(np.linalg.norm(pose_err))


def goal_reached(current_block_pose: np.ndarray, goal_block_pose: np.ndarray, threshold: float = 5.0) -> tuple[bool, float]:
    pose_dist = block_pose_distance(current_block_pose, goal_block_pose)
    return pose_dist <= threshold, pose_dist


def load_dataset_episode(
    dataset_path: Path,
    episode_idx: int,
) -> dict[str, np.ndarray | int]:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        return {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "action": np.asarray(h5["action"][rows], dtype=np.float32),
            "state": np.asarray(h5["state"][rows], dtype=np.float32),
            "proprio": np.asarray(h5["proprio"][rows], dtype=np.float32),
            "height": int(h5["pixels"].shape[1]),
            "width": int(h5["pixels"].shape[2]),
        }


def infer_dataset_state_format(state: np.ndarray) -> str:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.size < 7:
        raise ValueError(f"Expected dataset state vectors with at least 7 entries, got shape {state.shape}.")
    if np.all(np.abs(state[2:4]) <= 1.05):
        return "privileged_goal_state"
    return "env_state"


def dataset_row_to_env_state(state_row: np.ndarray, proprio_row: np.ndarray, state_format: str) -> np.ndarray:
    state_row = np.asarray(state_row, dtype=np.float32).reshape(-1)
    proprio_row = np.asarray(proprio_row, dtype=np.float32).reshape(-1)

    if state_format == "env_state":
        return state_row[:7].astype(np.float64, copy=True)
    if state_format != "privileged_goal_state":
        raise ValueError(f"Unsupported dataset state format: {state_format}")
    if proprio_row.size < 2:
        raise ValueError("Privileged goal-state dataset requires proprio to contain agent xy in the first two entries.")

    theta = float(np.arctan2(state_row[3], state_row[2]))
    return np.asarray(
        [
            proprio_row[0],
            proprio_row[1],
            state_row[0],
            state_row[1],
            theta,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def dataset_row_to_block_pose(state_row: np.ndarray, env_state_row: np.ndarray, state_format: str) -> np.ndarray:
    if state_format == "env_state":
        return np.asarray(env_state_row[2:5], dtype=np.float32)
    if state_format == "privileged_goal_state":
        state_row = np.asarray(state_row, dtype=np.float32).reshape(-1)
        theta = float(np.arctan2(state_row[3], state_row[2]))
        return np.asarray([state_row[0], state_row[1], theta], dtype=np.float32)
    raise ValueError(f"Unsupported dataset state format: {state_format}")


def dataset_row_to_goal_pose(state_row: np.ndarray, state_format: str) -> np.ndarray | None:
    if state_format != "privileged_goal_state":
        return None
    state_row = np.asarray(state_row, dtype=np.float32).reshape(-1)
    return np.asarray(state_row[4:7], dtype=np.float32)


def make_planning_env(*, width: int, height: int) -> Any:
    return make_no_target_env(height=height, width=width)


def make_visualization_env(*, width: int, height: int) -> Any:
    env = make_pusht_env(
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
    )
    env.reset(seed=0)
    return env


def reset_env_to_state(env: Any, state: np.ndarray) -> np.ndarray:
    return reset_pusht_env_to_state(env, state)


def current_block_pose(env: Any) -> np.ndarray:
    return get_pusht_block_pose(env)


def current_agent_pos(env: Any) -> np.ndarray:
    return get_pusht_agent_pos(env)


def set_goal_pose(env: Any, goal_pose: np.ndarray | None) -> None:
    if goal_pose is None:
        return
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "goal_pose"):
        base_env.goal_pose = np.asarray(goal_pose, dtype=np.float32).copy()


def extract_full_state(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "get_state"):
        return np.asarray(base_env.get_state(), dtype=np.float64).reshape(-1)

    if not (
        hasattr(base_env, "agent")
        and hasattr(base_env.agent, "position")
        and hasattr(base_env.agent, "velocity")
        and hasattr(base_env, "block")
        and hasattr(base_env.block, "position")
        and hasattr(base_env.block, "angle")
    ):
        raise AttributeError("PushT env does not expose get_state() and is missing agent/block bodies.")

    return np.asarray(
        [
            float(base_env.agent.position.x),
            float(base_env.agent.position.y),
            float(base_env.block.position.x),
            float(base_env.block.position.y),
            float(base_env.block.angle),
            float(base_env.agent.velocity.x),
            float(base_env.agent.velocity.y),
        ],
        dtype=np.float64,
    )


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
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        max_iters: int,
        tol: float,
        regularization: float,
        control_min: float,
        control_max: float,
        device: torch.device,
    ) -> None:
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim
        self.action_dim = dynamics.action_dim
        self.horizon = int(horizon)
        self.q_terminal = float(q_terminal)
        self.q_stage = float(q_stage)
        self.r_control = float(r_control)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.regularization = float(regularization)
        self.device = device
        self.eye_x = torch.eye(self.state_dim, dtype=torch.float32, device=device)
        self.eye_u = torch.eye(self.action_dim, dtype=torch.float32, device=device)
        self.u_min = torch.full((self.action_dim,), float(control_min), dtype=torch.float32, device=device)
        self.u_max = torch.full((self.action_dim,), float(control_max), dtype=torch.float32, device=device)
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)
        self.prev_u_guess = self._project_control(self.prev_u_guess)
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

    def _project_control(self, u: torch.Tensor) -> torch.Tensor:
        return torch.clamp(u, min=self.u_min, max=self.u_max)

    def _make_initial_action_guess(self) -> torch.Tensor:
        if self.horizon <= 1:
            return self._project_control(self.prev_u_guess.clone())
        guess = torch.empty_like(self.prev_u_guess)
        guess[:-1] = self.prev_u_guess[1:]
        guess[-1] = self.prev_u_guess[-1]
        return self._project_control(guess)

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x_traj = torch.empty((self.horizon + 1, self.state_dim), dtype=x0.dtype, device=self.device)
        x_traj[0] = x0
        x_curr = x0
        for step in range(self.horizon):
            u_step = self._project_control(u_seq[step])
            x_curr = self.dynamics.step(x_curr, u_step)
            x_traj[step + 1] = x_curr
        return x_traj

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = x_traj[step] - x_goal
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_goal
        cost = cost + self.q_terminal * torch.dot(terminal_err, terminal_err)
        return cost

    def _linearize_dynamics(self, x_traj: torch.Tensor, u_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _solve_box_qp(
        self,
        q_uu: torch.Tensor,
        q_u: torch.Tensor,
        q_ux: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_low = self.u_min - u
        delta_high = self.u_max - u
        best_obj: float | None = None
        best_k: torch.Tensor | None = None
        best_kk: torch.Tensor | None = None
        grad_tol = 1e-5
        bound_tol = 1e-6

        for status in itertools.product((-1, 0, 1), repeat=self.action_dim):
            k = torch.zeros((self.action_dim,), dtype=torch.float32, device=self.device)
            kk = torch.zeros((self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)
            free_idx = [idx for idx, tag in enumerate(status) if tag == 0]
            clamped_idx = [idx for idx, tag in enumerate(status) if tag != 0]

            for idx in clamped_idx:
                k[idx] = delta_low[idx] if status[idx] < 0 else delta_high[idx]

            if free_idx:
                free = torch.tensor(free_idx, dtype=torch.long, device=self.device)
                q_ff = q_uu.index_select(0, free).index_select(1, free)
                rhs_k = q_u.index_select(0, free)
                if clamped_idx:
                    clamped = torch.tensor(clamped_idx, dtype=torch.long, device=self.device)
                    q_fc = q_uu.index_select(0, free).index_select(1, clamped)
                    rhs_k = rhs_k + q_fc @ k.index_select(0, clamped)
                try:
                    k_free = -torch.linalg.solve(q_ff, rhs_k.unsqueeze(1)).squeeze(1)
                    kk_free = -torch.linalg.solve(q_ff, q_ux.index_select(0, free))
                except RuntimeError:
                    continue
                k[free] = k_free
                kk[free] = kk_free

            feasible = bool(torch.all(k >= delta_low - bound_tol) and torch.all(k <= delta_high + bound_tol))
            if not feasible:
                continue

            grad = q_u + q_uu @ k
            kkt_ok = True
            for idx, tag in enumerate(status):
                grad_i = float(grad[idx].item())
                if tag == 0 and abs(grad_i) > grad_tol:
                    kkt_ok = False
                    break
                if tag < 0 and grad_i < -grad_tol:
                    kkt_ok = False
                    break
                if tag > 0 and grad_i > grad_tol:
                    kkt_ok = False
                    break
            if not kkt_ok:
                continue

            obj = float((0.5 * torch.dot(k, q_uu @ k) + torch.dot(q_u, k)).item())
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_k = k
                best_kk = kk

        if best_k is None or best_kk is None:
            raise RuntimeError("Box-constrained control subproblem failed.")
        return best_k, best_kk

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

            terminal_err = x_traj[self.horizon] - x_goal
            v_x = 2.0 * self.q_terminal * terminal_err
            v_xx = 2.0 * self.q_terminal * self.eye_x
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = x_traj[step] - x_goal
                u = u_seq[step]
                a = a_seq[step]
                b = b_seq[step]

                l_x = 2.0 * self.q_stage * x_err
                l_u = 2.0 * self.r_control * u
                l_xx = 2.0 * self.q_stage * self.eye_x
                l_uu = 2.0 * self.r_control * self.eye_u

                q_x = l_x + a.T @ v_x
                q_u = l_u + b.T @ v_x
                q_xx = l_xx + a.T @ v_xx @ a
                q_ux = b.T @ v_xx @ a
                q_uu = l_uu + b.T @ v_xx @ b + reg * self.eye_u
                q_uu = 0.5 * (q_uu + q_uu.T)

                try:
                    k, kk = self._solve_box_qp(q_uu, q_u, q_ux, u)
                except RuntimeError:
                    backward_ok = False
                    break

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
                    u_candidate = u_seq[step] + alpha * k_seq[step] + kk_seq[step] @ dx
                    u_new[step] = self._project_control(u_candidate)
                    x_new[step + 1] = self.dynamics.step(x_new[step], u_new[step])
                new_cost = float(self._trajectory_cost(x_new, u_new, x_goal).item())
                if np.isfinite(new_cost) and new_cost < current_cost:
                    candidate_best = (x_new, u_new, new_cost)
                    accepted = True
                    break

            if not accepted:
                reg = min(reg * 10.0, 1e6)
                if reg >= 1e6:
                    break
                continue

            prev_u_seq = u_seq
            x_traj, u_seq, new_cost = candidate_best
            max_du = float(torch.max(torch.abs(u_seq - prev_u_seq)).item())
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
    model_dir = resolve_model_dir(args)
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

    markov_deriv = int(config.get("markov_deriv", 1))
    if markov_deriv < 0:
        raise ValueError(f"Expected non-negative markov_deriv for the MLP model, got {markov_deriv}.")

    img_size = int(config.get("img_size", 224))
    frameskip = int(config.get("frameskip", 1))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 48))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    if frameskip != 1:
        raise ValueError(
            f"This PushT MPC planner currently supports frameskip=1 only, but the model config has frameskip={frameskip}."
        )

    train_dataset_paths = resolve_dataset_paths(config.get("dataset_path"), dataset_path)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    action_mean, action_std = load_action_stats(train_dataset_paths, action_dim)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 2)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one trajectory with 2 or more frames.")

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
    state_np = np.asarray(episode["state"])
    proprio_np = np.asarray(episode["proprio"])
    height = int(episode["height"])
    width = int(episode["width"])
    state_format = infer_dataset_state_format(state_np[0])
    env_state_np = np.stack(
        [dataset_row_to_env_state(state_row, proprio_row, state_format) for state_row, proprio_row in zip(state_np, proprio_np)],
        axis=0,
    )
    goal_pose = dataset_row_to_goal_pose(state_np[-1], state_format)

    run_name = f"{int(time.time())}_episode_{episode_idx:05d}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    true_latents = encode_frames(
        model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    history_len = required_markov_history(markov_deriv)
    start_emb = true_latents[0]
    start_history = [start_emb] * history_len
    goal_history = [emb for emb in true_latents[-history_len:]]
    start_state = make_markov_state(start_history, markov_deriv)
    goal_state = make_markov_state(goal_history, markov_deriv)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

    save_rgb_image(out_dir / "start_image.png", pixels_np[0])
    save_rgb_image(out_dir / "goal_image.png", pixels_np[-1])

    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_env: list[np.ndarray] = []
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    stop_reason = "max_mpc_steps"
    video_path: str | None = None
    final_block = dataset_row_to_block_pose(state_np[0], env_state_np[0], state_format)
    final_agent = env_state_np[0, :2].astype(np.float32)
    goal_block = dataset_row_to_block_pose(state_np[-1], env_state_np[-1], state_format)
    rollout_frames: list[np.ndarray] = []
    latent_goal_distances: list[float] = []
    block_goal_distances: list[float] = []
    num_action_clips = 0

    plan_env = make_planning_env(width=width, height=height)
    viz_env = make_visualization_env(width=width, height=height)
    try:
        set_goal_pose(plan_env, goal_pose)
        set_goal_pose(viz_env, goal_pose)
        action_low, action_high = pusht_agent_action_bounds()
        hidden_start = reset_env_to_state(plan_env, env_state_np[0])
        visible_start = reset_env_to_state(viz_env.unwrapped, env_state_np[0])
        dynamics = MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
        mpc_solver = ILQRMPCSolver(
            dynamics,
            horizon=args.horizon,
            q_terminal=args.q_terminal,
            q_stage=args.q_stage,
            r_control=args.r_control,
            max_iters=args.ilqr_max_iters,
            tol=args.ilqr_tol,
            regularization=args.ilqr_regularization,
            control_min=CONTROL_MIN_NORM,
            control_max=CONTROL_MAX_NORM,
            device=device,
        )

        current_hidden_frame = hidden_start
        current_emb = encode_single_frame(
            model,
            current_hidden_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_history = [current_emb] * history_len
        current_state = make_markov_state(current_history, markov_deriv)
        goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
        current_block = current_block_pose(plan_env)

        rollout_frames = [visible_start.copy()]
        latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
        block_goal_distances = [block_pose_distance(current_block, goal_block)]

        pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
        try:
            for _ in pbar:
                current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
                _, u_plan, solve_time, n_iters, plan_cost = mpc_solver.solve(current_state_np, goal_state_np)
                solve_times_ms.append(solve_time * 1000.0)
                ilqr_iterations.append(int(n_iters))
                ilqr_costs.append(float(plan_cost))

                u0_norm = u_plan[0].astype(np.float32)
                u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
                unclipped_u0_env = raw_to_env_action(u0_raw, current_agent_pos(plan_env))
                u0_env = raw_to_env_action(
                    u0_raw,
                    current_agent_pos(plan_env),
                    action_low=action_low,
                    action_high=action_high,
                )
                if not np.allclose(u0_env, unclipped_u0_env):
                    num_action_clips += 1
                executed_actions_norm.append(u0_norm.copy())
                executed_actions_raw.append(u0_raw.copy())
                executed_actions_env.append(u0_env.copy())

                _, _, terminated, truncated, _ = plan_env.step(u0_env)
                current_hidden_frame = np.asarray(plan_env._render(visualize=False), dtype=np.uint8)
                next_emb = encode_single_frame(
                    model,
                    current_hidden_frame,
                    device=device,
                    img_size=img_size,
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                )
                current_history.append(next_emb)
                current_history = current_history[-history_len:]
                current_state = make_markov_state(current_history, markov_deriv)
                current_block = current_block_pose(plan_env)

                synced_state = extract_full_state(plan_env)
                visible_frame = reset_env_to_state(viz_env.unwrapped, synced_state)

                rollout_frames.append(visible_frame.copy())
                latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
                block_goal_distance = block_pose_distance(current_block, goal_block)
                latent_goal_distances.append(latent_goal_distance)
                block_goal_distances.append(block_goal_distance)

                pbar.set_postfix(
                    solve_ms=f"{solve_times_ms[-1]:.1f}",
                    iters=f"{ilqr_iterations[-1]}",
                    latent_goal=f"{latent_goal_distance:.3f}",
                    block_goal=f"{block_goal_distance:.3f}",
                )

                reached_goal, _ = goal_reached(current_block, goal_block)
                if reached_goal:
                    stop_reason = "goal_reached"
                    break
                if terminated or truncated:
                    stop_reason = "terminated" if terminated else "truncated"
                    break
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
        finally:
            pbar.close()

        final_block = current_block_pose(plan_env)
        final_agent = current_agent_pos(plan_env)
        video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    finally:
        plan_env.close()
        viz_env.close()

    metrics = {
        "episode_idx": episode_idx,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "config_path": str(model_dir / "config.json"),
        "dataset_path": str(dataset_path),
        "train_dataset_paths": [str(path) for path in train_dataset_paths],
        "dataset_state_format": state_format,
        "markov_deriv": markov_deriv,
        "action_history_size": 1,
        "state_space": "latent_plus_finite_differences",
        "markov_state_dim": markov_state_dim,
        "frameskip": frameskip,
        "horizon": args.horizon,
        "max_mpc_steps": args.max_mpc_steps,
        "stop_reason": stop_reason,
        "num_mpc_steps": len(executed_actions_norm),
        "action_space": "normalized_dataset_action_then_raw_then_absolute_env_target",
        "control_bound_low_norm": CONTROL_MIN_NORM,
        "control_bound_high_norm": CONTROL_MAX_NORM,
        "env_action_scale": ENV_ACTION_SCALE,
        "action_target_low": action_low.tolist(),
        "action_target_high": action_high.tolist(),
        "num_action_clips": num_action_clips,
        "start_agent_pos": env_state_np[0, :2].tolist(),
        "goal_block_pose": goal_block.tolist(),
        "goal_pose": goal_pose.tolist() if goal_pose is not None else None,
        "final_agent_pos": final_agent.tolist(),
        "final_block_pose": final_block.tolist(),
        "goal_proprio": proprio_np[-1].tolist(),
        "latent_goal_distance_initial": latent_goal_distances[0],
        "latent_goal_distance_final": latent_goal_distances[-1],
        "block_goal_distance_initial": block_goal_distances[0],
        "block_goal_distance_final": block_goal_distances[-1],
        "latent_goal_distances": latent_goal_distances,
        "block_goal_distances": block_goal_distances,
        "solve_times_ms": solve_times_ms,
        "ilqr_iterations": ilqr_iterations,
        "ilqr_costs": ilqr_costs,
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_env": [action.tolist() for action in executed_actions_env],
        "video_path": video_path,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        json.dumps(
            {
                "episode_idx": episode_idx,
                "latent_goal_distance_final": metrics["latent_goal_distance_final"],
                "block_goal_distance_final": metrics["block_goal_distance_final"],
                "stop_reason": stop_reason,
                "metrics_path": str(metrics_path),
                "video_path": video_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
