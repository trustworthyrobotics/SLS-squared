#!/usr/bin/env python3
"""Plan in OGBench cube pixel space with nominal iLQR MPC over a Markov-state MLP world model."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium
import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import ogbench.manipspace  # noqa: F401
import torch
from ogbench.manipspace import lie
from tqdm.auto import tqdm

from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset, build_markov_state, required_markov_history

DEFAULT_TEST_DATASET_PATH = "ogbench_cube/data/test_data/ogbench_cube_test.h5"
DEFAULT_MODEL_DIR = "ogbench_cube/models/mlpdyn_embd_12_strtn"
DEFAULT_OUT_DIR = "ogbench_cube/plan/ilqr_mpc_mlpdyn"

DEVICE = "auto"
HORIZON = 15
MAX_MPC_STEPS = 50
Q_TERMINAL = 15.0
Q_STAGE = 0.05
R_CONTROL = 0.5
VIDEO_FPS = 20
EPISODE_IDX = 284
MAX_ORACLE_STEPS = 80
ORACLE_SEGMENT_DT = 0.4
ORACLE_NOISE = 0.0
ORACLE_NOISE_SMOOTHING = 0.5
GRASP_CONTACT_THRESHOLD = 0.5
GRASP_ALIGNMENT_THRESHOLD = 0.03


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
    parser.add_argument("--env-max-episode-steps", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--max-oracle-steps", type=int, default=MAX_ORACLE_STEPS)
    parser.add_argument("--oracle-segment-dt", type=float, default=ORACLE_SEGMENT_DT)
    parser.add_argument("--oracle-noise", type=float, default=ORACLE_NOISE)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=ORACLE_NOISE_SMOOTHING)
    parser.add_argument("--grasp-contact-threshold", type=float, default=GRASP_CONTACT_THRESHOLD)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=GRASP_ALIGNMENT_THRESHOLD)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
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


def angular_distance(a: float, b: float) -> float:
    return float(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def action_to_standardized(action: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return ((action.astype(np.float32) - action_mean.reshape(-1)) / action_std.reshape(-1)).astype(np.float32)


def cube_is_grasped(
    info: dict[str, np.ndarray],
    *,
    contact_threshold: float,
    alignment_threshold: float,
) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_threshold and block_alignment <= alignment_threshold)


def load_dataset_episode(
    dataset_path: Path,
    episode_idx: int,
) -> dict[str, np.ndarray | int | float | str]:
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
            "effector_pos": np.asarray(h5["effector_pos"][rows], dtype=np.float32),
            "effector_yaw": np.asarray(h5["effector_yaw"][rows], dtype=np.float32),
            "block_pos": np.asarray(h5["block_pos"][rows], dtype=np.float32),
            "block_quat": np.asarray(h5["block_quat"][rows], dtype=np.float32),
            "block_yaw": np.asarray(h5["block_yaw"][rows], dtype=np.float32),
            "target_block_pos": np.asarray(h5["target_block_pos"][rows], dtype=np.float32),
            "target_block_yaw": np.asarray(h5["target_block_yaw"][rows], dtype=np.float32),
            "time": np.asarray(h5["time"][rows], dtype=np.float32),
            "episode_seed": int(h5["episode_seed"][episode_idx]),
            "env_name": str(h5.attrs.get("env_name", "cube-single-v0")),
            "camera": str(h5.attrs.get("camera", "front_pixels")),
            "width": int(h5["pixels"].shape[2]),
            "height": int(h5["pixels"].shape[1]),
            "physics_timestep": float(h5.attrs.get("physics_timestep", 1.0 / 500.0)),
            "control_timestep": float(h5.attrs.get("control_timestep", 25.0 / 500.0)),
            "max_episode_steps": int(h5.attrs.get("max_episode_steps", ep_len)),
            "video_fps": float(h5.attrs.get("video_fps", 1.0 / float(h5.attrs.get("control_timestep", 25.0 / 500.0)))),
        }


def make_env(
    *,
    env_name: str,
    physics_timestep: float,
    control_timestep: float,
    max_episode_steps: int,
    width: int,
    height: int,
) -> gymnasium.Env:
    return gymnasium.make(
        env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=False,
        max_episode_steps=max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=width,
        height=height,
    )


def restore_target_pose(
    env: gymnasium.Env,
    *,
    target_block_pos: np.ndarray,
    target_block_yaw: float,
) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )
    for geom_ids in unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            unwrapped._model.geom(gid).rgba[3] = 0.0


def reset_env_to_state(
    env: gymnasium.Env,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_block_pos: np.ndarray,
    target_block_yaw: float,
    camera: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    unwrapped._data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    restore_target_pose(
        env,
        target_block_pos=target_block_pos,
        target_block_yaw=target_block_yaw,
    )
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    frame = np.asarray(unwrapped.render(camera=camera), dtype=np.uint8)
    info = unwrapped.get_step_info()
    return frame, info


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
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)
        self.eye_x = torch.eye(self.state_dim, dtype=torch.float32, device=device)
        self.eye_u = torch.eye(self.action_dim, dtype=torch.float32, device=device)
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

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
            state_err = x_traj[step] - x_goal
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_goal
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

    markov_deriv = int(config.get("markov_deriv", 1))
    if markov_deriv < 0:
        raise ValueError(f"Expected non-negative markov_deriv for the MLP model, got {markov_deriv}.")

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 5))
    embed_dim = int(config.get("embed_dim", 24))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))

    train_dataset_path = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    train_stats_dataset = LeWMOGBenchCubeDataset(
        train_dataset_path,
        markov_deriv=markov_deriv,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)

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
    episode_seed = int(episode["episode_seed"])
    env_name = str(episode["env_name"])
    camera = str(episode["camera"])
    width = int(episode["width"])
    height = int(episode["height"])
    physics_timestep = float(episode["physics_timestep"])
    control_timestep = float(episode["control_timestep"])
    dataset_max_episode_steps = int(episode["max_episode_steps"])
    max_episode_steps = (
        int(args.env_max_episode_steps)
        if args.env_max_episode_steps is not None
        else max(dataset_max_episode_steps, int(args.max_oracle_steps) + int(args.max_mpc_steps) + 1)
    )
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    target_block_pos_np = np.asarray(episode["target_block_pos"], dtype=np.float32)
    target_block_yaw_np = np.asarray(episode["target_block_yaw"], dtype=np.float32)
    dataset_pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)

    run_name = f"{int(time.time())}_episode_{episode_idx:05d}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        env_name=env_name,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        max_episode_steps=max_episode_steps,
        width=width,
        height=height,
    )
    oracle = LocalCubePlanOracle(
        env=env,
        segment_dt=args.oracle_segment_dt,
        noise=args.oracle_noise,
        noise_smoothing=args.oracle_noise_smoothing,
    )

    start_frame, start_info = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        target_block_pos=target_block_pos_np[0],
        target_block_yaw=float(target_block_yaw_np[0, 0]),
        camera=camera,
    )
    goal_frame, goal_info = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[-1],
        qvel=qvel_np[-1],
        target_block_pos=target_block_pos_np[-1],
        target_block_yaw=float(target_block_yaw_np[-1, 0]),
        camera=camera,
    )
    current_frame, current_info = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        target_block_pos=target_block_pos_np[0],
        target_block_yaw=float(target_block_yaw_np[0, 0]),
        camera=camera,
    )

    goal_emb = encode_single_frame(
        model,
        goal_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_emb = encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    history_len = required_markov_history(markov_deriv)
    current_history = [current_emb] * history_len
    goal_history = [goal_emb] * history_len
    current_state = make_markov_state(current_history, markov_deriv)
    goal_state = make_markov_state(goal_history, markov_deriv)
    if int(current_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {current_state.numel()}.")

    dataset_pixels = preprocess_pixels(
        dataset_pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    true_latents = encode_frames(
        model,
        dataset_pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )

    save_rgb_image(out_dir / "start_image.png", start_frame)
    save_rgb_image(out_dir / "goal_image.png", goal_frame)

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
        device=device,
    )

    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_block_pos = np.asarray(goal_info["privileged/target_block_pos"], dtype=np.float32)
    goal_block_yaw = float(goal_info["privileged/target_block_yaw"][0])

    rollout_frames = [current_frame.copy()]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
    embedding_goal_distances = [float(torch.linalg.vector_norm(current_emb - goal_emb).item())]
    cube_goal_distances = [float(np.linalg.norm(np.asarray(current_info["privileged/block_0_pos"]) - goal_block_pos))]
    cube_yaw_errors = [
        angular_distance(
            float(current_info["privileged/block_0_yaw"][0]),
            goal_block_yaw,
        )
    ]
    used_oracle_before_mpc = False
    oracle_grasped = cube_is_grasped(
        current_info,
        contact_threshold=args.grasp_contact_threshold,
        alignment_threshold=args.grasp_alignment_threshold,
    )
    oracle_steps_executed = 0
    mpc_steps_executed = 0
    handoff_step = 0 if oracle_grasped else None
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    stop_reason = "max_mpc_steps"
    terminated = False
    truncated = False

    if not oracle_grasped:
        used_oracle_before_mpc = True
        oracle.reset(None, current_info)
        oracle_pbar = tqdm(range(args.max_oracle_steps), desc="Oracle Grasp")
        for _ in oracle_pbar:
            oracle_action = np.asarray(oracle.select_action(None, current_info), dtype=np.float32)
            oracle_action_std = action_to_standardized(oracle_action, action_mean, action_std)
            executed_actions_raw.append(oracle_action.copy())
            executed_actions_norm.append(oracle_action_std.copy())

            _, _, terminated, truncated, step_info = env.step(oracle_action)
            current_info = step_info
            current_frame = np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)
            next_emb = encode_single_frame(
                model,
                current_frame,
                device=device,
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            current_history.append(next_emb)
            current_history = current_history[-history_len:]
            current_state = make_markov_state(current_history, markov_deriv)
            current_emb = next_emb

            oracle_steps_executed += 1
            rollout_frames.append(current_frame.copy())
            latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
            embedding_goal_distance = float(torch.linalg.vector_norm(current_emb - goal_emb).item())
            cube_goal_distance = float(np.linalg.norm(np.asarray(current_info["privileged/block_0_pos"]) - goal_block_pos))
            cube_yaw_error = angular_distance(
                float(current_info["privileged/block_0_yaw"][0]),
                goal_block_yaw,
            )
            latent_goal_distances.append(latent_goal_distance)
            embedding_goal_distances.append(embedding_goal_distance)
            cube_goal_distances.append(cube_goal_distance)
            cube_yaw_errors.append(cube_yaw_error)

            oracle_grasped = cube_is_grasped(
                current_info,
                contact_threshold=args.grasp_contact_threshold,
                alignment_threshold=args.grasp_alignment_threshold,
            )
            oracle_pbar.set_postfix(
                grasped=f"{int(oracle_grasped)}",
                latent_goal=f"{latent_goal_distance:.3f}",
                cube_goal=f"{cube_goal_distance:.3f}",
            )
            if oracle_grasped:
                handoff_step = int(len(executed_actions_raw))
                break
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break
        oracle_pbar.close()

    if oracle_grasped and not (terminated or truncated):
        pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
        for _ in pbar:
            current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
            _, u_plan, solve_time, n_iters, plan_cost = mpc_solver.solve(current_state_np, goal_state_np)
            solve_times_ms.append(solve_time * 1000.0)
            ilqr_iterations.append(int(n_iters))
            ilqr_costs.append(float(plan_cost))

            u0_norm = u_plan[0].astype(np.float32)
            u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
            executed_actions_norm.append(u0_norm.copy())
            executed_actions_raw.append(u0_raw.copy())

            _, _, terminated, truncated, step_info = env.step(u0_raw)
            current_info = step_info
            current_frame = np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)
            next_emb = encode_single_frame(
                model,
                current_frame,
                device=device,
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            current_history.append(next_emb)
            current_history = current_history[-history_len:]
            current_state = make_markov_state(current_history, markov_deriv)
            current_emb = next_emb

            mpc_steps_executed += 1
            rollout_frames.append(current_frame.copy())
            latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
            embedding_goal_distance = float(torch.linalg.vector_norm(current_emb - goal_emb).item())
            cube_goal_distance = float(np.linalg.norm(np.asarray(current_info["privileged/block_0_pos"]) - goal_block_pos))
            cube_yaw_error = angular_distance(
                float(current_info["privileged/block_0_yaw"][0]),
                goal_block_yaw,
            )
            latent_goal_distances.append(latent_goal_distance)
            embedding_goal_distances.append(embedding_goal_distance)
            cube_goal_distances.append(cube_goal_distance)
            cube_yaw_errors.append(cube_yaw_error)

            pbar.set_postfix(
                solve_ms=f"{solve_times_ms[-1]:.1f}",
                iters=f"{ilqr_iterations[-1]}",
                latent_goal=f"{latent_goal_distance:.3f}",
                cube_goal=f"{cube_goal_distance:.3f}",
            )

            if terminated or truncated:
                if truncated and mpc_steps_executed >= int(args.max_mpc_steps):
                    stop_reason = "max_mpc_steps"
                else:
                    stop_reason = "terminated" if terminated else "truncated"
                break
        pbar.close()
    elif not oracle_grasped and not (terminated or truncated):
        stop_reason = "oracle_failed_to_grasp"

    if oracle_grasped and mpc_steps_executed >= int(args.max_mpc_steps) and not (terminated or truncated):
        stop_reason = "max_mpc_steps"

    final_info = current_info
    ilqr_start_frame_idx = None if handoff_step is None else int(handoff_step)
    metrics = {
        "episode_idx": int(episode_idx),
        "episode_seed": episode_seed,
        "checkpoint": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "env_name": env_name,
        "camera": camera,
        "img_size": img_size,
        "horizon": int(args.horizon),
        "max_mpc_steps": int(args.max_mpc_steps),
        "dataset_max_episode_steps": dataset_max_episode_steps,
        "env_max_episode_steps": max_episode_steps,
        "num_executed_steps": int(len(executed_actions_raw)),
        "oracle_steps_executed": int(oracle_steps_executed),
        "mpc_steps_executed": int(mpc_steps_executed),
        "used_oracle_before_mpc": bool(used_oracle_before_mpc),
        "oracle_grasped": bool(oracle_grasped),
        "handoff_step": None if handoff_step is None else int(handoff_step),
        "world_model_ilqr_start_frame_idx": ilqr_start_frame_idx,
        "world_model_ilqr_start_time_s": None
        if ilqr_start_frame_idx is None
        else float(ilqr_start_frame_idx) / float(args.video_fps),
        "stop_reason": stop_reason,
        "latent_goal_distance_initial": float(latent_goal_distances[0]),
        "latent_goal_distance_final": float(latent_goal_distances[-1]),
        "cube_goal_distance_initial": float(cube_goal_distances[0]),
        "cube_goal_distance_final": float(cube_goal_distances[-1]),
        "cube_yaw_error_initial": float(cube_yaw_errors[0]),
        "cube_yaw_error_final": float(cube_yaw_errors[-1]),
        "goal_block_pos": goal_block_pos.tolist(),
        "goal_block_yaw": goal_block_yaw,
        "final_block_pos": np.asarray(final_info["privileged/block_0_pos"], dtype=np.float32).tolist(),
        "final_block_yaw": float(final_info["privileged/block_0_yaw"][0]),
        "final_effector_pos": np.asarray(final_info["proprio/effector_pos"], dtype=np.float32).tolist(),
        "final_effector_yaw": float(final_info["proprio/effector_yaw"][0]),
        "final_qpos": np.asarray(final_info["qpos"], dtype=np.float32).tolist(),
        "final_qvel": np.asarray(final_info["qvel"], dtype=np.float32).tolist(),
        "solve_times_ms": solve_times_ms,
        "ilqr_iterations": ilqr_iterations,
        "ilqr_costs": ilqr_costs,
        "latent_goal_distances": latent_goal_distances,
        "embedding_goal_distances": embedding_goal_distances,
        "cube_goal_distances": cube_goal_distances,
        "cube_yaw_errors": cube_yaw_errors,
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "dataset_start_pixel_l2": float(np.linalg.norm(start_frame.astype(np.float32) - dataset_pixels_np[0].astype(np.float32))),
        "dataset_goal_pixel_l2": float(np.linalg.norm(goal_frame.astype(np.float32) - dataset_pixels_np[-1].astype(np.float32))),
        "dataset_goal_latent_distance": float(torch.linalg.vector_norm(true_latents[-1] - goal_emb).item()),
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    env.close()

    print(f"Saved to: {out_dir}")
    if video_path is not None:
        print(f"Video: {video_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
