#!/usr/bin/env python3
"""Plan in PushT pixel space with MPPI warm starts tracked by short-horizon iLQR."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import traceback
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_SLS_SRC = REPO_ROOT / "third_party" / "gpu_sls" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if GPU_SLS_SRC.is_dir() and str(GPU_SLS_SRC) not in sys.path:
    sys.path.insert(0, str(GPU_SLS_SRC))

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax

from gpu_sls.mppi_planner import MPPIPlanner
from pusht.plan.plan_ilqr_mpc import (
    CONTROL_MAX_NORM,
    CONTROL_MIN_NORM,
    DEFAULT_DATASET_PATH,
    ENV_ACTION_SCALE,
    block_pose_distance,
    current_agent_pos,
    current_block_pose,
    dataset_row_to_block_pose,
    dataset_row_to_env_state,
    dataset_row_to_goal_pose,
    encode_frames,
    encode_single_frame,
    extract_full_state,
    infer_dataset_state_format,
    latest_object_checkpoint,
    load_action_stats,
    load_config,
    load_dataset_episode,
    load_model,
    make_markov_state,
    make_planning_env,
    make_visualization_env,
    maybe_cuda_synchronize,
    MarkovDynamicsTorch,
    normalized_to_raw_action,
    preprocess_pixels,
    pusht_agent_action_bounds,
    raw_to_env_action,
    require_device,
    resolve_dataset_paths,
    resolve_model_dir,
    reset_env_to_state,
    save_rgb_image,
    save_rollout_video,
    set_goal_pose,
)
from pusht.train.mlpdyn_train import required_markov_history

DEFAULT_MODEL_DIR = "pusht/models/mlpdyn_embd_8_md_3"
DEFAULT_OUT_DIR = "pusht/plan/mppi_ilqr_track_mlpdyn"
DEVICE = "auto"
MPPI_HORIZON = 250
ILQR_HORIZON = 15
MAX_MPC_STEPS = 400
MPPI_Q_TERMINAL = 10.0
MPPI_Q_STAGE = 0.05
MPPI_R_CONTROL = 0.01
ILQR_Q_TERMINAL = 10.0
ILQR_Q_STAGE = 1.0
ILQR_R_CONTROL = 0.01
VIDEO_FPS = 10
EPISODE_IDX = None
MPPI_SAMPLES = 2048
MPPI_UPDATE_ITERS = 5
MPPI_REWARD_WEIGHT = 20.0
MPPI_NOISE_LEVEL = 0.35
MPPI_NOISE_DECAY = 1.0
MPPI_BETA_FILTER = 0.7
JAX_PLATFORM = "auto"
JAX_FALLBACK_ENV = "PUSHT_MPPI_JAX_FALLBACK"
POSITION_SUCCESS_THRESHOLD = 20.0
YAW_SUCCESS_THRESHOLD = 0.25
REQUIRE_YAW_SUCCESS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--episode-idx", type=int, default=EPISODE_IDX)
    parser.add_argument("--mppi-horizon", type=int, default=MPPI_HORIZON)
    parser.add_argument("--ilqr-horizon", type=int, default=ILQR_HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--mppi-q-terminal", type=float, default=MPPI_Q_TERMINAL)
    parser.add_argument("--mppi-q-stage", type=float, default=MPPI_Q_STAGE)
    parser.add_argument("--mppi-r-control", type=float, default=MPPI_R_CONTROL)
    parser.add_argument("--ilqr-q-terminal", type=float, default=ILQR_Q_TERMINAL)
    parser.add_argument("--ilqr-q-stage", type=float, default=ILQR_Q_STAGE)
    parser.add_argument("--ilqr-r-control", type=float, default=ILQR_R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=50)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--mppi-samples", type=int, default=MPPI_SAMPLES)
    parser.add_argument("--mppi-update-iters", type=int, default=MPPI_UPDATE_ITERS)
    parser.add_argument("--mppi-reward-weight", type=float, default=MPPI_REWARD_WEIGHT)
    parser.add_argument("--mppi-noise-level", type=float, default=MPPI_NOISE_LEVEL)
    parser.add_argument("--mppi-noise-decay", type=float, default=MPPI_NOISE_DECAY)
    parser.add_argument("--mppi-beta-filter", type=float, default=MPPI_BETA_FILTER)
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "gpu"), default=JAX_PLATFORM)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--position-success-threshold", type=float, default=POSITION_SUCCESS_THRESHOLD)
    parser.add_argument("--yaw-success-threshold", type=float, default=YAW_SUCCESS_THRESHOLD)
    parser.add_argument("--require-yaw-success", action=argparse.BooleanOptionalAction, default=REQUIRE_YAW_SUCCESS)
    return parser.parse_args()


def configure_jax_platform(device: torch.device, requested_platform: str) -> str:
    if requested_platform == "auto":
        return "gpu" if device.type == "cuda" else "cpu"
    return requested_platform


def initialize_jax(platform: str) -> tuple[Any, Any, Any]:
    os.environ["JAX_PLATFORMS"] = platform

    import jax
    import jax.numpy as jnp
    from jax import config, lax

    config.update("jax_default_matmul_precision", "highest")
    config.update("jax_enable_x64", False)
    return jax, jnp, lax


def build_batched_jax_dynamics(
    torch_dynamics_net: torch.nn.Module,
    device: torch.device,
    state_dim: int,
) -> Callable[[Any, Any, float, float], Any]:
    import jax
    import jax.numpy as jnp

    def _fwd_fn(x_np: np.ndarray, u_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float32).copy()).to(device)
            u_t = torch.from_numpy(np.asarray(u_np, dtype=np.float32).copy()).to(device)
            inp = torch.cat((x_t, u_t), dim=-1)
            out = torch_dynamics_net(inp)
            return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)

    def jax_dynamics(x: jnp.ndarray, u: jnp.ndarray, t: float, parameter: float) -> jnp.ndarray:
        del t, parameter
        result_shape = jax.ShapeDtypeStruct(x.shape[:-1] + (state_dim,), jnp.float32)
        return jax.pure_callback(_fwd_fn, result_shape, x, u)

    return jax_dynamics


def is_wrapped_keyboard_interrupt(exc: BaseException) -> bool:
    text = "".join(traceback.format_exception_only(type(exc), exc))
    return "KeyboardInterrupt" in text and (
        "CpuCallback error calling callback" in text or "jax.pure_callback failed" in text
    )


def make_mppi_rollout_and_eval(
    jax_dynamics_fn: Callable[[Any, Any, float, float], Any],
    *,
    q_stage: float,
    q_terminal: float,
    r_control: float,
):
    import jax.numpy as jnp
    from jax import lax

    def mppi_rollout_fn(state_cur: Any, act_seqs: Any, reach_config: dict | None = None):
        del reach_config
        batch_size = act_seqs.shape[0]
        state0 = jnp.broadcast_to(state_cur, (batch_size, state_cur.shape[0]))

        def step(state_batch: Any, u_batch: Any):
            next_state_batch = jax_dynamics_fn(state_batch, u_batch, 0.0, 1.0)
            return next_state_batch, next_state_batch

        _, state_seqs = lax.scan(step, state0, act_seqs.swapaxes(0, 1))
        state_seqs = state_seqs.swapaxes(0, 1)
        return state_seqs, {}

    def mppi_eval_fn(
        state_seqs: Any,
        act_seqs: Any,
        reach_config: dict | None = None,
        aux: dict | None = None,
        *,
        goal_state: Any,
    ):
        del reach_config, aux
        if state_seqs.shape[1] > 1:
            stage_delta = state_seqs[:, :-1, :] - goal_state[None, None, :]
            stage_costs = q_stage * jnp.sum(stage_delta**2, axis=-1).sum(axis=-1)
        else:
            stage_costs = jnp.zeros((state_seqs.shape[0],), dtype=state_seqs.dtype)
        terminal_delta = state_seqs[:, -1, :] - goal_state[None, :]
        terminal_costs = q_terminal * jnp.sum(terminal_delta**2, axis=-1)
        action_costs = r_control * jnp.sum(act_seqs**2, axis=(-1, -2))
        total_costs = stage_costs + terminal_costs + action_costs
        return {"rewards": -total_costs}

    return mppi_rollout_fn, mppi_eval_fn


def shift_warmstart(U: Any) -> Any:
    import jax.numpy as jnp

    return jnp.concatenate([U[1:], U[-1:]], axis=0)


def hard_pusht_goal_distances(current_block_pose: np.ndarray, goal_block_pose: np.ndarray) -> tuple[float, float, float]:
    position_distance = float(np.linalg.norm(current_block_pose[:2].astype(np.float64) - goal_block_pose[:2].astype(np.float64)))
    yaw_distance = abs(float((current_block_pose[2] - goal_block_pose[2] + np.pi) % (2.0 * np.pi) - np.pi))
    pose_distance = block_pose_distance(current_block_pose, goal_block_pose)
    return position_distance, yaw_distance, pose_distance


def hard_pusht_goal_success(
    current_block_pose: np.ndarray,
    goal_block_pose: np.ndarray,
    *,
    position_success_threshold: float,
    yaw_success_threshold: float,
    require_yaw_success: bool,
) -> tuple[bool, dict[str, float | bool]]:
    position_distance, yaw_distance, pose_distance = hard_pusht_goal_distances(current_block_pose, goal_block_pose)
    position_success = position_distance <= position_success_threshold
    yaw_success = yaw_distance <= yaw_success_threshold
    success = bool(position_success and (yaw_success or not require_yaw_success))
    return success, {
        "position_goal_distance": position_distance,
        "yaw_goal_distance": yaw_distance,
        "block_goal_distance": pose_distance,
        "position_success": bool(position_success),
        "yaw_success": bool(yaw_success),
        "success": success,
    }


class BoundedILQRTrajectoryTracker:
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
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

    def _project_control(self, u: torch.Tensor) -> torch.Tensor:
        return torch.clamp(u, min=self.u_min, max=self.u_max)

    def _make_initial_action_guess(self, warmstart_np: np.ndarray | None) -> torch.Tensor:
        if warmstart_np is not None:
            if warmstart_np.shape != (self.horizon, self.action_dim):
                raise ValueError(
                    f"Expected warmstart shape {(self.horizon, self.action_dim)}, got {warmstart_np.shape}."
                )
            return self._project_control(torch.tensor(warmstart_np, dtype=torch.float32, device=self.device))
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

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = x_traj[step] - x_ref[step]
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_ref[self.horizon]
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

    def solve(
        self,
        x0_np: np.ndarray,
        x_ref_np: np.ndarray,
        warmstart_np: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        if x_ref_np.shape != (self.horizon + 1, self.state_dim):
            raise ValueError(
                f"Expected x_ref_np with shape {(self.horizon + 1, self.state_dim)}, got {x_ref_np.shape}."
            )

        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_ref = torch.tensor(x_ref_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess(warmstart_np)

        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_cost(x_traj, u_seq, x_ref).item())
        iterations = 0
        reg = self.regularization

        for iteration in range(self.max_iters):
            iterations = iteration + 1
            a_seq, b_seq = self._linearize_dynamics(x_traj, u_seq)
            k_seq = torch.empty((self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            kk_seq = torch.empty((self.horizon, self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)

            terminal_err = x_traj[self.horizon] - x_ref[self.horizon]
            v_x = 2.0 * self.q_terminal * terminal_err
            v_xx = 2.0 * self.q_terminal * self.eye_x
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = x_traj[step] - x_ref[step]
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
                new_cost = float(self._trajectory_cost(x_new, u_new, x_ref).item())
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
    if args.ilqr_horizon < 1:
        raise ValueError(f"--ilqr-horizon must be positive, got {args.ilqr_horizon}.")
    if args.mppi_horizon < args.ilqr_horizon:
        raise ValueError(
            f"--mppi-horizon must be >= --ilqr-horizon, got {args.mppi_horizon} and {args.ilqr_horizon}."
        )
    if args.position_success_threshold < 0.0:
        raise ValueError("--position-success-threshold cannot be negative.")
    if args.yaw_success_threshold < 0.0:
        raise ValueError("--yaw-success-threshold cannot be negative.")
    device = require_device(args.device)
    requested_jax_platform = configure_jax_platform(device, args.jax_platform)
    fallback_enabled = args.jax_platform == "auto" and os.environ.get(JAX_FALLBACK_ENV) != "1"
    model_dir = resolve_model_dir(args)
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model_config = load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    model = load_model(checkpoint_path, device)

    markov_deriv = int(model_config.get("markov_deriv", 1))
    if markov_deriv < 0:
        raise ValueError(f"Expected non-negative markov_deriv for the MLP model, got {markov_deriv}.")

    img_size = int(model_config.get("img_size", 224))
    frameskip = int(model_config.get("frameskip", 1))
    action_dim = int(model_config.get("action_dim", 2))
    embed_dim = int(model_config.get("embed_dim", 48))
    markov_state_dim = int(model_config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    if frameskip != 1:
        raise ValueError(
            f"This PushT MPC planner currently supports frameskip=1 only, but the model config has frameskip={frameskip}."
        )

    train_dataset_paths = resolve_dataset_paths(model_config.get("dataset_path"), dataset_path)
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

    try:
        jax, jnp, _lax = initialize_jax(requested_jax_platform)
        # Force backend initialization here so a broken GPU setup fails before planning starts.
        _ = jnp.zeros((1,), dtype=jnp.float32)
        jax.block_until_ready(_)
        jax_platform = requested_jax_platform
    except Exception:
        if not fallback_enabled or requested_jax_platform != "gpu":
            raise
        traceback.print_exc = lambda *args, **kwargs: None
        sys.stderr.write("JAX GPU init failed, restarting with CPU backend.\n")
        sys.stderr.flush()
        restart_env = os.environ.copy()
        restart_env["JAX_PLATFORMS"] = "cpu"
        restart_env[JAX_FALLBACK_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], restart_env)

    jax_dynamics = build_batched_jax_dynamics(model.predictor.net, device, markov_state_dim)
    mppi_rollout_fn, mppi_eval_fn = make_mppi_rollout_and_eval(
        jax_dynamics,
        q_stage=args.mppi_q_stage,
        q_terminal=args.mppi_q_terminal,
        r_control=args.mppi_r_control,
    )
    mppi_config = {
        "planning": {
            "action_dim": action_dim,
            "n_sample": args.mppi_samples,
            "horizon": args.mppi_horizon,
            "n_update_iter": args.mppi_update_iters,
            "use_last": True,
            "reject_bad": False,
            "mppi": {
                "reward_weight": args.mppi_reward_weight,
                "noise_level": args.mppi_noise_level,
                "noise_decay": args.mppi_noise_decay,
                "beta_filter": args.mppi_beta_filter,
            },
        }
    }
    action_lower_lim = jnp.full((action_dim,), CONTROL_MIN_NORM, dtype=jnp.float32)
    action_upper_lim = jnp.full((action_dim,), CONTROL_MAX_NORM, dtype=jnp.float32)
    planner = MPPIPlanner(
        config=mppi_config,
        model_rollout_fn=mppi_rollout_fn,
        evaluate_traj_fn=mppi_eval_fn,
        action_lower_lim=action_lower_lim,
        action_upper_lim=action_upper_lim,
    )
    tracker_dynamics = MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    ilqr_tracker = BoundedILQRTrajectoryTracker(
        tracker_dynamics,
        horizon=args.ilqr_horizon,
        q_terminal=args.ilqr_q_terminal,
        q_stage=args.ilqr_q_stage,
        r_control=args.ilqr_r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        control_min=CONTROL_MIN_NORM,
        control_max=CONTROL_MAX_NORM,
        device=device,
    )

    def mppi_trajopt(
        key: jax.Array,
        state_cur: jnp.ndarray,
        init_action_seq: jnp.ndarray,
        goal_state_jax: jnp.ndarray,
    ):
        return planner.trajectory_optimization(
            key,
            state_cur,
            init_action_seq,
            skip=False,
            goal_state=goal_state_jax,
        )

    jit_mppi_trajopt = jax.jit(mppi_trajopt)

    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_env: list[np.ndarray] = []
    solve_times_ms: list[float] = []
    mppi_solve_times_ms: list[float] = []
    ilqr_solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    mppi_plan_rewards: list[float] = []
    latent_track_errors: list[float] = []
    stop_reason = "max_mpc_steps"
    video_path: str | None = None
    metrics_path = out_dir / "metrics.json"
    final_block = dataset_row_to_block_pose(state_np[0], env_state_np[0], state_format)
    final_agent = env_state_np[0, :2].astype(np.float32)
    goal_block = dataset_row_to_block_pose(state_np[-1], env_state_np[-1], state_format)
    rollout_frames: list[np.ndarray] = []
    latent_goal_distances: list[float] = []
    block_goal_distances: list[float] = []
    position_goal_distances: list[float] = []
    yaw_goal_distances: list[float] = []
    success_log: list[bool] = []
    position_success_log: list[bool] = []
    yaw_success_log: list[bool] = []
    success = False
    num_action_clips = 0
    prev_u = jnp.zeros((args.mppi_horizon, action_dim), dtype=jnp.float32)
    goal_state_jax = jnp.asarray(goal_state.detach().cpu().numpy().astype(np.float32))
    jax_key = jax.random.PRNGKey(0 if args.seed is None else int(args.seed))
    action_low, action_high = pusht_agent_action_bounds()

    def scalar_or_nan(values: list[float], index: int) -> float:
        return values[index] if values else float("nan")

    def save_outputs() -> dict[str, Any]:
        nonlocal video_path
        metrics = {
            "episode_idx": episode_idx,
            "seed": args.seed,
            "planner": "mppi_ilqr_track",
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
            "mppi_horizon": args.mppi_horizon,
            "ilqr_horizon": args.ilqr_horizon,
            "max_mpc_steps": args.max_mpc_steps,
            "position_success_threshold": float(args.position_success_threshold),
            "yaw_success_threshold": float(args.yaw_success_threshold),
            "require_yaw_success": bool(args.require_yaw_success),
            "stop_reason": stop_reason,
            "success": bool(success),
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
            "latent_goal_distance_initial": scalar_or_nan(latent_goal_distances, 0),
            "latent_goal_distance_final": scalar_or_nan(latent_goal_distances, -1),
            "block_goal_distance_initial": scalar_or_nan(block_goal_distances, 0),
            "block_goal_distance_final": scalar_or_nan(block_goal_distances, -1),
            "block_goal_distance_min": min(block_goal_distances) if block_goal_distances else float("nan"),
            "position_goal_distance_initial": scalar_or_nan(position_goal_distances, 0),
            "position_goal_distance_final": scalar_or_nan(position_goal_distances, -1),
            "position_goal_distance_min": min(position_goal_distances) if position_goal_distances else float("nan"),
            "yaw_goal_distance_initial": scalar_or_nan(yaw_goal_distances, 0),
            "yaw_goal_distance_final": scalar_or_nan(yaw_goal_distances, -1),
            "yaw_goal_distance_min": min(yaw_goal_distances) if yaw_goal_distances else float("nan"),
            "latent_goal_distances": latent_goal_distances,
            "block_goal_distances": block_goal_distances,
            "position_goal_distances": position_goal_distances,
            "yaw_goal_distances": yaw_goal_distances,
            "success_log": success_log,
            "position_success_log": position_success_log,
            "yaw_success_log": yaw_success_log,
            "solve_times_ms": solve_times_ms,
            "mppi_solve_times_ms": mppi_solve_times_ms,
            "ilqr_solve_times_ms": ilqr_solve_times_ms,
            "ilqr_iterations": ilqr_iterations,
            "ilqr_costs": ilqr_costs,
            "latent_track_errors": latent_track_errors,
            "mppi_samples": args.mppi_samples,
            "mppi_update_iters": args.mppi_update_iters,
            "mppi_reward_weight": args.mppi_reward_weight,
            "mppi_noise_level": args.mppi_noise_level,
            "mppi_noise_decay": args.mppi_noise_decay,
            "mppi_beta_filter": args.mppi_beta_filter,
            "jax_platform": jax_platform,
            "mppi_q_stage": args.mppi_q_stage,
            "mppi_q_terminal": args.mppi_q_terminal,
            "mppi_r_control": args.mppi_r_control,
            "ilqr_q_stage": args.ilqr_q_stage,
            "ilqr_q_terminal": args.ilqr_q_terminal,
            "ilqr_r_control": args.ilqr_r_control,
            "mppi_plan_rewards": mppi_plan_rewards,
            "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
            "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
            "executed_actions_env": [action.tolist() for action in executed_actions_env],
            "video_path": video_path,
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        if rollout_frames and video_path is None:
            video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps))
            metrics["video_path"] = video_path
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
        return metrics

    plan_env = make_planning_env(width=width, height=height)
    viz_env = make_visualization_env(width=width, height=height)
    try:
        set_goal_pose(plan_env, goal_pose)
        set_goal_pose(viz_env, goal_pose)
        hidden_start = reset_env_to_state(plan_env, env_state_np[0])
        visible_start = reset_env_to_state(viz_env.unwrapped, env_state_np[0])

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
        current_block = current_block_pose(plan_env)

        rollout_frames = [visible_start.copy()]
        latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
        initial_success, initial_goal_metrics = hard_pusht_goal_success(
            current_block,
            goal_block,
            position_success_threshold=float(args.position_success_threshold),
            yaw_success_threshold=float(args.yaw_success_threshold),
            require_yaw_success=bool(args.require_yaw_success),
        )
        block_goal_distances = [float(initial_goal_metrics["block_goal_distance"])]
        position_goal_distances = [float(initial_goal_metrics["position_goal_distance"])]
        yaw_goal_distances = [float(initial_goal_metrics["yaw_goal_distance"])]
        success_log = [bool(initial_goal_metrics["success"])]
        position_success_log = [bool(initial_goal_metrics["position_success"])]
        yaw_success_log = [bool(initial_goal_metrics["yaw_success"])]
        success = bool(initial_success)

        pbar = tqdm(range(args.max_mpc_steps), desc="MPPI+iLQR Steps")
        try:
            if success:
                stop_reason = "goal_reached"
            for _ in pbar:
                if success:
                    break
                current_state_np = current_state.detach().cpu().numpy().astype(np.float32)
                init_action_seq = shift_warmstart(prev_u)
                jax_key, subkey = jax.random.split(jax_key)

                t0 = time.perf_counter()
                plan = jit_mppi_trajopt(
                    subkey,
                    jnp.asarray(current_state_np, dtype=jnp.float32),
                    init_action_seq,
                    goal_state_jax,
                )
                jax.block_until_ready(plan["state_seq"])
                mppi_ms = (time.perf_counter() - t0) * 1000.0
                mppi_solve_times_ms.append(mppi_ms)

                mppi_u_plan = np.asarray(plan["act_seq"], dtype=np.float32)
                mppi_state_seq = np.asarray(plan["state_seq"], dtype=np.float32)
                prev_u = jnp.asarray(mppi_u_plan, dtype=jnp.float32)
                mppi_plan_rewards.append(float(plan["reward"]))

                x_ref_np = np.concatenate(
                    [current_state_np[None, :], mppi_state_seq[: args.ilqr_horizon]],
                    axis=0,
                ).astype(np.float64)
                ilqr_warmstart_np = mppi_u_plan[: args.ilqr_horizon].astype(np.float64)
                _x_track, ilqr_u_plan, ilqr_solve_time, n_iters, plan_cost = ilqr_tracker.solve(
                    current_state_np.astype(np.float64),
                    x_ref_np,
                    ilqr_warmstart_np,
                )
                ilqr_ms = ilqr_solve_time * 1000.0
                ilqr_solve_times_ms.append(ilqr_ms)
                solve_times_ms.append(mppi_ms + ilqr_ms)
                ilqr_iterations.append(int(n_iters))
                ilqr_costs.append(float(plan_cost))

                u0_norm = ilqr_u_plan[0].astype(np.float32)
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
                step_success, goal_metrics = hard_pusht_goal_success(
                    current_block,
                    goal_block,
                    position_success_threshold=float(args.position_success_threshold),
                    yaw_success_threshold=float(args.yaw_success_threshold),
                    require_yaw_success=bool(args.require_yaw_success),
                )
                block_goal_distance = float(goal_metrics["block_goal_distance"])
                position_goal_distance = float(goal_metrics["position_goal_distance"])
                yaw_goal_distance = float(goal_metrics["yaw_goal_distance"])
                latent_track_error = float(torch.linalg.vector_norm(current_state - torch.from_numpy(x_ref_np[1]).to(current_state)).item())
                latent_goal_distances.append(latent_goal_distance)
                block_goal_distances.append(block_goal_distance)
                position_goal_distances.append(position_goal_distance)
                yaw_goal_distances.append(yaw_goal_distance)
                success_log.append(bool(goal_metrics["success"]))
                position_success_log.append(bool(goal_metrics["position_success"]))
                yaw_success_log.append(bool(goal_metrics["yaw_success"]))
                latent_track_errors.append(latent_track_error)
                success = bool(step_success)

                pbar.set_postfix(
                    solve_ms=f"{solve_times_ms[-1]:.1f}",
                    mppi=f"{mppi_ms:.1f}",
                    ilqr=f"{ilqr_ms:.1f}",
                    iters=f"{ilqr_iterations[-1]}",
                    reward=f"{mppi_plan_rewards[-1]:.3f}",
                    track=f"{latent_track_error:.3f}",
                    latent_goal=f"{latent_goal_distance:.3f}",
                    pos_goal=f"{position_goal_distance:.2f}",
                    yaw_goal=f"{yaw_goal_distance:.3f}",
                    success=int(success),
                )

                if success:
                    stop_reason = "goal_reached"
                    break
                if terminated or truncated:
                    stop_reason = "terminated" if terminated else "truncated"
                    break
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
        except Exception as exc:
            if not is_wrapped_keyboard_interrupt(exc):
                raise
            stop_reason = "keyboard_interrupt"
        finally:
            pbar.close()

        final_block = current_block_pose(plan_env)
        final_agent = current_agent_pos(plan_env)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        plan_env.close()
        viz_env.close()

    metrics = save_outputs()

    print(
        json.dumps(
            {
                "episode_idx": episode_idx,
                "success": metrics["success"],
                "position_goal_distance_min": metrics["position_goal_distance_min"],
                "latent_goal_distance_final": metrics["latent_goal_distance_final"],
                "block_goal_distance_final": metrics["block_goal_distance_final"],
                "position_goal_distance_final": metrics["position_goal_distance_final"],
                "yaw_goal_distance_final": metrics["yaw_goal_distance_final"],
                "stop_reason": stop_reason,
                "metrics_path": str(metrics_path),
                "video_path": video_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
