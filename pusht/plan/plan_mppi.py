#!/usr/bin/env python3
"""Plan in PushT pixel space with pure MPPI MPC over a Markov-state MLP world model."""

from __future__ import annotations

import argparse
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
    goal_reached,
    infer_dataset_state_format,
    latest_object_checkpoint,
    load_action_stats,
    load_config,
    load_dataset_episode,
    load_model,
    make_markov_state,
    make_planning_env,
    make_visualization_env,
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

DEFAULT_MODEL_DIR = "pusht/models/mlpdyn_embd_48"
DEFAULT_OUT_DIR = "pusht/plan/mppi_mlpdyn"
DEVICE = "auto"
HORIZON = 45
MAX_MPC_STEPS = 400
Q_TERMINAL = 5.0
Q_STAGE = 0.005
R_CONTROL = 0.01
VIDEO_FPS = 10
EPISODE_IDX = 727
MPPI_SAMPLES = 2048
MPPI_UPDATE_ITERS = 5
MPPI_REWARD_WEIGHT = 20.0
MPPI_NOISE_LEVEL = 0.35
MPPI_NOISE_DECAY = 1.0
MPPI_BETA_FILTER = 0.7
JAX_PLATFORM = "auto"
JAX_FALLBACK_ENV = "PUSHT_MPPI_JAX_FALLBACK"


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
    parser.add_argument("--mppi-samples", type=int, default=MPPI_SAMPLES)
    parser.add_argument("--mppi-update-iters", type=int, default=MPPI_UPDATE_ITERS)
    parser.add_argument("--mppi-reward-weight", type=float, default=MPPI_REWARD_WEIGHT)
    parser.add_argument("--mppi-noise-level", type=float, default=MPPI_NOISE_LEVEL)
    parser.add_argument("--mppi-noise-decay", type=float, default=MPPI_NOISE_DECAY)
    parser.add_argument("--mppi-beta-filter", type=float, default=MPPI_BETA_FILTER)
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "gpu"), default=JAX_PLATFORM)
    parser.add_argument("--seed", type=int, default=None)
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


def main() -> None:
    args = parse_args()
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
        q_stage=args.q_stage,
        q_terminal=args.q_terminal,
        r_control=args.r_control,
    )
    mppi_config = {
        "planning": {
            "action_dim": action_dim,
            "n_sample": args.mppi_samples,
            "horizon": args.horizon,
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
    mppi_plan_rewards: list[float] = []
    stop_reason = "max_mpc_steps"
    video_path: str | None = None
    metrics_path = out_dir / "metrics.json"
    final_block = dataset_row_to_block_pose(state_np[0], env_state_np[0], state_format)
    final_agent = env_state_np[0, :2].astype(np.float32)
    goal_block = dataset_row_to_block_pose(state_np[-1], env_state_np[-1], state_format)
    rollout_frames: list[np.ndarray] = []
    latent_goal_distances: list[float] = []
    block_goal_distances: list[float] = []
    num_action_clips = 0
    prev_u = jnp.zeros((args.horizon, action_dim), dtype=jnp.float32)
    goal_state_jax = jnp.asarray(goal_state.detach().cpu().numpy().astype(np.float32))
    jax_key = jax.random.PRNGKey(0 if args.seed is None else int(args.seed))
    action_low, action_high = pusht_agent_action_bounds()

    def scalar_or_nan(values: list[float], index: int) -> float:
        return values[index] if values else float("nan")

    def save_outputs() -> dict[str, Any]:
        nonlocal video_path
        metrics = {
            "episode_idx": episode_idx,
            "planner": "mppi",
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
            "latent_goal_distance_initial": scalar_or_nan(latent_goal_distances, 0),
            "latent_goal_distance_final": scalar_or_nan(latent_goal_distances, -1),
            "block_goal_distance_initial": scalar_or_nan(block_goal_distances, 0),
            "block_goal_distance_final": scalar_or_nan(block_goal_distances, -1),
            "latent_goal_distances": latent_goal_distances,
            "block_goal_distances": block_goal_distances,
            "solve_times_ms": solve_times_ms,
            "mppi_samples": args.mppi_samples,
            "mppi_update_iters": args.mppi_update_iters,
            "mppi_reward_weight": args.mppi_reward_weight,
            "mppi_noise_level": args.mppi_noise_level,
            "mppi_noise_decay": args.mppi_noise_decay,
            "mppi_beta_filter": args.mppi_beta_filter,
            "jax_platform": jax_platform,
            "mppi_q_stage": args.q_stage,
            "mppi_q_terminal": args.q_terminal,
            "mppi_r_control": args.r_control,
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
        block_goal_distances = [block_pose_distance(current_block, goal_block)]

        pbar = tqdm(range(args.max_mpc_steps), desc="MPPI Steps")
        try:
            for _ in pbar:
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
                jax.block_until_ready(plan["act_seq"])
                solve_times_ms.append((time.perf_counter() - t0) * 1000.0)

                u_plan = np.asarray(plan["act_seq"], dtype=np.float32)
                prev_u = jnp.asarray(u_plan, dtype=jnp.float32)
                mppi_plan_rewards.append(float(plan["reward"]))

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
                    reward=f"{mppi_plan_rewards[-1]:.3f}",
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
