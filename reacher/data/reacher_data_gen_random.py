#!/usr/bin/env python3
"""Collect random-action DM Control Reacher trajectories directly into LE-WM HDF5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.reacher_policy_train import DmControlGymEnv

DEFAULT_OUTDIR = "reacher/data/random_rollouts"
DEFAULT_OUTPUT_NAME = "reacher_random.h5"
DEFAULT_ACTION_NOISE_SCALE = 10.0
ROLLOUT_MODES = ("random",)

# Local timing/video knobs.
PHYSICS_FREQ_HZ = 50.0
CONTROL_FREQ_HZ = 50.0
STATE_DIM = 6
ACTION_DIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("easy", "hard"), default="hard")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-transitions",
        type=int,
        default=None,
        help="Collect variable-length trajectories until this many action transitions are stored.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help="Fallback collection target when --target-transitions is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=PHYSICS_FREQ_HZ)
    parser.add_argument("--control-freq-hz", type=float, default=CONTROL_FREQ_HZ)
    parser.add_argument(
        "--trajectory-length",
        "--max-steps-per-episode",
        dest="trajectory_length",
        type=int,
        default=100,
        help="Number of random control steps to apply per trajectory.",
    )
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--action-noise-scale", type=float, default=DEFAULT_ACTION_NOISE_SCALE)
    parser.add_argument("--action-cost-weight", type=float, default=0.0)
    parser.add_argument("--action-rate-cost-weight", type=float, default=0.0)
    parser.add_argument("--velocity-cost-weight", type=float, default=0.0)
    return parser.parse_args()


def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def compute_control_substeps(physics_freq_hz: float, control_freq_hz: float) -> int:
    ratio = physics_freq_hz / control_freq_hz
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, rtol=0.0, atol=1e-8):
        raise ValueError(
            "--physics-freq-hz must be an integer multiple of --control-freq-hz "
            f"(got {physics_freq_hz:g} Hz and {control_freq_hz:g} Hz)."
        )
    return substeps


def configure_dm_control_timing(
    env: DmControlGymEnv,
    *,
    physics_timestep: float,
    time_limit: float,
) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def create_resizable_dataset(
    h5: h5py.File,
    name: str,
    shape_tail: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    compression: str | None = None,
    chunks: tuple[int, ...] | bool | None = True,
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=(0, *shape_tail),
        maxshape=(None, *shape_tail),
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )


def append_rows(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = int(dataset.shape[0])
    end = start + int(values.shape[0])
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values
    return start, end


def valid_training_windows(ep_len: np.ndarray, *, history_size: int = 3, num_preds: int = 1, frameskip: int = 1) -> int:
    num_steps = history_size + num_preds
    required_last_frame_offset = (num_steps - 1) * frameskip
    required_action_end_offset = history_size * frameskip
    required_offset = max(required_last_frame_offset, required_action_end_offset)
    return int(np.maximum(ep_len - 1 - required_offset + 1, 0).sum())


def should_continue(args: argparse.Namespace, num_trajectories: int, total_transitions: int) -> bool:
    if args.target_transitions is not None:
        return total_transitions < args.target_transitions
    return num_trajectories < args.num_trajectories


def collect_random_trajectory(
    *,
    env: DmControlGymEnv,
    trajectory_seed: int,
    width: int,
    height: int,
    trajectory_length: int,
    action_noise_scale: float,
    physics_timestep: float,
    time_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    rng = np.random.default_rng(trajectory_seed)
    obs, _ = env.reset(seed=trajectory_seed)
    configure_dm_control_timing(
        env,
        physics_timestep=physics_timestep,
        time_limit=time_limit,
    )
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)

    states = [np.asarray(obs, dtype=np.float32)]
    actions: list[np.ndarray] = []
    frames = [env._env.physics.render(height=height, width=width, camera_id=0)]

    total_reward = 0.0
    terminated = False

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    scaled_action_low = action_low * float(action_noise_scale)
    scaled_action_high = action_high * float(action_noise_scale)
    for _ in range(trajectory_length):
        action = rng.uniform(scaled_action_low, scaled_action_high).astype(np.float32)
        obs, reward, env_terminated, env_truncated, _ = env.step(action)
        total_reward += float(reward)

        actions.append(action.copy())
        states.append(np.asarray(obs, dtype=np.float32))
        frames.append(env._env.physics.render(height=height, width=width, camera_id=0))

        if env_terminated or env_truncated:
            terminated = bool(env_terminated)
            break

    return (
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        np.stack(frames, axis=0),
        total_reward,
        terminated,
    )


def main() -> None:
    args = parse_args()
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.trajectory_length < 1:
        raise ValueError("--trajectory-length must be positive.")
    if args.physics_freq_hz <= 0.0:
        raise ValueError("--physics-freq-hz must be positive.")
    if args.control_freq_hz <= 0.0:
        raise ValueError("--control-freq-hz must be positive.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.action_noise_scale < 0.0:
        raise ValueError("--action-noise-scale must be non-negative.")
    if args.action_cost_weight < 0.0:
        raise ValueError("--action-cost-weight must be non-negative.")
    if args.action_rate_cost_weight < 0.0:
        raise ValueError("--action-rate-cost-weight must be non-negative.")
    if args.velocity_cost_weight < 0.0:
        raise ValueError("--velocity-cost-weight must be non-negative.")

    physics_timestep = 1.0 / args.physics_freq_hz
    control_timestep = 1.0 / args.control_freq_hz
    control_substeps = compute_control_substeps(args.physics_freq_hz, args.control_freq_hz)
    video_fps = args.physics_freq_hz

    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    env = DmControlGymEnv(
        domain_name="reacher",
        task_name=args.task,
        seed=args.seed,
        time_limit=args.time_limit,
        action_cost_weight=args.action_cost_weight,
        action_rate_cost_weight=args.action_rate_cost_weight,
        velocity_cost_weight=args.velocity_cost_weight,
    )
    env.reset(seed=args.seed)

    compression = None if args.compression == "none" else args.compression
    if output_path.exists():
        output_path.unlink()

    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["source"] = "data/reacher_data_gen_random.py"
        h5.attrs["state_keys"] = json.dumps(["position(2)", "to_target(2)", "velocity(2)"])
        h5.attrs["task"] = args.task
        h5.attrs["seed"] = args.seed
        h5.attrs["video_dir"] = str(video_dir)
        h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
        h5.attrs["video_fps"] = video_fps
        h5.attrs["physics_freq_hz"] = args.physics_freq_hz
        h5.attrs["control_freq_hz"] = args.control_freq_hz
        h5.attrs["physics_timestep"] = physics_timestep
        h5.attrs["control_timestep"] = control_timestep
        h5.attrs["control_substeps"] = control_substeps
        h5.attrs["frames_per_physics_step"] = 1
        h5.attrs["time_limit"] = args.time_limit
        h5.attrs["trajectory_length"] = args.trajectory_length
        h5.attrs["action_noise_scale"] = args.action_noise_scale
        h5.attrs["rollout_modes"] = json.dumps(list(ROLLOUT_MODES))
        h5.attrs["rollout_mode_probabilities"] = json.dumps([1.0])

        ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
        ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
        reward_ds = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
        seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
        terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
        rollout_mode_ds = create_resizable_dataset(h5, "rollout_mode", (), np.int64, chunks=True)
        pixels_ds = create_resizable_dataset(
            h5,
            "pixels",
            (args.height, args.width, 3),
            np.uint8,
            compression=compression,
            chunks=(1, args.height, args.width, 3),
        )
        action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
        obs_ds = create_resizable_dataset(h5, "observation", (STATE_DIM,), np.float32, chunks=True)
        qpos_ds = create_resizable_dataset(h5, "qpos", (2,), np.float32, chunks=True)
        qvel_ds = create_resizable_dataset(h5, "qvel", (2,), np.float32, chunks=True)
        episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
        step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

        progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
        progress_desc = "Collecting transitions" if args.target_transitions is not None else "Collecting trajectories"
        progress_unit = "step" if args.target_transitions is not None else "traj"
        with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
            while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                trajectory_seed = args.seed + seed_offset
                seed_offset += 1
                states, actions, frames, total_reward, terminated = collect_random_trajectory(
                    env=env,
                    trajectory_seed=trajectory_seed,
                    width=args.width,
                    height=args.height,
                    trajectory_length=args.trajectory_length,
                    action_noise_scale=args.action_noise_scale,
                    physics_timestep=physics_timestep,
                    time_limit=args.time_limit,
                )
                if actions.shape[0] < args.min_steps:
                    skipped_short += 1
                    continue

                episode_idx = len(step_counts)
                video_path = video_dir / f"trajectory_{episode_idx:07d}.mp4"
                imageio.mimwrite(
                    video_path,
                    frames,
                    fps=video_fps,
                    quality=args.quality,
                    macro_block_size=1,
                )

                padded_actions = np.empty((states.shape[0], ACTION_DIM), dtype=np.float32)
                padded_actions[:-1] = actions
                padded_actions[-1] = np.nan

                offset, _ = append_rows(pixels_ds, frames)
                append_rows(action_ds, padded_actions)
                append_rows(obs_ds, states)
                append_rows(qpos_ds, states[:, :2])
                append_rows(qvel_ds, states[:, 4:6])
                append_rows(episode_idx_ds, np.full((states.shape[0],), episode_idx, dtype=np.int64))
                append_rows(step_idx_ds, np.arange(states.shape[0], dtype=np.int64))
                append_rows(rollout_mode_ds, np.zeros((states.shape[0],), dtype=np.int64))
                append_rows(ep_len_ds, np.asarray([states.shape[0]], dtype=np.int64))
                append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                append_rows(reward_ds, np.asarray([total_reward], dtype=np.float32))
                append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))

                rewards.append(total_reward)
                step_counts.append(int(actions.shape[0]))
                terminated_flags.append(terminated)
                progress.update(actions.shape[0] if args.target_transitions is not None else 1)
                progress.set_postfix(
                    episodes=len(step_counts),
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                    skipped=skipped_short,
                )

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        total_transitions = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = total_transitions
        h5.attrs["skipped_short_episodes"] = skipped_short
        h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
        h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
        h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)

    env.close()

    summary = {
        "output_path": str(output_path),
        "video_dir": str(video_dir),
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "total_frames": int(np.sum(step_counts, dtype=np.int64) + len(step_counts)),
        "min_episode_steps": int(np.min(step_counts)) if step_counts else 0,
        "mean_episode_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "max_episode_steps": int(np.max(step_counts)) if step_counts else 0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "terminated_episodes": int(np.sum(terminated_flags, dtype=np.int64)),
        "skipped_short_episodes": skipped_short,
        "random_episodes": len(step_counts),
        "action_noise_scale": float(args.action_noise_scale),
        "usable_train_windows_default": valid_training_windows(np.asarray(step_counts, dtype=np.int64) + 1)
        if step_counts
        else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
