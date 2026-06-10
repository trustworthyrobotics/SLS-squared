#!/usr/bin/env python3
"""Collect expert DM Control Reacher trajectories directly into LE-WM HDF5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# os.environ.setdefault("MUJOCO_GL", "egl")
# If macOS, use glfw, otherwise default to egl (Linux/headless)
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC
from tqdm.auto import tqdm

from reacher.eval.reacher_policy_viz import (
    configure_offscreen_framebuffer,
    get_render_env,
    make_eval_env,
    require_device,
)

DEFAULT_MODEL_PATH = "reacher/models/reacher-dm-control-sac/best_model/best_model.zip"
DEFAULT_VECNORMALIZE_PATH = "reacher/models/reacher-dm-control-sac/vecnormalize.pkl"
DEFAULT_OUTDIR = "reacher/data/crazy_noise_rollouts"
DEFAULT_OUTPUT_NAME = "reacher_train.h5"
ROLLOUT_MODES = ("expert", "expert_plus_noise", "obtuse")
DEFAULT_ROLLOUT_RATIOS = (0.0, 1.0, 0.0)
EXPERT_NOISE_STD = 2.0
JOINT2_OBTUSE_MARGIN_RAD = 0.35

# Local timing/video knobs.
PHYSICS_FREQ_HZ = 50.0
CONTROL_FREQ_HZ = 50.0
STATE_DIM = 6
ACTION_DIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize-path", type=Path, default=DEFAULT_VECNORMALIZE_PATH)
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
        default=1_00,
        help="Fallback collection target when --target-transitions is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=PHYSICS_FREQ_HZ)
    parser.add_argument("--control-freq-hz", type=float, default=CONTROL_FREQ_HZ)
    parser.add_argument("--max-steps-per-episode", type=int, default=100)
    parser.add_argument("--goal-threshold", type=float, default=0.03)
    parser.add_argument("--max-blank-steps", type=int, default=10)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument(
        "--expert-ratio",
        type=float,
        default=DEFAULT_ROLLOUT_RATIOS[0],
        help="Relative amount of clean expert trajectories to collect.",
    )
    parser.add_argument(
        "--expert-plus-noise-ratio",
        type=float,
        default=DEFAULT_ROLLOUT_RATIOS[1],
        help="Relative amount of noised expert trajectories to collect.",
    )
    parser.add_argument(
        "--obtuse-ratio",
        type=float,
        default=DEFAULT_ROLLOUT_RATIOS[2],
        help="Relative amount of trajectories that must flip joint 2 across the acute/obtuse boundary.",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy output by default. Pass --no-deterministic for stochastic rollouts.",
    )
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return parser.parse_args()


def get_original_obs(env: object) -> np.ndarray:
    return np.asarray(env.get_original_obs()[0], dtype=np.float32)


def hide_target(render_env: object) -> None:
    target_geom_id = render_env._env.physics.model.name2id("target", "geom")
    render_env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def reached_goal(state: np.ndarray, threshold: float) -> bool:
    return bool(np.linalg.norm(state[2:4]) <= threshold)


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
    render_env: object,
    *,
    physics_timestep: float,
    time_limit: float,
) -> None:
    dm_env = render_env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def clip_action_to_space(action: np.ndarray, env: object) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    action = np.asarray(action, dtype=np.float32)
    if high.shape != action.shape[1:] or low.shape != action.shape[1:]:
        return action
    return np.clip(action, low, high).astype(np.float32)


def rollout_probabilities(args: argparse.Namespace) -> np.ndarray:
    ratios = np.asarray(
        [args.expert_ratio, args.expert_plus_noise_ratio, args.obtuse_ratio],
        dtype=np.float64,
    )
    if ratios.shape != (len(ROLLOUT_MODES),):
        raise ValueError(f"Expected {len(ROLLOUT_MODES)} rollout ratios.")
    if np.any(ratios < 0.0):
        raise ValueError("Rollout ratios cannot be negative.")
    total = float(ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one rollout ratio must be positive.")
    return ratios / total


def sample_rollout_mode(args: argparse.Namespace, rng: np.random.Generator) -> str:
    probabilities = rollout_probabilities(args)
    return str(rng.choice(ROLLOUT_MODES, p=probabilities))


def is_obtuse_rollout(states: np.ndarray, margin_rad: float = JOINT2_OBTUSE_MARGIN_RAD) -> bool:
    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError(f"Expected states with shape (T, >=2), got {states.shape}.")
    joint2 = np.asarray(states[:, 1], dtype=np.float32)
    return bool(np.min(joint2) <= -margin_rad and np.max(joint2) >= margin_rad)


def collect_trajectory(
    *,
    model: SAC,
    env: object,
    render_env: object,
    trajectory_seed: int,
    deterministic: bool,
    width: int,
    height: int,
    max_steps: int,
    physics_timestep: float,
    control_substeps: int,
    time_limit: float,
    goal_threshold: float,
    max_blank_steps: int,
    rollout_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    env.seed(trajectory_seed)
    obs = env.reset()
    rng = np.random.default_rng(trajectory_seed)
    configure_dm_control_timing(
        render_env,
        physics_timestep=physics_timestep,
        time_limit=time_limit,
    )
    hide_target(render_env)
    configure_offscreen_framebuffer(render_env, width, height)

    states = [get_original_obs(env)]
    actions: list[np.ndarray] = []
    frames = [render_env._env.physics.render(height=height, width=width, camera_id=0)]

    total_reward = 0.0
    goal_seen = reached_goal(states[-1], goal_threshold)
    post_goal_steps = 0
    terminated = False

    action: np.ndarray | None = None
    for physics_step in range(max_steps):
        was_already_at_goal = goal_seen
        if physics_step % control_substeps == 0:
            action, _ = model.predict(obs, deterministic=deterministic)
            if rollout_mode == "expert_plus_noise":
                noise = rng.normal(loc=0.0, scale=EXPERT_NOISE_STD, size=action.shape).astype(np.float32)
                action = clip_action_to_space(action + noise, env)
        if action is None:
            raise RuntimeError("Policy action was not initialized.")
        obs, rewards, dones, _ = env.step(action)
        total_reward += float(rewards[0])

        actions.append(np.asarray(action[0], dtype=np.float32))
        state = get_original_obs(env)
        states.append(state)
        frames.append(render_env._env.physics.render(height=height, width=width, camera_id=0))

        if was_already_at_goal:
            post_goal_steps += 1
        elif reached_goal(state, goal_threshold):
            goal_seen = True

        if bool(dones[0]):
            terminated = True
            break
        if goal_seen and post_goal_steps >= max_blank_steps:
            break

    return (
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        np.stack(frames, axis=0),
        total_reward,
        terminated,
    )


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


def main() -> None:
    args = parse_args()
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.max_steps_per_episode < 1:
        raise ValueError("--max-steps-per-episode must be positive.")
    if args.physics_freq_hz <= 0.0:
        raise ValueError("--physics-freq-hz must be positive.")
    if args.control_freq_hz <= 0.0:
        raise ValueError("--control-freq-hz must be positive.")
    if args.max_blank_steps < 0:
        raise ValueError("--max-blank-steps cannot be negative.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    rollout_mode_probs = rollout_probabilities(args)

    device = require_device(args.device)
    physics_timestep = 1.0 / args.physics_freq_hz
    control_timestep = 1.0 / args.control_freq_hz
    control_substeps = compute_control_substeps(args.physics_freq_hz, args.control_freq_hz)
    video_fps = args.physics_freq_hz

    model_path = args.model_path.expanduser().resolve()
    vecnormalize_path = args.vecnormalize_path.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    env = make_eval_env(
        task=args.task,
        seed=args.seed,
        time_limit=args.time_limit,
        vecnormalize_path=vecnormalize_path,
    )
    render_env = get_render_env(env)
    model = SAC.load(str(model_path), env=env, device=device)

    compression = None if args.compression == "none" else args.compression
    if output_path.exists():
        output_path.unlink()

    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    skipped_short = 0
    skipped_obtuse_mismatch = 0
    seed_offset = 0
    rollout_mode_rng = np.random.default_rng(args.seed)
    rollout_mode_by_episode: list[str] = []

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["source"] = "data/reacher_data_gen.py"
        h5.attrs["state_keys"] = json.dumps(["position(2)", "to_target(2)", "velocity(2)"])
        h5.attrs["task"] = args.task
        h5.attrs["deterministic"] = args.deterministic
        h5.attrs["seed"] = args.seed
        h5.attrs["model_path"] = str(model_path)
        h5.attrs["vecnormalize_path"] = str(vecnormalize_path)
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
        h5.attrs["goal_threshold"] = args.goal_threshold
        h5.attrs["max_blank_steps"] = args.max_blank_steps
        h5.attrs["max_steps_per_episode"] = args.max_steps_per_episode
        h5.attrs["rollout_modes"] = json.dumps(list(ROLLOUT_MODES))
        h5.attrs["rollout_mode_probabilities"] = json.dumps(rollout_mode_probs.tolist())
        h5.attrs["expert_noise_std"] = EXPERT_NOISE_STD
        h5.attrs["joint2_obtuse_margin_rad"] = JOINT2_OBTUSE_MARGIN_RAD

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
                rollout_mode = sample_rollout_mode(args, rollout_mode_rng)
                states, actions, frames, total_reward, terminated = collect_trajectory(
                    model=model,
                    env=env,
                    render_env=render_env,
                    trajectory_seed=trajectory_seed,
                    deterministic=args.deterministic,
                    width=args.width,
                    height=args.height,
                    max_steps=args.max_steps_per_episode,
                    physics_timestep=physics_timestep,
                    control_substeps=control_substeps,
                    time_limit=args.time_limit,
                    goal_threshold=args.goal_threshold,
                    max_blank_steps=args.max_blank_steps,
                    rollout_mode=rollout_mode,
                )
                if rollout_mode == "obtuse" and not is_obtuse_rollout(states):
                    skipped_obtuse_mismatch += 1
                    continue
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
                append_rows(
                    rollout_mode_ds,
                    np.full((states.shape[0],), ROLLOUT_MODES.index(rollout_mode), dtype=np.int64),
                )
                append_rows(ep_len_ds, np.asarray([states.shape[0]], dtype=np.int64))
                append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                append_rows(reward_ds, np.asarray([total_reward], dtype=np.float32))
                append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))

                rewards.append(total_reward)
                step_counts.append(int(actions.shape[0]))
                terminated_flags.append(terminated)
                rollout_mode_by_episode.append(rollout_mode)
                progress.update(actions.shape[0] if args.target_transitions is not None else 1)
                progress.set_postfix(
                    episodes=len(step_counts),
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                    skipped=skipped_short,
                    obtuse_skipped=skipped_obtuse_mismatch,
                )

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        total_transitions = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = total_transitions
        h5.attrs["skipped_short_episodes"] = skipped_short
        h5.attrs["skipped_obtuse_mismatch_episodes"] = skipped_obtuse_mismatch
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
        "skipped_obtuse_mismatch_episodes": skipped_obtuse_mismatch,
        "expert_episodes": int(sum(mode == "expert" for mode in rollout_mode_by_episode)),
        "expert_plus_noise_episodes": int(sum(mode == "expert_plus_noise" for mode in rollout_mode_by_episode)),
        "obtuse_episodes": int(sum(mode == "obtuse" for mode in rollout_mode_by_episode)),
        "expert_noise_std": EXPERT_NOISE_STD,
        "joint2_obtuse_margin_rad": JOINT2_OBTUSE_MARGIN_RAD,
        "usable_train_windows_default": valid_training_windows(np.asarray(step_counts, dtype=np.int64) + 1)
        if step_counts
        else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
