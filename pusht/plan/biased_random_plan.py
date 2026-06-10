#!/usr/bin/env python3
"""Visualize PushT rollouts from a T-biased random action planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_pusht_env,
)
from pusht.shared.utils import render_frame

DEFAULT_OUT_DIR = Path("pusht/plan/biased_random_plan")
DEFAULT_MAX_STEPS = 50
ENV_ACTION_SCALE = 100.0
TEE_SCALE = 30.0
TEE_LENGTH = 4.0
TEE_BAR_X_MIN = -TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_X_MAX = TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_Y_MIN = 0.0
TEE_BAR_Y_MAX = TEE_SCALE
TEE_STEM_X_MIN = -TEE_SCALE / 2.0
TEE_STEM_X_MAX = TEE_SCALE / 2.0
TEE_STEM_Y_MIN = TEE_SCALE
TEE_STEM_Y_MAX = TEE_LENGTH * TEE_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--video-name", default="biased_random_plan.mp4")
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Sample a new biased random action every N env steps and hold it in between.",
    )
    parser.add_argument(
        "--direction-kappa",
        type=float,
        default=1.5,
        help="Von Mises concentration around the direction to the T. Higher means more strongly biased.",
    )
    parser.add_argument(
        "--magnitude-min",
        type=float,
        default=15.0,
        help="Minimum sampled target displacement magnitude in pixels.",
    )
    parser.add_argument(
        "--magnitude-max",
        type=float,
        default=35.0,
        help="Maximum sampled target displacement magnitude in pixels.",
    )
    parser.add_argument(
        "--aim-mode",
        choices=("center", "surface"),
        default="surface",
        help="Bias directions toward the block center or toward a random point on the T surface.",
    )
    return parser.parse_args()


def _clip_action_to_space(action: np.ndarray, env: Any) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    if high.shape != action.shape or low.shape != action.shape:
        return np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


def _target_xy_to_env_action(env: Any, agent_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        high = np.asarray(action_space.high)
        low = np.asarray(action_space.low)
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return np.clip((target_xy - agent_xy) / ENV_ACTION_SCALE, low, high).astype(np.float32)
    return target_xy.astype(np.float32)


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _sample_point_on_t(block_pose: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | str]]:
    bar_area = (TEE_BAR_X_MAX - TEE_BAR_X_MIN) * (TEE_BAR_Y_MAX - TEE_BAR_Y_MIN)
    stem_area = (TEE_STEM_X_MAX - TEE_STEM_X_MIN) * (TEE_STEM_Y_MAX - TEE_STEM_Y_MIN)
    if float(rng.uniform()) < bar_area / (bar_area + stem_area):
        local_xy = np.asarray(
            [
                rng.uniform(TEE_BAR_X_MIN, TEE_BAR_X_MAX),
                rng.uniform(TEE_BAR_Y_MIN, TEE_BAR_Y_MAX),
            ],
            dtype=np.float32,
        )
        region = "bar"
    else:
        local_xy = np.asarray(
            [
                rng.uniform(TEE_STEM_X_MIN, TEE_STEM_X_MAX),
                rng.uniform(TEE_STEM_Y_MIN, TEE_STEM_Y_MAX),
            ],
            dtype=np.float32,
        )
        region = "stem"
    world_xy = (_rotation_matrix(float(block_pose[2])) @ local_xy) + block_pose[:2]
    return world_xy.astype(np.float32), {
        "region": region,
        "local_x": float(local_xy[0]),
        "local_y": float(local_xy[1]),
    }


def _sample_biased_action(
    env: Any,
    rng: np.random.Generator,
    *,
    direction_kappa: float,
    magnitude_min: float,
    magnitude_max: float,
    aim_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    agent_xy = get_pusht_agent_pos(env)
    block_pose = get_pusht_block_pose(env)
    if aim_mode == "center":
        aim_xy = block_pose[:2].astype(np.float32)
        aim_meta: dict[str, Any] = {"region": "center"}
    else:
        aim_xy, aim_meta = _sample_point_on_t(block_pose, rng)

    base_delta = aim_xy - agent_xy
    base_angle = float(np.arctan2(base_delta[1], base_delta[0]))
    sampled_angle = float(rng.vonmises(base_angle, direction_kappa))
    magnitude = float(rng.uniform(magnitude_min, magnitude_max))
    direction = np.asarray([np.cos(sampled_angle), np.sin(sampled_angle)], dtype=np.float32)
    sampled_delta = direction * magnitude
    target_xy = (agent_xy + sampled_delta).astype(np.float32)
    env_action = _clip_action_to_space(_target_xy_to_env_action(env, agent_xy, target_xy), env)
    return env_action, {
        "agent_xy": agent_xy.tolist(),
        "block_pose": block_pose.tolist(),
        "aim_xy": aim_xy.tolist(),
        "aim_mode": aim_mode,
        "aim_meta": aim_meta,
        "base_angle_rad": base_angle,
        "sampled_angle_rad": sampled_angle,
        "angle_error_rad": float(np.arctan2(np.sin(sampled_angle - base_angle), np.cos(sampled_angle - base_angle))),
        "magnitude": magnitude,
        "sampled_delta": sampled_delta.tolist(),
        "target_xy": target_xy.tolist(),
        "direction_kappa": float(direction_kappa),
    }


def _get_video_fps(env: Any, control_interval: int) -> int:
    metadata = getattr(env, "metadata", None) or getattr(getattr(env, "unwrapped", None), "metadata", None) or {}
    base_fps = metadata.get("render_fps")
    if base_fps is None:
        dt = getattr(getattr(env, "unwrapped", None), "dt", None)
        if dt is not None and dt > 1e-6:
            base_fps = float(round(1.0 / dt))
    if base_fps is None:
        base_fps = 10.0
    return max(1, int(round(float(base_fps) / float(control_interval))))


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).item())
    if "success" in info:
        return bool(np.asarray(info["success"]).item())
    return bool(terminated or reward >= 0.95)


def rollout_episode(args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1.")
    if args.direction_kappa < 0.0:
        raise ValueError("--direction-kappa must be >= 0.")
    if args.magnitude_min < 0.0:
        raise ValueError("--magnitude-min must be >= 0.")
    if args.magnitude_max < args.magnitude_min:
        raise ValueError("--magnitude-max must be >= --magnitude-min.")

    episode_seed = None if args.seed is None else args.seed + episode_idx
    rng = np.random.default_rng(episode_seed)
    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    try:
        _, _ = env.reset(seed=episode_seed)

        frames = [render_frame(env)]
        rewards: list[float] = []
        contact_counts: list[int] = []
        action_history: list[dict[str, Any]] = []
        action = None
        success = False
        terminated = False
        truncated = False
        steps_taken = 0
        control_updates = 0

        for step_idx in range(args.max_steps):
            if action is None or step_idx % args.control_interval == 0:
                action, action_meta = _sample_biased_action(
                    env,
                    rng,
                    direction_kappa=args.direction_kappa,
                    magnitude_min=args.magnitude_min,
                    magnitude_max=args.magnitude_max,
                    aim_mode=args.aim_mode,
                )
                action_history.append(action_meta)
                control_updates += 1

            _, reward, terminated, truncated, info = env.step(action)
            success = success or _extract_success(terminated, float(reward), info)
            steps_taken = step_idx + 1

            if (step_idx + 1) % args.control_interval == 0 or terminated or truncated:
                frames.append(render_frame(env))
                rewards.append(float(reward))
                contact_counts.append(int(np.asarray(info.get("n_contacts", 0)).item()))

            if terminated or truncated:
                break

        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.out_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            video_path,
            frames,
            fps=args.fps,
            quality=8,
            macro_block_size=1,
        )

        final_block_pose = get_pusht_block_pose(env)
        final_agent_xy = get_pusht_agent_pos(env)
        goal_pose = np.asarray(getattr(env.unwrapped, "goal_pose", None), dtype=np.float32).tolist()
        return {
            "episode": episode_idx,
            "seed": episode_seed,
            "env_steps": steps_taken,
            "stored_steps": len(frames),
            "control_updates": control_updates,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "direction_kappa": float(args.direction_kappa),
            "magnitude_min": float(args.magnitude_min),
            "magnitude_max": float(args.magnitude_max),
            "aim_mode": args.aim_mode,
            "goal_pose": goal_pose,
            "final_agent_xy": final_agent_xy.tolist(),
            "final_block_pose": final_block_pose.tolist(),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_contacts": float(np.mean(contact_counts)) if contact_counts else 0.0,
            "actions": action_history,
            "video_path": str(video_path),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [rollout_episode(args, episode_idx) for episode_idx in range(args.episodes)]
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for result in results:
        print(
            f"episode={result['episode']} env_steps={result['env_steps']} "
            f"stored_steps={result['stored_steps']} control_updates={result['control_updates']} "
            f"success={result['success']} mean_reward={result['mean_reward']:.3f} "
            f"mean_contacts={result['mean_contacts']:.3f} video={result['video_path']}"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
