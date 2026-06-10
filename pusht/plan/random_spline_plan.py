#!/usr/bin/env python3
"""Visualize random spline-based PushT rollouts that intentionally pass through the block."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_pusht_env,
    set_pusht_state,
)
from pusht.shared.utils import render_frame

DEFAULT_OUT_DIR = Path("pusht/plan/random_spline_plan")
ARENA_MIN = 32.0
ARENA_MAX = 480.0
ARENA_CENTER = np.asarray([(ARENA_MIN + ARENA_MAX) / 2.0, (ARENA_MIN + ARENA_MAX) / 2.0], dtype=np.float32)
ENV_ACTION_SCALE = 100.0
DEFAULT_MAX_STEPS = 500
DEFAULT_SPLINE_SAMPLING_ATTEMPTS = 500
DEFAULT_NUM_CHAINED_SPLINES = 10
DEFAULT_BLOCK_CENTER_JITTER = 48.0
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
    parser.add_argument("--video-name", default="random_spline_plan.mp4")
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Update the spline-tracking controller every N env steps and hold the last action in between.",
    )
    parser.add_argument("--num-spline-points", type=int, default=80)
    parser.add_argument("--circle-min-radius", type=float, default=65.0)
    parser.add_argument("--circle-max-radius", type=float, default=95.0)
    parser.add_argument("--min-angle-separation-deg", type=float, default=20.0)
    parser.add_argument("--angle-jitter-deg", type=float, default=160.0)
    parser.add_argument(
        "--spline-sampling-attempts",
        type=int,
        default=DEFAULT_SPLINE_SAMPLING_ATTEMPTS,
        help="How many times to resample contact/circle geometry before giving up.",
    )
    parser.add_argument(
        "--num-chained-splines",
        type=int,
        default=DEFAULT_NUM_CHAINED_SPLINES,
        help="How many spline segments to chain together before the rollout stops.",
    )
    parser.add_argument("--contact-tol", type=float, default=14.0)
    parser.add_argument("--waypoint-tol", type=float, default=18.0)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--kd", type=float, default=0.18)
    parser.add_argument("--max-action-delta", type=float, default=50.0)
    return parser.parse_args()


def _clip_xy(xy: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(xy, dtype=np.float32), ARENA_MIN, ARENA_MAX)


def _sample_xy(rng: np.random.Generator, *, margin: float = 48.0) -> np.ndarray:
    return rng.uniform(ARENA_MIN + margin, ARENA_MAX - margin, size=(2,)).astype(np.float32)


def _sample_centered_xy(rng: np.random.Generator, *, jitter: float = DEFAULT_BLOCK_CENTER_JITTER) -> np.ndarray:
    low = np.maximum(ARENA_CENTER - jitter, ARENA_MIN)
    high = np.minimum(ARENA_CENTER + jitter, ARENA_MAX)
    return rng.uniform(low, high).astype(np.float32)


def _sample_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(1024):
        agent_xy = _sample_xy(rng, margin=56.0)
        block_xy = _sample_centered_xy(rng)
        if np.linalg.norm(agent_xy - block_xy) >= 90.0:
            theta = float(rng.uniform(-np.pi, np.pi))
            state = np.asarray(
                [agent_xy[0], agent_xy[1], block_xy[0], block_xy[1], theta, 0.0, 0.0],
                dtype=np.float64,
            )
            return state, np.asarray([block_xy[0], block_xy[1], theta], dtype=np.float32)
    raise RuntimeError("Failed to sample a valid random PushT state.")


def _make_catmull_rom_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    *,
    num_points: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)[:, None]
    t2 = t * t
    t3 = t2 * t
    curve = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return np.asarray(curve, dtype=np.float32)


def _resample_polyline(points: np.ndarray, *, num_points: int) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(np.asarray(points, dtype=np.float32), num_points, axis=0)

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float32)))
    total_length = float(cumulative[-1])
    if total_length < 1e-6:
        return np.repeat(np.asarray(points[:1], dtype=np.float32), num_points, axis=0)

    targets = np.linspace(0.0, total_length, num_points, dtype=np.float32)
    resampled = np.empty((num_points, 2), dtype=np.float32)
    seg_idx = 0
    for target_idx, target in enumerate(targets):
        while seg_idx < len(lengths) - 1 and target > cumulative[seg_idx + 1]:
            seg_idx += 1
        seg_length = float(lengths[seg_idx])
        if seg_length < 1e-6:
            resampled[target_idx] = points[seg_idx + 1]
            continue
        alpha = float((target - cumulative[seg_idx]) / seg_length)
        resampled[target_idx] = (1.0 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1]
    return resampled


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


def _in_bounds(xy: np.ndarray) -> bool:
    return bool(np.all(xy >= ARENA_MIN) and np.all(xy <= ARENA_MAX))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _sample_circle_points(
    center_xy: np.ndarray,
    push_dir: np.ndarray,
    rng: np.random.Generator,
    *,
    min_radius: float,
    max_radius: float,
    min_angle_separation_deg: float,
    angle_jitter_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if min_radius <= 0.0:
        raise ValueError("--circle-min-radius must be > 0.")
    if max_radius < min_radius:
        raise ValueError("--circle-max-radius must be >= --circle-min-radius.")

    min_angle_sep = np.deg2rad(min_angle_separation_deg)
    jitter = np.deg2rad(angle_jitter_deg)
    push_angle = float(np.arctan2(push_dir[1], push_dir[0]))

    for _ in range(1024):
        radius = float(rng.uniform(min_radius, max_radius))
        entry_angle = push_angle + np.pi + float(rng.uniform(-jitter, jitter))
        exit_angle = push_angle + float(rng.uniform(-jitter, jitter))
        angle_gap = abs(_wrap_angle(exit_angle - entry_angle))
        if angle_gap < min_angle_sep:
            continue

        entry_xy = center_xy + radius * np.asarray([np.cos(entry_angle), np.sin(entry_angle)], dtype=np.float32)
        exit_xy = center_xy + radius * np.asarray([np.cos(exit_angle), np.sin(exit_angle)], dtype=np.float32)
        if _in_bounds(entry_xy) and _in_bounds(exit_xy):
            return entry_xy.astype(np.float32), exit_xy.astype(np.float32), {
                "radius": radius,
                "entry_angle_rad": float(entry_angle),
                "exit_angle_rad": float(exit_angle),
                "angle_gap_rad": float(angle_gap),
            }
    raise RuntimeError("Failed to sample in-bounds circle points around the chosen T contact point.")


def _make_catmull_rom_spline(points: np.ndarray, *, num_points: int) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("Need at least two points for a spline.")
    if len(points) == 2:
        return np.linspace(points[0], points[1], num_points, dtype=np.float32)

    segments: list[np.ndarray] = []
    points_per_segment = max(8, num_points // (len(points) - 1) + 2)
    for idx in range(len(points) - 1):
        p0 = points[idx - 1] if idx > 0 else points[idx]
        p1 = points[idx]
        p2 = points[idx + 1]
        p3 = points[idx + 2] if idx + 2 < len(points) else points[idx + 1]
        segment = _make_catmull_rom_segment(p0, p1, p2, p3, num_points=points_per_segment)
        segments.append(segment[:-1] if idx < len(points) - 2 else segment)
    dense_curve = np.concatenate(segments, axis=0)
    return _resample_polyline(dense_curve, num_points=num_points)


def _make_spline(
    agent_xy: np.ndarray,
    block_pose: np.ndarray,
    goal_pose: np.ndarray,
    rng: np.random.Generator,
    *,
    circle_min_radius: float,
    circle_max_radius: float,
    min_angle_separation_deg: float,
    angle_jitter_deg: float,
    num_points: int,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    push_dir = goal_pose[:2] - block_pose[:2]
    push_norm = float(np.linalg.norm(push_dir))
    if push_norm < 1e-6:
        push_dir = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        push_dir = (push_dir / push_norm).astype(np.float32)

    contact_xy, contact_meta = _sample_point_on_t(block_pose, rng)
    entry_xy, exit_xy, circle_meta = _sample_circle_points(
        contact_xy,
        push_dir,
        rng,
        min_radius=circle_min_radius,
        max_radius=circle_max_radius,
        min_angle_separation_deg=min_angle_separation_deg,
        angle_jitter_deg=angle_jitter_deg,
    )
    waypoints = np.asarray(
        [
            agent_xy,
            entry_xy,
            contact_xy,
            exit_xy,
        ],
        dtype=np.float32,
    )
    curve = _clip_xy(_make_catmull_rom_spline(waypoints, num_points=num_points))
    meta = {
        "entry_xy": entry_xy.tolist(),
        "contact_xy": contact_xy.tolist(),
        "exit_xy": exit_xy.tolist(),
        "goal_xy": goal_pose[:2].tolist(),
        "contact_region": str(contact_meta["region"]),
        "contact_local_xy": [contact_meta["local_x"], contact_meta["local_y"]],
        "circle_radius": circle_meta["radius"],
        "entry_angle_rad": circle_meta["entry_angle_rad"],
        "exit_angle_rad": circle_meta["exit_angle_rad"],
        "angle_gap_rad": circle_meta["angle_gap_rad"],
    }
    return curve, meta


def _make_spline_with_retries(
    agent_xy: np.ndarray,
    block_pose: np.ndarray,
    goal_pose: np.ndarray,
    rng: np.random.Generator,
    *,
    circle_min_radius: float,
    circle_max_radius: float,
    min_angle_separation_deg: float,
    angle_jitter_deg: float,
    num_points: int,
    max_attempts: int,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    if max_attempts < 1:
        raise ValueError("--spline-sampling-attempts must be >= 1.")

    last_error: RuntimeError | None = None
    for _ in range(max_attempts):
        try:
            return _make_spline(
                agent_xy,
                block_pose,
                goal_pose,
                rng,
                circle_min_radius=circle_min_radius,
                circle_max_radius=circle_max_radius,
                min_angle_separation_deg=min_angle_separation_deg,
                angle_jitter_deg=angle_jitter_deg,
                num_points=num_points,
            )
        except RuntimeError as exc:
            if "Failed to sample in-bounds circle points" not in str(exc):
                raise
            last_error = exc

    raise RuntimeError(
        f"Failed to sample a valid random spline after {max_attempts} attempts. "
        "Try reducing circle radii, reducing min angle separation, or increasing angle jitter."
    ) from last_error


def _target_xy_to_env_action(env: Any, agent_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        high = np.asarray(action_space.high)
        low = np.asarray(action_space.low)
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return np.clip((target_xy - agent_xy) / ENV_ACTION_SCALE, low, high).astype(np.float32)
    return target_xy.astype(np.float32)


def _target_xy_to_relative_action(
    agent_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> np.ndarray:
    low = np.asarray(action_low if action_low is not None else [-1.0, -1.0], dtype=np.float32)
    high = np.asarray(action_high if action_high is not None else [1.0, 1.0], dtype=np.float32)
    return np.clip((target_xy - agent_xy) / ENV_ACTION_SCALE, low, high).astype(np.float32)


def _pd_target(
    agent_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    prev_error: np.ndarray,
    *,
    kp: float,
    kd: float,
    max_action_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    error = waypoint_xy - agent_xy
    correction = kp * error + kd * (error - prev_error)
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > max_action_delta and correction_norm > 1e-6:
        correction = correction * (max_action_delta / correction_norm)
    return agent_xy + correction, error


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(info["is_success"])
    if "success" in info:
        return bool(info["success"])
    return bool(terminated or reward >= 0.95)


def _observation_agent_xy(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray(observation["agent_pos"], dtype=np.float32).reshape(-1)[:2]


def _observation_block_pose(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray(observation["block_pose"], dtype=np.float32).reshape(-1)[:3]


def _observation_goal_pose(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray(observation["goal_pose"], dtype=np.float32).reshape(-1)[:3]


@dataclass
class RandomSplineController:
    spline_xy: np.ndarray
    spline_meta: dict[str, Any]
    rng: np.random.Generator
    num_points: int
    circle_min_radius: float
    circle_max_radius: float
    min_angle_separation_deg: float
    angle_jitter_deg: float
    spline_sampling_attempts: int
    num_chained_splines: int
    waypoint_tol: float
    kp: float
    kd: float
    max_action_delta: float
    action_low: np.ndarray
    action_high: np.ndarray
    prev_error: np.ndarray
    waypoint_idx: int = 0
    completed_splines: int = 0
    spline_history: list[dict[str, Any]] | None = None

    @classmethod
    def from_observation(
        cls,
        observation: dict[str, Any],
        *,
        rng: np.random.Generator,
        num_points: int,
        circle_min_radius: float,
        circle_max_radius: float,
        min_angle_separation_deg: float,
        angle_jitter_deg: float,
        waypoint_tol: float,
        kp: float,
        kd: float,
        max_action_delta: float,
        spline_sampling_attempts: int = DEFAULT_SPLINE_SAMPLING_ATTEMPTS,
        num_chained_splines: int = DEFAULT_NUM_CHAINED_SPLINES,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> "RandomSplineController":
        if num_chained_splines < 1:
            raise ValueError("--num-chained-splines must be >= 1.")
        spline_xy, spline_meta = _make_spline_with_retries(
            _observation_agent_xy(observation),
            _observation_block_pose(observation),
            _observation_goal_pose(observation),
            rng,
            circle_min_radius=circle_min_radius,
            circle_max_radius=circle_max_radius,
            min_angle_separation_deg=min_angle_separation_deg,
            angle_jitter_deg=angle_jitter_deg,
            num_points=num_points,
            max_attempts=spline_sampling_attempts,
        )
        return cls(
            spline_xy=spline_xy,
            spline_meta=spline_meta,
            rng=rng,
            num_points=int(num_points),
            circle_min_radius=float(circle_min_radius),
            circle_max_radius=float(circle_max_radius),
            min_angle_separation_deg=float(min_angle_separation_deg),
            angle_jitter_deg=float(angle_jitter_deg),
            spline_sampling_attempts=int(spline_sampling_attempts),
            num_chained_splines=int(num_chained_splines),
            waypoint_tol=float(waypoint_tol),
            kp=float(kp),
            kd=float(kd),
            max_action_delta=float(max_action_delta),
            action_low=np.asarray(action_low if action_low is not None else [-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray(action_high if action_high is not None else [1.0, 1.0], dtype=np.float32),
            prev_error=np.zeros(2, dtype=np.float32),
            spline_history=[dict(spline_meta)],
        )

    def _current_spline_reached(self, observation: dict[str, Any]) -> bool:
        agent_xy = _observation_agent_xy(observation)
        return bool(
            self.waypoint_idx == len(self.spline_xy) - 1
            and np.linalg.norm(agent_xy - self.spline_xy[-1]) <= self.waypoint_tol
        )

    def _start_next_spline(self, observation: dict[str, Any]) -> None:
        spline_xy, spline_meta = _make_spline_with_retries(
            _observation_agent_xy(observation),
            _observation_block_pose(observation),
            _observation_goal_pose(observation),
            self.rng,
            circle_min_radius=self.circle_min_radius,
            circle_max_radius=self.circle_max_radius,
            min_angle_separation_deg=self.min_angle_separation_deg,
            angle_jitter_deg=self.angle_jitter_deg,
            num_points=self.num_points,
            max_attempts=self.spline_sampling_attempts,
        )
        self.spline_xy = spline_xy
        self.spline_meta = spline_meta
        self.waypoint_idx = 0
        self.prev_error = np.zeros(2, dtype=np.float32)
        if self.spline_history is None:
            self.spline_history = []
        self.spline_history.append(dict(spline_meta))

    def _advance_chain_if_needed(self, observation: dict[str, Any]) -> bool:
        while self._current_spline_reached(observation):
            self.completed_splines += 1
            if self.completed_splines >= self.num_chained_splines:
                return True
            self._start_next_spline(observation)
        return False

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        self._advance_chain_if_needed(observation)
        agent_xy = _observation_agent_xy(observation)
        while self.waypoint_idx < len(self.spline_xy) - 1 and np.linalg.norm(agent_xy - self.spline_xy[self.waypoint_idx]) <= self.waypoint_tol:
            self.waypoint_idx += 1

        waypoint_xy = self.spline_xy[self.waypoint_idx]
        target_xy, self.prev_error = _pd_target(
            agent_xy,
            waypoint_xy,
            self.prev_error,
            kp=self.kp,
            kd=self.kd,
            max_action_delta=self.max_action_delta,
        )
        return _target_xy_to_relative_action(
            agent_xy,
            _clip_xy(target_xy),
            action_low=self.action_low,
            action_high=self.action_high,
        )

    def reached_goal(self, observation: dict[str, Any]) -> bool:
        return self._advance_chain_if_needed(observation)


def rollout_episode(args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.spline_sampling_attempts < 1:
        raise ValueError("--spline-sampling-attempts must be >= 1.")
    if args.num_chained_splines < 1:
        raise ValueError("--num-chained-splines must be >= 1.")

    episode_seed = None if args.seed is None else args.seed + episode_idx
    rng = np.random.default_rng(episode_seed)
    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    try:
        env.reset(seed=episode_seed)
        state, block_pose = _sample_state(rng)
        set_pusht_state(env.unwrapped, state)
        goal_pose = np.asarray(env.unwrapped.goal_pose, dtype=np.float32).copy()

        frames = [render_frame(env)]
        agent_positions = [get_pusht_agent_pos(env).tolist()]
        block_poses = [get_pusht_block_pose(env).tolist()]
        waypoint_errors: list[float] = []
        rewards: list[float] = []

        controller = RandomSplineController.from_observation(
            {
                "agent_pos": get_pusht_agent_pos(env),
                "block_pose": get_pusht_block_pose(env),
                "goal_pose": goal_pose,
            },
            rng=rng,
            num_points=args.num_spline_points,
            circle_min_radius=args.circle_min_radius,
            circle_max_radius=args.circle_max_radius,
            min_angle_separation_deg=args.min_angle_separation_deg,
            angle_jitter_deg=args.angle_jitter_deg,
            waypoint_tol=args.waypoint_tol,
            kp=args.kp,
            kd=args.kd,
            max_action_delta=args.max_action_delta,
            spline_sampling_attempts=args.spline_sampling_attempts,
            num_chained_splines=args.num_chained_splines,
        )
        contacted_block = False
        success = False
        terminated = False
        truncated = False
        reached_pusher_goal = False
        action = None
        control_updates = 0
        steps_taken = 0

        for step_idx in range(args.max_steps):
            agent_xy = get_pusht_agent_pos(env)
            block_pose_now = get_pusht_block_pose(env)
            if np.linalg.norm(agent_xy - block_pose_now[:2]) <= args.contact_tol:
                contacted_block = True

            if action is None or step_idx % args.control_interval == 0:
                observation = {
                    "agent_pos": agent_xy,
                    "block_pose": block_pose_now,
                    "goal_pose": goal_pose,
                }
                relative_action = controller.select_action(observation)
                action = _target_xy_to_env_action(env, agent_xy, agent_xy + relative_action * ENV_ACTION_SCALE)
                control_updates += 1
            _, reward, terminated, truncated, info = env.step(action)
            steps_taken = step_idx + 1
            success = _extract_success(terminated, float(reward), info)

            if (step_idx + 1) % args.control_interval == 0 or terminated or truncated:
                frames.append(render_frame(env))
                agent_positions.append(get_pusht_agent_pos(env).tolist())
                block_poses.append(get_pusht_block_pose(env).tolist())
                waypoint_errors.append(float(np.linalg.norm(controller.prev_error)))
                rewards.append(float(reward))

            if controller.reached_goal(
                {
                    "agent_pos": get_pusht_agent_pos(env),
                    "block_pose": get_pusht_block_pose(env),
                    "goal_pose": goal_pose,
                }
            ):
                reached_pusher_goal = True
                break

            if terminated or truncated:
                break

        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.out_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            video_path,
            frames,
            fps=max(1, int(args.fps)),
            quality=8,
            macro_block_size=1,
        )

        final_block_pose = get_pusht_block_pose(env)
        final_agent_xy = get_pusht_agent_pos(env)
        block_goal_distance = float(np.linalg.norm(final_block_pose[:2] - goal_pose[:2]))
        return {
            "episode": episode_idx,
            "seed": episode_seed,
            "start_agent_xy": state[:2].tolist(),
            "start_block_pose": block_pose.tolist(),
            "goal_block_pose": goal_pose.tolist(),
            "final_agent_xy": final_agent_xy.tolist(),
            "final_block_pose": final_block_pose.tolist(),
            "contacted_block": contacted_block,
            "reached_pusher_goal": reached_pusher_goal,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "env_steps": steps_taken,
            "stored_steps": len(frames),
            "control_updates": control_updates,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "final_waypoint_error": waypoint_errors[-1] if waypoint_errors else 0.0,
            "block_goal_distance": block_goal_distance,
            "num_chained_splines": args.num_chained_splines,
            "completed_splines": controller.completed_splines,
            "splines": controller.spline_history or [],
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
            f"contacted_block={result['contacted_block']} reached_pusher_goal={result['reached_pusher_goal']} "
            f"success={result['success']} "
            f"block_goal_distance={result['block_goal_distance']:.3f} "
            f"video={result['video_path']}"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
