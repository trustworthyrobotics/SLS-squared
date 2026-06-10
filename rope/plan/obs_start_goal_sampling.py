#!/usr/bin/env python3
"""Generate obstacle-crossing start/goal pairs with low rope endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.plan.obs_data_collect import (
    DEFAULT_CAMERA,
    DEFAULT_OBSTACLE_BASE_HEIGHT,
    DEFAULT_OBSTACLE_REACH_LOWER,
    DEFAULT_OBSTACLE_REACH_UPPER,
    DISABLE_SHADOWS,
    compute_width_sag_profile,
    estimate_low_rope_height,
    half_ellipse_obstacle_height,
    render_dataset_images,
    states_to_qpos_and_control,
    task_bounds_arrays,
)
from rope.shared.lab_env import LabEnv

DEFAULT_OUT_DIR = "rope/plan/random_endpoint_pairs"
DEFAULT_DATASET_NAME = "random_endpoint_pairs.pt"
DEFAULT_SUMMARY_NAME = "summary.json"
DEFAULT_DIAGNOSTIC_NAME = "reach_low_height.png"
DEFAULT_PAIR_COUNT = 1000
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_LOW_ROPE_HEIGHT_MAX = 0.95
DEFAULT_MIDPOINT_CLEARANCE = 0.15
DEFAULT_MIDPOINT_BUFFER = 0.01
DEFAULT_OBSTACLE_HEIGHT = DEFAULT_OBSTACLE_BASE_HEIGHT + DEFAULT_MIDPOINT_CLEARANCE + DEFAULT_MIDPOINT_BUFFER
DEFAULT_WIDTH_STEPS = 41
DEFAULT_LEFT_REACH_LOWER = -0.05
DEFAULT_LEFT_REACH_UPPER = 0.0
DEFAULT_RIGHT_REACH_LOWER = 0.2
DEFAULT_RIGHT_REACH_UPPER = 0.25
DIAGNOSTIC_Y_MAX = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help="Directory where the endpoint pair artifact, summary, and diagnostic plot are saved.",
    )
    parser.add_argument("--out-path", type=Path, default=None, help="Override the .pt artifact path.")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--diagnostic-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        default=DISABLE_SHADOWS,
        help="Disable shadows for saved renders.",
    )
    parser.add_argument(
        "--obstacle-reach-lower",
        type=float,
        default=DEFAULT_OBSTACLE_REACH_LOWER,
        help="Upper reach bound for left endpoints.",
    )
    parser.add_argument(
        "--obstacle-reach-upper",
        type=float,
        default=DEFAULT_OBSTACLE_REACH_UPPER,
        help="Lower reach bound for right endpoints.",
    )
    parser.add_argument("--left-reach-lower", type=float, default=DEFAULT_LEFT_REACH_LOWER)
    parser.add_argument("--left-reach-upper", type=float, default=DEFAULT_LEFT_REACH_UPPER)
    parser.add_argument("--right-reach-lower", type=float, default=DEFAULT_RIGHT_REACH_LOWER)
    parser.add_argument("--right-reach-upper", type=float, default=DEFAULT_RIGHT_REACH_UPPER)
    parser.add_argument(
        "--low-rope-height-max",
        type=float,
        default=DEFAULT_LOW_ROPE_HEIGHT_MAX,
        help="Maximum estimated low-rope height for both start and goal endpoints.",
    )
    parser.add_argument(
        "--obstacle-base-height",
        type=float,
        default=DEFAULT_OBSTACLE_BASE_HEIGHT,
        help="Base height where the half-ellipse obstacle meets its reach boundaries.",
    )
    parser.add_argument(
        "--obstacle-height",
        type=float,
        default=DEFAULT_OBSTACLE_HEIGHT,
        help="Peak height of the half-ellipse obstacle shown in the diagnostic plot.",
    )
    parser.add_argument(
        "--width-steps",
        type=int,
        default=DEFAULT_WIDTH_STEPS,
        help="Number of widths used to estimate the width-dependent sag profile.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)


def endpoint_boxes(
    lower: np.ndarray,
    upper: np.ndarray,
    left_reach: tuple[float, float],
    right_reach: tuple[float, float],
) -> tuple[dict[str, np.ndarray | str], dict[str, np.ndarray | str]]:
    left_lower = lower.copy()
    left_upper = upper.copy()
    left_lower[0] = float(left_reach[0])
    left_upper[0] = float(left_reach[1])

    right_lower = lower.copy()
    right_upper = upper.copy()
    right_lower[0] = float(right_reach[0])
    right_upper[0] = float(right_reach[1])

    if float(left_upper[0]) <= float(left_lower[0]):
        raise ValueError("Left endpoint box is empty.")
    if float(right_upper[0]) <= float(right_lower[0]):
        raise ValueError("Right endpoint box is empty.")
    return (
        {"name": "left_low_reach", "lower": left_lower.astype(np.float32), "upper": left_upper.astype(np.float32)},
        {"name": "right_high_reach", "lower": right_lower.astype(np.float32), "upper": right_upper.astype(np.float32)},
    )


def sample_low_rope_states(
    box: dict[str, np.ndarray | str],
    count: int,
    *,
    low_rope_height_max: float,
    width_values: np.ndarray,
    sag_drop_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(box["lower"], dtype=np.float64)
    upper = np.asarray(box["upper"], dtype=np.float64)
    states: list[np.ndarray] = []
    low_heights: list[np.ndarray] = []
    sag_drops: list[np.ndarray] = []
    attempts = 0
    chunk_size = max(4096, 2 * int(count))
    progress = tqdm(total=int(count), desc=f"Sampling {box['name']}", unit="state")
    try:
        while sum(batch.shape[0] for batch in states) < int(count):
            attempts += chunk_size
            candidates = rng.uniform(lower, upper, size=(chunk_size, 3)).astype(np.float64)
            candidate_low = estimate_low_rope_height(candidates, width_values, sag_drop_values)
            candidate_sag = np.asarray(candidates[:, 1], dtype=np.float64) - candidate_low
            keep = candidate_low < float(low_rope_height_max)
            if np.any(keep):
                kept_states = candidates[keep]
                states.append(kept_states)
                low_heights.append(candidate_low[keep])
                sag_drops.append(candidate_sag[keep])
                progress.update(min(kept_states.shape[0], int(count) - progress.n))
            if attempts > 10_000_000 and not states:
                raise RuntimeError(f"Could not sample any endpoints for {box['name']} below the low-rope cutoff.")
    finally:
        progress.close()

    sampled_states = np.concatenate(states, axis=0)[:count]
    sampled_low = np.concatenate(low_heights, axis=0)[:count]
    sampled_sag = np.concatenate(sag_drops, axis=0)[:count]
    return sampled_states, sampled_low, sampled_sag


def save_start_goal_diagnostic(
    path: Path,
    start_states: np.ndarray,
    start_low_rope_height: np.ndarray,
    goal_states: np.ndarray,
    goal_low_rope_height: np.ndarray,
    *,
    task_reach: tuple[float, float],
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_height: float,
    low_rope_height_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    ax.scatter(
        start_states[:, 0],
        start_low_rope_height,
        s=18.0,
        c="#0072b2",
        alpha=0.7,
        edgecolors="none",
        label="start",
    )
    ax.scatter(
        goal_states[:, 0],
        goal_low_rope_height,
        s=18.0,
        c="#d55e00",
        alpha=0.7,
        edgecolors="none",
        label="goal",
    )
    curve_reach = np.linspace(float(obstacle_reach[0]), float(obstacle_reach[1]), num=256, dtype=np.float64)
    curve_height = half_ellipse_obstacle_height(
        curve_reach,
        obstacle_reach,
        float(obstacle_base_height),
        float(obstacle_height),
    )
    ax.fill_between(
        curve_reach,
        float(obstacle_base_height),
        curve_height,
        color="#d55e00",
        alpha=0.08,
        label="half-ellipse obstacle region",
    )
    ax.plot(curve_reach, curve_height, color="#4d4d4d", linestyle="--", linewidth=1.2, label="obstacle boundary")
    ax.axhline(float(obstacle_base_height), color="#0072b2", linestyle=":", linewidth=1.2, label="table top")
    ax.axhline(float(low_rope_height_max), color="#009e73", linestyle=":", linewidth=1.3, label="low-rope cutoff")
    sampled_min = float(min(np.min(start_low_rope_height), np.min(goal_low_rope_height)))
    y_padding = max(0.005, 0.04 * max(float(DIAGNOSTIC_Y_MAX) - sampled_min, 1e-6))
    ax.set_ylim(sampled_min - y_padding, float(DIAGNOSTIC_Y_MAX))
    ax.set_xlim(float(task_reach[0]), float(task_reach[1]))
    ax.set_title("Start/goal endpoint samples by reach and estimated low-rope height")
    ax.set_xlabel("reach")
    ax.set_ylabel("estimated low-rope height")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def validate_args(args: argparse.Namespace, lower: np.ndarray, upper: np.ndarray, env: LabEnv) -> None:
    if int(args.pair_count) <= 0:
        raise ValueError("Pair count must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("Render width and height must both be positive.")
    if int(args.width_steps) <= 1:
        raise ValueError("Width steps must be greater than one.")
    obstacle_reach = (float(args.obstacle_reach_lower), float(args.obstacle_reach_upper))
    if not (float(lower[0]) <= obstacle_reach[0] <= obstacle_reach[1] <= float(upper[0])):
        raise ValueError("Obstacle reach bounds must lie inside task reach bounds.")
    left_reach = (float(args.left_reach_lower), float(args.left_reach_upper))
    right_reach = (float(args.right_reach_lower), float(args.right_reach_upper))
    if not (float(lower[0]) <= left_reach[0] < left_reach[1] <= float(upper[0])):
        raise ValueError("Left reach range must be non-empty and lie inside task reach bounds.")
    if not (float(lower[0]) <= right_reach[0] < right_reach[1] <= float(upper[0])):
        raise ValueError("Right reach range must be non-empty and lie inside task reach bounds.")
    if left_reach[1] > obstacle_reach[0]:
        raise ValueError("Left reach upper bound must be at or below the obstacle lower reach bound.")
    if right_reach[0] < obstacle_reach[1]:
        raise ValueError("Right reach lower bound must be at or above the obstacle upper reach bound.")
    if float(args.obstacle_base_height) > float(args.obstacle_height):
        raise ValueError("Obstacle base height cannot exceed obstacle height.")
    try:
        env.model.camera(str(args.camera)).id
    except KeyError as exc:
        raise ValueError(f"Unknown camera {args.camera!r}.") from exc


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = args.out_dir.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve() if args.out_path is not None else out_dir / DEFAULT_DATASET_NAME
    summary_path = args.summary_path.expanduser().resolve() if args.summary_path is not None else out_dir / DEFAULT_SUMMARY_NAME
    diagnostic_path = (
        args.diagnostic_path.expanduser().resolve()
        if args.diagnostic_path is not None
        else out_dir / DEFAULT_DIAGNOSTIC_NAME
    )

    env = LabEnv()
    lower, upper = task_bounds_arrays(env)
    validate_args(args, lower, upper, env)

    obstacle_reach = (float(args.obstacle_reach_lower), float(args.obstacle_reach_upper))
    left_reach = (float(args.left_reach_lower), float(args.left_reach_upper))
    right_reach = (float(args.right_reach_lower), float(args.right_reach_upper))
    left_box, right_box = endpoint_boxes(lower, upper, left_reach, right_reach)
    width_values = np.linspace(float(lower[2]), float(upper[2]), num=int(args.width_steps), dtype=np.float64)
    sag_drop_values = compute_width_sag_profile(width_values, progress_desc="Estimating width-dependent rope sag")

    start_states, start_low, start_sag = sample_low_rope_states(
        left_box,
        int(args.pair_count),
        low_rope_height_max=float(args.low_rope_height_max),
        width_values=width_values,
        sag_drop_values=sag_drop_values,
        rng=rng,
    )
    goal_states, goal_low, goal_sag = sample_low_rope_states(
        right_box,
        int(args.pair_count),
        low_rope_height_max=float(args.low_rope_height_max),
        width_values=width_values,
        sag_drop_values=sag_drop_values,
        rng=rng,
    )

    start_qpos, start_control = states_to_qpos_and_control(start_states, progress_desc="Solving IK for starts")
    goal_qpos, goal_control = states_to_qpos_and_control(goal_states, progress_desc="Solving IK for goals")
    start_pixels = render_dataset_images(
        start_states,
        start_qpos,
        start_control,
        camera_name=str(args.camera),
        image_width=int(args.width),
        image_height=int(args.height),
        disable_shadows=bool(args.disable_shadows),
    )
    goal_pixels = render_dataset_images(
        goal_states,
        goal_qpos,
        goal_control,
        camera_name=str(args.camera),
        image_width=int(args.width),
        image_height=int(args.height),
        disable_shadows=bool(args.disable_shadows),
    )

    save_start_goal_diagnostic(
        diagnostic_path,
        start_states,
        start_low,
        goal_states,
        goal_low,
        task_reach=(float(lower[0]), float(upper[0])),
        obstacle_reach=obstacle_reach,
        obstacle_base_height=float(args.obstacle_base_height),
        obstacle_height=float(args.obstacle_height),
        low_rope_height_max=float(args.low_rope_height_max),
    )

    payload = {
        "metadata": {
            "seed": int(args.seed),
            "pair_count": int(args.pair_count),
            "camera": str(args.camera),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "disable_shadows": bool(args.disable_shadows),
            "obstacle_reach": np.asarray(obstacle_reach, dtype=np.float32),
            "left_reach": np.asarray(left_reach, dtype=np.float32),
            "right_reach": np.asarray(right_reach, dtype=np.float32),
            "task_lower": lower.astype(np.float32),
            "task_upper": upper.astype(np.float32),
            "low_rope_height_max": float(args.low_rope_height_max),
            "obstacle_base_height": float(args.obstacle_base_height),
            "obstacle_height": float(args.obstacle_height),
            "endpoint_sampling_rule": (
                "sample starts left of obstacle and goals right of obstacle, with estimated low-rope height below cutoff"
            ),
            "left_sampling_box": left_box,
            "right_sampling_box": right_box,
            "sag_width_steps": int(args.width_steps),
            "diagnostic_path": str(diagnostic_path),
            "summary_path": str(summary_path),
            "out_dir": str(out_dir),
        },
        "start": {
            "task_target": start_states.astype(np.float32),
            "low_rope_height": start_low.astype(np.float32),
            "sag_drop": start_sag.astype(np.float32),
            "qpos": start_qpos.astype(np.float32),
            "control": start_control.astype(np.float32),
            "pixels": start_pixels.astype(np.uint8),
        },
        "goal": {
            "task_target": goal_states.astype(np.float32),
            "low_rope_height": goal_low.astype(np.float32),
            "sag_drop": goal_sag.astype(np.float32),
            "qpos": goal_qpos.astype(np.float32),
            "control": goal_control.astype(np.float32),
            "pixels": goal_pixels.astype(np.uint8),
        },
        "sag_profile": {
            "width_values": width_values.astype(np.float32),
            "sag_drop": sag_drop_values.astype(np.float32),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    save_json(
        summary_path,
        {
            "out_path": str(out_path),
            "out_dir": str(out_dir),
            "diagnostic_path": str(diagnostic_path),
            "pair_count": int(args.pair_count),
            "camera": str(args.camera),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "obstacle_reach": list(obstacle_reach),
            "left_reach": list(left_reach),
            "right_reach": list(right_reach),
            "low_rope_height_max": float(args.low_rope_height_max),
            "obstacle_base_height": float(args.obstacle_base_height),
            "obstacle_height": float(args.obstacle_height),
            "left_sampling_box": left_box,
            "right_sampling_box": right_box,
            "start_low_rope_height_min": float(np.min(start_low)),
            "start_low_rope_height_max": float(np.max(start_low)),
            "goal_low_rope_height_min": float(np.min(goal_low)),
            "goal_low_rope_height_max": float(np.max(goal_low)),
            "sag_drop_min": float(np.min(sag_drop_values)),
            "sag_drop_max": float(np.max(sag_drop_values)),
        },
    )

    print(f"Saved endpoint pairs: {out_path}")
    print(f"Saved summary:        {summary_path}")
    print(f"Saved diagnostic:     {diagnostic_path}")


if __name__ == "__main__":
    main()
