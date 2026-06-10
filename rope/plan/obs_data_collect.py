#!/usr/bin/env python3
"""Collect a balanced rope obstacle image dataset from a reach-gated low-rope-height rule."""

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
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.shared.lab_env import BaseEnvConfig, LabEnv, TABLE_TOP_Z, TaskState

DEFAULT_OUT_DIR = "rope/plan/obstacle_data"
DEFAULT_CAMERA = "video_cam"
DISABLE_SHADOWS = True
TRAIN_FRACTION = 0.9
DIAGNOSTIC_PLOT_NAME = "balanced_obstacle_dataset_reach_height.png"
DEFAULT_SAMPLES_PER_CLASS = 8192
DEFAULT_OBSTACLE_REACH_LOWER = 0.05
DEFAULT_OBSTACLE_REACH_UPPER = 0.15
DEFAULT_OBSTACLE_BASE_HEIGHT = float(TABLE_TOP_Z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        default=DISABLE_SHADOWS,
        help="Disable shadows for saved renders.",
    )
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument(
        "--obstacle-reach-lower",
        type=float,
        default=DEFAULT_OBSTACLE_REACH_LOWER,
        help="Minimum reach coordinate where the low-rope-height obstacle is active.",
    )
    parser.add_argument(
        "--obstacle-reach-upper",
        type=float,
        default=DEFAULT_OBSTACLE_REACH_UPPER,
        help="Maximum reach coordinate where the low-rope-height obstacle is active.",
    )
    parser.add_argument(
        "--width-steps",
        type=int,
        default=41,
        help="Number of widths used to estimate the width-dependent sag profile.",
    )
    parser.add_argument("--reach-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--height-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--midpoint-clearance",
        type=float,
        default=0.15,
        help="Minimum allowed low-rope clearance above the table.",
    )
    parser.add_argument(
        "--midpoint-buffer",
        type=float,
        default=0.01,
        help="Extra clearance margin for proxy/model error.",
    )
    parser.add_argument(
        "--obstacle-base-height",
        type=float,
        default=DEFAULT_OBSTACLE_BASE_HEIGHT,
        help="Base height where the half-ellipse obstacle meets its left/right reach boundaries.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def task_bounds_arrays(env: LabEnv) -> tuple[np.ndarray, np.ndarray]:
    bounds = env.task_bounds
    lower = np.array([bounds.reach[0], bounds.height[0], bounds.width[0]], dtype=np.float64)
    upper = np.array([bounds.reach[1], bounds.height[1], bounds.width[1]], dtype=np.float64)
    return lower, upper


def obstacle_reach_bounds(lower: np.ndarray, upper: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    del lower, upper
    reach_lower = float(args.obstacle_reach_lower)
    reach_upper = float(args.obstacle_reach_upper)
    return reach_lower, reach_upper


def endpoint_sampling_boxes(
    lower: np.ndarray,
    upper: np.ndarray,
    obstacle_reach: tuple[float, float],
) -> list[dict[str, np.ndarray | str]]:
    boxes: list[dict[str, np.ndarray | str]] = []
    left_upper = upper.copy()
    left_upper[0] = min(float(left_upper[0]), float(obstacle_reach[0]))
    if float(left_upper[0]) > float(lower[0]):
        boxes.append(
            {
                "name": "reach_below_obstacle",
                "lower": lower.astype(np.float32),
                "upper": left_upper.astype(np.float32),
            }
        )

    right_lower = lower.copy()
    right_lower[0] = max(float(right_lower[0]), float(obstacle_reach[1]))
    if float(upper[0]) > float(right_lower[0]):
        boxes.append(
            {
                "name": "reach_above_obstacle",
                "lower": right_lower.astype(np.float32),
                "upper": upper.astype(np.float32),
            }
        )

    if not boxes:
        raise ValueError(
            "Obstacle reach interval covers the full task reach range; no outside-reach endpoint sampling box exists."
        )
    return boxes


def validate_args(lower: np.ndarray, upper: np.ndarray, args: argparse.Namespace, env: LabEnv) -> None:
    if int(args.samples_per_class) <= 0:
        raise ValueError("Samples per class must be positive.")
    if int(args.width_steps) <= 1:
        raise ValueError("Width steps must be greater than one.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("Render width and height must both be positive.")
    if float(args.midpoint_clearance) < 0.0 or float(args.midpoint_buffer) < 0.0:
        raise ValueError("Low-rope clearance and proxy/model-error buffer must be non-negative.")
    reach_lower, reach_upper = obstacle_reach_bounds(lower, upper, args)
    if not (float(lower[0]) <= reach_lower <= reach_upper <= float(upper[0])):
        raise ValueError(
            f"Obstacle reach bounds [{reach_lower}, {reach_upper}] must lie inside task reach bounds "
            f"[{float(lower[0])}, {float(upper[0])}]."
        )
    try:
        env.model.camera(str(args.camera)).id
    except KeyError as exc:
        raise ValueError(f"Unknown camera {args.camera!r}.") from exc


def sample_uniform_task_states(
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(lower, upper, size=(count, 3)).astype(np.float64)


def states_to_qpos_and_control(
    task_states: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if task_states.shape[0] == 0:
        env = LabEnv()
        return (
            np.zeros((0, env.model.nq), dtype=np.float32),
            np.zeros((0, env.model.nu), dtype=np.float32),
        )
    env = LabEnv()
    qpos_batch: list[np.ndarray] = []
    control_batch: list[np.ndarray] = []
    iterator = tqdm(task_states, desc=progress_desc, unit="state", leave=False) if progress_desc else task_states
    for state_vec in iterator:
        env.reset(TaskState.from_array(state_vec))
        qpos_batch.append(env.data.qpos.copy().astype(np.float32))
        control_batch.append(env.data.ctrl.copy().astype(np.float32))
    return np.stack(qpos_batch, axis=0), np.stack(control_batch, axis=0)


def compute_width_sag_profile(
    width_values: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> np.ndarray:
    proxy_env = LabEnv(base_config=BaseEnvConfig(enable_proxy_rope=True))
    reach = 0.5 * (float(proxy_env.task_bounds.reach[0]) + float(proxy_env.task_bounds.reach[1]))
    height = float(proxy_env.task_bounds.height[1])
    sag_drop = np.zeros((width_values.shape[0],), dtype=np.float64)
    iterator = (
        tqdm(enumerate(width_values), total=width_values.shape[0], desc=progress_desc, unit="width")
        if progress_desc
        else enumerate(width_values)
    )
    for index, width in iterator:
        proxy_env.reset(TaskState(reach=reach, height=height, width=float(width)))
        points = proxy_env.get_proxy_rope_points()
        endpoint_height = 0.5 * (float(points[0, 2]) + float(points[-1, 2]))
        sag_drop[index] = max(0.0, endpoint_height - float(np.min(points[:, 2])))
    return sag_drop


def interpolate_sag_drop(widths: np.ndarray, width_values: np.ndarray, sag_drop_values: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(widths, dtype=np.float64),
        np.asarray(width_values, dtype=np.float64),
        np.asarray(sag_drop_values, dtype=np.float64),
    )


def half_ellipse_obstacle_height(
    reach: np.ndarray,
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_peak_height: float,
) -> np.ndarray:
    reach_values = np.asarray(reach, dtype=np.float64)
    center = 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1]))
    half_width = 0.5 * (float(obstacle_reach[1]) - float(obstacle_reach[0]))
    if half_width <= 0.0:
        raise ValueError("Obstacle reach interval must have positive width.")
    normalized = (reach_values - center) / half_width
    boundary = 1.0 - normalized**2
    profile = np.sqrt(np.clip(boundary, 0.0, None))
    return float(obstacle_base_height) + (float(obstacle_peak_height) - float(obstacle_base_height)) * profile


def estimate_low_rope_height(
    states: np.ndarray,
    width_values: np.ndarray,
    sag_drop_values: np.ndarray,
) -> np.ndarray:
    sag_drop = interpolate_sag_drop(states[:, 2], width_values, sag_drop_values)
    return np.asarray(states[:, 1], dtype=np.float64) - sag_drop


def classify_obstacle_states(
    states: np.ndarray,
    *,
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_peak_height: float,
    width_values: np.ndarray,
    sag_drop_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_rope_height = estimate_low_rope_height(states, width_values, sag_drop_values)
    sag_drop = interpolate_sag_drop(states[:, 2], width_values, sag_drop_values)
    profile_height = half_ellipse_obstacle_height(states[:, 0], obstacle_reach, obstacle_base_height, obstacle_peak_height)
    in_reach = (states[:, 0] >= obstacle_reach[0]) & (states[:, 0] <= obstacle_reach[1])
    labels = (in_reach & (low_rope_height <= profile_height)).astype(np.int64)
    return labels, low_rope_height, sag_drop


def sample_labeled_task_states(
    label: int,
    count: int,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_peak_height: float,
    width_values: np.ndarray,
    sag_drop_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    low_heights: list[np.ndarray] = []
    sag_drops: list[np.ndarray] = []
    attempts = 0
    chunk_size = max(4096, 2 * count)
    while sum(batch.shape[0] for batch in states) < count:
        attempts += chunk_size
        candidates = sample_uniform_task_states(lower, upper, chunk_size, rng)
        candidate_labels, candidate_low, candidate_sag = classify_obstacle_states(
            candidates,
            obstacle_reach=obstacle_reach,
            obstacle_base_height=obstacle_base_height,
            obstacle_peak_height=obstacle_peak_height,
            width_values=width_values,
            sag_drop_values=sag_drop_values,
        )
        keep = candidate_labels == int(label)
        if np.any(keep):
            states.append(candidates[keep])
            low_heights.append(candidate_low[keep])
            sag_drops.append(candidate_sag[keep])
        if attempts > 10_000_000 and not states:
            class_name = "obstacle" if label == 1 else "non-obstacle"
            raise RuntimeError(f"Could not sample any {class_name} states with the configured reach/height rule.")

    sampled_states = np.concatenate(states, axis=0)[:count]
    sampled_low = np.concatenate(low_heights, axis=0)[:count]
    sampled_sag = np.concatenate(sag_drops, axis=0)[:count]
    return sampled_states, sampled_low, sampled_sag


def render_rgb_frame(
    renderer: mujoco.Renderer,
    env: LabEnv,
    camera_id: int,
    *,
    disable_shadows: bool,
) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    if disable_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def render_dataset_images(
    task_states: np.ndarray,
    qpos_batch: np.ndarray,
    control_batch: np.ndarray,
    *,
    camera_name: str,
    image_width: int,
    image_height: int,
    disable_shadows: bool,
) -> np.ndarray:
    env = LabEnv()
    camera_id = env.model.camera(camera_name).id
    qvel = np.zeros((env.model.nv,), dtype=np.float32)
    frames: list[np.ndarray] = []
    with mujoco.Renderer(env.model, height=image_height, width=image_width) as renderer:
        iterator = tqdm(
            zip(task_states, qpos_batch, control_batch, strict=True),
            total=task_states.shape[0],
            desc="Rendering dataset images",
            unit="image",
        )
        for state_vec, qpos, control in iterator:
            env.reset(TaskState.from_array(state_vec))
            env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
            env.data.qvel[: qvel.shape[0]] = qvel
            env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
            env.task_controller.set_target(TaskState.from_array(state_vec))
            env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
            mujoco.mj_forward(env.model, env.data)
            frames.append(render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows))
    return np.stack(frames, axis=0)


def split_indices(indices: np.ndarray, rng: np.random.Generator, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    if shuffled.size == 1:
        return shuffled.copy(), np.zeros((0,), dtype=np.int64)
    train_count = int(np.floor(train_fraction * float(shuffled.size)))
    train_count = min(max(train_count, 1), shuffled.size - 1)
    return shuffled[:train_count], shuffled[train_count:]


def build_balanced_dataset_from_samples(
    obstacle_states: np.ndarray,
    obstacle_low_rope_height: np.ndarray,
    obstacle_sag_drop: np.ndarray,
    non_obstacle_states: np.ndarray,
    non_obstacle_low_rope_height: np.ndarray,
    non_obstacle_sag_drop: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    selected_states = np.concatenate((obstacle_states, non_obstacle_states), axis=0).astype(np.float32)
    selected_labels = np.concatenate(
        (
            np.ones((obstacle_states.shape[0],), dtype=np.int64),
            np.zeros((non_obstacle_states.shape[0],), dtype=np.int64),
        ),
        axis=0,
    )
    selected_low_rope_height = np.concatenate((obstacle_low_rope_height, non_obstacle_low_rope_height), axis=0)
    selected_sag_drop = np.concatenate((obstacle_sag_drop, non_obstacle_sag_drop), axis=0)

    obstacle_local_idx = np.flatnonzero(selected_labels == 1)
    non_obstacle_local_idx = np.flatnonzero(selected_labels == 0)
    obstacle_train, obstacle_cal = split_indices(obstacle_local_idx, rng, TRAIN_FRACTION)
    non_obstacle_train, non_obstacle_cal = split_indices(non_obstacle_local_idx, rng, TRAIN_FRACTION)
    train_idx = np.concatenate((obstacle_train, non_obstacle_train), axis=0)
    cal_idx = np.concatenate((obstacle_cal, non_obstacle_cal), axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(cal_idx)

    return {
        "task_target": selected_states,
        "label": selected_labels.astype(np.int64),
        "low_rope_height": selected_low_rope_height.astype(np.float32),
        "sag_drop": selected_sag_drop.astype(np.float32),
        "train_idx": train_idx.astype(np.int64),
        "calibration_idx": cal_idx.astype(np.int64),
    }


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def save_balanced_dataset_diagnostic(
    out_path: Path,
    task_states: np.ndarray,
    labels: np.ndarray,
    low_rope_height: np.ndarray,
    *,
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_peak_height: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    obstacle_mask = labels == 1
    safe_mask = ~obstacle_mask

    ax.scatter(
        task_states[safe_mask, 0],
        low_rope_height[safe_mask],
        s=18.0,
        c="#009e73",
        alpha=0.7,
        edgecolors="none",
        label="non-obstacle",
    )
    ax.scatter(
        task_states[obstacle_mask, 0],
        low_rope_height[obstacle_mask],
        s=18.0,
        c="#d55e00",
        alpha=0.75,
        edgecolors="none",
        label="obstacle",
    )
    curve_reach = np.linspace(float(obstacle_reach[0]), float(obstacle_reach[1]), num=256, dtype=np.float64)
    curve_height = half_ellipse_obstacle_height(curve_reach, obstacle_reach, obstacle_base_height, obstacle_peak_height)
    ax.fill_between(
        curve_reach,
        float(obstacle_base_height),
        curve_height,
        color="#d55e00",
        alpha=0.08,
        label="half-ellipse obstacle region",
    )
    ax.plot(curve_reach, curve_height, color="#4d4d4d", linestyle="--", linewidth=1.2, label="obstacle boundary")
    ax.axhline(float(TABLE_TOP_Z), color="#0072b2", linestyle=":", linewidth=1.2, label="table top")
    ax.set_title("Balanced obstacle dataset by reach and estimated low-rope height")
    ax.set_xlabel("reach")
    ax.set_ylabel("estimated low-rope height")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = LabEnv()
    lower, upper = task_bounds_arrays(env)
    validate_args(lower, upper, args, env)

    reach_bounds = obstacle_reach_bounds(lower, upper, args)
    endpoint_boxes = endpoint_sampling_boxes(lower, upper, reach_bounds)
    width_values = np.linspace(float(lower[2]), float(upper[2]), num=int(args.width_steps), dtype=np.float64)
    sag_drop_values = compute_width_sag_profile(
        width_values,
        progress_desc="Estimating width-dependent rope sag",
    )

    low_rope_target = float(TABLE_TOP_Z + float(args.midpoint_clearance))
    obstacle_base_height = float(args.obstacle_base_height)
    obstacle_peak_height = float(low_rope_target + float(args.midpoint_buffer))
    if obstacle_base_height > obstacle_peak_height:
        raise ValueError(
            f"Obstacle base height {obstacle_base_height} cannot exceed obstacle peak height {obstacle_peak_height}."
        )

    obstacle_states, obstacle_low_rope_height, obstacle_sag_drop = sample_labeled_task_states(
        1,
        int(args.samples_per_class),
        lower=lower,
        upper=upper,
        obstacle_reach=reach_bounds,
        obstacle_base_height=obstacle_base_height,
        obstacle_peak_height=obstacle_peak_height,
        width_values=width_values,
        sag_drop_values=sag_drop_values,
        rng=rng,
    )
    non_obstacle_states, non_obstacle_low_rope_height, non_obstacle_sag_drop = sample_labeled_task_states(
        0,
        int(args.samples_per_class),
        lower=lower,
        upper=upper,
        obstacle_reach=reach_bounds,
        obstacle_base_height=obstacle_base_height,
        obstacle_peak_height=obstacle_peak_height,
        width_values=width_values,
        sag_drop_values=sag_drop_values,
        rng=rng,
    )

    balanced = build_balanced_dataset_from_samples(
        obstacle_states,
        obstacle_low_rope_height,
        obstacle_sag_drop,
        non_obstacle_states,
        non_obstacle_low_rope_height,
        non_obstacle_sag_drop,
        rng,
    )

    dataset_states = balanced["task_target"].astype(np.float64)
    dataset_qpos, dataset_control = states_to_qpos_and_control(
        dataset_states,
        progress_desc="Solving IK for sampled states",
    )
    dataset_pixels = render_dataset_images(
        dataset_states,
        dataset_qpos,
        dataset_control,
        camera_name=str(args.camera),
        image_width=int(args.width),
        image_height=int(args.height),
        disable_shadows=bool(args.disable_shadows),
    )

    save_balanced_dataset_diagnostic(
        out_dir / DIAGNOSTIC_PLOT_NAME,
        balanced["task_target"],
        balanced["label"],
        balanced["low_rope_height"],
        obstacle_reach=reach_bounds,
        obstacle_base_height=obstacle_base_height,
        obstacle_peak_height=obstacle_peak_height,
    )

    torch.save(
        {
            "metadata": {
                "seed": int(args.seed),
                "camera": str(args.camera),
                "image_width": int(args.width),
                "image_height": int(args.height),
                "disable_shadows": bool(args.disable_shadows),
                "train_fraction": float(TRAIN_FRACTION),
                "calibration_fraction": float(1.0 - TRAIN_FRACTION),
                "table_top_z": float(TABLE_TOP_Z),
                "low_rope_clearance": float(args.midpoint_clearance),
                "model_error_buffer": float(args.midpoint_buffer),
                "low_rope_target": float(low_rope_target),
                "low_rope_cutoff": float(obstacle_peak_height),
                "obstacle_profile": "half_ellipse",
                "obstacle_base_height": float(obstacle_base_height),
                "obstacle_height": float(obstacle_peak_height),
                "obstacle_reach": np.array(reach_bounds, dtype=np.float32),
                "task_lower": lower.astype(np.float32),
                "task_upper": upper.astype(np.float32),
                "endpoint_sampling_rule": "force obstacle crossing by sampling start and goal on opposite sides of the obstacle reach interval",
                "endpoint_sampling_boxes": endpoint_boxes,
                "samples_per_class": int(args.samples_per_class),
                "sag_width_steps": int(args.width_steps),
                "balanced_total_count": int(balanced["task_target"].shape[0]),
            },
            "dataset": {
                "pixels": dataset_pixels.astype(np.uint8),
                "task_target": balanced["task_target"].astype(np.float32),
                "label": balanced["label"].astype(np.int64),
                "low_rope_height": balanced["low_rope_height"].astype(np.float32),
                "midpoint_height": balanced["low_rope_height"].astype(np.float32),
                "sag_drop": balanced["sag_drop"].astype(np.float32),
                "qpos": dataset_qpos.astype(np.float32),
                "control": dataset_control.astype(np.float32),
                "train_idx": balanced["train_idx"].astype(np.int64),
                "calibration_idx": balanced["calibration_idx"].astype(np.int64),
            },
            "sag_profile": {
                "width_values": width_values.astype(np.float32),
                "sag_drop": sag_drop_values.astype(np.float32),
            },
        },
        out_dir / "obstacle_classifier_data.pt",
    )

    save_json(
        out_dir / "summary.json",
        {
            "out_dir": str(out_dir),
            "camera": str(args.camera),
            "sampling": {
                "samples_per_class": int(args.samples_per_class),
                "width_steps": int(args.width_steps),
            },
            "counts": {
                "balanced_obstacle_count": int(np.sum(balanced["label"] == 1)),
                "balanced_non_obstacle_count": int(np.sum(balanced["label"] == 0)),
                "balanced_total_count": int(balanced["label"].shape[0]),
                "train_count": int(balanced["train_idx"].shape[0]),
                "calibration_count": int(balanced["calibration_idx"].shape[0]),
            },
            "split": {
                "train_fraction": float(TRAIN_FRACTION),
                "calibration_fraction": float(1.0 - TRAIN_FRACTION),
            },
            "task_lower": lower.tolist(),
            "task_upper": upper.tolist(),
            "obstacle_reach": list(reach_bounds),
            "endpoint_sampling_rule": "force obstacle crossing by sampling start and goal on opposite sides of the obstacle reach interval",
            "endpoint_sampling_boxes": [
                {
                    "name": str(box["name"]),
                    "lower": np.asarray(box["lower"], dtype=np.float64).tolist(),
                    "upper": np.asarray(box["upper"], dtype=np.float64).tolist(),
                }
                for box in endpoint_boxes
            ],
            "low_rope_target": float(low_rope_target),
            "low_rope_cutoff": float(obstacle_peak_height),
            "obstacle_profile": "half_ellipse",
            "obstacle_base_height": float(obstacle_base_height),
            "obstacle_height": float(obstacle_peak_height),
            "sag_drop_min": float(np.min(sag_drop_values)),
            "sag_drop_max": float(np.max(sag_drop_values)),
        },
    )

    print(f"Saved diagnostic: {out_dir / DIAGNOSTIC_PLOT_NAME}")
    print(f"Saved dataset:    {out_dir / 'obstacle_classifier_data.pt'}")
    print(f"Saved summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
