#!/usr/bin/env python3
"""Generate a balanced Reacher obstacle dataset from a joint-space box."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from reacher.plan.obs_data_collect import (
    OBSTACLE_DATA_NAME,
    OBSTACLE_OVERLAY_NAME,
    OUTSIDE_OVERLAY_NAME,
    TRAIN_FRACTION,
    infer_planar_arm_geometry,
    joint_limits_with_unbounded_fixed,
    log_progress,
    make_render_env,
    qpos_sampling_bounds,
    render_qpos_batch,
    save_json,
    save_obstacle_overlay,
    split_indices,
)

DEFAULT_OUT_DIR = Path("reacher/plan/obstacle_data_joint_box")
DEFAULT_SAMPLES_PER_CLASS = 4096
DIAGNOSTIC_PLOT_NAME = "joint_space_samples.png"
DEFAULT_Q1_MIN = 0.0
DEFAULT_Q1_MAX = 3.1415
DEFAULT_Q2_MIN = -2.88
DEFAULT_Q2_MAX = -2.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--negative-sampling-budget", type=int, default=200_000)
    parser.add_argument("--q1-min", type=float, default=DEFAULT_Q1_MIN)
    parser.add_argument("--q1-max", type=float, default=DEFAULT_Q1_MAX)
    parser.add_argument("--q2-min", type=float, default=DEFAULT_Q2_MIN)
    parser.add_argument("--q2-max", type=float, default=DEFAULT_Q2_MAX)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if int(args.negative_sampling_budget) <= 0:
        raise ValueError("--negative-sampling-budget must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if float(args.q1_min) > float(args.q1_max):
        raise ValueError("--q1-min must be <= --q1-max.")
    if float(args.q2_min) > float(args.q2_max):
        raise ValueError("--q2-min must be <= --q2-max.")


def make_positive_box(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([float(args.q1_min), float(args.q2_min)], dtype=np.float64)
    upper = np.array([float(args.q1_max), float(args.q2_max)], dtype=np.float64)
    return lower, upper


def sample_positive_qpos(rng: np.random.Generator, *, count: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    side = int(np.ceil(np.sqrt(count)))
    q1_values = np.linspace(lower[0], upper[0], side, dtype=np.float64)
    q2_values = np.linspace(lower[1], upper[1], side, dtype=np.float64)
    q1_mesh, q2_mesh = np.meshgrid(q1_values, q2_values, indexing="xy")
    qpos = np.stack((q1_mesh.reshape(-1), q2_mesh.reshape(-1)), axis=1)
    if qpos.shape[0] > count:
        indices = np.arange(qpos.shape[0], dtype=np.int64)
        rng.shuffle(indices)
        qpos = qpos[indices[:count]]
    return qpos


def inside_box_mask(qpos: np.ndarray, *, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all((qpos >= lower[None, :]) & (qpos <= upper[None, :]), axis=1)


def sample_negative_qpos(
    rng: np.random.Generator,
    *,
    count: int,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    sampling_budget: int,
) -> np.ndarray:
    sample_lower, sample_upper = qpos_sampling_bounds(joint_lower, joint_upper)
    qpos_chunks: list[np.ndarray] = []
    remaining = int(count)
    draws = 0
    while remaining > 0 and draws < sampling_budget:
        batch_count = min(max(remaining * 4, 1024), sampling_budget - draws)
        qpos_batch = rng.uniform(sample_lower, sample_upper, size=(batch_count, sample_lower.shape[0]))
        draws += batch_count
        valid_mask = ~inside_box_mask(qpos_batch, lower=box_lower, upper=box_upper)
        if not np.any(valid_mask):
            continue
        take = min(remaining, int(np.sum(valid_mask)))
        qpos_chunks.append(qpos_batch[valid_mask][:take])
        remaining -= take
    if remaining > 0:
        raise RuntimeError(f"Failed to collect {count} outside-box samples after {draws} qpos draws; missing {remaining}.")
    return np.concatenate(qpos_chunks, axis=0)


def save_joint_space_plot(
    *,
    out_path: Path,
    positive_qpos: np.ndarray,
    negative_qpos: np.ndarray,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)
    ax.scatter(negative_qpos[:, 0], negative_qpos[:, 1], s=6, c="#009e73", alpha=0.22, label="outside obstacle")
    ax.scatter(positive_qpos[:, 0], positive_qpos[:, 1], s=8, c="#d55e00", alpha=0.42, label="inside obstacle")

    q1_rect = [box_lower[0], box_upper[0], box_upper[0], box_lower[0], box_lower[0]]
    q2_rect = [box_lower[1], box_lower[1], box_upper[1], box_upper[1], box_lower[1]]
    ax.plot(q1_rect, q2_rect, color="#000000", linewidth=1.5, label="obstacle box")

    finite_lower = np.where(np.isfinite(joint_lower), joint_lower, -np.pi)
    finite_upper = np.where(np.isfinite(joint_upper), joint_upper, np.pi)
    pad = 0.08
    ax.set_xlim(float(finite_lower[0] - pad), float(finite_upper[0] + pad))
    ax.set_ylim(float(finite_lower[1] - pad), float(finite_upper[1] + pad))
    ax.grid(alpha=0.2)
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title("Reacher joint-space obstacle samples")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def split_balanced_indices(labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    obstacle_train, obstacle_cal = split_indices(np.flatnonzero(labels == 1), rng, TRAIN_FRACTION)
    outside_train, outside_cal = split_indices(np.flatnonzero(labels == 0), rng, TRAIN_FRACTION)
    train_idx = np.concatenate((obstacle_train, outside_train), axis=0)
    calibration_idx = np.concatenate((obstacle_cal, outside_cal), axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(calibration_idx)
    return train_idx.astype(np.int64), calibration_idx.astype(np.int64)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    box_lower, box_upper = make_positive_box(args)
    positive_qpos = sample_positive_qpos(rng, count=int(args.samples_per_class), lower=box_lower, upper=box_upper)

    log_progress("Inferring arm geometry and sampling joint-space obstacle states.")
    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        geom = infer_planar_arm_geometry(env)
        joint_lower, joint_upper = joint_limits_with_unbounded_fixed(env)
    finally:
        env.close()

    negative_qpos = sample_negative_qpos(
        rng,
        count=int(args.samples_per_class),
        box_lower=box_lower,
        box_upper=box_upper,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        sampling_budget=int(args.negative_sampling_budget),
    )

    labels = np.concatenate(
        (
            np.ones((positive_qpos.shape[0],), dtype=np.int64),
            np.zeros((negative_qpos.shape[0],), dtype=np.int64),
        ),
        axis=0,
    )
    qpos = np.concatenate((positive_qpos, negative_qpos), axis=0).astype(np.float32)
    indices = np.arange(qpos.shape[0], dtype=np.int64)
    rng.shuffle(indices)
    qpos = qpos[indices]
    labels = labels[indices]
    train_idx, calibration_idx = split_balanced_indices(labels, rng)

    log_progress("Rendering balanced dataset images.")
    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        pixels = render_qpos_batch(
            env,
            int(args.seed),
            qpos,
            height=int(args.height),
            width=int(args.width),
            progress_desc="Rendering obstacle dataset",
        )
    finally:
        env.close()

    obstacle_center_qpos = 0.5 * (box_lower + box_upper)
    save_joint_space_plot(
        out_path=out_dir / DIAGNOSTIC_PLOT_NAME,
        positive_qpos=positive_qpos,
        negative_qpos=negative_qpos,
        box_lower=box_lower,
        box_upper=box_upper,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )
    save_obstacle_overlay(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
        center_qpos=obstacle_center_qpos.astype(np.float32),
        sample_qpos=positive_qpos.astype(np.float32),
        out_path=out_dir / OBSTACLE_OVERLAY_NAME,
        progress_desc="Rendering obstacle overlay",
    )
    save_obstacle_overlay(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
        center_qpos=obstacle_center_qpos.astype(np.float32),
        sample_qpos=negative_qpos.astype(np.float32),
        out_path=out_dir / OUTSIDE_OVERLAY_NAME,
        progress_desc="Rendering outside overlay",
    )

    metadata = {
        "seed": int(args.seed),
        "image_width": int(args.width),
        "image_height": int(args.height),
        "time_limit": float(args.time_limit),
        "physics_freq_hz": float(args.physics_freq_hz),
        "task_lower": joint_lower.astype(np.float32),
        "task_upper": joint_upper.astype(np.float32),
        "samples_per_class": int(args.samples_per_class),
        "inside_sample_count": int(positive_qpos.shape[0]),
        "outside_sample_count": int(negative_qpos.shape[0]),
        "train_fraction": float(TRAIN_FRACTION),
        "calibration_fraction": float(1.0 - TRAIN_FRACTION),
        "obstacle_label": 1,
        "non_obstacle_label": 0,
        "label_rule": "1 iff qpos lies inside the configured joint-space box; 0 otherwise",
        "box_q1_min": float(box_lower[0]),
        "box_q1_max": float(box_upper[0]),
        "box_q2_min": float(box_lower[1]),
        "box_q2_max": float(box_upper[1]),
        "tip_source": str(geom["tip_source"]),
        "link1": float(geom["link1"]),
        "link2": float(geom["link2"]),
        "reach_min": float(geom["reach_min"]),
        "reach_max": float(geom["reach_max"]),
    }

    torch.save(
        {
            "metadata": metadata,
            "dataset": {
                "pixels": pixels.astype(np.uint8),
                "task_target": qpos.astype(np.float32),
                "qpos": qpos.astype(np.float32),
                "qvel": np.zeros_like(qpos, dtype=np.float32),
                "label": labels.astype(np.int64),
                "train_idx": train_idx,
                "calibration_idx": calibration_idx,
            },
            "obstacle_data": {
                "obstacle_center_qpos": obstacle_center_qpos.astype(np.float32),
                "obstacle_qpos": positive_qpos.astype(np.float32),
                "outside_qpos": negative_qpos.astype(np.float32),
                "box_lower": box_lower.astype(np.float32),
                "box_upper": box_upper.astype(np.float32),
            },
        },
        out_dir / OBSTACLE_DATA_NAME,
    )

    summary = {
        "out_dir": str(out_dir),
        "dataset_path": str(out_dir / OBSTACLE_DATA_NAME),
        "joint_space_plot_path": str(out_dir / DIAGNOSTIC_PLOT_NAME),
        "obstacle_overlay_path": str(out_dir / OBSTACLE_OVERLAY_NAME),
        "outside_overlay_path": str(out_dir / OUTSIDE_OVERLAY_NAME),
        "counts": {
            "obstacle": int(np.sum(labels == 1)),
            "non_obstacle": int(np.sum(labels == 0)),
            "total": int(labels.shape[0]),
            "train": int(train_idx.shape[0]),
            "calibration": int(calibration_idx.shape[0]),
        },
        "box_lower": box_lower.astype(np.float64).tolist(),
        "box_upper": box_upper.astype(np.float64).tolist(),
        "obstacle_center_qpos": obstacle_center_qpos.astype(np.float64).tolist(),
        "link1": float(geom["link1"]),
        "link2": float(geom["link2"]),
        "reach_min": float(geom["reach_min"]),
        "reach_max": float(geom["reach_max"]),
        "tip_source": str(geom["tip_source"]),
    }
    save_json(out_dir / "summary.json", summary)

    print(f"Saved dataset:   {out_dir / OBSTACLE_DATA_NAME}")
    print(f"Saved joint plot: {out_dir / DIAGNOSTIC_PLOT_NAME}")
    print(f"Saved obstacle overlay: {out_dir / OBSTACLE_OVERLAY_NAME}")
    print(f"Saved outside overlay:  {out_dir / OUTSIDE_OVERLAY_NAME}")
    print(f"Saved summary:   {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
