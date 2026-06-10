#!/usr/bin/env python3
"""Collect a balanced OGBench cube grasped-state dataset split by gripper-height threshold."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from ogbench_cube.data.ogbench_cube_data_gen import THETA_SAMPLING_BOUNDS, XY_SAMPLING_BOUNDS, Z_SAMPLING_BOUNDS
from ogbench_cube.plan.obs_data_collect_3d_ellipsoid import (
    DEFAULT_CAMERA,
    DEFAULT_CONTROL_DECIMATION,
    DEFAULT_ENV_NAME,
    DEFAULT_MAX_ATTEMPTS_PER_CLASS,
    DEFAULT_MAX_ORACLE_STEPS,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_SIM_FREQ_HZ,
    DEFAULT_ACCEPTANCE_POS_TOL,
    DEFAULT_ACCEPTANCE_YAW_TOL,
    capture_grasp_reference,
    cube_is_grasped,
    jsonable,
    make_env,
    render_without_target_cube,
    synthesize_grasped_state,
)

DEFAULT_OUT_DIR = Path("ogbench_cube/plan/height_data")
DEFAULT_OUT_NAME = "height_classifier_data.pt"
DEFAULT_SUMMARY_NAME = "summary.json"
DEFAULT_REFERENCE_IMAGE_NAME = "reference_grasp.png"
DEFAULT_XY_PLOT_NAME = "height_dataset_xy.png"
DEFAULT_HIST_PLOT_NAME = "height_dataset_hist.png"
DEFAULT_SAMPLES_PER_CLASS = 8192
DEFAULT_HEIGHT_THRESHOLD = 0.09
DEFAULT_HEIGHT_MARGIN = 0.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--height-threshold", type=float, default=DEFAULT_HEIGHT_THRESHOLD)
    parser.add_argument("--height-margin", type=float, default=DEFAULT_HEIGHT_MARGIN)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--oracle-segment-dt", type=float, default=0.4)
    parser.add_argument("--oracle-noise", type=float, default=0.0)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=0.5)
    parser.add_argument("--max-oracle-steps", type=int, default=DEFAULT_MAX_ORACLE_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--grasp-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=0.03)
    parser.add_argument("--acceptance-pos-tol", type=float, default=DEFAULT_ACCEPTANCE_POS_TOL)
    parser.add_argument("--acceptance-yaw-tol", type=float, default=DEFAULT_ACCEPTANCE_YAW_TOL)
    parser.add_argument(
        "--require-grasped",
        action="store_true",
        help="Reject samples unless the synthesized state exceeds the grasp/contact thresholds.",
    )
    parser.add_argument(
        "--require-yaw-match",
        action="store_true",
        help="Reject samples unless the synthesized cube yaw is within --acceptance-yaw-tol of the request.",
    )
    parser.add_argument("--max-attempts-per-class", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_CLASS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if float(args.height_threshold) <= 0.0:
        raise ValueError("--height-threshold must be positive.")
    if float(args.height_margin) < 0.0:
        raise ValueError("--height-margin must be non-negative.")
    if int(args.max_oracle_steps) <= 0 or int(args.settle_steps) < 0:
        raise ValueError("--max-oracle-steps must be positive and --settle-steps must be non-negative.")
    if float(args.acceptance_pos_tol) < 0.0 or float(args.acceptance_yaw_tol) < 0.0:
        raise ValueError("Acceptance tolerances must be non-negative.")
    if int(args.max_attempts_per_class) <= 0:
        raise ValueError("--max-attempts-per-class must be positive.")


def angular_distance(a: float, b: float) -> float:
    return float(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def height_label_from_value(gripper_height: float, *, threshold: float) -> int:
    return int(float(gripper_height) > float(threshold))


def sample_xy(rng: np.random.Generator) -> np.ndarray:
    x = float(rng.uniform(float(XY_SAMPLING_BOUNDS[0, 0]), float(XY_SAMPLING_BOUNDS[1, 0])))
    y = float(rng.uniform(float(XY_SAMPLING_BOUNDS[0, 1]), float(XY_SAMPLING_BOUNDS[1, 1])))
    return np.array([x, y], dtype=np.float64)


def sample_pose_for_height_class(
    rng: np.random.Generator,
    *,
    label: int,
    block_height_threshold: float,
    height_margin: float,
) -> tuple[np.ndarray, float]:
    z_min = float(Z_SAMPLING_BOUNDS[0])
    z_max = float(Z_SAMPLING_BOUNDS[1])
    if label == 0:
        class_z_max = min(z_max, block_height_threshold - height_margin)
        if class_z_max <= z_min:
            raise ValueError(
                "No feasible below-threshold region remains in block z-space. "
                "Reduce --height-threshold or --height-margin."
            )
        z = float(rng.uniform(z_min, class_z_max))
    else:
        class_z_min = max(z_min, block_height_threshold + height_margin)
        if class_z_min >= z_max:
            raise ValueError(
                "No feasible above-threshold region remains in block z-space. "
                "Increase --height-threshold range or reduce --height-margin."
            )
        z = float(rng.uniform(class_z_min, z_max))
    xy = sample_xy(rng)
    yaw = float(rng.uniform(float(THETA_SAMPLING_BOUNDS[0]), float(THETA_SAMPLING_BOUNDS[1])))
    return np.array([xy[0], xy[1], z], dtype=np.float64), yaw


def evaluate_sample(
    info: dict[str, np.ndarray],
    *,
    desired_block_pos: np.ndarray,
    desired_block_yaw: float,
    height_threshold: float,
    contact_threshold: float,
    alignment_threshold: float,
) -> dict[str, Any]:
    block_pos = np.asarray(info["privileged/block_0_pos"], dtype=np.float64)
    block_yaw = float(info["privileged/block_0_yaw"][0])
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
    gripper_height = float(effector_pos[2])
    gripper_contact = float(info["proprio/gripper_contact"][0])
    grasp_alignment_error = float(np.linalg.norm(block_pos - effector_pos))
    return {
        "block_pos": block_pos.astype(np.float32),
        "block_yaw": np.float32(block_yaw),
        "qpos": np.asarray(info["qpos"], dtype=np.float32),
        "qvel": np.asarray(info["qvel"], dtype=np.float32),
        "control": np.asarray(info["control"], dtype=np.float32),
        "effector_pos": effector_pos.astype(np.float32),
        "effector_yaw": np.float32(float(info["proprio/effector_yaw"][0])),
        "gripper_height": np.float32(gripper_height),
        "gripper_contact": np.float32(gripper_contact),
        "grasp_alignment_error": np.float32(grasp_alignment_error),
        "block_pos_error": np.float32(np.linalg.norm(block_pos - desired_block_pos)),
        "block_yaw_error": np.float32(angular_distance(block_yaw, desired_block_yaw)),
        "actual_label": np.int64(height_label_from_value(gripper_height, threshold=height_threshold)),
        "grasped": bool(gripper_contact >= contact_threshold and grasp_alignment_error <= alignment_threshold),
    }


def sample_balanced_dataset(
    *,
    env: object,
    args: argparse.Namespace,
    reference: object,
    rng: np.random.Generator,
    block_height_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    class_names = {0: "below", 1: "above"}
    accepted: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    attempt_counts = {0: 0, 1: 0}
    rejected_counts = {
        0: {"synthesis_failed": 0, "wrong_label": 0, "pose_error": 0, "not_grasped": 0},
        1: {"synthesis_failed": 0, "wrong_label": 0, "pose_error": 0, "not_grasped": 0},
    }

    for label in (0, 1):
        progress = tqdm(total=int(args.samples_per_class), desc=f"Sampling {class_names[label]}", unit="image")
        try:
            while len(accepted[label]) < int(args.samples_per_class):
                attempt_counts[label] += 1
                if attempt_counts[label] > int(args.max_attempts_per_class):
                    raise RuntimeError(
                        f"Exceeded max attempts while collecting {class_names[label]} samples: "
                        f"{attempt_counts[label]} attempts for {len(accepted[label])} accepted."
                    )
                env.reset(seed=int(args.seed) + 100_000 * label + attempt_counts[label])
                desired_block_pos, desired_block_yaw = sample_pose_for_height_class(
                    rng,
                    label=label,
                    block_height_threshold=float(block_height_threshold),
                    height_margin=float(args.height_margin),
                )
                try:
                    info = synthesize_grasped_state(
                        env,
                        block_pos=desired_block_pos,
                        block_yaw=desired_block_yaw,
                        reference=reference,
                        settle_steps=int(args.settle_steps),
                    )
                except RuntimeError:
                    rejected_counts[label]["synthesis_failed"] += 1
                    continue

                metrics = evaluate_sample(
                    info,
                    desired_block_pos=desired_block_pos,
                    desired_block_yaw=desired_block_yaw,
                    height_threshold=float(args.height_threshold),
                    contact_threshold=float(args.grasp_contact_threshold),
                    alignment_threshold=float(args.grasp_alignment_threshold),
                )
                if int(metrics["actual_label"]) != int(label):
                    rejected_counts[label]["wrong_label"] += 1
                    continue
                if float(metrics["block_pos_error"]) > float(args.acceptance_pos_tol):
                    rejected_counts[label]["pose_error"] += 1
                    continue
                if args.require_yaw_match and float(metrics["block_yaw_error"]) > float(args.acceptance_yaw_tol):
                    rejected_counts[label]["pose_error"] += 1
                    continue
                if args.require_grasped and not metrics["grasped"]:
                    rejected_counts[label]["not_grasped"] += 1
                    continue

                frame = render_without_target_cube(env, str(args.camera))
                accepted[label].append(
                    {
                        "pixels": frame.astype(np.uint8),
                        "task_target": np.asarray(desired_block_pos, dtype=np.float32),
                        "yaw": np.float32(desired_block_yaw),
                        "label": np.int64(label),
                        **metrics,
                    }
                )
                progress.update(1)
        finally:
            progress.close()

    all_samples = accepted[0] + accepted[1]
    labels = np.asarray([sample["label"] for sample in all_samples], dtype=np.int64)
    dataset = {
        "pixels": np.stack([sample["pixels"] for sample in all_samples], axis=0).astype(np.uint8),
        "task_target": np.stack([sample["task_target"] for sample in all_samples], axis=0).astype(np.float32),
        "yaw": np.asarray([sample["yaw"] for sample in all_samples], dtype=np.float32),
        "label": labels.astype(np.int64),
        "block_pos": np.stack([sample["block_pos"] for sample in all_samples], axis=0).astype(np.float32),
        "block_yaw": np.asarray([sample["block_yaw"] for sample in all_samples], dtype=np.float32),
        "qpos": np.stack([sample["qpos"] for sample in all_samples], axis=0).astype(np.float32),
        "qvel": np.stack([sample["qvel"] for sample in all_samples], axis=0).astype(np.float32),
        "control": np.stack([sample["control"] for sample in all_samples], axis=0).astype(np.float32),
        "effector_pos": np.stack([sample["effector_pos"] for sample in all_samples], axis=0).astype(np.float32),
        "effector_yaw": np.asarray([sample["effector_yaw"] for sample in all_samples], dtype=np.float32),
        "gripper_height": np.asarray([sample["gripper_height"] for sample in all_samples], dtype=np.float32),
        "gripper_contact": np.asarray([sample["gripper_contact"] for sample in all_samples], dtype=np.float32),
        "grasp_alignment_error": np.asarray(
            [sample["grasp_alignment_error"] for sample in all_samples],
            dtype=np.float32,
        ),
        "block_pos_error": np.asarray([sample["block_pos_error"] for sample in all_samples], dtype=np.float32),
        "block_yaw_error": np.asarray([sample["block_yaw_error"] for sample in all_samples], dtype=np.float32),
    }
    stats = {"attempt_counts": attempt_counts, "rejected_counts": rejected_counts}
    return dataset, stats


def save_xy_plot(path: Path, *, block_pos: np.ndarray, labels: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    below_mask = labels == 0
    above_mask = labels == 1
    ax.scatter(
        block_pos[below_mask, 0],
        block_pos[below_mask, 1],
        s=16.0,
        c="#0072b2",
        alpha=0.7,
        edgecolors="none",
        label="below threshold",
    )
    ax.scatter(
        block_pos[above_mask, 0],
        block_pos[above_mask, 1],
        s=16.0,
        c="#d55e00",
        alpha=0.7,
        edgecolors="none",
        label="above threshold",
    )
    ax.set_title("Balanced height dataset by cube x/y position")
    ax.set_xlabel("cube x")
    ax.set_ylabel("cube y")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_height_histogram(path: Path, *, gripper_height: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    below_mask = labels == 0
    above_mask = labels == 1
    bins = 60
    ax.hist(gripper_height[below_mask], bins=bins, color="#0072b2", alpha=0.75, label="below threshold")
    ax.hist(gripper_height[above_mask], bins=bins, color="#d55e00", alpha=0.65, label="above threshold")
    ax.axvline(float(threshold), color="#1a1a1a", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}")
    ax.set_title("Balanced gripper-height dataset")
    ax.set_xlabel("gripper height")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    try:
        reference, reference_info = capture_grasp_reference(env, args)
        reference_frame = render_without_target_cube(env, str(args.camera))
        reference_block_pos = np.asarray(reference.block_pos, dtype=np.float64)
        reference_effector_pos = np.asarray(reference.effector_pos, dtype=np.float64)
        gripper_block_height_offset = float(reference_effector_pos[2] - reference_block_pos[2])
        block_height_threshold = float(args.height_threshold) - gripper_block_height_offset
        dataset, stats = sample_balanced_dataset(
            env=env,
            args=args,
            reference=reference,
            rng=rng,
            block_height_threshold=block_height_threshold,
        )
    finally:
        env.close()

    imageio.imwrite(out_dir / DEFAULT_REFERENCE_IMAGE_NAME, reference_frame)
    save_xy_plot(out_dir / DEFAULT_XY_PLOT_NAME, block_pos=dataset["block_pos"], labels=dataset["label"])
    save_height_histogram(
        out_dir / DEFAULT_HIST_PLOT_NAME,
        gripper_height=dataset["gripper_height"],
        labels=dataset["label"],
        threshold=float(args.height_threshold),
    )

    payload = {
        "metadata": {
            "seed": int(args.seed),
            "camera": str(args.camera),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "samples_per_class": int(args.samples_per_class),
            "height_threshold": float(args.height_threshold),
            "height_margin": float(args.height_margin),
            "task_xy_bounds": np.asarray(XY_SAMPLING_BOUNDS, dtype=np.float32),
            "task_z_bounds": np.asarray(Z_SAMPLING_BOUNDS, dtype=np.float32),
            "task_yaw_bounds": np.asarray(THETA_SAMPLING_BOUNDS, dtype=np.float32),
            "acceptance_pos_tol": float(args.acceptance_pos_tol),
            "acceptance_yaw_tol": float(args.acceptance_yaw_tol),
            "require_grasped": bool(args.require_grasped),
            "require_yaw_match": bool(args.require_yaw_match),
            "settle_steps": int(args.settle_steps),
            "balanced_total_count": int(dataset["label"].shape[0]),
            "reference_block_pos": reference_block_pos.astype(np.float32),
            "reference_effector_pos": reference_effector_pos.astype(np.float32),
            "gripper_block_height_offset": np.float32(gripper_block_height_offset),
            "derived_block_height_threshold": np.float32(block_height_threshold),
        },
        "dataset": dataset,
        "reference_grasp": jsonable(reference),
    }
    torch.save(payload, out_dir / args.out_name)

    summary = {
        "out_dir": out_dir,
        "counts": {
            "below_count": int(np.sum(dataset["label"] == 0)),
            "above_count": int(np.sum(dataset["label"] == 1)),
            "total_count": int(dataset["label"].shape[0]),
        },
        "sampling": {
            "height_threshold": float(args.height_threshold),
            "height_margin": float(args.height_margin),
            "gripper_block_height_offset": float(gripper_block_height_offset),
            "derived_block_height_threshold": float(block_height_threshold),
            "task_xy_bounds": np.asarray(XY_SAMPLING_BOUNDS, dtype=np.float32),
            "task_z_bounds": np.asarray(Z_SAMPLING_BOUNDS, dtype=np.float32),
            "task_yaw_bounds": np.asarray(THETA_SAMPLING_BOUNDS, dtype=np.float32),
        },
        "acceptance": {
            "grasp_contact_threshold": float(args.grasp_contact_threshold),
            "grasp_alignment_threshold": float(args.grasp_alignment_threshold),
            "position_tolerance": float(args.acceptance_pos_tol),
            "yaw_tolerance": float(args.acceptance_yaw_tol),
            "require_grasped": bool(args.require_grasped),
            "require_yaw_match": bool(args.require_yaw_match),
        },
        "attempt_counts": stats["attempt_counts"],
        "rejected_counts": stats["rejected_counts"],
        "quality": {
            "gripper_height_min": float(np.min(dataset["gripper_height"])),
            "gripper_height_max": float(np.max(dataset["gripper_height"])),
            "gripper_height_mean": float(np.mean(dataset["gripper_height"])),
            "block_pos_error_mean": float(np.mean(dataset["block_pos_error"])),
            "block_pos_error_max": float(np.max(dataset["block_pos_error"])),
            "block_yaw_error_mean": float(np.mean(dataset["block_yaw_error"])),
            "block_yaw_error_max": float(np.max(dataset["block_yaw_error"])),
            "gripper_contact_mean": float(np.mean(dataset["gripper_contact"])),
            "grasp_alignment_error_mean": float(np.mean(dataset["grasp_alignment_error"])),
        },
        "reference_grasped": cube_is_grasped(
            reference_info,
            contact_threshold=float(args.grasp_contact_threshold),
            alignment_threshold=float(args.grasp_alignment_threshold),
        ),
    }
    save_json(out_dir / args.summary_name, summary)

    print(f"Saved dataset:            {out_dir / args.out_name}")
    print(f"Saved summary:            {out_dir / args.summary_name}")
    print(f"Saved reference image:    {out_dir / DEFAULT_REFERENCE_IMAGE_NAME}")
    print(f"Saved XY diagnostic:      {out_dir / DEFAULT_XY_PLOT_NAME}")
    print(f"Saved height histogram:   {out_dir / DEFAULT_HIST_PLOT_NAME}")


if __name__ == "__main__":
    main()
