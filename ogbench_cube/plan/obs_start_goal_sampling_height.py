#!/usr/bin/env python3
"""Sample OGBench cube grasped start/goal endpoint pairs from fixed x-bands for height-constrained experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

USER_SITE_FRAGMENT = ".local/lib/python"
sys.path = [path for path in sys.path if USER_SITE_FRAGMENT not in path]

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm.auto import tqdm

from ogbench_cube.plan.obs_data_collect_3d_ellipsoid import (
    DEFAULT_CAMERA,
    DEFAULT_CONTROL_DECIMATION,
    DEFAULT_ENV_NAME,
    DEFAULT_SETTLE_STEPS,
    DEFAULT_SIM_FREQ_HZ,
    capture_grasp_reference,
    make_env,
    render_without_target_cube,
    synthesize_grasped_state,
)

DEFAULT_OUT_DIR = Path("ogbench_cube/plan/random_endpoint_pairs_height")
DEFAULT_PLOT_NAME = "start_goal_height_bands.png"
DEFAULT_DATASET_NAME = "start_goal_height.pt"
DEFAULT_NUM_POINTS = 1024
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
FIXED_YAW = 0.0
TABLE_Z = 0.02

X_BOUNDS = (0.30, 0.50)
Y_BOUNDS = (-0.25, 0.25)
START_X_BOUNDS = (0.45, 0.475)
GOAL_X_BOUNDS = (0.325, 0.35)
SHARED_Y_BOUNDS = (-0.2, 0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plot-name", default=DEFAULT_PLOT_NAME)
    parser.add_argument("--out-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--oracle-segment-dt", type=float, default=0.4)
    parser.add_argument("--oracle-noise", type=float, default=0.0)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=0.5)
    parser.add_argument("--max-oracle-steps", type=int, default=80)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--grasp-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=0.03)
    parser.add_argument("--start-x-min", type=float, default=float(START_X_BOUNDS[0]))
    parser.add_argument("--start-x-max", type=float, default=float(START_X_BOUNDS[1]))
    parser.add_argument("--goal-x-min", type=float, default=float(GOAL_X_BOUNDS[0]))
    parser.add_argument("--goal-x-max", type=float, default=float(GOAL_X_BOUNDS[1]))
    parser.add_argument("--y-min", type=float, default=float(SHARED_Y_BOUNDS[0]))
    parser.add_argument("--y-max", type=float, default=float(SHARED_Y_BOUNDS[1]))
    return parser.parse_args()


def validate_bounds(bounds: tuple[float, float], *, name: str) -> tuple[float, float]:
    lower = float(bounds[0])
    upper = float(bounds[1])
    if not lower < upper:
        raise ValueError(f"{name} must satisfy min < max, got {bounds}.")
    return lower, upper


def sample_points(
    rng: np.random.Generator,
    count: int,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> np.ndarray:
    if int(count) <= 0:
        raise ValueError("--num-points must be positive.")
    x = rng.uniform(float(x_bounds[0]), float(x_bounds[1]), size=int(count))
    y = rng.uniform(float(y_bounds[0]), float(y_bounds[1]), size=int(count))
    z = np.full((int(count),), TABLE_Z, dtype=np.float64)
    return np.stack((x, y, z), axis=1)


def save_plot(
    path: Path,
    *,
    start_points: np.ndarray,
    goal_points: np.ndarray,
    start_x_bounds: tuple[float, float],
    goal_x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    x_values = np.linspace(float(X_BOUNDS[0]), float(X_BOUNDS[1]), num=80, dtype=np.float64)
    y_values = np.linspace(float(Y_BOUNDS[0]), float(Y_BOUNDS[1]), num=160, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    plane_z = np.full_like(grid_x, TABLE_Z, dtype=np.float64)

    fig = plt.figure(figsize=(8.0, 6.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        grid_x,
        grid_y,
        plane_z,
        color="#d9d9d9",
        alpha=0.25,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )
    ax.scatter(
        start_points[:, 0],
        start_points[:, 1],
        start_points[:, 2],
        c="#0072b2",
        s=8.0,
        depthshade=False,
        label="start",
    )
    ax.scatter(
        goal_points[:, 0],
        goal_points[:, 1],
        goal_points[:, 2],
        c="#d55e00",
        s=8.0,
        depthshade=False,
        label="goal",
    )
    ax.set_xlim(X_BOUNDS)
    ax.set_ylim(Y_BOUNDS)
    ax.set_zlim(0.0, 0.30)
    ax.set_xlabel("cube x")
    ax.set_ylabel("cube y")
    ax.set_zlabel("cube z")
    ax.view_init(elev=28, azim=-25)
    ax.legend(loc="upper left")
    ax.set_title(
        "OGBench height start/goal endpoint bands\n"
        f"start x in [{start_x_bounds[0]:.3f}, {start_x_bounds[1]:.3f}], "
        f"goal x in [{goal_x_bounds[0]:.3f}, {goal_x_bounds[1]:.3f}], "
        f"y in [{y_bounds[0]:.3f}, {y_bounds[1]:.3f}]"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    # plt.show()
    plt.close(fig)


def render_grasped_images(
    points: np.ndarray,
    *,
    env: object,
    reference: object,
    yaw: float,
    camera: str,
    settle_steps: int,
    seed_offset: int,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for index, point in enumerate(tqdm(points, desc=f"Rendering {camera}", unit="image")):
        env.reset(seed=seed_offset + index)
        synthesize_grasped_state(
            env,
            block_pos=np.asarray(point, dtype=np.float64),
            block_yaw=float(yaw),
            reference=reference,
            settle_steps=int(settle_steps),
        )
        frames.append(render_without_target_cube(env, str(camera)))
    return np.stack(frames, axis=0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    start_x_bounds = validate_bounds((args.start_x_min, args.start_x_max), name="start x bounds")
    goal_x_bounds = validate_bounds((args.goal_x_min, args.goal_x_max), name="goal x bounds")
    y_bounds = validate_bounds((args.y_min, args.y_max), name="y bounds")

    start_points = sample_points(rng, args.num_points, x_bounds=start_x_bounds, y_bounds=y_bounds)
    goal_points = sample_points(rng, args.num_points, x_bounds=goal_x_bounds, y_bounds=y_bounds)

    plot_path = args.out_dir / args.plot_name
    out_path = args.out_dir / args.out_name
    save_plot(
        plot_path,
        start_points=start_points,
        goal_points=goal_points,
        start_x_bounds=start_x_bounds,
        goal_x_bounds=goal_x_bounds,
        y_bounds=y_bounds,
    )

    env = make_env(args)
    try:
        reference, _ = capture_grasp_reference(env, args)
        start_pixels = render_grasped_images(
            start_points,
            env=env,
            reference=reference,
            yaw=FIXED_YAW,
            camera=str(args.camera),
            settle_steps=int(args.settle_steps),
            seed_offset=int(args.seed) + 10_000,
        )
        goal_pixels = render_grasped_images(
            goal_points,
            env=env,
            reference=reference,
            yaw=FIXED_YAW,
            camera=str(args.camera),
            settle_steps=int(args.settle_steps),
            seed_offset=int(args.seed) + 20_000,
        )
    finally:
        env.close()

    payload = {
        "metadata": {
            "seed": int(args.seed),
            "num_points": int(args.num_points),
            "env_name": str(args.env_name),
            "camera": str(args.camera),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "workspace_x_bounds": np.asarray(X_BOUNDS, dtype=np.float32),
            "workspace_y_bounds": np.asarray(Y_BOUNDS, dtype=np.float32),
            "start_x_bounds": np.asarray(start_x_bounds, dtype=np.float32),
            "goal_x_bounds": np.asarray(goal_x_bounds, dtype=np.float32),
            "shared_y_bounds": np.asarray(y_bounds, dtype=np.float32),
            "table_z": float(TABLE_Z),
            "fixed_yaw": float(FIXED_YAW),
            "plot_path": str(plot_path),
        },
        "start": {
            "task_target": start_points.astype(np.float32),
            "yaw": np.full((int(args.num_points),), FIXED_YAW, dtype=np.float32),
            "pixels": start_pixels,
        },
        "goal": {
            "task_target": goal_points.astype(np.float32),
            "yaw": np.full((int(args.num_points),), FIXED_YAW, dtype=np.float32),
            "pixels": goal_pixels,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"Saved plot to {plot_path}")
    print(f"Saved dataset to {out_path}")
    print(f"sampled {start_points.shape[0]} start points and {goal_points.shape[0]} goal points")
    print(f"first start = {np.array2string(start_points[0], precision=4)}")
    print(f"first goal  = {np.array2string(goal_points[0], precision=4)}")


if __name__ == "__main__":
    main()
