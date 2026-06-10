#!/usr/bin/env python3
"""Animate saved OGBench cube SLS latent tubes from tube_data.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_saved_latent_tubes import (
    FIGURE_GREEN,
    FIGURE_GREEN_DARK,
    alpha_for_order,
    data_axis_limits,
    fill_tube_with_horizon_fade,
    load_ellipsoid_axis_limits,
    parse_dims,
    resolve_tube_data,
    select_plan_indices,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 300,
    }
)


def draw_tube_plan(
    ax,
    *,
    plan_step: int,
    center: np.ndarray,
    width: np.ndarray,
    time_scale: float,
    fill_alpha: float,
    horizon_alpha_decay: float,
    fill_color: str,
    line_color: str,
    line_alpha: float,
    line_width: float,
    draw_center_line: bool,
    clip_after_x: float | None = None,
) -> None:
    valid = np.isfinite(center) & np.isfinite(width)
    if not np.any(valid):
        return
    horizon_x = (int(plan_step) + np.arange(center.shape[0])) * time_scale
    if clip_after_x is not None:
        valid = valid & (horizon_x <= float(clip_after_x))
        if not np.any(valid):
            return
    horizon_x = horizon_x[valid]
    center = center[valid]
    width = np.maximum(width[valid], 0.0)
    fill_tube_with_horizon_fade(
        ax,
        horizon_x,
        center - width,
        center + width,
        color=fill_color,
        base_alpha=fill_alpha,
        horizon_alpha_decay=horizon_alpha_decay,
    )
    if draw_center_line:
        ax.plot(
            horizon_x,
            center,
            color=line_color,
            linestyle=":",
            linewidth=line_width,
            alpha=line_alpha,
        )


def render_animation(args: argparse.Namespace, dims: list[int], out_path: Path) -> Path:
    tube_path = resolve_tube_data(args.tube_data)
    data = np.load(tube_path, allow_pickle=False)
    plan_steps = np.asarray(data["plan_steps"], dtype=np.int64)
    centers = np.asarray(data["nominal_centers"], dtype=np.float64)
    widths = np.asarray(data["tube_widths"], dtype=np.float64)
    executed = np.asarray(data["executed_markov_states"], dtype=np.float64)
    state_dim = int(np.asarray(data["state_dim"]))
    dt = float(args.dt if args.dt is not None else np.asarray(data["control_timestep"]))

    selected = select_plan_indices(
        plan_steps,
        start_step=args.start_step,
        plan_stride=args.plan_stride,
        max_plans=args.max_plans,
    )
    step_to_plan_idx = {int(step): int(idx) for idx, step in enumerate(plan_steps)}

    time_scale = dt if args.use_seconds else 1.0
    x_label = "Time [s]" if args.use_seconds else "MPC step"
    exec_x = np.arange(executed.shape[0]) * time_scale

    stacked_for_limits = np.concatenate(
        [
            executed,
            centers.reshape(-1, state_dim),
            (centers - widths).reshape(-1, state_dim),
            (centers + widths).reshape(-1, state_dim),
        ],
        axis=0,
    )
    data_low, data_high = data_axis_limits(stacked_for_limits, args.data_axis_padding)
    axis_low = None
    axis_high = None
    if args.axis_scale == "ellipsoid":
        try:
            axis_low, axis_high = load_ellipsoid_axis_limits(args.ellipsoid_path, state_dim, args.axis_padding)
            if not args.strict_ellipsoid_axis:
                axis_low = np.minimum(axis_low, data_low)
                axis_high = np.maximum(axis_high, data_high)
        except Exception as exc:
            if not args.allow_axis_fallback:
                raise
            print(f"[WARN] Falling back to data-scaled axes because ellipsoid scaling failed: {exc}")
    if axis_low is None or axis_high is None:
        axis_low, axis_high = data_low, data_high

    all_dims_selected = dims == list(range(state_dim))
    first_half_selected = dims == list(range(state_dim // 2))
    compact_dims_selected = state_dim == 24 and (all_dims_selected or first_half_selected)
    n_cols = 6 if compact_dims_selected else min(3, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    panel_width = 2.45 if compact_dims_selected else 4.4
    panel_height = 1.35 if compact_dims_selected else 2.65

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(out_path, fps=args.fps, quality=args.quality, macro_block_size=1)
    try:
        for frame_step in range(executed.shape[0]):
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(panel_width * n_cols, panel_height * n_rows),
                dpi=args.dpi,
                sharex=True,
            )
            axes = np.atleast_1d(axes).reshape(-1)
            persistent = [int(idx) for idx in selected if int(plan_steps[idx]) <= frame_step]
            current_idx = step_to_plan_idx.get(frame_step)
            show_current = (
                current_idx is not None
                and args.show_current_plan
                and frame_step < executed.shape[0] - 1
            )
            current_start_x = frame_step * time_scale if show_current else None
            if show_current and current_idx is not None:
                persistent = [plan_idx for plan_idx in persistent if plan_idx != current_idx]

            for panel_idx, dim in enumerate(dims):
                ax = axes[panel_idx]
                ax.plot(
                    exec_x[: frame_step + 1],
                    executed[: frame_step + 1, dim],
                    color=FIGURE_GREEN_DARK,
                    linewidth=2.0,
                    label="executed" if panel_idx == 0 else None,
                )

                for order, plan_idx in enumerate(persistent):
                    alpha = alpha_for_order(order, len(selected), args.alpha, args.alpha_decay, args.alpha_mode)
                    draw_tube_plan(
                        ax,
                        plan_step=int(plan_steps[plan_idx]),
                        center=centers[plan_idx, :, dim],
                        width=widths[plan_idx, :, dim],
                        time_scale=time_scale,
                        fill_alpha=alpha,
                        horizon_alpha_decay=args.horizon_alpha_decay,
                        fill_color=FIGURE_GREEN,
                        line_color=FIGURE_GREEN_DARK,
                        line_alpha=0.55,
                        line_width=1.0,
                        draw_center_line=False,
                        clip_after_x=current_start_x,
                    )

                if show_current and current_idx is not None:
                    draw_tube_plan(
                        ax,
                        plan_step=int(plan_steps[current_idx]),
                        center=centers[current_idx, :, dim],
                        width=widths[current_idx, :, dim],
                        time_scale=time_scale,
                        fill_alpha=args.current_alpha,
                        horizon_alpha_decay=args.horizon_alpha_decay,
                        fill_color=args.current_color,
                        line_color=args.current_line_color,
                        line_alpha=0.85,
                        line_width=1.2,
                        draw_center_line=True,
                    )

                ax.text(
                    0.97,
                    0.91,
                    f"Dim. {dim}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=13,
                    color="0.15",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
                )
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.set_ylim(float(axis_low[dim]), float(axis_high[dim]))
                if args.ylim is not None:
                    lo, hi = (float(v) for v in args.ylim.split(","))
                    ax.set_ylim(lo, hi)

            for ax in axes[len(dims) :]:
                ax.axis("off")

            axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
            y_label = args.ylabel
            if y_label is None:
                y_label = "Markovian Latent Rollout" if all_dims_selected else "Latent Rollout"
            x_label_y = 0.005 if first_half_selected else (0.025 if compact_dims_selected else 0.02)
            y_label_x = 0.004 if first_half_selected else (0.012 if compact_dims_selected else 0.01)
            fig.supxlabel(x_label, y=x_label_y, fontsize=13)
            fig.supylabel(y_label, x=y_label_x, fontsize=13)

            if first_half_selected:
                fig.subplots_adjust(left=0.038, right=0.995, bottom=0.22, top=0.975, wspace=0.34, hspace=0.30)
            elif compact_dims_selected:
                fig.subplots_adjust(left=0.055, right=0.995, bottom=0.105, top=0.985, wspace=0.34, hspace=0.28)
            else:
                fig.tight_layout(rect=(0.04, 0.055, 1.0, 1.0))

            fig.canvas.draw()
            width_px, height_px = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height_px, width_px, 4)
            rgb = frame[:, :, :3]
            pad_h = rgb.shape[0] % 2
            pad_w = rgb.shape[1] % 2
            if pad_h or pad_w:
                rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            writer.append_data(rgb.copy())
            plt.close(fig)
    finally:
        writer.close()

    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tube_data", type=Path, help="Path to a tube_data.npz file or its run directory.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for generated animations.")
    parser.add_argument("--out", type=Path, default=None, help="Output path when --dims is supplied.")
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--plan-stride", type=int, default=5)
    parser.add_argument("--max-plans", type=int, default=7)
    parser.add_argument("--dims", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--alpha-decay", type=float, default=0.86)
    parser.add_argument("--horizon-alpha-decay", type=float, default=0.89)
    parser.add_argument("--alpha-mode", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--show-current-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--current-alpha", type=float, default=0.34)
    parser.add_argument("--current-color", type=str, default="#fdae6b")
    parser.add_argument("--current-line-color", type=str, default="#d95f0e")
    parser.add_argument("--axis-scale", choices=("ellipsoid", "data"), default="ellipsoid")
    parser.add_argument(
        "--ellipsoid-path",
        type=Path,
        default=Path("ogbench_cube/eval/latent_ellipsoid/latent_ellipsoid.npz"),
    )
    parser.add_argument("--axis-padding", type=float, default=1.05)
    parser.add_argument("--data-axis-padding", type=float, default=0.08)
    parser.add_argument("--strict-ellipsoid-axis", action="store_true")
    parser.add_argument("--allow-axis-fallback", action="store_true")
    parser.add_argument("--use-seconds", action="store_true")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--ylim", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--quality", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tube_path = resolve_tube_data(args.tube_data)
    data = np.load(tube_path, allow_pickle=False)
    state_dim = int(np.asarray(data["state_dim"]))
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else tube_path.parent

    if args.dims is not None or args.out is not None:
        dims = parse_dims(args.dims, state_dim)
        out_path = args.out
        if out_path is None:
            out_path = out_dir / f"animated_latent_tubes_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.mp4"
        saved = render_animation(args, dims, out_path)
        print(f"Saved latent tube animation to {saved}")
        return

    specs = [
        ("all_dims", list(range(state_dim))),
        ("first_half", list(range(state_dim // 2))),
    ]
    for suffix, dims in specs:
        out_path = out_dir / f"ogbench_animated_latent_tubes_{suffix}_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.mp4"
        saved = render_animation(args, dims, out_path)
        print(f"Saved latent tube animation to {saved}")


if __name__ == "__main__":
    main()
