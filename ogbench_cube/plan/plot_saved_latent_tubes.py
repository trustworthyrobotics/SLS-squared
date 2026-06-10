#!/usr/bin/env python3
"""Plot saved OGBench SLS latent tubes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 300,
    }
)

DEFAULT_ELLIPSOID_PATH = Path("ogbench_cube/eval/latent_ellipsoid/latent_ellipsoid.npz")
FIGURE_GREEN = "#2ca25f"
FIGURE_GREEN_DARK = "#167a43"


def parse_dims(value: str | None, state_dim: int) -> list[int]:
    if value is None or value.strip() == "":
        return list(range(state_dim))
    dims: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim < 0 or dim >= state_dim:
            raise ValueError(f"Dimension {dim} is outside [0, {state_dim - 1}].")
        dims.append(dim)
    if not dims:
        raise ValueError("At least one dimension must be selected.")
    return dims


def resolve_tube_data(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "tube_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find tube data file: {path}")
    return path


def select_plan_indices(
    plan_steps: np.ndarray,
    *,
    start_step: int,
    plan_stride: int,
    max_plans: int | None,
) -> np.ndarray:
    if plan_stride <= 0:
        raise ValueError("--plan-stride must be positive.")
    selected = np.flatnonzero(plan_steps >= int(start_step))[:: int(plan_stride)]
    if max_plans is not None:
        selected = selected[: int(max_plans)]
    if selected.size == 0:
        raise ValueError(
            f"No tube plans selected. Available step range is "
            f"{int(plan_steps.min())}..{int(plan_steps.max())}."
        )
    return selected


def alpha_for_order(order: int, total: int, start: float, decay: float, mode: str) -> float:
    if mode == "reverse":
        order = total - order - 1
    return float(max(0.015, min(1.0, start * (decay ** order))))


def fill_tube_with_horizon_fade(
    ax,
    horizon_x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str,
    base_alpha: float,
    horizon_alpha_decay: float,
) -> None:
    if horizon_x.shape[0] < 2:
        ax.fill_between(horizon_x, lower, upper, color=color, alpha=base_alpha, linewidth=0.0)
        return

    for horizon_idx in range(horizon_x.shape[0] - 1):
        segment_alpha = float(
            max(0.01, min(1.0, base_alpha * (float(horizon_alpha_decay) ** horizon_idx)))
        )
        segment_slice = slice(horizon_idx, horizon_idx + 2)
        ax.fill_between(
            horizon_x[segment_slice],
            lower[segment_slice],
            upper[segment_slice],
            color=color,
            alpha=segment_alpha,
            linewidth=0.0,
        )


def load_ellipsoid_axis_limits(path: Path, state_dim: int, padding: float) -> tuple[np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Could not find ellipsoid artifact: {path}")

    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if "markov_unit_precision" in payload:
            unit_precision = np.asarray(payload["markov_unit_precision"], dtype=np.float64)
            center = (
                np.asarray(payload["markov_center"], dtype=np.float64)
                if "markov_center" in payload
                else np.zeros(unit_precision.shape[0], dtype=np.float64)
            )
        elif "unit_precision" in payload:
            unit_precision = np.asarray(payload["unit_precision"], dtype=np.float64)
            center = (
                np.asarray(payload["center"], dtype=np.float64)
                if "center" in payload
                else np.zeros(unit_precision.shape[0], dtype=np.float64)
            )
        else:
            raise KeyError(f"{path} must contain 'markov_unit_precision' or 'unit_precision'.")
    else:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "markov_unit_precision" in payload:
            unit_precision = payload["markov_unit_precision"]
            center = payload.get("markov_center", None)
        elif "unit_precision" in payload:
            unit_precision = payload["unit_precision"]
            center = payload.get("center", None)
        else:
            raise KeyError(f"{path} must contain 'markov_unit_precision' or 'unit_precision'.")
        if hasattr(unit_precision, "detach"):
            unit_precision = unit_precision.detach().cpu().numpy()
        unit_precision = np.asarray(unit_precision, dtype=np.float64)
        if center is None:
            center = np.zeros(unit_precision.shape[0], dtype=np.float64)
        elif hasattr(center, "detach"):
            center = center.detach().cpu().numpy()
        center = np.asarray(center, dtype=np.float64)

    if unit_precision.shape != (state_dim, state_dim):
        raise ValueError(
            f"Expected ellipsoid precision shape {(state_dim, state_dim)}, got {unit_precision.shape}."
        )
    diagonal = np.maximum(np.diag(unit_precision), 1e-12)
    radius = np.sqrt(1.0 / diagonal) * float(padding)
    return center - radius, center + radius


def data_axis_limits(values: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanmin(values, axis=0)
    high = np.nanmax(values, axis=0)
    span = np.maximum(high - low, 1e-6)
    return low - float(padding) * span, high + float(padding) * span


def plot_tubes(args: argparse.Namespace) -> Path:
    tube_path = resolve_tube_data(args.tube_data)
    data = np.load(tube_path, allow_pickle=False)
    plan_steps = np.asarray(data["plan_steps"], dtype=np.int64)
    centers = np.asarray(data["nominal_centers"], dtype=np.float64)
    widths = np.asarray(data["tube_widths"], dtype=np.float64)
    executed = np.asarray(data["executed_markov_states"], dtype=np.float64)
    goal_state = np.asarray(data["goal_state"], dtype=np.float64)
    state_dim = int(np.asarray(data["state_dim"]))
    dt = float(args.dt if args.dt is not None else np.asarray(data["control_timestep"]))

    dims = parse_dims(args.dims, state_dim)
    selected = select_plan_indices(
        plan_steps,
        start_step=args.start_step,
        plan_stride=args.plan_stride,
        max_plans=args.max_plans,
    )

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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for panel_idx, dim in enumerate(dims):
        ax = axes[panel_idx]
        ax.plot(exec_x, executed[:, dim], color=FIGURE_GREEN_DARK, linewidth=2.0, label="executed")
        if args.show_goal:
            ax.axhline(goal_state[dim], color="0.25", linestyle="-", linewidth=0.9, alpha=0.55, label="goal")

        for order, plan_idx in enumerate(selected):
            center = centers[plan_idx, :, dim]
            width = widths[plan_idx, :, dim]
            valid = np.isfinite(center) & np.isfinite(width)
            if not np.any(valid):
                continue
            horizon_x = (plan_steps[plan_idx] + np.arange(center.shape[0])) * time_scale
            horizon_x = horizon_x[valid]
            center = center[valid]
            width = np.maximum(width[valid], 0.0)
            alpha = alpha_for_order(order, len(selected), args.alpha, args.alpha_decay, args.alpha_mode)
            fill_tube_with_horizon_fade(
                ax,
                horizon_x,
                center - width,
                center + width,
                color=FIGURE_GREEN,
                base_alpha=alpha,
                horizon_alpha_decay=args.horizon_alpha_decay,
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
        if not compact_dims_selected and panel_idx % n_cols == 0:
            ax.set_ylabel("state")
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
        fig.subplots_adjust(
            left=0.038,
            right=0.995,
            bottom=0.22,
            top=0.975,
            wspace=0.34,
            hspace=0.30,
        )
    elif compact_dims_selected:
        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            bottom=0.105,
            top=0.985,
            wspace=0.34,
            hspace=0.28,
        )
    else:
        fig.tight_layout(rect=(0.04, 0.055, 1.0, 1.0))

    out_path = args.out
    if out_path is None:
        out_path = tube_path.with_name(
            f"latent_tubes_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.pdf"
        )
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tube_data", type=Path, help="Path to a tube_data.npz file or its run directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output image path.")
    parser.add_argument("--start-step", type=int, default=1, help="First MPC step whose predicted tube can be plotted.")
    parser.add_argument("--plan-stride", type=int, default=5, help="Plot one saved tube plan every this many saved plans.")
    parser.add_argument("--max-plans", type=int, default=None, help="Maximum number of tube plans to overlay.")
    parser.add_argument("--dims", type=str, default=None, help="Comma-separated state dimensions, for example 0,1,2,18.")
    parser.add_argument("--alpha", type=float, default=0.50, help="Initial tube fill alpha.")
    parser.add_argument("--alpha-decay", type=float, default=0.86, help="Multiplicative alpha decay between plotted plans.")
    parser.add_argument(
        "--horizon-alpha-decay",
        type=float,
        default=0.89,
        help="Multiplicative alpha decay along each tube prediction horizon.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=("forward", "reverse"),
        default="forward",
        help="Use forward to fade later plans, reverse to fade earlier plans.",
    )
    parser.add_argument(
        "--axis-scale",
        choices=("ellipsoid", "data"),
        default="ellipsoid",
        help="Scale y axes from the fitted latent ellipsoid or from plotted data.",
    )
    parser.add_argument(
        "--ellipsoid-path",
        type=Path,
        default=DEFAULT_ELLIPSOID_PATH,
        help="Fitted latent ellipsoid artifact used when --axis-scale=ellipsoid.",
    )
    parser.add_argument("--axis-padding", type=float, default=1.05, help="Padding multiplier for ellipsoid axis limits.")
    parser.add_argument("--data-axis-padding", type=float, default=0.08, help="Fractional padding for data-scaled axes.")
    parser.add_argument(
        "--strict-ellipsoid-axis",
        action="store_true",
        help="Use exact ellipsoid-derived y-limits even if plotted tubes extend beyond them.",
    )
    parser.add_argument(
        "--allow-axis-fallback",
        action="store_true",
        help="Fall back to data-scaled axes if ellipsoid scaling cannot be loaded.",
    )
    parser.add_argument("--show-goal", action="store_true", help="Draw a thin solid goal-state reference line.")
    parser.add_argument("--use-seconds", action="store_true", help="Use control_timestep for the x-axis.")
    parser.add_argument("--dt", type=float, default=None, help="Override timestep used with --use-seconds.")
    parser.add_argument("--ylim", type=str, default=None, help="Optional y limits as lo,hi.")
    parser.add_argument("--ylabel", type=str, default=None, help="Shared y-axis label.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=300, help="Saved figure DPI.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dims is not None or args.out is not None:
        out_path = plot_tubes(args)
        print(f"Saved latent tube plot to {out_path}")
        return

    tube_path = resolve_tube_data(args.tube_data)
    data = np.load(tube_path, allow_pickle=False)
    state_dim = int(np.asarray(data["state_dim"]))
    plot_specs = (
        ("all_dims", list(range(state_dim))),
        ("first_half", list(range(state_dim // 2))),
    )
    for suffix, dims in plot_specs:
        plot_args = argparse.Namespace(**vars(args))
        plot_args.dims = ",".join(str(dim) for dim in dims)
        plot_args.out = tube_path.with_name(
            f"ogbench_latent_tubes_{suffix}_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.pdf"
        )
        out_path = plot_tubes(plot_args)
        print(f"Saved latent tube plot to {out_path}")


if __name__ == "__main__":
    main()
