#!/usr/bin/env python3
"""Plot saved Reacher SLS latent tubes."""

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

GREEN = "#2ca25f"
GREEN_DARK = "#167a43"


def resolve_tube_data(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "tube_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find tube data file: {path}")
    return path


def alpha_for_order(order: int, total: int, start: float, decay: float) -> float:
    return float(max(0.015, min(1.0, start * (decay ** order))))


def fill_tube(ax, horizon_x, lower, upper, *, alpha, horizon_alpha_decay):
    if horizon_x.shape[0] < 2:
        ax.fill_between(horizon_x, lower, upper, color=GREEN, alpha=alpha, linewidth=0.0)
        return
    for idx in range(horizon_x.shape[0] - 1):
        seg = slice(idx, idx + 2)
        seg_alpha = float(max(0.01, min(1.0, alpha * (horizon_alpha_decay ** idx))))
        ax.fill_between(horizon_x[seg], lower[seg], upper[seg], color=GREEN, alpha=seg_alpha, linewidth=0.0)


def parse_dims(value: str | None, state_dim: int) -> list[int]:
    if value is None or value.strip() == "":
        return list(range(min(10, state_dim)))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tube_data", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--plan-stride", type=int, default=5)
    parser.add_argument("--max-plans", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--alpha-decay", type=float, default=0.86)
    parser.add_argument("--horizon-alpha-decay", type=float, default=0.89)
    parser.add_argument("--dims", type=str, default=None, help="Comma-separated state dimensions.")
    parser.add_argument("--ylabel", type=str, default=None, help="Shared y-axis label.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tube_path = resolve_tube_data(args.tube_data)
    if args.dims is None and args.out is None:
        data = np.load(tube_path, allow_pickle=False)
        state_dim = int(np.asarray(data["state_dim"]))
        plot_specs = (
            ("2x5", list(range(min(10, state_dim)))),
            ("first_half", list(range(min(5, state_dim)))),
        )
        for suffix, dims in plot_specs:
            plot_args = argparse.Namespace(**vars(args))
            plot_args.dims = ",".join(str(dim) for dim in dims)
            plot_args.out = tube_path.with_name(
                f"reacher_latent_tubes_{suffix}_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.pdf"
            )
            plot_single(plot_args, tube_path)
        return

    plot_single(args, tube_path)


def plot_single(args: argparse.Namespace, tube_path: Path) -> None:
    data = np.load(tube_path, allow_pickle=False)
    plan_steps = np.asarray(data["plan_steps"], dtype=np.int64)
    centers = np.asarray(data["nominal_centers"], dtype=np.float64)
    widths = np.asarray(data["tube_widths"], dtype=np.float64)
    executed = np.asarray(data["executed_markov_states"], dtype=np.float64)

    state_dim = int(np.asarray(data["state_dim"]))
    dims = parse_dims(args.dims, state_dim)
    selected = np.flatnonzero(plan_steps >= args.start_step)[:: args.plan_stride]
    if args.max_plans is not None:
        selected = selected[: args.max_plans]
    if selected.size == 0:
        raise ValueError("No plans selected for plotting.")

    stacked = np.concatenate(
        [
            executed,
            centers.reshape(-1, centers.shape[-1]),
            (centers - widths).reshape(-1, centers.shape[-1]),
            (centers + widths).reshape(-1, centers.shape[-1]),
        ],
        axis=0,
    )
    low = np.nanmin(stacked, axis=0)
    high = np.nanmax(stacked, axis=0)
    span = np.maximum(high - low, 1e-6)
    low = low - 0.08 * span
    high = high + 0.08 * span

    full_markov_dims = dims == list(range(min(10, state_dim)))
    latent_only_dims = dims == list(range(min(5, state_dim)))
    n_cols = 5 if (full_markov_dims or latent_only_dims) else min(5, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    fig_height = 3.0 if full_markov_dims else 1.75
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.45 * n_cols, fig_height), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    exec_x = np.arange(executed.shape[0])
    for panel_idx, dim in enumerate(dims):
        ax = axes[panel_idx]
        ax.plot(exec_x, executed[:, dim], color=GREEN_DARK, linewidth=2.0, label="executed")
        for order, plan_idx in enumerate(selected):
            center = centers[plan_idx, :, dim]
            width = np.maximum(widths[plan_idx, :, dim], 0.0)
            valid = np.isfinite(center) & np.isfinite(width)
            if not np.any(valid):
                continue
            horizon_x = plan_steps[plan_idx] + np.arange(center.shape[0])
            horizon_x = horizon_x[valid]
            center = center[valid]
            width = width[valid]
            fill_tube(
                ax,
                horizon_x,
                center - width,
                center + width,
                alpha=alpha_for_order(order, len(selected), args.alpha, args.alpha_decay),
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
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_ylim(float(low[dim]), float(high[dim]))

    for ax in axes[len(dims) :]:
        ax.axis("off")

    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    y_label = args.ylabel
    if y_label is None:
        y_label = "Latent Rollout" if latent_only_dims else "Markovian Latent Rollout"
    fig.supxlabel("MPC step", y=0.005, fontsize=13)
    fig.supylabel(y_label, x=0.004, fontsize=13)
    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        bottom=0.34 if latent_only_dims else 0.22,
        top=0.96 if latent_only_dims else 0.975,
        wspace=0.34,
        hspace=0.30,
    )

    out_path = args.out
    if out_path is None:
        out_path = tube_path.with_name(f"reacher_latent_tubes_start_{args.start_step:03d}_stride_{args.plan_stride:03d}.pdf")
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved Reacher latent tube plot to {out_path}")


if __name__ == "__main__":
    main()
