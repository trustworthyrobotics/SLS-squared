import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


# --- Styling palette (muted, readable) ---
PALETTE = {
    "plan":      "#1f77b4",   # blue
    "random":    "#ff7f0e",   # orange
    "adversary": "#d62728",   # red
    "tube_face": "#2ca02c",   # green
    "tube_edge": "#1b7f1b",   # darker green edge
    "obs_face":  "#7f7f7f",   # gray
    "obs_edge":  "#4d4d4d",   # dark gray edge
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
    ],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def plot_rollouts_tubes_centers(
    xs,
    centers=None,
    radii=None,
    plans_xy=None,
    lowers_xy=None,
    uppers_xy=None,
    step_idx: int | None = 0,
    tube_stride: int = 2,
    tube_alpha: float = 0.15,
    rollout_alpha: float = 0.35,
    show_plan: bool = True,
    margin: float = 0.5,
    filename: str | None = "rollouts_tubes_centers.png",
    dpi: int = 300,
):
    """
    Static plot with obstacle centers, tube rectangles, and all rollout trajectories.

    Expected shapes:
      xs:        (n_rollouts, T, 3) OR (T, 3)
      plans_xy:  (n_steps, N+1, 2)  (optional)
      lowers_xy: (n_steps, N+1, 2)  (optional)
      uppers_xy: (n_steps, N+1, 2)  (optional)
      centers:   (K, 2)             (optional)
      radii:     (K,)               (optional)
    """
    xs = np.asarray(xs)

    # Normalize xs to (n_rollouts, T, 3)
    if xs.ndim == 2 and xs.shape[1] == 3:
        xs = xs[None, :, :]
    elif xs.ndim == 2 and xs.shape[1] != 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected last dim=3.")
    elif xs.ndim == 3 and xs.shape[2] != 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected xs[...,2] to be theta.")
    elif xs.ndim != 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected 2D or 3D array.")

    n_rollouts, T, _ = xs.shape

    if plans_xy is not None:
        plans_xy = np.asarray(plans_xy)
    if lowers_xy is not None:
        lowers_xy = np.asarray(lowers_xy)
    if uppers_xy is not None:
        uppers_xy = np.asarray(uppers_xy)

    if centers is not None:
        centers = np.asarray(centers)
        if centers.ndim == 1:
            centers = centers[None, :]
    if radii is not None:
        radii = np.asarray(radii).reshape(-1)

    # pick tube/plan frame
    if lowers_xy is not None and uppers_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, lowers_xy.shape[0] - 1))
        lo = lowers_xy[step_idx]
        up = uppers_xy[step_idx]
    else:
        lo = up = None

    # axis limits (use nan-aware because rollouts may be padded with NaN)
    all_x = [xs[:, :, 0].ravel()]
    all_y = [xs[:, :, 1].ravel()]

    if plans_xy is not None:
        all_x.append(plans_xy[:, :, 0].ravel())
        all_y.append(plans_xy[:, :, 1].ravel())
    if lo is not None and up is not None:
        all_x.append(lo[:, 0].ravel())
        all_x.append(up[:, 0].ravel())
        all_y.append(lo[:, 1].ravel())
        all_y.append(up[:, 1].ravel())
    if centers is not None and centers.size:
        all_x.append(centers[:, 0].ravel())
        all_y.append(centers[:, 1].ravel())

    all_x = np.concatenate(all_x) if len(all_x) else np.array([0.0])
    all_y = np.concatenate(all_y) if len(all_y) else np.array([0.0])

    xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
    ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # obstacles
    if centers is not None and centers.size and radii is not None and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), alpha=0.5, color="tab:red"))

    # tubes
    if lo is not None and up is not None:
        for k in range(0, lo.shape[0], max(1, int(tube_stride))):
            w = up[k, 0] - lo[k, 0]
            h = up[k, 1] - lo[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            rect = Rectangle((lo[k, 0], lo[k, 1]), w, h, alpha=tube_alpha)
            ax.add_patch(rect)
        ax.plot([], [], alpha=tube_alpha, label=f"Tube boxes (step {step_idx})")

    # plan
    if show_plan and plans_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, plans_xy.shape[0] - 1))
        ax.plot(
            plans_xy[step_idx, :, 0],
            plans_xy[step_idx, :, 1],
            linestyle="--",
            linewidth=2,
            label="Planned (open-loop)",
        )

    # rollouts (nan-padded -> line breaks automatically)
    for i in range(n_rollouts):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=rollout_alpha, color="tab:orange")
    ax.plot([], [], alpha=rollout_alpha, label=f"Rollouts (n={n_rollouts})")

    ax.set_title("Dubins: Rollouts + Robust Tube + Obstacle Centers")
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_tube_graph(disturbed, tube, dt):
    # -----------------------------
    # Deviation vs tube size plots (nan-safe)
    # -----------------------------
    dx_np_all  = disturbed[:, :, 0]
    dy_np_all  = disturbed[:, :, 1]
    dth_np_all = disturbed[:, :, 2]

    tube_x_np  = np.asarray(tube[1:, 0])
    tube_y_np  = np.asarray(tube[1:, 1])
    tube_th_np = np.asarray(tube[1:, 2])

    t = np.arange(dx_np_all.shape[1]) * dt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # ---- X direction ----
    ax1.plot(t, tube_x_np, label="tube size (x)", linewidth=4)
    for r, dx_np in enumerate(dx_np_all):
        m = np.isfinite(dx_np)
        ax1.plot(t[m], dx_np[m], label="|x - x_nominal|" if r == 0 else None)
    ax1.set_ylabel("meters")
    ax1.set_title("X-direction: Deviation vs Tube Size")
    ax1.grid(True)
    ax1.legend()

    # ---- Y direction ----
    ax2.plot(t, tube_y_np, label="tube size (y)", linewidth=4)
    for r, dy_np in enumerate(dy_np_all):
        m = np.isfinite(dy_np)
        ax2.plot(t[m], dy_np[m], label="|y - y_nominal|" if r == 0 else None)
    ax2.set_ylabel("meters")
    ax2.set_title("Y-direction: Deviation vs Tube Size")
    ax2.grid(True)
    ax2.legend()

    # ---- Theta direction ----
    ax3.plot(t, tube_th_np, label="tube size (theta)", linewidth=4)
    for r, dth_np in enumerate(dth_np_all):
        m = np.isfinite(dth_np)
        ax3.plot(t[m], dth_np[m], label="|wrap(theta - theta_nominal)|" if r == 0 else None)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("radians")
    ax3.set_title("Theta-direction: Deviation vs Tube Size")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("disturbance_vs_tube_size_xytheta_dubins.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

