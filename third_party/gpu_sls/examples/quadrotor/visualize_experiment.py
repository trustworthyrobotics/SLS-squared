import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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


PROJECTION_AXES = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}


def _normalize_plan_array(arr, name: str):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{name} has shape {arr.shape}. Expected (n_steps, N+1, 2).")
    return arr


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
    filename: str | None = "quadrotor_3d_rollouts_projection.png",
    dpi: int = 300,
    projection: str = "xy",
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "3D Quadrotor: Rollouts + Robust Tube + Obstacle Centers",
):
    """
    Static projected plot with obstacle centers, tube rectangles, and rollout trajectories.

    Expected shapes:
      xs:        (n_rollouts, T, n) OR (T, n), with n >= 3 for 3D quadrotor
      plans_xy:  (n_steps, N+1, 2) OR (N+1, 2)  (optional)
      lowers_xy: (n_steps, N+1, 2) OR (N+1, 2)  (optional)
      uppers_xy: (n_steps, N+1, 2) OR (N+1, 2)  (optional)
      centers:   (K, 2)                         (optional)
      radii:     (K,)                           (optional)

    Notes:
      - For the 3D quadrotor state:
          0: px, 1: py, 2: pz, 3: phi, 4: theta, 5: psi,
          6: vx, 7: vy, 8: vz, 9: p, 10: q, 11: r
      - xs is projected using `projection` unless you already pass preprojected plans/tubes.
      - plans_xy / lowers_xy / uppers_xy are assumed already projected into 2D.
    """
    xs = np.asarray(xs)

    if projection not in PROJECTION_AXES:
        raise ValueError(f"projection must be one of {list(PROJECTION_AXES.keys())}, got {projection!r}")

    ax_i, ax_j, default_x_label, default_y_label = PROJECTION_AXES[projection]
    if x_label is None:
        x_label = default_x_label
    if y_label is None:
        y_label = default_y_label

    # Normalize xs to (n_rollouts, T, n)
    if xs.ndim == 2:
        if xs.shape[1] < max(ax_i, ax_j) + 1:
            raise ValueError(f"xs has shape {xs.shape}. Expected last dim large enough for projection {projection}.")
        xs = xs[None, :, :]
    elif xs.ndim == 3:
        if xs.shape[2] < max(ax_i, ax_j) + 1:
            raise ValueError(f"xs has shape {xs.shape}. Expected xs[..., :] large enough for projection {projection}.")
    else:
        raise ValueError(f"xs has shape {xs.shape}. Expected 2D or 3D array.")

    n_rollouts, T, nx = xs.shape

    plans_xy = _normalize_plan_array(plans_xy, "plans_xy")
    lowers_xy = _normalize_plan_array(lowers_xy, "lowers_xy")
    uppers_xy = _normalize_plan_array(uppers_xy, "uppers_xy")

    if centers is not None:
        centers = np.asarray(centers)
        if centers.ndim == 1:
            centers = centers[None, :]
        if centers.shape[-1] != 2:
            raise ValueError(f"centers has shape {centers.shape}. Expected (K, 2).")

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

    # axis limits (nan-aware)
    all_x = [xs[:, :, ax_i].ravel()]
    all_y = [xs[:, :, ax_j].ravel()]

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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # obstacles
    if centers is not None and centers.size and radii is not None and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            circ = plt.Circle(
                (float(c[0]), float(c[1])),
                float(r),
                alpha=0.5,
                color="tab:red",
            )
            ax.add_patch(circ)

    # tubes
    if lo is not None and up is not None:
        for k in range(0, lo.shape[0], max(1, int(tube_stride))):
            w = up[k, 0] - lo[k, 0]
            h = up[k, 1] - lo[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            rect = Rectangle(
                (lo[k, 0], lo[k, 1]),
                w,
                h,
                facecolor=PALETTE["tube_face"],
                edgecolor=PALETTE["tube_edge"],
                alpha=tube_alpha,
            )
            ax.add_patch(rect)
        ax.plot([], [], color=PALETTE["tube_face"], alpha=tube_alpha, label=f"Tube boxes (step {step_idx})")

    # plan
    if show_plan and plans_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, plans_xy.shape[0] - 1))
        ax.plot(
            plans_xy[step_idx, :, 0],
            plans_xy[step_idx, :, 1],
            linestyle="--",
            linewidth=2,
            color=PALETTE["plan"],
            label="Planned (open-loop)",
        )

    # rollouts
    for i in range(n_rollouts):
        ax.plot(
            xs[i, :, ax_i],
            xs[i, :, ax_j],
            alpha=rollout_alpha,
            color=PALETTE["random"],
        )
    ax.plot([], [], alpha=rollout_alpha, color=PALETTE["random"], label=f"Rollouts (n={n_rollouts})")

    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_tube_graph_quadrotor(
    disturbed,
    tube,
    dt,
    filename: str = "disturbance_vs_tube_size_quadrotor_3d.png",
    state_labels: list | None = None,
):
    """
    Plot deviation vs tube size for quadrotor states.

    disturbed: (n_rollouts, T, n_states)
    tube:      (T+1, n_states)
    """
    disturbed = np.asarray(disturbed)
    tube = np.asarray(tube)

    n_states = disturbed.shape[2]

    if disturbed.ndim != 3:
        raise ValueError(
            f"disturbed has shape {disturbed.shape}. Expected (n_rollouts, T, n_states)."
        )
    if tube.ndim != 2 or tube.shape[1] != n_states:
        raise ValueError(
            f"tube has shape {tube.shape}. Expected (T+1, {n_states})."
        )

    default_labels_6 = [
        ("px", "meters"), ("pz", "meters"), ("theta", "radians"),
        ("vx", "m/s"),    ("vz", "m/s"),    ("omega", "rad/s"),
    ]
    default_labels_12 = [
        ("px", "meters"), ("py", "meters"), ("pz", "meters"),
        ("phi", "radians"), ("theta", "radians"), ("psi", "radians"),
        ("vx", "m/s"), ("vy", "m/s"), ("vz", "m/s"),
        ("p", "rad/s"), ("q", "rad/s"), ("r", "rad/s"),
    ]

    if state_labels is None:
        if n_states == 6:
            state_labels = default_labels_6
        elif n_states == 12:
            state_labels = default_labels_12
        else:
            state_labels = [(f"x{i}", "units") for i in range(n_states)]

    T = disturbed.shape[1]
    tube_trim = tube[1:, :]
    if tube_trim.shape[0] != T:
        raise ValueError(
            f"tube[1:] has length {tube_trim.shape[0]}, but disturbed time dimension is {T}."
        )

    t = np.arange(T) * dt

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2 * n_states + 2), sharex=True)
    if n_states == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        tube_i = tube_trim[:, idx]
        dev_all = disturbed[:, :, idx]

        ax.plot(t, tube_i, label=f"tube size ({state_labels[idx][0]})", linewidth=3)

        for r_idx, dev in enumerate(dev_all):
            m = np.isfinite(dev)
            ax.plot(
                t[m],
                dev[m],
                label=f"|{state_labels[idx][0]} - nominal|" if r_idx == 0 else None,
                alpha=0.8,
            )

        ax.set_ylabel(state_labels[idx][1])
        ax.set_title(f"{state_labels[idx][0]}: Deviation vs Tube Size")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)