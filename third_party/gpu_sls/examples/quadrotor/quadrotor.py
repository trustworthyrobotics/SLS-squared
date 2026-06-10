from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import config

import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_control_box_constraints, make_state_box_constraints, make_constant_disturbance
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_rollouts_tubes_centers, plot_tube_graph_quadrotor

config.update("jax_enable_x64", False)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.25  # meters in xyz

def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def reached_goal_xyz(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dpos = x[:3] - x_goal[:3]
    return (dpos @ dpos) <= (tol * tol)

# -----------------------------
# 3D quadrotor parameters
# x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]
# u = [T, tau_phi, tau_theta, tau_psi]
# -----------------------------
MASS = 1.0
GRAVITY = 9.81

JX = 0.02
JY = 0.02
JZ = 0.04
J = jnp.diag(jnp.array([JX, JY, JZ], dtype=jnp.float64))
J_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=jnp.float64))

NUM_RANDOM = 5
NUM_ADV = 26


def rotation_matrix(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)

    # R = Rz(psi) Ry(theta) Rx(phi)
    return jnp.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                          cth * cphi],
    ], dtype=jnp.float64)

def euler_angle_rates_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    tth = jnp.tan(theta)
    cth = jnp.cos(theta)

    return jnp.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi,       -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=jnp.float64)

def rigid_body_3d_step(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x
    T, tau_phi, tau_theta, tau_psi = u

    v = jnp.array([vx, vy, vz], dtype=x.dtype)
    omega = jnp.array([p, q, r], dtype=x.dtype)
    tau = jnp.array([tau_phi, tau_theta, tau_psi], dtype=x.dtype)

    R = rotation_matrix(phi, theta, psi)
    E = euler_angle_rates_matrix(phi, theta)

    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)

    # Translational dynamics
    pos_dot = v
    v_dot = (R @ (T * e3)) / MASS - GRAVITY * e3

    # Rotational dynamics
    euler_dot = E @ omega
    omega_dot = J_INV @ (tau - jnp.cross(omega, J @ omega))

    x_dot = jnp.concatenate([pos_dot, euler_dot, v_dot, omega_dot], axis=0)
    x_next = x + dt * x_dot

    x_next = x_next.at[3].set(wrap_to_pi(x_next[3]))  # phi
    x_next = x_next.at[4].set(wrap_to_pi(x_next[4]))  # theta
    x_next = x_next.at[5].set(wrap_to_pi(x_next[5]))  # psi

    return x_next

def quadrotor_step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,      # (12,)
    u: jnp.ndarray,      # (4,)
    E: jnp.ndarray,      # (12,12)
    dt: float,
    i: int
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    x_{k+1} = f(x_k, u_k) + E w, with ||w||_2 <= 1
    """
    x_nom = rigid_body_3d_step(x, u, dt)

    # Random disturbance in unit ball
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (x.shape[0],), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(x.shape[0], dtype=x.dtype)
    a = jnp.asarray(0.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = r * z

    # Deterministic adversarial-ish directions
    start = i - NUM_RANDOM + 5
    if 5 <= start <= 16:
        idx = start - 5
        w = jnp.zeros((12,), dtype=x.dtype).at[idx].set(1.0)
    if 17 <= start <= 28:
        idx = start - 17
        w = jnp.zeros((12,), dtype=x.dtype).at[idx].set(-1.0)
    if start == 29:
        w = jnp.ones((12,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(12.0, dtype=x.dtype))
    if start == 30:
        w = -jnp.ones((12,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(12.0, dtype=x.dtype))

    x_next = x_nom + E @ w
    x_next = x_next.at[3].set(wrap_to_pi(x_next[3]))
    x_next = x_next.at[4].set(wrap_to_pi(x_next[4]))
    x_next = x_next.at[5].set(wrap_to_pi(x_next[5]))
    return key, x_next, w

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dt = parameter
    return rigid_body_3d_step(x, u, dt)

def cost(W, reference, x, u, t):
    """
    W =
    [wpx, wpy, wpz,
     wphi, wtheta, wpsi,
     wvx, wvy, wvz,
     wp, wq, wr,
     wT, wtau_phi, wtau_theta, wtau_psi]
    """
    (
        wpx, wpy, wpz,
        wphi, wtheta, wpsi,
        wvx, wvy, wvz,
        wp, wq, wr,
        wT, wtau_phi, wtau_theta, wtau_psi
    ) = W

    xref = reference[t]

    dpos = x[:3] - xref[:3]
    dang = wrap_to_pi(x[3:6] - xref[3:6])
    dvel = x[6:9] - xref[6:9]
    drates = x[9:12] - xref[9:12]

    T_hover = MASS * GRAVITY
    du = jnp.array([
        u[0] - T_hover,
        u[1],
        u[2],
        u[3],
    ], dtype=x.dtype)

    angle_cost = (
        wphi * (1.0 - jnp.cos(dang[0]))
        + wtheta * (1.0 - jnp.cos(dang[1]))
        + wpsi * (1.0 - jnp.cos(dang[2]))
    )

    return (
        wpx * dpos[0] ** 2
        + wpy * dpos[1] ** 2
        + wpz * dpos[2] ** 2
        + angle_cost
        + wvx * dvel[0] ** 2
        + wvy * dvel[1] ** 2
        + wvz * dvel[2] ** 2
        + wp * drates[0] ** 2
        + wq * drates[1] ** 2
        + wr * drates[2] ** 2
        + wT * du[0] ** 2
        + wtau_phi * du[1] ** 2
        + wtau_theta * du[2] ** 2
        + wtau_psi * du[3] ** 2
    )

def build_piecewise_reference(x0: jnp.ndarray, x_goal: jnp.ndarray, N: int, dt: float) -> jnp.ndarray:
    """
    Build a straight-line reference trajectory from x0 to x_goal.

    Only position (px,py,pz) and yaw are interpolated.
    All other states are set to zero reference.
    """

    t = jnp.linspace(0.0, 1.0, N + 1)

    # Linear interpolation for position
    pos = (1.0 - t[:, None]) * x0[:3] + t[:, None] * x_goal[:3]

    # Shortest-path yaw interpolation
    dpsi = wrap_to_pi(x_goal[5] - x0[5])
    psi = wrap_to_pi(x0[5] + t * dpsi)

    X_ref = jnp.zeros((N + 1, 12), dtype=jnp.float64)

    X_ref = X_ref.at[:, :3].set(pos)
    X_ref = X_ref.at[:, 5].set(psi)

    # Optionally compute velocity reference from the line
    vel = (x_goal[:3] - x0[:3]) / (N * dt)
    X_ref = X_ref.at[:, 6:9].set(vel)

    return X_ref

def main():
    # -----------------------------
    # Dimensions
    # -----------------------------
    n = 12
    nu = 4

    # -----------------------------
    # Horizon and dt
    # -----------------------------
    N = 110
    dt = 0.03

    # -----------------------------
    # Cost weights
    # -----------------------------
    W = jnp.array([
        25.0, 25.0, 25.0,     # position
        2.0, 2.0, 0.5,        # roll, pitch, yaw
        0.5, 0.5, 0.5,        # velocities
        0.05, 0.05, 0.05,     # body rates
        0.01, 0.01, 0.01, 0.01  # control
    ], dtype=jnp.float64)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=jnp.float64),
        dt=dt,
    )

    parameter = dt

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    T_max = 2.0 * T_hover
    tau_max = 10.0

    u_min = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=jnp.float64)
    u_max = jnp.array([T_max, tau_max, tau_max, tau_max], dtype=jnp.float64)
    constraints_u = make_control_box_constraints(u_min, u_max)

    # -----------------------------
    # State limits
    # -----------------------------
    x_max = jnp.array([
        15.0, 15.0, 15.0,       # px, py, pz
        jnp.pi / 2.0,           # phi
        jnp.pi / 2.0,           # theta
        10.0 * jnp.pi,          # psi
        5.0, 5.0, 5.0,          # vx, vy, vz
        8.0, 8.0, 8.0           # p, q, r
    ], dtype=jnp.float64)
    x_min = -x_max
    x_min = x_min.at[2].set(-1.0)

    constraints_x = make_state_box_constraints(x_min, x_max)
    constraints_all = combine_constraints(constraints_x, constraints_u)

    obstacles = jnp.array([
        [0.0, 0.0, 0.35],
    ], dtype=jnp.float64)

    # obstacles = jnp.array([])

    n_obs = obstacles.shape[0]
    nc = 2 * nu + 2 * n + n_obs

    E_mag = 0.03
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    # -----------------------------
    # Initial / goal
    # -----------------------------
    x0 = jnp.array([
        -0.75, -0.75, 0.25,    # px, py, pz
        0.0, 0.0, 0.0,          # phi, theta, psi
        0.0, 0.0, 0.0,          # vx, vy, vz
        0.0, 0.0, 0.0           # p, q, r
    ], dtype=jnp.float64)

    x_goal = jnp.array([
        1.0, 0.8, 0.5,          # px, py, pz
        0.0, 0.0, 0.0,          # phi, theta, psi
        0.0, 0.0, 0.0,          # vx, vy, vz
        0.0, 0.0, 0.0           # p, q, r
    ], dtype=jnp.float64)

    X_ref = build_piecewise_reference(x0, x_goal, N, dt)
    reference = X_ref
    T_steps = N

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(n, dtype=jnp.float64)

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-1,
        eps_rel=1e-2,
        rho_max=1e6,
        max_iterations=400,
        rho_update_frequency=20,
        initial_rho=1.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=False,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=10,
        warm_start=True,
        feas_tol=0.01,
        step_tol=0.0001,
        line_search=True,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, 0].set(T_hover),
    )

    # -----------------------------
    # Robust plan
    # -----------------------------
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )

    # -----------------------------
    # Rollout simulations
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)

    for i in range(N_ROLLOUTS):
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]
        x = x0
        jax.debug.print("Rolling out iteration {}", i)

        for k in range(T_steps):
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback
            u = jnp.clip(u, u_min, u_max)

            key, x, w = quadrotor_step_with_disturbance(key, x, u, E_sim, dt, i)

            err = np.abs(np.asarray(X_pred[k + 1] - x))
            err[3] = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 3] - x[3])))
            err[4] = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 4] - x[4])))
            err[5] = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 5] - x[5])))

            disturbed[i, k, :] = err
            disturbance_history.append(w)
            xs[i, k] = np.asarray(x)

            # if bool(reached_goal_xyz(x, x_goal, GOAL_TOL)):
            #     stop_steps[i] = k
            #     break

    # -----------------------------
    # Tube visualization
    # NOTE:
    # plotting is still 2D, using xy projection
    # -----------------------------
    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    tube = get_trajectory_tubes(Phi_x)

    plan_xy = X_pred[:, :2]       # (px, py)
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    plot_rollouts_tubes_centers(
        xs=xs,   # updated plotting helper can take nx >= 2
        centers=np.asarray(obstacles[:, :2]),
        radii=np.asarray(obstacles[:, 2]),
        plans_xy=np.asarray(plans_xy),
        lowers_xy=np.asarray(lowers_xy),
        uppers_xy=np.asarray(uppers_xy),
        step_idx=0,
        tube_stride=1,
        filename="quadrotor_3d_rollouts_xy_projection.png",
        show_plan=False,
        tube_alpha=0.1,
        margin=0.2,
        rollout_alpha=0.5,
        x_label="x",
        y_label="y",
        title="3D Quadrotor: XY Projection of Rollouts + Tube",
    )

    plot_tube_graph_quadrotor(
        disturbed=disturbed[:, :, :6],   # position + Euler angles only
        tube=tube[:, :6],
        dt=dt,
        filename="quadrotor_3d_disturbance_vs_tube_size_pose.png",
    )


if __name__ == "__main__":
    main()