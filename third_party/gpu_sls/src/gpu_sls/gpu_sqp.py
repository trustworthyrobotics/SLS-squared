from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from trajax.optimizers import linearize, quadratize, vectorize

from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.optimizers import (
    line_search,
    merit_rho,
    model_evaluator_helper,
    slope,
)
from gpu_sls.gpu_admm import ADMMConfig, constrained_solve
from gpu_sls.gpu_sls import SLSConfig, sls_solve_gpu


@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True

    def tree_flatten(self):
        children = (self.max_sqp_iterations, self.feas_tol, self.step_tol, self.warm_start, self.line_search)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# TODO: Add constraints to this?
def lagrangian(cost, dynamics, x0):
    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun

@jax.jit
def add_obstacle_constraints(C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                             obstacles: jnp.ndarray, x_curr: jnp.ndarray, eps=1e-5):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _,  _, nu = D.shape

    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    pos = x_curr[:, :2]
    diff = pos[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + eps
    n = diff / dist[..., None]
    coeffs = -n

    C_obstacle = jnp.zeros((Tp1, centers.shape[0], nx), dtype=C.dtype)
    D_obstacle = jnp.zeros((Tp1, centers.shape[0], nu), dtype=D.dtype)

    C_obstacle = C_obstacle.at[..., 0:2].set(coeffs)

    f_obstacle = (dist - radii[None, :]).astype(f.dtype)

    C_all = jnp.concatenate([C, C_obstacle], axis=1)
    D_all = jnp.concatenate([D, D_obstacle], axis=1)
    f_all = jnp.concatenate([f, f_obstacle], axis=1)
    
    return C_all, D_all, f_all

def merit_function_factory(rho_merit):
    def merit_fn(V, g, c):
        return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c)
    return merit_fn

@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def compute_search_direction(
    sls_config: SLSConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    obstacles,
    x0, X, U, V, c,
    w, y, rho,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws,
    sqp_iteration
):
    T = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc = w.shape[1]
    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), jnp.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), jnp.arange(T + 1))

    R = R_pad[:-1]
    M = M_pad[:-1]

    linearizer = linearize(lagrangian(cost, dynamics, x0), argnums=5)
    dynamics_linearizer = linearize(dynamics)
    q, r_pad = linearizer(X, pad(U), jnp.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]
    A_pad, B_pad = dynamics_linearizer(X, pad(U), jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    pad = lambda A: jnp.pad(A, ((0, 1), (0, 0)))
    U_pad = pad(U)
    t = jnp.arange(X.shape[0])
    g = vectorize(constraints)(X, U_pad, t)
    f = -g
    C, D = linearize(constraints)(X, U_pad, t)
    C_all, D_all, f_all = add_obstacle_constraints(C, D, f, obstacles, X)
    E = disturbance(X, U_pad) # <-- PASSES BOTH STATE AND ACTION

    if sls_config.Q_bar is None:
        nx = Q.shape[1]
        Q_bar_2d = jnp.eye(nx)

        # 2. Modify the first half (this works because it's a normal array, not a broadcast view)
        half_nx = nx // 2
        Q_bar_2d = Q_bar_2d.at[:half_nx, :half_nx].set(1.0 * jnp.eye(half_nx))

        # 3. Add the batch dimension at the front to match Q.shape
        # (Assuming Q.shape is (batch_size, nx, nx))
        batch_size = Q.shape[0]
        Q_bar = jnp.tile(Q_bar_2d, (batch_size, 1, 1))
    else:
        Q_bar = sls_config.Q_bar
    if sls_config.R_bar is None:
        R_bar = jnp.broadcast_to(jnp.eye(R.shape[1]), R.shape)
    else:        
        R_bar = sls_config.R_bar

    n_obs = obstacles.shape[0]

    def run_nominal(_):
        dX, dU, dV, w1, y1, rho1, _, converged_admm = constrained_solve(
            admm_config, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho
        )
        backoffs = jnp.zeros((T + 1, nc - n_obs))
        Phi_x   = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u   = jnp.zeros((T, T + 1, nu, nx))
        betaN   = jnp.ones((T + 1, T + 1, nc - n_obs)) * 1e-10
        muN     = jnp.zeros((T + 1, nc))
        return dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN

    def run_sls(_):
        dX, dU, dV, w1, y1, rho1, converged, converged_admm, backoffs, Phi_x, Phi_u, betaN, muN = sls_solve_gpu(
            admm_config,
            Q, q, R, r, M, A, B, c,
            C_all, D_all, f_all, w, y, rho, sls_config,
            E, Q_bar, R_bar, obstacles, X, h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws
        )
        return dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN

    use_nominal = jnp.logical_or(
        jnp.logical_not(sls_config.enable_fastsls),
        jnp.logical_and(sls_config.enable_fastsls, sqp_iteration < sls_config.max_initial_sqp_iterations)
    )
    dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN = lax.cond(
        use_nominal, run_nominal, run_sls, operand=None
    )

    return dX, dU, dV, q, r, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN


@partial(jit, static_argnums=(0,1,2,3,4,5,6,7))
def sqp(
    sls_config: SLSConfig, sqp_config: SQPConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    reference, parameter,
    W,
    x0, X_in, U_in, V_in,
    w, y, rho,
    obstacles,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws,
):
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    _dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics, x0)

    def body(i, carry):
        i, X_curr, U_curr, V_curr, w, y, rho, converged, backoffs, Phi_x, Phi_u, beta_ws, mu_w = carry

        def do_nothing(_):
            return carry

        def do_iter(_):
            g, c = model_evaluator(X_curr, U_curr)
            feas = jnp.max(jnp.abs(c))
            warm_flag = jnp.array(bool(sqp_config.warm_start))

            w0   = lax.select(warm_flag, w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            # TODO: Make the defualt rho a parameter
            rho0 = lax.select(warm_flag, rho, jnp.asarray(admm_config.initial_rho, dtype=rho.dtype))
            h_ct_ws = backoffs
            dX, dU, dV, q, r, w1, y1, rho1, backoffs1, Phi_x1, Phi_u1, betaN, muN = compute_search_direction(
                sls_config, admm_config,
                _cost, _dynamics, _hessian_approx,
                constraints, disturbance,
                obstacles,
                x0, X_curr, U_curr, V_curr, c,
                w0, y0, rho0,
                h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, i
            )

            step = jnp.maximum(
                jnp.max(jnp.abs(dX)),
                jnp.max(jnp.abs(dU))
            )
            z_norm = jnp.maximum(
                jnp.max(jnp.abs(X_curr)),
                jnp.max(jnp.abs(U_curr))
            )

            feas_ok = feas <= sqp_config.feas_tol
            step_ok = step <= sqp_config.step_tol * (1.0 + z_norm)
            # jax.debug.print("SQP Iteration {} Feas {} (<= {}) Step {} (<= {})", i, feas, sqp_config.feas_tol, step, sqp_config.step_tol)
            converged1 = jnp.logical_and(feas_ok, step_ok)
            X_next = lax.select(converged1, X_curr, X_curr + dX)
            U_next = lax.select(converged1, U_curr, U_curr + dU)
            V_next = lax.select(converged1, V_curr, V_curr + dV)

            g, c = model_evaluator(X_curr, U_curr)

            rho_merit = merit_rho(c, dV)
            merit_fn  = merit_function_factory(rho_merit)
            current_merit = merit_fn(V_curr, g, c)
            merit_slope = slope(dX, dU, dV, c, q, r, rho_merit)
            last_iter = (i == (sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations - 1))
            do_ls = jnp.logical_and(jnp.array(bool(sqp_config.line_search)), jnp.logical_not(last_iter))

            def ls_branch(_):
                Xn, Un, Vn, g_new, c_new, ok = line_search(
                    merit_fn, model_evaluator,
                    X_curr, U_curr, V_curr,
                    dX, dU, dV,
                    current_merit, g, c,
                    merit_slope,
                    armijo_factor=1e-4,
                    alpha_0=1.0,
                    alpha_mult=0.5,
                    alpha_min=1e-6,
                )
                return Xn, Un, Vn

            def fullstep_branch(_):
                return (X_curr + dX, U_curr + dU, V_curr + dV)

            X_next, U_next, V_next = lax.cond(do_ls, ls_branch, fullstep_branch, operand=None)

            w_next = lax.select(converged1, w, w1)
            y_next = lax.select(converged1, y, y1)
            rho_next = lax.select(converged1, rho, rho1)
            rho_next = jnp.minimum(rho, admm_config.initial_rho)
            y_next = rho / rho_next * y_next
            backoffs_next = lax.select(converged1, backoffs, backoffs1)
            Phi_x_next = lax.select(converged1, Phi_x, Phi_x1)
            Phi_u_next = lax.select(converged1, Phi_u, Phi_u1)

            return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
                    jnp.logical_or(converged, converged1),
                    backoffs_next, Phi_x_next, Phi_u_next, betaN, muN)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    backoffs0 = h_ct_ws
    carry0 = (0, X_in, U_in, V_in, w, y, rho, jnp.array(False), backoffs0, Phi_x_ws, Phi_u_ws, beta_ws, mu_ws)
    total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out, converged, backoffs, Phi_x, Phi_u, betaN, muN = lax.fori_loop(
        0, sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations, body, carry0
    )
    return X_out, U_out, V_out, w_out, y_out, rho_out, backoffs, Phi_x, Phi_u, betaN, muN
