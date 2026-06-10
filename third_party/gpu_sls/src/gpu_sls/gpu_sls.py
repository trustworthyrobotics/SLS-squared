from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, vmap

from gpu_sls.gpu_admm import constrained_solve
from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import tvlqr_gpu

@dataclass(frozen=True)
class SLSConfig:
    max_sls_iterations: int = 2
    sls_primal_tol: float = 1e-2
    enable_fastsls: bool = True
    warm_start: bool = True
    rti: bool = False
    initialize_nominal: bool = True
    max_initial_sqp_iterations: int = 0
    Q_bar: jnp.ndarray = None
    R_bar: jnp.ndarray = None


@jax.jit
def controller_pas(Q, R, M, A, B):
    T = Q.shape[0] - 1
    n = Q.shape[1]
    I = jnp.eye(n, dtype=Q.dtype)

    def op(next_elem, prev_elem):
        def decompose(elem):
            A_blk = elem[:n, :]
            C_blk = elem[n:2*n, :]
            P_blk = elem[2*n:3*n, :]
            return A_blk, C_blk, P_blk

        A_l, C_l, P_l = decompose(prev_elem)
        A_r, C_r, P_r = decompose(next_elem)
        X1 = jnp.linalg.solve(I + C_l @ P_r, I)
        X2 = jnp.linalg.solve(I + P_r @ C_l, I)

        ArIClPr = A_r @ X1
        AlTIPrCl = A_l.T @ X2

        A_new = ArIClPr @ A_l
        C_new = ArIClPr @ C_l @ A_r.T + C_r
        P_new = AlTIPrCl @ P_r @ A_l + P_l

        return jnp.concatenate([A_new, C_new, P_new], axis=0)

    def chol_inv(mat):
        f = jsp.linalg.cho_factor(mat, lower=True)
        m = mat.shape[0]
        return jsp.linalg.cho_solve(f, jnp.eye(m, dtype=mat.dtype))

    Rinv = vmap(chol_inv)(R)
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))
    MRinv = vmap(lambda t: M[t] @ Rinv[t])(jnp.arange(T))

    A_bar = A - vmap(lambda t: BRinv[t] @ M[t].T)(jnp.arange(T))
    C_bar = vmap(lambda t: BRinv[t] @ B[t].T)(jnp.arange(T))
    P_bar = Q[:T] - vmap(lambda t: MRinv[t] @ M[t].T)(jnp.arange(T))
    P_T   = Q[T]

    elems = jnp.concatenate(
        [
            jnp.concatenate([A_bar, jnp.zeros((1, n, n), dtype=Q.dtype)], axis=0),
            jnp.concatenate([C_bar, jnp.zeros((1, n, n), dtype=Q.dtype)], axis=0),
            jnp.concatenate([P_bar, P_T[None, :, :]], axis=0),
        ],
        axis=1,
    )

    result = lax.associative_scan(lambda r, l: vmap(op)(r, l), elems, reverse=True)
    P = result[:, 2*n:3*n, :]

    def gain_at(t):
        BtP = B[t].T @ P[t + 1]
        G = R[t] + BtP @ B[t]
        H = BtP @ A[t] + M[t].T
        return -jsp.linalg.solve(G, H, assume_a="pos")

    K = vmap(gain_at)(jnp.arange(T))
    return K

@jax.jit
def calculate_cost(Q_bar, R_bar, C, D, eta):
    eta = jnp.asarray(eta).reshape(-1)
    eta = jnp.maximum(eta, 0.0)

    s = jnp.sqrt(eta)
    Cs = C * s[:, None]
    Ds = D * s[:, None]

    Cx  = Cs.T @ Cs + Q_bar
    Cxu = Cs.T @ Ds
    Cu  = Ds.T @ Ds + R_bar

    return Cx, Cxu, Cu

@jax.jit
def calculate_phis(A, B, Cx, Cxu, Cu, E):
    T = Cu.shape[0]
    nx = A.shape[1]
    nu = B.shape[-1]
    Tp1 = T + 1
    nw = E.shape[-1]
    A = A[:T]
    B = B[:T]
    # def solve_one_j(j):
    #     Qj = Cx[:, j, :, :]
    #     Rj = Cu[:, j, :, :]
    #     Mj = Cxu[:, j, :, :]
    #     K = controller_pas(Qj, Rj, Mj, A, B)
    #     return K
    zeros_q = jnp.zeros((Tp1, nx), dtype=A.dtype)
    zeros_r = jnp.zeros((T,  nu), dtype=A.dtype)
    zeros_c = jnp.zeros((T,  nx), dtype=A.dtype)
    def solve_one_j(j):
        Qj = Cx[:, j, :, :]     # [T+1, nx, nx]
        Rj = Cu[:, j, :, :]     # [T,   nu, nu]
        Mj = Cxu[:, j, :, :]    # [T,   nx, nu]
        K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
        return K   

    K_all = jax.vmap(solve_one_j)(jnp.arange(T))
    K_kj_core = jnp.swapaxes(K_all, 0, 1)
    K_lastcol = jnp.zeros((T, 1, nu, nx), dtype=A.dtype)
    K_kj = jnp.concatenate([K_kj_core, K_lastcol], axis=1)

    BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)
    F  = A[:, None, :, :] + BK

    I = jnp.eye(nx, dtype=A.dtype)
    F = F.at[:, T].set(I)

    t_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    use_F = (t_idx >= j_idx)
    elems = jnp.where(use_F[:, :, None, None], F, I)

    def compose(l, r):
        return jnp.einsum("...ab,...bc->...ac", r, l)

    P = lax.associative_scan(compose, elems, axis=0)
    Phix_1toT = jnp.einsum("tjab,jbn->tjan", P, E)
    Phi_x = jnp.concatenate(
        [jnp.zeros((1, Tp1, nx, nw), dtype=A.dtype), Phix_1toT],
        axis=0
    )

    Phi_x = Phi_x.at[jnp.arange(Tp1), jnp.arange(Tp1)].set(E)
    k_idx_full = jnp.arange(Tp1)[:, None]
    valid_x = (k_idx_full >= j_idx)
    Phi_x = Phi_x * valid_x[:, :, None, None]

    Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])
    k_idx = jnp.arange(T)[:, None]
    valid_u = (k_idx >= j_idx)
    Phi_u = Phi_u * valid_u[:, :, None, None]

    return Phi_x, Phi_u

@jax.jit
def get_controller(Q, R, A, B, C, D, E, eta_stage, eta_f):
    T, nx, _ = A.shape

    js = jnp.arange(T)
    ks = jnp.arange(T)

    def blocks_for_k(k):
        def blocks_for_j(j):
            return calculate_cost(Q[k], R[k], C[k], D[k], eta_stage[k, j])
        return vmap(blocks_for_j)(js)

    Cx_kj, Cxu_kj, Cu_kj = vmap(blocks_for_k)(ks)

    Cterm = C[-1]
    def terminal_Cx_for_j(j):
        w = eta_f[j]
        return (Cterm.T * w[None, :]) @ Cterm + Q[T]

    Cx_Nj = vmap(terminal_Cx_for_j)(jnp.arange(T))
    Cx = jnp.concatenate([Cx_kj, Cx_Nj[None, ...]], axis=0)

    Phi_x, Phi_u = calculate_phis(A, B, Cx, Cxu_kj, Cu_kj, E)
    return Phi_x, Phi_u

@jax.jit
def get_betas(C, D, Phi_x, Phi_u):
    T = Phi_u.shape[0]
    Tp1 = T + 1
    nc = C.shape[1]

    term_x = jnp.einsum("kix,kjxn->kjin", C[:-1], Phi_x[:-1])
    term_u = jnp.einsum("kiu,kjun->kjin", D[:-1], Phi_u)
    gPhi = term_x + term_u
    beta_stage = jnp.sum(gPhi * gPhi, axis=-1)

    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    mask = (j_idx <= k_idx)
    beta_stage = beta_stage * mask[:, :, None]

    gPhi_term = jnp.einsum("ix,jxn->jin", C[-1], Phi_x[-1])
    beta_term = jnp.sum(gPhi_term * gPhi_term, axis=-1)

    beta = jnp.zeros((Tp1, Tp1, nc), dtype=Phi_x.dtype)
    beta = beta.at[:-1, :, :].set(beta_stage)
    beta = beta.at[-1, :, :].set(beta_term)

    return beta

@jax.jit
def get_constraint_tightenings(betas, eps_beta=1e-6):
    T1, _, _ = betas.shape

    s = jnp.sqrt(jnp.maximum(betas, 0.0))

    k_idx = jnp.arange(T1)[:, None]
    j_idx = jnp.arange(T1)[None, :]
    valid = (j_idx <= k_idx)
    s = s * valid[:, :, None]

    h_ct = jnp.sum(s, axis=1)
    h_ct = h_ct + eps_beta
    return h_ct

@jax.jit
def get_etas(mus, betas, eps=1e-12):
    Tp1 = mus.shape[0]
    T = Tp1 - 1

    mu_k = mus[:-1]
    beta_kj = betas[:-1, :-1, :]
    eta = (mu_k[:, None, :] /
           (2.0 * jnp.sqrt(jnp.maximum(beta_kj, eps))))

    k_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(T)[None, :]
    eta = eta * (k_idx >= j_idx)[:, :, None]
    eta = jnp.maximum(eta, 0.0)

    mu_f = mus[-1]
    beta_f = betas[-1, :, :]
    eta_f = (mu_f[None, :] /
             (2.0 * jnp.sqrt(jnp.maximum(beta_f, eps))))
    eta_f = jnp.maximum(eta_f, 0.0)

    return eta, eta_f

@jax.jit
def _scaled_primal_diff(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Returns a scaled infinity-norm difference:
        ||a-b||_inf / max(1, ||b||_inf)
    """
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(1.0, jnp.max(jnp.abs(b)))
    return num / (den + eps)

@jax.jit
def primal_convergence_metric(
    X_new: jnp.ndarray, U_new: jnp.ndarray,
    X_old: jnp.ndarray, U_old: jnp.ndarray
) -> jnp.ndarray:
    mX = _scaled_primal_diff(X_new, X_old)
    mU = _scaled_primal_diff(U_new, U_old)
    return jnp.maximum(mX, mU)

@jax.jit
def add_obstacle_tightenings(
    obstacles: jnp.ndarray,
    primal_pos: jnp.ndarray,
    h_ct: jnp.ndarray,
    tightened_constraints: jnp.ndarray,
    idx_px: int = 0,
    idx_py: int = 1,
    eps: float = 1e-6,
):
    pos = primal_pos[:, :2]
    centers = obstacles[:, :2]
    radii = obstacles[:, 2]

    diff = pos[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + eps
    n = diff / dist[..., None]

    hx = jnp.abs(h_ct[:, idx_px])
    hy = jnp.abs(h_ct[:, idx_py])

    over = jnp.abs(n[..., 0]) * hx[:, None] + jnp.abs(n[..., 1]) * hy[:, None]

    tightened = dist - radii[None, :] - over
    return jnp.concatenate([tightened_constraints, tightened], axis=1)

@partial(jit, static_argnums=(0, 15))
def sls_solve_gpu(cfg, Q: jnp.ndarray, q: jnp.ndarray,
                       R: jnp.ndarray, r: jnp.ndarray,
                       M: jnp.ndarray,
                       A: jnp.ndarray, B: jnp.ndarray, c: jnp.ndarray,
                       C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                       w: jnp.ndarray, y: jnp.ndarray, rho: jnp.ndarray, # ADMM Params
                       sls_config: SLSConfig, E: jnp.ndarray, Q_bar: jnp.ndarray, R_bar: jnp.ndarray,
                       obstacles: jnp.ndarray, primal_pos: jnp.ndarray, h_ct_ws: jnp.ndarray,
                       beta_ws: jnp.ndarray, mu_ws: jnp.ndarray, Phi_x_ws: jnp.ndarray, Phi_u_ws: jnp.ndarray):
    Tp1 = Q.shape[0]
    nx  = Q.shape[1]
    nu  = R.shape[1]
    nc  = w.shape[1]
    num_obstacles = obstacles.shape[0]
    T   = Tp1 - 1

    # beta0 = jnp.ones((Tp1, Tp1, nc - num_obstacles), dtype=Q.dtype) * 1e-10
    # h_ct0 = jnp.zeros((Tp1, nc - num_obstacles))
    x0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)
    u0 = jnp.zeros((T, nu),  dtype=Q.dtype)
    v0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)

    i0 = jnp.array(0, dtype=rho.dtype)
    converged0 = jnp.array(False)

    max_iter = jnp.array(sls_config.max_sls_iterations, dtype=jnp.int32)
    tol = jnp.array(sls_config.sls_primal_tol, dtype=Q.dtype)

    h_ct0 = h_ct_ws
    carry0 = (i0, beta_ws, x0, u0, v0, w, y, rho, converged0, converged0, h_ct0, Phi_x_ws, Phi_u_ws, mu_ws)

    def cond_fn(carry):
        i, _, _, _, _, _, _, _, converged, _, _, _, _, _ = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_fn(carry):
        i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _, h_ct, _, _, mu = carry

        prev_rho = rho
        x_prev = x_curr
        u_prev = u_curr
        # if sls_config.rti:
        #     mu_nominal = mu[: , :-num_obstacles]
        #     eta_stage, eta_f = get_etas(mu_nominal, beta)
        num_regular_constraints = f.shape[1] - num_obstacles
        
        if sls_config.rti:
            # Use the pre-calculated num_regular_constraints to avoid the :-0 bug
            mu_nominal = mu[:, :num_regular_constraints]
            eta_stage, eta_f = get_etas(mu_nominal, beta)
            C_box = C[:, :nc - num_obstacles, :]
            D_box = D[:, :nc - num_obstacles, :]
            Phi_x, Phi_u = get_controller(Q_bar, R_bar, A, B, C_box, D_box, E, eta_stage, eta_f)
            beta = get_betas(C_box, D_box, Phi_x, Phi_u)
            h_ct = get_constraint_tightenings(beta)
        
        tightened_constraints = f[:, :num_regular_constraints] - h_ct
        tightened_constraints_all = add_obstacle_tightenings(obstacles, primal_pos, h_ct, tightened_constraints)
        warm_flag = jnp.array(bool(sls_config.warm_start))

        w   = lax.select(warm_flag, w, jnp.zeros_like(w))
        y   = lax.select(warm_flag, y, jnp.zeros_like(y))
        rho = lax.select(warm_flag, rho, jnp.array(cfg.initial_rho, dtype=rho.dtype))
        x_curr, u_curr, v_curr, w, y, rho, mu, converged_admm = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C, D, tightened_constraints_all, w, y, rho
        )

        metric = primal_convergence_metric(x_curr, u_curr, x_prev, u_prev)
        mu_nominal = mu[: , :num_regular_constraints]
        eta_stage, eta_f = get_etas(mu_nominal, beta)
        C_box = C[:, :num_regular_constraints, :]
        D_box = D[:, :num_regular_constraints, :]
        Phi_x, Phi_u = get_controller(Q_bar, R_bar, A, B, C_box, D_box, E, eta_stage, eta_f)
        beta = get_betas(C_box, D_box, Phi_x, Phi_u)
        h_ct = get_constraint_tightenings(beta) # .at[5:].set(0.0)
        # jax.debug.print("{}", h_ct)
        y = prev_rho / rho * y
        rho = jnp.asarray(rho, dtype=prev_rho.dtype)
        w   = jnp.asarray(w,   dtype=w.dtype)
        y   = jnp.asarray(y,   dtype=y.dtype)
        converged_now = metric <= tol
        converged = jnp.logical_or(converged, converged_now)

        return (i + jnp.array(1, dtype=jnp.int32),
                beta, x_curr, u_curr, v_curr, w, y, rho, converged, converged_admm, h_ct, Phi_x, Phi_u, mu)

    carryN = jax.lax.while_loop(cond_fn, body_fn, carry0)

    _, betaN, xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct, Phi_x, Phi_u, muN = carryN
    return xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct, Phi_x, Phi_u, betaN, muN