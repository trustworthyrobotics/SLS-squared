from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, scipy, vmap
from jax.tree_util import register_pytree_node_class

from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.dual_tvlqr import dual_lqr
from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import rollout_gpu

class ACPScanCache(NamedTuple):
    Ar:  jnp.ndarray
    Al:  jnp.ndarray
    ArC: jnp.ndarray
    AlP: jnp.ndarray
    Cn:  jnp.ndarray
    Pn:  jnp.ndarray

@register_pytree_node_class
@dataclass(frozen=True)
class ADMMConfig:
    rho_update_frequency: int = 25
    max_iterations: int = 400
    eps_abs: float = 1e-2
    eps_rel: float = 1e-2
    rho_max: int = 1e5
    initial_rho: int = 1.0

    def tree_flatten(self):
        children = (self.rho_update_frequency, self.max_iterations, self.eps_abs, self.eps_rel, self.rho_max, self.initial_rho)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

def _shift_down(x, step):
    pad = jnp.zeros((step,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([pad, x[:-step]], axis=0)

def _mask_upsweep(T, step):
    t = jnp.arange(T)
    period = 2 * step
    return (t % period) == (period - 1)

def _mask_downsweep(T, step):
    t = jnp.arange(T)
    period = 2 * step
    return (t >= (3 * step - 1)) & ((t % period) == (step - 1))

def _masked_write_level(cache, level, vals, mask):
    m = mask.astype(vals.dtype)[:, None, None]
    return cache.at[level].set(vals * m)

def _combine_acp_all(next_block, prev_block, n):
    dtype = prev_block.dtype
    I = jnp.eye(n, dtype=dtype)[None, :, :]

    A_l = prev_block[:, 0:n, :]
    C_l = prev_block[:, n:2*n, :]
    P_l = prev_block[:, 2*n:3*n, :]

    A_r = next_block[:, 0:n, :]
    C_r = next_block[:, n:2*n, :]
    P_r = next_block[:, 2*n:3*n, :]

    # inv1 = jnp.linalg.inv(I + C_l @ P_r)
    # inv2 = jnp.linalg.inv(I + P_r @ C_l)

    # Ar = A_r @ inv1
    # Al = jnp.swapaxes(A_l, -1, -2) @ inv2
    M1 = I + C_l @ P_r
    M2 = I + P_r @ C_l
    Ar = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(M1, -1, -2),
                                  jnp.swapaxes(A_r, -1, -2)), -1, -2)

    Al = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(M2, -1, -2), A_l), -1, -2)

    ArC = Ar @ C_l
    AlP = Al @ P_r

    A_new = Ar @ A_l
    C_new = ArC @ jnp.swapaxes(A_r, -1, -2) + C_r
    P_new = AlP @ A_l + P_l

    combined = jnp.concatenate([A_new, C_new, P_new], axis=1)
    return combined, Ar, Al, ArC, AlP, C_new, P_new


@partial(jax.jit, static_argnums=(1, 2, 3))
def associative_scan_cache_acp_jax(elems_acp, T: int, n: int, reverse: bool = False):
    dtype = elems_acp.dtype
    L = int(math.ceil(math.log2(max(T, 1))))

    Ar  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Al  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    ArC = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    AlP = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Cn  = jnp.zeros((2 * L, T, n, n), dtype=dtype)
    Pn  = jnp.zeros((2 * L, T, n, n), dtype=dtype)

    x = elems_acp[::-1] if reverse else elems_acp
    out = x

    # Upsweep
    step = 1
    level = 0
    while step < T and level < L:
        mask = _mask_upsweep(T, step)
        next_block = _shift_down(out, step)
        prev_block = out

        combined, aR, aL, aRC, aLP, cN, pN = _combine_acp_all(next_block, prev_block, n)

        m = mask[:, None, None]
        out = jnp.where(m, combined, out)

        Ar  = _masked_write_level(Ar,  level, aR,  mask)
        Al  = _masked_write_level(Al,  level, aL,  mask)
        ArC = _masked_write_level(ArC, level, aRC, mask)
        AlP = _masked_write_level(AlP, level, aLP, mask)
        Cn  = _masked_write_level(Cn,  level, cN,  mask)
        Pn  = _masked_write_level(Pn,  level, pN,  mask)

        step *= 2
        level += 1

    # Downsweep-style fill
    step //= 4
    level2 = 0
    while step >= 1 and level2 < L:
        mask = _mask_downsweep(T, step)
        next_block = _shift_down(out, step)
        prev_block = out

        combined, aR, aL, aRC, aLP, cN, pN = _combine_acp_all(next_block, prev_block, n)

        m = mask[:, None, None]
        out = jnp.where(m, combined, out)

        lvl = L + level2
        Ar  = _masked_write_level(Ar,  lvl, aR,  mask)
        Al  = _masked_write_level(Al,  lvl, aL,  mask)
        ArC = _masked_write_level(ArC, lvl, aRC, mask)
        AlP = _masked_write_level(AlP, lvl, aLP, mask)
        Cn  = _masked_write_level(Cn,  lvl, cN,  mask)
        Pn  = _masked_write_level(Pn,  lvl, pN,  mask)

        step //= 2
        level2 += 1

    if reverse:
        out = out[::-1]

    return out, ACPScanCache(Ar=Ar, Al=Al, ArC=ArC, AlP=AlP, Cn=Cn, Pn=Pn)


@partial(jax.jit, static_argnums=(2, 4))
def associative_scan_use_cache_cp_jax(c, p, T: int, cache: ACPScanCache, reverse: bool = False):
    Ar, Al, ArC, AlP = cache.Ar, cache.Al, cache.ArC, cache.AlP
    L = Ar.shape[0] // 2

    c_out = c[::-1] if reverse else c
    p_out = p[::-1] if reverse else p

    # Upsweep
    step = 1
    level = 0
    while step < T and level < L:
        mask = _mask_upsweep(T, step)
        m = mask.astype(c_out.dtype)[:, None]

        c_l = c_out
        p_l = p_out
        c_r = _shift_down(c_out, step)
        p_r = _shift_down(p_out, step)

        c_new = (Ar[level]  @ c_l[..., None]).squeeze(-1) - (ArC[level] @ p_r[..., None]).squeeze(-1) + c_r
        p_new = (Al[level]  @ p_r[..., None]).squeeze(-1) + (AlP[level] @ c_l[..., None]).squeeze(-1) + p_l

        c_out = c_out + m * (c_new - c_out)
        p_out = p_out + m * (p_new - p_out)

        step *= 2
        level += 1

    # Downsweep
    step //= 4
    level2 = 0
    while step >= 1 and level2 < L:
        mask = _mask_downsweep(T, step)
        m = mask.astype(c_out.dtype)[:, None]
        lvl = L + level2

        c_l = c_out
        p_l = p_out
        c_r = _shift_down(c_out, step)
        p_r = _shift_down(p_out, step)

        c_new = (Ar[lvl]  @ c_l[..., None]).squeeze(-1) - (ArC[lvl] @ p_r[..., None]).squeeze(-1) + c_r
        p_new = (Al[lvl]  @ p_r[..., None]).squeeze(-1) + (AlP[lvl] @ c_l[..., None]).squeeze(-1) + p_l

        c_out = c_out + m * (c_new - c_out)
        p_out = p_out + m * (p_new - p_out)

        step //= 2
        level2 += 1

    if reverse:
        c_out = c_out[::-1]
        p_out = p_out[::-1]

    return c_out, p_out

def admm_augment_xu(Q, q, R, r, M, C, D, w_bar, y_bar, rho):
    s_bar = w_bar - y_bar   

    CtC = jnp.einsum('tmi,tmj->tij', C, C)
    DtD = jnp.einsum('tmi,tmj->tij', D, D)
    CtD = jnp.einsum('tmi,tmj->tij', C, D)

    Ct_s = jnp.einsum('tmi,tm->ti', C, s_bar)
    Dt_s = jnp.einsum('tmi,tm->ti', D, s_bar)

    tilde_Q = Q + rho * CtC
    tilde_q = q - rho * Ct_s

    tilde_R = R + rho * DtD
    tilde_r = r - rho * Dt_s

    tilde_M = M + rho * CtD

    return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M

# def admm_residuals(z, w, w_prev, y, rho, eps_abs=1e-2, eps_rel=1e-2):
#     """
#     z, w, w_prev, y: (T+1, m)
#     Returns scalar norms and (optional) thresholds.
#     """
#     r = z - w                          # primal residual
#     s = rho * (w - w_prev)             # dual residual (A=I)

#     # Norms over all time/constraints
#     r_norm = jnp.linalg.norm(r.reshape(-1), ord=2)
#     s_norm = jnp.linalg.norm(s.reshape(-1), ord=2)

#     # Stopping thresholds (Boyd et al., scaled form)
#     n = r.size
#     z_norm = jnp.linalg.norm(z.reshape(-1), ord=2)
#     w_norm = jnp.linalg.norm(w.reshape(-1), ord=2)
#     y_norm = jnp.linalg.norm(y.reshape(-1), ord=2)

#     eps_pri = jnp.sqrt(n) * eps_abs + eps_rel * z_norm
#     eps_dual = jnp.sqrt(n) * eps_abs + eps_rel * (rho * y_norm)

#     return r_norm, s_norm, eps_pri, eps_dual

def admm_residuals(z, w, w_prev, y, rho, eps_abs=1e-2, eps_rel=1e-2):
    r = z - w
    s = rho * (w - w_prev)

    r_norm = jnp.linalg.norm(r.reshape(-1), ord=jnp.inf)
    s_norm = jnp.linalg.norm(s.reshape(-1), ord=jnp.inf)

    z_norm = jnp.linalg.norm(z.reshape(-1), ord=jnp.inf)
    w_norm = jnp.linalg.norm(w.reshape(-1), ord=jnp.inf)
    y_norm = jnp.linalg.norm(y.reshape(-1), ord=jnp.inf)

    eps_pri = eps_abs + eps_rel * jnp.maximum(z_norm, w_norm)
    eps_dual = eps_abs + eps_rel * (rho * y_norm)

    return r_norm, s_norm, eps_pri, eps_dual

def adaptive_rho_update(rp_norm, rd_norm, rho,
                        clip_min=0.2, clip_max=5,
                        rho_min=1e-4, rho_max=1e5,
                        eps=1e-12):
    """
    Adaptive rho update using residual ratio directly as scaling,
    clipped to a bounded range.
    """

    # Residual ratio as scaling factor
    # scale = rp_norm / (rd_norm + eps)

    # # Clip scaling factor
    # scale = jnp.clip(scale, clip_min, clip_max)

    # # Update rho with hard bounds
    # rho_new = jnp.clip(rho * scale, rho_min, rho_max)


    rd_eff = jnp.maximum(rd_norm, 1e-10)
    # scale = jnp.sqrt(rp_norm / rd_eff)
    scale = jnp.sqrt(rp_norm / rd_eff)
    # scale = rp_norm / rd_eff
    # scale = jnp.clip(scale, 0.5, 2.0)
    rho_new = jnp.clip(rho * scale, rho_min, rho_max)
    updated = rho_new != rho
    return rho_new, updated
    
    
# def adaptive_rho_update(rp, rd, rho,
#                         mu=10.0, tau=5.0,
#                         rho_min=1e-3, rho_max=1e5):
#     inc = rp > mu * rd
#     dec = rd > mu * rp
#     rho_new = jnp.where(inc, rho * tau, rho)
#     rho_new = jnp.where(dec, rho / tau, rho_new)
#     rho_new = jnp.clip(rho_new, rho_min, rho_max)
#     updated = rho_new != rho
#     return rho_new, updated


def rho_update_y(rp_norm, rd_norm, rho, y, rho_max):
    rho_new, updated = adaptive_rho_update(rp_norm, rd_norm, rho, rho_max=rho_max)
    y_new = lax.cond(
        updated,
        lambda _: (rho / rho_new) * y,
        lambda _: y,
        operand=None
    )
    return rho_new, y_new, updated

def generate_leaf(tilde_Q, tilde_R, tilde_M, A, B, reg=1e-8):
    T = tilde_Q.shape[0] - 1
    n = tilde_Q.shape[1]
    nu = tilde_R.shape[-1]
    # def chol_inv(t):
    #     Rt = 0.5 * (tilde_R[t] + tilde_R[t].T)
    #     Rt = Rt + 1e-8 * jnp.eye(Rt.shape[0], dtype=Rt.dtype)
    #     I = jnp.eye(tilde_R[t].shape[0])
    #     Rinv = jnp.linalg.solve(Rt, I)
    #     return Rinv
    #     # f = scipy.linalg.cho_factor(tilde_R[t])
    #     # m = tilde_R[t].shape[0]
    #     # return scipy.linalg.cho_solve(f, jnp.eye(m))

    # Rinv = vmap(chol_inv)(jnp.arange(T))
    # BRinv = vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))
    # MRinv = vmap(lambda t: tilde_M[t] @ Rinv[t])(jnp.arange(T))

    def make_R(t):
        Rt = 0.5 * (tilde_R[t] + tilde_R[t].T)
        Rt = Rt + reg * jnp.eye(nu, dtype=Rt.dtype)
        return Rt

    def solve_right(Rt, X):
        return jnp.linalg.solve(Rt.T, X.T).T

    def one(t):
        Rt = make_R(t)
        BR = solve_right(Rt, B[t])
        MR = solve_right(Rt, tilde_M[t])

        return BR, MR

    BRinv, MRinv = vmap(one)(jnp.arange(T))

    elems = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    A - vmap(lambda t: BRinv[t] @ tilde_M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            jnp.concatenate(
                [
                    vmap(lambda t: BRinv[t] @ B[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            tilde_Q
            - jnp.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ tilde_M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
        ],
        axis=1,
    )

    return elems, BRinv, MRinv

def generate_leaf_bp(c, BRinv, MRinv, tilde_r, tilde_q, T, n):
    c_stage = c - vmap(lambda t: BRinv[t] @ tilde_r[t])(jnp.arange(T))
    c0 = jnp.concatenate([c_stage, jnp.zeros((1, n), dtype=c.dtype)], axis=0)

    p_stage = vmap(lambda t: MRinv[t] @ tilde_r[t])(jnp.arange(T))
    p0 = tilde_q - jnp.concatenate([p_stage, jnp.zeros((1, n), dtype=tilde_q.dtype)], axis=0)

    return c0, p0

def get_k(tilde_R, tilde_r, B, P, p, b):
    T = B.shape[0]
    def one(t):
        BtP = B[t].T @ P[t + 1]
        G = tilde_R[t] + BtP @ B[t]
        h = B[t].T @ p[t + 1] + BtP @ b[t] + tilde_r[t]
        return scipy.linalg.solve(G, -h)
    return vmap(one)(jnp.arange(T))

def get_K(tilde_R, tilde_M, A, B, P):
    T = B.shape[0]
    def one(t):
        BtP = B[t].T @ P[t + 1]
        H  = BtP @ A[t] + tilde_M[t].T
        G  = tilde_R[t] + BtP @ B[t]
        return scipy.linalg.solve(G, -H)
    return vmap(one)(jnp.arange(T))

def constrained_solve(cfg: ADMMConfig, Q, q, R, r, M, A, B, c, C, D, f, w, y, rho):
    rho_max = cfg.rho_max
    def one_iter(carry):
        (it, tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, 
         x_bar, u_bar, y_bar, w_prev, rho, cache, BRinv, MRinv, P, _, K,
         _, _, _, _, _) = carry


        # -------- Solve unconstrained LQR subproblem --------
        T = Q.shape[0] - 1
        n = Q.shape[1]
        c0, p0 = generate_leaf_bp(c[1:], BRinv, MRinv, tilde_r, tilde_q, T, n)
        b, p = associative_scan_use_cache_cp_jax(c0, p0, T + 1, cache, reverse=True)
        k = get_k(tilde_R, tilde_r, B, P, p, c[1:])

        x_bar, u_stage = rollout_gpu(K, k, c[0], A, B, c[1:])
        u_bar = jnp.pad(u_stage, ((0, 1), (0, 0)))

        z_bar = (
            jnp.einsum('tmi,ti->tm', C, x_bar) +
            jnp.einsum('tmi,ti->tm', D, u_bar)
        )
        # -------- Project onto constraint set -------- 
        w_new = jnp.minimum(z_bar + y_bar, f)

        # # -------- Dual update (scaled form) -------- 
        y_new = y_bar + (z_bar - w_new)


        # -------- Termination + Rho/Cache Update -------- 
        rp_norm, rd_norm, eps_pri, eps_dual = admm_residuals(
            z_bar, w_new, w_prev, y_new, rho,
            eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel
        )

        # Convergence check
        converged = jnp.logical_and(rp_norm <= eps_pri, rd_norm <= eps_dual)
        do_rho_update = (it % cfg.rho_update_frequency) == 0

        def update_fn(_):
            rho_upd, y_upd, updated = rho_update_y(
                rp_norm, rd_norm,
                rho, y_new, rho_max
            )
            return rho_upd, y_upd, updated

        def no_update_fn(_):
            return rho, y_new, jnp.array(False)

        rho_new, y_new, rho_updated = lax.cond(do_rho_update, update_fn, no_update_fn, operand=None)
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M = admm_augment_xu(
            Q, q, R, r, M, C, D, w_new, y_new, rho_new
        )

        def cache_update(_):
            Tp1 = Q.shape[0]
            n   = Q.shape[1]

            elems_acp, BRinv, MRinv = generate_leaf(tilde_Q, tilde_R, tilde_M, A, B)
            out_acp, cache = associative_scan_cache_acp_jax(elems_acp, Tp1, n, reverse=True)

            P = out_acp[:, -n:, :]
            K = get_K(tilde_R, tilde_M, A, B, P)
            return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache, BRinv, MRinv, P, K

        def no_cache_update(_):
            return tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache, BRinv, MRinv, P, K

        # jax.debug.print(
        #     "ADMM: it={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e})",
        #     it, rho, rp_norm, eps_pri, rd_norm, eps_dual
        # )
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, cache_new, BRinv, MRinv, P, K = lax.cond(rho_updated, cache_update, no_cache_update, operand=None)

        return (it + 1, tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M, x_bar, u_bar, y_new, w_new,
                rho_new, cache_new, BRinv, MRinv, P, p, K,
                rp_norm, rd_norm, eps_pri, eps_dual, converged)

    # --- loop condition: keep going until max_iters OR converged ---
    def cond_fun(carry):
        it = carry[0]
        converged = carry[-1]
        return jnp.logical_and(it < cfg.max_iterations, jnp.logical_not(converged))
    T = A.shape[0]
    n = Q.shape[1]
    nx = Q.shape[-1]
    nu = R.shape[-1]
    f = f - cfg.eps_abs
    R = jnp.concatenate([R, jnp.zeros((1, nu, nu), dtype=R.dtype)], axis=0)
    r = jnp.concatenate([r, jnp.zeros((1, nu), dtype=r.dtype)], axis=0)
    M = jnp.concatenate([M, jnp.zeros((1, nx, nu), dtype=M.dtype)], axis=0)
    init_x = jnp.zeros((T + 1, nx), dtype=Q.dtype)
    init_u = jnp.zeros((T + 1, nu), dtype=Q.dtype)
    init_w = w
    init_y = y
    rho0 = rho
    p_init = jnp.zeros((T + 1, nx), dtype=Q.dtype)
    tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M = admm_augment_xu(
        Q, q, R, r, M, C, D, init_w, init_y, rho0
    )
    elems_acp, BRinv, MRinv = generate_leaf(tilde_Q, tilde_R, tilde_M, A, B)
    out_acp, cache = associative_scan_cache_acp_jax(elems_acp, T + 1, n, reverse=True)
    P = out_acp[:, -n:, :]
    K = get_K(tilde_R, tilde_M, A, B, P)
    init = (
        jnp.array(1, dtype=jnp.int32),
        tilde_Q, tilde_q, tilde_R, tilde_r, tilde_M,
        init_x, init_u, init_y, init_w,                          
        jnp.array(rho0, dtype=Q.dtype),
        cache, BRinv, MRinv, P, p_init, K,
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(jnp.inf, dtype=Q.dtype),
        jnp.array(False)
    )

    out = jax.lax.while_loop(cond_fun, one_iter, init)

    it, _, _, _, _, _, x_bar, u_bar, y_bar, w_bar, rho_final, _, _, _, P_final, p_final, K, rp_norm, rd_norm, eps_pri, eps_dual, converged = out
    v = dual_lqr(x_bar, P_final, p_final)
    jax.debug.print(
        "ADMM done: Total Iterations={} converged={} rho={:.3e} rp={:.3e} (<= {:.3e}) rd={:.3e} (<= {:.3e}) Rho0 {:.3e}",
        it - 1, converged, rho_final, rp_norm, eps_pri, rd_norm, eps_dual, rho0
    )
    mu = rho_final * y_bar
    return x_bar, u_bar[:-1], v, w_bar, y_bar, rho_final, mu, converged