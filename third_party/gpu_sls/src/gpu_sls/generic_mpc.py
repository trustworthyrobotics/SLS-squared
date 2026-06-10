
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import gpu_sls.gpu_sqp

@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray
    dt: float


class GenericMPC:
    def __init__(
        self,
        sls_config, sqp_config, admm_config,
        config, dynamics, constraints, obstacles,
        cost,
        num_constraints: int,
        disturbance,
        X_in, U_in,
        shift: int = 1,
    ):
        self.sls_config = sls_config
        self.sqp_config = sqp_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift
        self.obstacles = obstacles
        num_obstacles = self.obstacles.shape[0]
        self.h_ct_ws = jnp.zeros((config.N + 1, num_constraints - num_obstacles))
        self.beta_ws = jnp.ones((config.N + 1, config.N + 1, num_constraints - num_obstacles)) * 1e-10
        self.mu_ws = jnp.zeros((config.N + 1, num_constraints))
        self.Phi_x_ws = jnp.zeros((config.N + 1, config.N + 1, config.n, config.n))
        self.Phi_u_ws = jnp.zeros((config.N, config.N + 1, config.nu, config.n))

        self.U0 = U_in
        self.X0 = X_in
        self.V0 = jnp.zeros((config.N + 1, config.n))
        self.w = jnp.zeros((config.N + 1, num_constraints))
        self.y = jnp.zeros((config.N + 1, num_constraints))
        # TODO: Make this a parameter
        self.rho = jnp.asarray(self.admm_config.initial_rho, dtype=self.w.dtype)

        self.dynamics = dynamics
        self.constraints = constraints
        self.cost = cost
        self.disturbance = disturbance

        work = partial(
            gpu_sls.gpu_sqp.sqp,
            self.sls_config, self.sqp_config, self.admm_config,
            cost, dynamics,
            None,
            constraints, disturbance,
        )
        self._solve = jax.jit(work)

    def run(self, x0: jnp.ndarray, reference: jnp.ndarray, parameter: Any):
        X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, betaN, muN = self._solve(
            reference,
            parameter,
            self.config.W,
            x0, self.X0, self.U0, self.V0,
            self.w, self.y, self.rho,
            self.obstacles,
            self.h_ct_ws, self.beta_ws, self.mu_ws, self.Phi_x_ws, self.Phi_u_ws
        )

        s = self.shift

        invalid = (
            jnp.any(~jnp.isfinite(U)) |
            jnp.any(~jnp.isfinite(X)) |
            jnp.any(~jnp.isfinite(V)) |
            jnp.any(~jnp.isfinite(w)) |
            jnp.any(~jnp.isfinite(y)) |
            jnp.any(~jnp.isfinite(backoffs)) |
            jnp.any(~jnp.isfinite(betaN)) |
            jnp.any(~jnp.isfinite(muN)) |
            jnp.any(~jnp.isfinite(Phi_x)) |
            jnp.any(~jnp.isfinite(Phi_u))
        )

        def shift_and_pad(arr, pad_value=None):
            if pad_value is None:
                tail = jnp.tile(arr[-1:], (s,) + (1,) * (arr.ndim - 1))
            else:
                tail = jnp.broadcast_to(
                    pad_value,
                    (s,) + arr.shape[1:]
                )
            return jnp.concatenate([arr[s:], tail], axis=0)

        # ---- primal warm starts ----
        self.U0 = jax.lax.cond(
            invalid,
            lambda _: jnp.tile(self.config.u_ref, (self.config.N, 1)),
            lambda _: shift_and_pad(U),
            operand=None,
        )

        self.X0 = jax.lax.cond(
            invalid,
            lambda _: jnp.tile(x0, (self.config.N + 1, 1)),
            lambda _: shift_and_pad(X),
            operand=None,
        )

        self.V0 = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros((self.config.N + 1, self.config.n), dtype=V.dtype),
            lambda _: shift_and_pad(V),
            operand=None,
        )

        # ---- constraint / tube warm starts ----
        self.h_ct_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.h_ct_ws),
            lambda _: shift_and_pad(backoffs),
            operand=None,
        )

        self.beta_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.ones_like(self.beta_ws) * 1e-10,
            lambda _: shift_and_pad(betaN),
            operand=None,
        )

        self.mu_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.mu_ws),
            lambda _: shift_and_pad(muN),
            operand=None,
        )

        # ---- ADMM-ish dual warm starts ----
        self.w = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.w),
            lambda _: shift_and_pad(w),
            operand=None,
        )

        self.y = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.y),
            lambda _: shift_and_pad(y),
            operand=None,
        )

        rho = jnp.asarray(rho, dtype=self.rho.dtype)

        # Only rescale y if the solve was valid
        self.y = jax.lax.cond(
            invalid,
            lambda _: self.y,
            lambda _: rho / self.rho * self.y,
            operand=None,
        )

        self.rho = jax.lax.cond(
            invalid,
            lambda _: jnp.asarray(self.admm_config.initial_rho, dtype=self.rho.dtype),
            lambda _: rho,
            operand=None,
        )

        self.Phi_x_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.Phi_x_ws),
            lambda _: Phi_x,
            operand=None,
        )

        self.Phi_u_ws = jax.lax.cond(
            invalid,
            lambda _: jnp.zeros_like(self.Phi_u_ws),
            lambda _: Phi_u,
            operand=None,
        )

        return U[0], X, U, V, backoffs, Phi_x, Phi_u