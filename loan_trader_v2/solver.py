from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.hermite import hermgauss
from tqdm.auto import tqdm
import pickle
from pathlib import Path

# Try to import original interpolation helpers.
try:
    from src.swap_trader.interpolation import (
        interp3_uniform_numpy,
        build_interp3_stencil,
        interp3_from_stencil,
    )
except ImportError:
    interp3_uniform_numpy = None
    build_interp3_stencil = None
    interp3_from_stencil = None



@dataclass(frozen=True)
class LoanParams:
    # Time
    dt: float = 1.0 / 12.0
    T: float = 10.0

    # Annual rates / percentages
    cost_of_equity_annual: float = 0.10

    # short rate dynamics (annualized)
    mu_annual: float = 0.05
    a_annual: float = 0.35
    sigma_annual: float = 0.012

    # loan portfolio
    alpha_annual: float = 0.20
    s0_annual: float = 0.05
    s_max_annual: float = 0.20
    kappa: float = 0.20

    # capital thresholds
    c0: float = 0.12
    c_star: float = 0.135
    delta: float = 0.8

    # terminal value
    theta_mode: Literal["fixed", "steady_state"] = "steady_state"
    theta_fixed: float | None = None
    theta_spread_ref_annual: float | None = None
    theta_k_ref: float | None = None

    # actions
    g_span_mult: int = 5
    g_step_mult: int = 1
    g_action_multipliers: tuple[float, ...] | None = None
    p_actions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 1.0)

    # grids (annual units)
    r_min_annual: float = 0.0
    r_max_annual: float = 0.12
    c_min: float = 0.0
    c_max: float = 0.20
    k_min: float = 0.0
    k_max: float = 2.0

    def spread(self, g: np.ndarray | float):
        raw = self.s0_annual - self.kappa * (g - self.alpha_annual)
        return np.maximum(np.minimum(raw, self.s_max_annual), -0.01)

    @property
    def N(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def mean_reversion_factor(self) -> float:
        return math.exp(-self.a_annual * self.dt)

    @property
    def alpha_step(self) -> float:
        return self.alpha_annual * self.dt

    @property
    def gamma(self) -> float:
        return math.exp(-self.cost_of_equity_annual * self.dt)

    @property
    def base_sd(self) -> float:
        a = self.a_annual
        s = self.sigma_annual
        dt = self.dt
        return s * math.sqrt((1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a))

    @property
    def r_min(self) -> float:
        return self.r_min_annual

    @property
    def r_max(self) -> float:
        return self.r_max_annual
        
    def annual_to_step(self, x: float | np.ndarray):
        return x * self.dt

    @property
    def g_actions(self) -> np.ndarray:
        """
        Annualized growth rates.
        """
        if self.g_action_multipliers is not None:
            mult = np.asarray(self.g_action_multipliers, dtype=np.float64)
            return mult * self.alpha_annual

        mults = np.arange(
            -self.g_span_mult,
            self.g_span_mult + self.g_step_mult,
            self.g_step_mult,
            dtype=np.float64,
        )
        return mults * self.alpha_annual

    @property
    def theta(self) -> float:
        """
        Terminal franchise value per unit assets.
        """
        if self.theta_mode == "fixed":
            if self.theta_fixed is None:
                raise ValueError("theta_fixed must be set when theta_mode='fixed'")
            return float(self.theta_fixed)

        if self.cost_of_equity_annual <= 0:
            raise ValueError("cost_of_equity_annual must be positive.")

        s_ref = self.s0_annual if self.theta_spread_ref_annual is None else self.theta_spread_ref_annual
        k_ref = self.c_star if self.theta_k_ref is None else self.theta_k_ref

        steady_state_profit_annual = s_ref + self.mu_annual * k_ref
        return steady_state_profit_annual / self.cost_of_equity_annual

    # Compatibility aliases so older code does not break.
    @property
    def rho(self) -> float:
        return self.mean_reversion_factor

    @property
    def mu(self) -> float:
        return self.mu_annual

    @property
    def s0(self) -> float:
        return self.s0_annual

    @property
    def alpha(self) -> float:
        return self.alpha_step


class LoanTraderSolver:
    def __init__(self, params: LoanParams | None = None, nr: int = 40, nc: int = 30, nk: int = 40):
        self.params = params or LoanParams()
        self.nr = nr
        self.nc = nc
        self.nk = nk

        # grids
        self.r_grid = np.linspace(self.params.r_min, self.params.r_max, nr)
        self.c_grid = np.linspace(self.params.c_min, self.params.c_max, nc)
        self.k_grid = np.linspace(self.params.k_min, self.params.k_max, nk)

        self.r_mesh, self.c_mesh, self.k_mesh = np.meshgrid(
            self.r_grid, self.c_grid, self.k_grid, indexing="ij"
        )

        # Gauss-Hermite quadrature
        gh_x, gh_w = hermgauss(5)
        self.z_nodes = np.sqrt(2.0) * gh_x
        self.z_probs = gh_w / np.sqrt(np.pi)

        self.dr = self.r_grid[1] - self.r_grid[0]
        self.dc = self.c_grid[1] - self.c_grid[0]
        self.dk = self.k_grid[1] - self.k_grid[0]

        self.g_actions = np.asarray(self.params.g_actions, dtype=np.float64)
        self.p_actions = np.asarray(self.params.p_actions, dtype=np.float64)

        self._action_cache: dict[tuple[float, float], dict] | None = None
        self._stencil_cache: dict[tuple[float, float, int], tuple] | None = None

        self._init_interpolation_functions()

    def _restore_runtime_state(self):
        # Ensure the interpolation function is defined even if the external module is absent
        self._init_interpolation_functions()

        if hasattr(self, "_build_action_cache"):
            self._build_action_cache()

    @classmethod
    def load(cls, path: str | Path):
        """
        Load previously saved solver.
        """
        path = Path(path)

        with open(path, "rb") as f:
            obj = pickle.load(f)

        obj._restore_runtime_state()

        return obj

    def _init_interpolation_functions(self):
        global interp3_uniform_numpy, build_interp3_stencil, interp3_from_stencil

        if interp3_uniform_numpy is None:
            def interp3_uniform_numpy(values, x, y, z, r_grid, q_grid, k_grid):
                x = np.clip(x, r_grid[0], r_grid[-1])
                y = np.clip(y, q_grid[0], q_grid[-1])
                z = np.clip(z, k_grid[0], k_grid[-1])

                dx = r_grid[1] - r_grid[0]
                dy = q_grid[1] - q_grid[0]
                dz = k_grid[1] - k_grid[0]

                ix = np.floor((x - r_grid[0]) / dx).astype(np.int64)
                iy = np.floor((y - q_grid[0]) / dy).astype(np.int64)
                iz = np.floor((z - k_grid[0]) / dz).astype(np.int64)

                ix = np.clip(ix, 0, len(r_grid) - 2)
                iy = np.clip(iy, 0, len(q_grid) - 2)
                iz = np.clip(iz, 0, len(k_grid) - 2)

                tx = (x - r_grid[ix]) / dx
                ty = (y - q_grid[iy]) / dy
                tz = (z - k_grid[iz]) / dz

                c000 = values[ix, iy, iz]
                c001 = values[ix, iy, iz + 1]
                c010 = values[ix, iy + 1, iz]
                c011 = values[ix, iy + 1, iz + 1]
                c100 = values[ix + 1, iy, iz]
                c101 = values[ix + 1, iy, iz + 1]
                c110 = values[ix + 1, iy + 1, iz]
                c111 = values[ix + 1, iy + 1, iz + 1]

                c00 = c000 * (1.0 - tz) + c001 * tz
                c01 = c010 * (1.0 - tz) + c011 * tz
                c10 = c100 * (1.0 - tz) + c101 * tz
                c11 = c110 * (1.0 - tz) + c111 * tz
                c0 = c00 * (1.0 - ty) + c01 * ty
                c1 = c10 * (1.0 - ty) + c11 * ty
                return c0 * (1.0 - tx) + c1 * tx

            def build_interp3_stencil(x, y, z, r_grid, q_grid, k_grid):
                x = np.clip(x, r_grid[0], r_grid[-1])
                y = np.clip(y, q_grid[0], q_grid[-1])
                z = np.clip(z, k_grid[0], k_grid[-1])

                dx = r_grid[1] - r_grid[0]
                dy = q_grid[1] - q_grid[0]
                dz = k_grid[1] - k_grid[0]

                ix = np.floor((x - r_grid[0]) / dx).astype(np.int64)
                iy = np.floor((y - q_grid[0]) / dy).astype(np.int64)
                iz = np.floor((z - k_grid[0]) / dz).astype(np.int64)

                ix = np.clip(ix, 0, len(r_grid) - 2)
                iy = np.clip(iy, 0, len(q_grid) - 2)
                iz = np.clip(iz, 0, len(k_grid) - 2)

                tx = (x - r_grid[ix]) / dx
                ty = (y - q_grid[iy]) / dy
                tz = (z - k_grid[iz]) / dz
                return ix, iy, iz, tx, ty, tz

            def interp3_from_stencil(values, stencil):
                ix, iy, iz, tx, ty, tz = stencil

                c000 = values[ix, iy, iz]
                c001 = values[ix, iy, iz + 1]
                c010 = values[ix, iy + 1, iz]
                c011 = values[ix, iy + 1, iz + 1]
                c100 = values[ix + 1, iy, iz]
                c101 = values[ix + 1, iy, iz + 1]
                c110 = values[ix + 1, iy + 1, iz]
                c111 = values[ix + 1, iy + 1, iz + 1]

                c00 = c000 * (1.0 - tz) + c001 * tz
                c01 = c010 * (1.0 - tz) + c011 * tz
                c10 = c100 * (1.0 - tz) + c101 * tz
                c11 = c110 * (1.0 - tz) + c111 * tz
                c0 = c00 * (1.0 - ty) + c01 * ty
                c1 = c10 * (1.0 - ty) + c11 * ty
                return c0 * (1.0 - tx) + c1 * tx

        self._interp3_uniform = interp3_uniform_numpy
        self._build_stencil = build_interp3_stencil
        self._interp_from_stencil = interp3_from_stencil

    @property
    def action_cache(self):
        if self._action_cache is None:
            self._action_cache = self._build_action_cache()
        return self._action_cache

    def _build_action_cache(self):
        """
        Precompute deterministic quantities for every (g, p):
        m, pi, d, default, k_next, c_next
        """
        cache = {}
        dt = self.params.dt
        alpha_step = self.params.alpha_step
        s0 = self.params.s0_annual
        kappa = self.params.kappa
        c0 = self.params.c0
        c_star = self.params.c_star

        r_mesh = self.r_mesh
        c_mesh = self.c_mesh
        k_mesh = self.k_mesh

        for g in self.g_actions:
            g_step = self.params.annual_to_step(g)

            for p in self.p_actions:
                m = 1.0 - alpha_step + g_step
                if np.any(m <= 0):
                    raise ValueError(
                        f"Invalid action g={g:.6f}: 1 - alpha_step + g must stay positive."
                    )

                # Keep the original structure but in consistent units.
                # spread = s0 - kappa * (g - alpha)
                spread = self.params.spread(g)
                R_new = r_mesh + spread

                # per-step PnL per unit of current assets
                pi = (c_mesh - r_mesh) * dt + r_mesh * k_mesh * dt

                pos_pi = np.maximum(pi, 0.0)
                div_cap = np.maximum(k_mesh + pi - c_star * m, 0.0)
                d = np.minimum(p * pos_pi, div_cap)

                default_cond = (k_mesh + pi) / m < c0

                k_next = (k_mesh + pi - d) / m
                c_next = ((1.0 - alpha_step) * c_mesh + g_step * R_new) / m

                k_next = np.clip(k_next, self.params.k_min, self.params.k_max)
                c_next = np.clip(c_next, self.params.c_min, self.params.c_max)

                cache[(g, p)] = {
                    "m": m,
                    "pi": pi,
                    "d": d,
                    "default": default_cond,
                    "k_next": k_next,
                    "c_next": c_next,
                    "spread": spread,
                }
        return cache

    def _diffuse(self, value_next: np.ndarray, stencils: tuple):
        cont = np.zeros_like(self.r_mesh)
        for w, stencil in zip(self.z_probs, stencils):
            cont += w * self._interp_from_stencil(value_next, stencil)
        return cont

    def solve(self):
        params = self.params
        N = params.N
        nr, nc, nk = self.nr, self.nc, self.nk

        value = np.zeros((N + 1, nr, nc, nk), dtype=np.float64)
        value[N] = self.k_mesh + params.theta

        opt_g = np.zeros((N, nr, nc, nk), dtype=np.float64)
        opt_p = np.zeros((N, nr, nc, nk), dtype=np.float64)

        self._stencil_cache = {}
        rng = range(len(self.z_nodes))
        phi = params.mean_reversion_factor

        for g in self.g_actions:
            for p in self.p_actions:
                cache = self.action_cache[(g, p)]
                k_next = cache["k_next"]
                c_next = cache["c_next"]

                for qi in rng:
                    r_prime = (
                        params.mu_annual
                        + phi * (self.r_mesh - params.mu_annual)
                        + params.base_sd * self.z_nodes[qi]
                    )
                    r_prime = np.clip(r_prime, params.r_min, params.r_max)

                    stencil = self._build_stencil(
                        r_prime, c_next, k_next,
                        self.r_grid, self.c_grid, self.k_grid
                    )
                    self._stencil_cache[(g, p, qi)] = stencil

        for t in tqdm(range(N - 1, -1, -1), desc="Backward induction"):
            value_next = value[t + 1]
            n_actions = len(self.g_actions) * len(self.p_actions)
            Q = np.zeros((n_actions, nr, nc, nk), dtype=np.float64)

            idx = 0
            for g in self.g_actions:
                for p in self.p_actions:
                    cache = self.action_cache[(g, p)]
                    m = cache["m"]
                    pi = cache["pi"]
                    d = cache["d"]
                    default_flag = cache["default"]

                    stencils = tuple(self._stencil_cache[(g, p, qi)] for qi in rng)
                    continuation = self._diffuse(value_next, stencils)

                    recovery = (1.0 - params.delta) * np.maximum(self.k_mesh + pi, 0.0)
                    val = np.where(default_flag, recovery, d + params.gamma * m * continuation)

                    Q[idx] = val
                    idx += 1

            best_idx = np.argmax(Q, axis=0)
            n_p = len(self.p_actions)

            g_idx = best_idx // n_p
            p_idx = best_idx % n_p

            opt_g[t] = self.g_actions[g_idx]
            opt_p[t] = self.p_actions[p_idx]
            value[t] = np.take_along_axis(Q, best_idx[None, ...], axis=0)[0]

        self.value = value
        self.opt_g = opt_g
        self.opt_p = opt_p
        return value, opt_g, opt_p

    def simulate(self, r0: float, c0: float, k0: float, nsim: int = 1000) -> dict:
        params = self.params
        N = params.N

        if not hasattr(self, "opt_g"):
            raise RuntimeError("Call solve() first to obtain optimal policy.")

        rates = np.zeros((N + 1, nsim), dtype=np.float64)
        coupons = np.zeros((N + 1, nsim), dtype=np.float64)
        kappa_sim = np.zeros((N + 1, nsim), dtype=np.float64)
        divs = np.zeros((N + 1, nsim), dtype=np.float64)
        default_time = np.full(nsim, N + 1, dtype=int)

        rates[0] = r0
        coupons[0] = c0
        kappa_sim[0] = k0

        eps = np.random.randn(N, nsim)

        for t in range(N):
            r_cur = rates[t]
            c_cur = coupons[t]
            k_cur = kappa_sim[t]

            g_interp = self._interp3_uniform(
                self.opt_g[t], r_cur, c_cur, k_cur,
                self.r_grid, self.c_grid, self.k_grid
            )
            p_interp = self._interp3_uniform(
                self.opt_p[t], r_cur, c_cur, k_cur,
                self.r_grid, self.c_grid, self.k_grid
            )

            alpha_step = params.alpha_step
            g_step = params.annual_to_step(g_interp)
            m = 1.0 - alpha_step + g_step
            # spread = params.s0_annual - params.kappa * (g_interp - alpha)
            spread = self.params.spread(g_interp)
            R_new = r_cur + spread

            dt = params.dt
            pi = (c_cur - r_cur) * dt + r_cur * k_cur * dt
            pos_pi = np.maximum(pi, 0.0)
            div_cap = np.maximum(k_cur + pi - params.c_star * m, 0.0)
            d = np.minimum(p_interp * pos_pi, div_cap)

            default_cond = (k_cur + pi) / m < params.c0
            new_default = default_cond & (default_time == N + 1)
            default_time[new_default] = t + 1

            k_next = (k_cur + pi - d) / m
            c_next = ((1.0 - alpha_step) * c_cur + g_step * R_new) / m
            r_next = params.mu_annual + params.mean_reversion_factor * (r_cur - params.mu_annual) + params.base_sd * eps[t]

            rates[t + 1] = np.where(default_time <= t + 1, rates[t], r_next)
            coupons[t + 1] = np.where(default_time <= t + 1, coupons[t], c_next)
            kappa_sim[t + 1] = np.where(default_time <= t + 1, kappa_sim[t], k_next)
            divs[t] = d

        return {
            "rates": rates,
            "coupons": coupons,
            "capital_ratio": kappa_sim,
            "dividends_norm": divs,
            "default_times": default_time,
            "time_grid": np.arange(N + 1) * params.dt,
            "dt": params.dt,
        }
        
    def save(self, path: str | Path) -> None:
        """
        Save fully solved solver object.

        Includes:
            - params
            - grids
            - value
            - optimal policies
            - action cache
            - interpolation stencils

        After loading the object is immediately usable.
        """
        path = Path(path)

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    solver = LoanTraderSolver()
    value, opt_g, opt_p = solver.solve()
    print("Done. value shape:", value.shape)
    i_mid = (solver.nr // 2, solver.nc // 2, solver.nk // 2)
    print("Example optimal g at t=0:", opt_g[0][i_mid])