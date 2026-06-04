from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from tqdm.auto import tqdm

from src import interpolation


@dataclass(frozen=True)
class Params:
    dt: float = 1.0 / 12.0
    T: float = 7.0
    a: float = 0.35
    rbar: float = 0.025
    sigma: float = 0.012
    q0: float = 50.0
    K0: float = 11.0
    CAR: float = 0.10
    eps: float = 0.25
    lam: float = 0.002
    q_max: float = 140.0
    u_actions: tuple[float, ...] = (-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)

    @property
    def N(self) -> int:
        return int(round(self.T / self.dt))


class Solver:
    def __init__(self, params: Params | None = None, nr: int = 45, nq: int = 25, nk: int = 26):
        self.params = params or Params()
        self.fixed_rate = self.params.rbar + 0.01

        self.nr = nr
        self.nq = nq
        self.nk = nk

        self.r_grid = np.linspace(-0.03, 0.08, nr)
        self.q_grid = np.linspace(0.0, self.params.q_max, nq)
        self.k_grid = np.linspace(-25.0, 35.0, nk)
        self.r_mesh, self.q_mesh, self.k_mesh = np.meshgrid(
            self.r_grid, self.q_grid, self.k_grid, indexing="ij"
        )

        self.timegrid = np.arange(self.params.N + 1) * self.params.dt

        self.phi = math.exp(-self.params.a * self.params.dt)
        self.base_sd = self.params.sigma * math.sqrt(
            (1.0 - math.exp(-2.0 * self.params.a * self.params.dt)) / (2.0 * self.params.a)
        )
        gh_x, gh_w = hermgauss(5)
        self.z_nodes = np.sqrt(2.0) * gh_x
        self.z_probs = gh_w / np.sqrt(np.pi)

        self.r_min, self.r_max = float(self.r_grid[0]), float(self.r_grid[-1])
        self.q_min, self.q_max = float(self.q_grid[0]), float(self.q_grid[-1])
        self.k_min, self.k_max = float(self.k_grid[0]), float(self.k_grid[-1])
        self.r_step = float(self.r_grid[1] - self.r_grid[0])
        self.q_step = float(self.q_grid[1] - self.q_grid[0])
        self.k_step = float(self.k_grid[1] - self.k_grid[0])
        self.disc_grid = np.exp(-self.r_mesh * self.params.dt)
        self.u_actions = np.asarray(self.params.u_actions, dtype=np.float64)

        self._swap_mtm: np.ndarray | None = None
        self._action_cache: dict[tuple[int, float], dict[str, object]] | None = None

        self.swap_mtm
        self.action_cache
        

    @property
    def swap_mtm(self) -> np.ndarray:
        if self._swap_mtm is None:
            self._swap_mtm = self.get_swap_mtm()
        return self._swap_mtm

    @property
    def action_cache(self) -> dict[tuple[int, float], dict[str, object]]:
        if self._action_cache is None:
            self._action_cache = self.build_action_cache()
        return self._action_cache

    def interp3_uniform(self, values: np.ndarray, x, y, z) -> np.ndarray:
        return interpolation.interp3_uniform_numpy(values, x, y, z, self.r_grid, self.q_grid, self.k_grid)

    def build_interp3_stencil(self, x, y, z):
        return interpolation.build_interp3_stencil_numba(x, y, z, self.r_grid, self.q_grid, self.k_grid)

    def interp3_from_stencil(self, values: np.ndarray, stencil) -> np.ndarray:
        return interpolation.interp3_from_stencil(values, stencil)

    def get_next_position(self, q, u):
        return np.clip(q + u * self.params.dt, 0.0, self.params.q_max)

    def get_flows_on_grid(self, t: int, u):
        q_next = self.get_next_position(self.q_mesh, u)
        dq = q_next - self.q_mesh
        u_eff = dq / self.params.dt

        coupon = q_next * (self.fixed_rate - self.r_mesh) * self.params.dt
        q_clip = np.maximum(q_next, 1e-8)
        penalty = -(
            ((q_next > 1e-8) & (self.k_mesh / q_clip < self.params.CAR)).astype(float)
            * (self.params.dt / self.params.eps)
        )
        liq = -0.5 * (u_eff ** 2) * self.params.lam * self.params.dt
        trade = -dq * self.swap_mtm[t]
        return coupon, penalty, liq, trade

    def get_flows(self, t: int, r, q, K, u):
        q_next = self.get_next_position(q, u)
        dq = q_next - q
        u_eff = dq / self.params.dt

        coupon = q_next * (self.fixed_rate - r) * self.params.dt
        q_clip = np.maximum(q_next, 1e-8)
        penalty = -((q_next > 1e-8) & (K / q_clip < self.params.CAR)).astype(float) * (
            self.params.dt / self.params.eps
        )
        liq = -0.5 * (u_eff ** 2) * self.params.lam * self.params.dt
        trade = -dq * self.interp3_uniform(self.swap_mtm[t], r, q, K)
        return coupon, penalty, liq, trade

    def get_delta_K_on_grid(self, t: int, u):
        coupon, penalty, liq, trade = self.get_flows_on_grid(t, u)
        return coupon + penalty + liq + trade

    def get_delta_K_bar_on_grid(self, t: int, u):
        coupon, _, _, trade = self.get_flows_on_grid(t, u)
        return coupon + trade

    def get_delta_K(self, t: int, r, q, K, u):
        coupon, penalty, liq, trade = self.get_flows(t, r, q, K, u)
        return coupon + penalty + liq + trade

    def get_delta_K_bar(self, t: int, r, q, K, u):
        coupon, _, _, trade = self.get_flows(t, r, q, K, u)
        return coupon + trade

    def horizon_to_steps(self, horizon) -> int:
        if isinstance(horizon, (int, np.integer)):
            return int(horizon)
        return max(1, int(round(float(horizon) / self.params.dt)))

    def diffuse(self, surf: np.ndarray, qn, Kn):
        continuation_value = np.zeros_like(self.r_mesh)
        for z, p in zip(self.z_nodes, self.z_probs):
            rn = self.params.rbar + self.phi * (self.r_mesh - self.params.rbar) + self.base_sd * z
            continuation_value += p * self.interp3_uniform(surf, rn, qn, Kn)
        return continuation_value

    def diffuse_1d(self, y: np.ndarray):
        cv = np.zeros_like(y)
        for z, p in zip(self.z_nodes, self.z_probs):
            rn = self.params.rbar + self.phi * (self.r_grid - self.params.rbar) + self.base_sd * z
            cv += p * np.interp(rn, self.r_grid, y, left=y[0], right=y[-1])
        return cv

    def get_swap_mtm(self):
        nt = self.params.N
        swap_mtm = np.zeros((nt + 1, self.nr))
        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            cf_t = (self.fixed_rate - self.r_grid) * self.params.dt
            swap_mtm[t] = cf_t + np.exp(-self.r_grid * self.params.dt) * self.diffuse_1d(
                swap_mtm[t + 1]
            )
        res = np.zeros((nt + 1,) + self.r_mesh.shape)
        res[:, :, :, :] = swap_mtm[:, :, None, None]
        return res

    def build_action_cache(self):
        cache = {}
        rn_nodes = tuple(
            self.params.rbar + self.phi * (self.r_mesh - self.params.rbar) + self.base_sd * z
            for z in self.z_nodes
        )
        for t in tqdm(range(self.params.N), leave=False):
            for u in self.u_actions:
                delta_K = self.get_delta_K_on_grid(t, u)
                delta_K_bar = self.get_delta_K_bar_on_grid(t, u)
                qn = self.get_next_position(self.q_mesh, u)
                Kn = self.k_mesh + delta_K
                stencils = tuple(self.build_interp3_stencil(rn, qn, Kn) for rn in rn_nodes)
                cache[(t, float(u))] = {
                    "delta_K": delta_K,
                    "delta_K_bar": delta_K_bar,
                    "stencils": stencils,
                }
        return cache

    def diffuse_cached(self, surf: np.ndarray, stencils):
        continuation_value = np.zeros_like(self.r_mesh)
        for p, stencil in zip(self.z_probs, stencils):
            continuation_value += p * self.interp3_from_stencil(surf, stencil)
        return continuation_value

    def get_optimal_policy(self):
        nt = self.params.N
        optimal_policy = np.zeros((nt + 1,) + self.r_mesh.shape)
        value_function = np.zeros((nt + 1,) + self.r_mesh.shape)

        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            q_function = np.zeros((len(self.u_actions),) + self.r_mesh.shape)
            for i, u in enumerate(self.u_actions):
                action = self.action_cache[(t, float(u))]
                continuation_value = self.diffuse_cached(value_function[t + 1], action["stencils"])
                q_function[i] = action["delta_K"] + self.disc_grid * continuation_value
            idx = np.argmax(q_function, axis=0)
            optimal_policy[t] = self.u_actions[idx]
            value_function[t] = np.take_along_axis(q_function, idx[None, ...], axis=0)[0]
        return value_function, optimal_policy

    def get_optimal_policy_capital(self, horizon):
        horizon = self.horizon_to_steps(horizon)
        nt = self.params.N
        optimal_policy = np.zeros((nt + 1,) + self.r_mesh.shape)
        capital = np.zeros((nt + 1,) + self.r_mesh.shape)
        flows = np.zeros((nt + 1,) + self.r_mesh.shape)

        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            q_function = np.zeros((len(self.u_actions),) + self.r_mesh.shape)
            s_stop = min(self.params.N, t + horizon + 1)
            s_len = max(0, s_stop - t)
            flows_ = np.zeros((len(self.u_actions), s_len) + self.r_mesh.shape)
            for i, u in enumerate(self.u_actions):
                action = self.action_cache[(t, float(u))]
                dK = action["delta_K"]
                stencils = action["stencils"]
                for local_s, s in enumerate(range(t, s_stop)):
                    flows_[i, local_s] = dK if s == t else 0.0
                    flows_[i, local_s] += self.diffuse_cached(flows[s], stencils)
                q_function[i] = flows_[i, : min(horizon, s_len)].sum(axis=0)
            idx = np.argmax(q_function, axis=0)
            for local_s, _ in enumerate(range(t, s_stop)):
                flows[t + local_s] = np.take_along_axis(flows_[:, local_s], idx[None, ...], axis=0)[0]
            optimal_policy[t] = self.u_actions[idx]
            capital[t] = np.take_along_axis(q_function, idx[None, ...], axis=0)[0]
        return capital, optimal_policy

    def get_optimal_policy_capital_short(self, horizon):
        horizon = self.horizon_to_steps(horizon)
        nt = min(self.params.N, horizon)
        optimal_policy = np.zeros((nt + 1,) + self.r_mesh.shape)
        capital = np.zeros((nt + 1,) + self.r_mesh.shape)

        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            q_function = np.zeros((len(self.u_actions),) + self.r_mesh.shape)
            for i, u in enumerate(self.u_actions):
                action = self.action_cache[(t, float(u))]
                cv = self.diffuse_cached(capital[t + 1], action["stencils"])
                q_function[i] = action["delta_K"] + cv
            idx = np.argmax(q_function, axis=0)
            optimal_policy[t] = self.u_actions[idx]
            capital[t] = np.take_along_axis(q_function, idx[None, ...], axis=0)[0]
        return capital, optimal_policy

    def get_capital(self, policy: np.ndarray, horizon):
        horizon = self.horizon_to_steps(horizon)
        nt = self.params.N
        capital = np.zeros((nt + 1,) + self.r_mesh.shape)
        flows = np.zeros((nt + 1,) + self.r_mesh.shape)
        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            dK = self.get_delta_K_on_grid(t, policy[t])
            qn = self.get_next_position(self.q_mesh, policy[t])
            Kn = self.k_mesh + dK
            for s in range(t, min(self.params.N, t + horizon + 1)):
                flows[s] = (s == t) * dK + self.diffuse(flows[s], qn, Kn)
            capital[t] = flows[t : min(nt + 1, t + horizon)].sum(axis=0)
        return capital

    def get_capital_bar(self, policy: np.ndarray, horizon):
        horizon = self.horizon_to_steps(horizon)
        nt = self.params.N
        capital = np.zeros((nt + 1,) + self.r_mesh.shape)
        flows = np.zeros((nt + 1,) + self.r_mesh.shape)
        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            dK = self.get_delta_K_bar_on_grid(t, policy[t])
            qn = self.get_next_position(self.q_mesh, policy[t])
            Kn = self.k_mesh + dK
            for s in range(t, min(self.params.N, t + horizon + 1)):
                flows[s] = (s == t) * dK + self.diffuse(flows[s], qn, Kn)
            capital[t] = flows[t : min(nt + 1, t + horizon)].sum(axis=0)
        return capital

    def get_value(self, policy: np.ndarray):
        nt = self.params.N
        value = np.zeros((nt + 1,) + self.r_mesh.shape)
        for t in tqdm(range(nt - 1, -1, -1), leave=False):
            dK = self.get_delta_K_on_grid(t, policy[t])
            qn = self.get_next_position(self.q_mesh, policy[t])
            Kn = self.k_mesh + dK
            value[t] = dK + self.disc_grid * self.diffuse(value[t + 1], qn, Kn)
        return value

    def get_scenario(self, rate: np.ndarray, q0: float, K0: float, policy: np.ndarray):
        if rate.ndim == 1:
            rate = rate[:, None]
        nt, nsim = rate.shape
        nt -= 1
        assert nt == self.params.N

        q = np.zeros((nt + 1, nsim))
        K = np.zeros((nt + 1, nsim))
        q[0] = q0
        K[0] = K0

        dK = np.zeros((nt, nsim))
        dKbar = np.zeros((nt, nsim))
        discounts = np.ones((nt + 1, nsim))
        for t in range(nt):
            u = self.interp3_uniform(policy[t], rate[t], q[t], K[t])
            dK[t] = self.get_delta_K(t, rate[t], q[t], K[t], u)
            dKbar[t] = self.get_delta_K_bar(t, rate[t], q[t], K[t], u)
            K[t + 1] = K[t] + dK[t]
            q[t + 1] = self.get_next_position(q[t], u)
            discounts[t + 1] = discounts[t] * np.exp(-rate[t] * self.params.dt)

        return {
            "rate": rate.squeeze(),
            "positions": q.squeeze(),
            "capital": K.squeeze(),
            "CAR": (K / q.clip(0.01)).squeeze(),
            "dK": dK.squeeze(),
            "dKbar": dKbar.squeeze(),
            "discounts": discounts.squeeze(),
        }


    def sample_rates(self, r0=None, nsim=None):
        if r0 is None:
            r0 = self.params.rbar
        if nsim is None:
            nsim = 10_000
        nt = self.params.N
        rates = np.empty((nt + 1, nsim))
        rates[0, :] = r0
        for t in range(nt):
            rates[t + 1] = self.params.rbar + \
                          self.phi * (rates[t] - self.params.rbar) + \
                          self.base_sd * np.random.randn(nsim)
        rates = rates.clip(min=self.r_grid[0], max=self.r_grid[-1])
        return rates
