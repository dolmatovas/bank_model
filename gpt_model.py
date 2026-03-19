#!/usr/bin/env python3
"""Toy comparison of derivative control policies under five rate-risk metrics.

Generates figures, LaTeX tables, CSV summaries, and the main report.tex.
The setup is intentionally simplified but internally consistent:
- short rate follows one-factor Hull-White (OU) dynamics,
- the instrument is a 7Y payer swap with quarterly payments,
- capital evolves from swap carry minus CAR breach penalty minus liquidity cost,
- policies act through the notional speed u_t = dq_t/dt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numba import njit

# -----------------------------
# Global formatting preferences
# -----------------------------
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass(frozen=True)
class Params:
    dt: float = 1.0/12.0
    T: float = 7.0
    a: float = 0.35
    rbar: float = 0.025
    sigma: float = 0.012
    q0: float = 100.0
    K0: float = 14.0
    CAR: float = 0.10
    eps: float = 0.25         # penalty rate = 1/eps = 4 capital units per year in breach
    lam: float = 0.002        # liquidity coefficient
    q_max: float = 140.0
    u_actions: Tuple[float, ...] = (-40.0, -30.0, -20.0, -10.0, 0.0, 10, 20.0, 30.0, 40.0)

    @property
    def N(self) -> int:
        return int(round(self.T / self.dt))


P = Params()
PAYMENT_TIMES = np.arange(P.dt, P.T + 1e-12, P.dt)
FIXED_RATE = float((1.0 - np.exp(-P.rbar * P.T)) / (P.dt * np.exp(-P.rbar * PAYMENT_TIMES).sum()))

NR = 45
NQ = 51
NK = 55
# state grids for value surfaces
R_GRID = np.linspace(-0.03, 0.08, NR)   # 25bp spacing; wide enough for stylized scenarios
Q_GRID = np.linspace(0.0, 140.0, NQ)    # 5 notional units
K_GRID = np.linspace(-25.0, 35.0, NK)   # wide enough to avoid clipping in stressed paths
R_MESH, Q_MESH, K_MESH = np.meshgrid(R_GRID, Q_GRID, K_GRID, indexing="ij")

PHI = math.exp(-P.a * P.dt)
BASE_SD = P.sigma * math.sqrt((1.0 - math.exp(-2.0 * P.a * P.dt)) / (2.0 * P.a))
GH_X, GH_W = hermgauss(5)
Z_NODES = np.sqrt(2.0) * GH_X
Z_PROBS = GH_W / np.sqrt(np.pi)
R_MIN, R_MAX = float(R_GRID[0]), float(R_GRID[-1])
Q_MIN, Q_MAX = float(Q_GRID[0]), float(Q_GRID[-1])
K_MIN, K_MAX = float(K_GRID[0]), float(K_GRID[-1])
R_STEP = float(R_GRID[1] - R_GRID[0])
Q_STEP = float(Q_GRID[1] - Q_GRID[0])
K_STEP = float(K_GRID[1] - K_GRID[0])
DISC_GRID = np.exp(-R_MESH * P.dt)


def interp3_uniform(values: np.ndarray, x, y, z) -> np.ndarray:
    """Fast trilinear interpolation on a uniform 3D grid."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x = np.clip(x, R_GRID[0], R_GRID[-1])
    y = np.clip(y, Q_GRID[0], Q_GRID[-1])
    z = np.clip(z, K_GRID[0], K_GRID[-1])

    dx = R_GRID[1] - R_GRID[0]
    dy = Q_GRID[1] - Q_GRID[0]
    dz = K_GRID[1] - K_GRID[0]

    ix = np.floor((x - R_GRID[0]) / dx).astype(int)
    iy = np.floor((y - Q_GRID[0]) / dy).astype(int)
    iz = np.floor((z - K_GRID[0]) / dz).astype(int)

    ix = np.clip(ix, 0, len(R_GRID) - 2)
    iy = np.clip(iy, 0, len(Q_GRID) - 2)
    iz = np.clip(iz, 0, len(K_GRID) - 2)

    tx = (x - R_GRID[ix]) / dx
    ty = (y - Q_GRID[iy]) / dy
    tz = (z - K_GRID[iz]) / dz

    c000 = values[ix, iy, iz]
    c001 = values[ix, iy, iz + 1]
    c010 = values[ix, iy + 1, iz]
    c011 = values[ix, iy + 1, iz + 1]
    c100 = values[ix + 1, iy, iz]
    c101 = values[ix + 1, iy, iz + 1]
    c110 = values[ix + 1, iy + 1, iz]
    c111 = values[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1 - tz) + c001 * tz
    c01 = c010 * (1 - tz) + c011 * tz
    c10 = c100 * (1 - tz) + c101 * tz
    c11 = c110 * (1 - tz) + c111 * tz
    c0 = c00 * (1 - ty) + c01 * ty
    c1 = c10 * (1 - ty) + c11 * ty
    return c0 * (1 - tx) + c1 * tx


def build_interp3_stencil(x, y, z):
    """Precompute trilinear interpolation indices and barycentric weights."""
    x = np.clip(np.asarray(x), R_MIN, R_MAX)
    y = np.clip(np.asarray(y), Q_MIN, Q_MAX)
    z = np.clip(np.asarray(z), K_MIN, K_MAX)

    ix = np.floor((x - R_MIN) / R_STEP).astype(np.int64)
    iy = np.floor((y - Q_MIN) / Q_STEP).astype(np.int64)
    iz = np.floor((z - K_MIN) / K_STEP).astype(np.int64)

    ix = np.clip(ix, 0, len(R_GRID) - 2)
    iy = np.clip(iy, 0, len(Q_GRID) - 2)
    iz = np.clip(iz, 0, len(K_GRID) - 2)

    tx = (x - R_GRID[ix]) / R_STEP
    ty = (y - Q_GRID[iy]) / Q_STEP
    tz = (z - K_GRID[iz]) / K_STEP
    return ix, iy, iz, tx, ty, tz


@njit(cache=True)
def interp3_from_stencil_numba(values, ix, iy, iz, tx, ty, tz):
    out = np.empty(ix.shape, dtype=np.float64)
    out_flat = out.ravel()
    values_flat = values
    ix_flat = ix.ravel()
    iy_flat = iy.ravel()
    iz_flat = iz.ravel()
    tx_flat = tx.ravel()
    ty_flat = ty.ravel()
    tz_flat = tz.ravel()

    for n in range(out_flat.size):
        i = ix_flat[n]
        j = iy_flat[n]
        k = iz_flat[n]
        wx = tx_flat[n]
        wy = ty_flat[n]
        wz = tz_flat[n]

        c000 = values_flat[i, j, k]
        c001 = values_flat[i, j, k + 1]
        c010 = values_flat[i, j + 1, k]
        c011 = values_flat[i, j + 1, k + 1]
        c100 = values_flat[i + 1, j, k]
        c101 = values_flat[i + 1, j, k + 1]
        c110 = values_flat[i + 1, j + 1, k]
        c111 = values_flat[i + 1, j + 1, k + 1]

        c00 = c000 * (1.0 - wz) + c001 * wz
        c01 = c010 * (1.0 - wz) + c011 * wz
        c10 = c100 * (1.0 - wz) + c101 * wz
        c11 = c110 * (1.0 - wz) + c111 * wz
        c0 = c00 * (1.0 - wy) + c01 * wy
        c1 = c10 * (1.0 - wy) + c11 * wy
        out_flat[n] = c0 * (1.0 - wx) + c1 * wx

    return out


def interp3_from_stencil(values: np.ndarray, stencil) -> np.ndarray:
    return interp3_from_stencil_numba(values, *stencil)

def remaining_taus(n: int) -> np.ndarray:
    t = n * P.dt
    return PAYMENT_TIMES[PAYMENT_TIMES > t] - t

def swap_unit_mtm(n: int, r) -> np.ndarray:
    """Local-flat-curve proxy for a payer swap remaining MtM per unit notional."""
    tau = remaining_taus(n)
    if len(tau) == 0:
        return np.zeros_like(r, dtype=float)
    r = np.asarray(r)
    discounts = np.exp(-np.outer(np.ravel(r), tau))
    annuity = P.dt * discounts.sum(axis=1)
    p_final = discounts[:, -1]
    value = 1.0 - p_final - FIXED_RATE * annuity
    return value.reshape(r.shape)


def swap_unit_delta(n: int, r) -> np.ndarray:
    tau = remaining_taus(n)
    if len(tau) == 0:
        return np.zeros_like(r, dtype=float)
    r = np.asarray(r)
    discounts = np.exp(-np.outer(np.ravel(r), tau))
    d_annuity = P.dt * (discounts * tau).sum(axis=1)
    dp_final = tau[-1] * discounts[:, -1]
    delta = dp_final + FIXED_RATE * d_annuity
    return delta.reshape(r.shape)


def next_q(q, u):
    return np.clip(q + u * P.dt, 0.0, P.q_max)


def stage_dK(r, q, Kstate, u, dt=P.dt):
    """Trade first, then accrue coupon/penalty over the quarter."""
    qh = np.clip(q + u * dt, 0.0, P.q_max)
    coupon = qh * (FIXED_RATE - r) * dt
    penalty = (((qh > 1e-8) & ((Kstate / np.maximum(qh, 1e-8)) < P.CAR)).astype(float)) * (dt / P.eps)
    liq = 0.5 * P.lam * (u ** 2) * dt
    return coupon - penalty - liq


def horizon_to_steps(horizont) -> int:
    if isinstance(horizont, (int, np.integer)):
        return int(horizont)
    return max(1, int(round(float(horizont) / P.dt)))

def get_discount(discount):
    if discount is True:
        return np.exp(-R_MESH * P.dt)
    elif discount is False:
        return 1.0
    else:
        return discount

# numba here???
def diffuse(surf, qn, Kn):
    continuation_value = np.zeros_like(R_MESH)
    for z, p in zip(Z_NODES, Z_PROBS):
        rn = P.rbar + PHI * (R_MESH - P.rbar) + BASE_SD * z
        continuation_value += p * interp3_uniform(surf, rn, qn, Kn)
    return continuation_value


def build_action_cache():
    cache = {}
    rn_nodes = tuple(P.rbar + PHI * (R_MESH - P.rbar) + BASE_SD * z for z in Z_NODES)
    for u in P.u_actions:
        dK = stage_dK(R_MESH, Q_MESH, K_MESH, u)
        qn = next_q(Q_MESH, u)
        Kn = K_MESH + dK
        stencils = tuple(build_interp3_stencil(rn, qn, Kn) for rn in rn_nodes)
        cache[u] = {"dK": dK, "stencils": stencils}
    return cache


ACTION_CACHE = build_action_cache()


def diffuse_cached(surf, stencils):
    continuation_value = np.zeros_like(R_MESH)
    for p, stencil in zip(Z_PROBS, stencils):
        continuation_value += p * interp3_from_stencil(surf, stencil)
    return continuation_value


def backup_fixed(next_surface: np.ndarray, control_array: np.ndarray, discount=False) -> np.ndarray:
    dK = stage_dK(R_MESH, Q_MESH, K_MESH, control_array)
    qn = next_q(Q_MESH, control_array)
    Kn = K_MESH + dK
    continuation_value = diffuse(next_surface, qn, Kn)
    disc = get_discount(discount)
    return dK + disc * continuation_value


def backup_optimal(next_surface: np.ndarray, discount=False):
    q_function = []
    for u in P.u_actions:
        # capital increment
        dK = stage_dK(R_MESH, Q_MESH, K_MESH, u)
        # next position
        qn = next_q(Q_MESH, u)
        # next capital
        Kn = K_MESH + dK
        # continuation value for policy u
        continuation_value = diffuse(next_surface, qn, Kn)
        disc = get_discount(discount)
        q_function.append(dK + disc * continuation_value)
    q_function = np.stack(q_function, axis=0)
    # choosing optimal policy
    idx = np.argmax(q_function, axis=0)
    value_function = np.take_along_axis(q_function, idx[None, ...], axis=0)[0]
    optimal_control = np.asarray(P.u_actions)[idx]
    return value_function, optimal_control, q_function


from tqdm.auto import tqdm

def get_optimal_policy():
    nt = P.N
    optimal_policy = np.zeros((nt + 1, ) + R_MESH.shape)
    value_function = np.zeros((nt + 1, ) + R_MESH.shape)

    for t in tqdm(range(nt)[::-1]):
        n_actions = len(P.u_actions) 
        q_funtion = np.zeros((n_actions, ) + R_MESH.shape)
        # interating over policies
        for i, u in enumerate(P.u_actions):    
            action = ACTION_CACHE[u]
            cv = diffuse_cached(value_function[t + 1], action["stencils"])
            q_funtion[i] = action["dK"] + DISC_GRID * cv
        idx = np.argmax(q_funtion, axis=0)
        optimal_policy[t] = np.asarray(P.u_actions)[idx]
        value_function[t] = np.take_along_axis(q_funtion, idx[None, ...], axis=0)[0]
    return value_function, optimal_policy


def get_optimal_policy_capital(horizont):
    horizont = horizon_to_steps(horizont)
    nt = P.N
    optimal_policy = np.zeros((nt + 1, ) + R_MESH.shape)
    capital = np.zeros((nt + 1, ) + R_MESH.shape)
    flows = np.zeros((nt + 1, ) + R_MESH.shape)

    for t in tqdm(range(nt)[::-1]):
        n_actions = len(P.u_actions) 
        q_funtion = np.zeros((n_actions, ) + R_MESH.shape)
        s_stop = min(P.N, t + horizont + 1)
        s_len = max(0, s_stop - t)
        # Keep only the active horizon window instead of allocating the whole timeline.
        flows_ = np.zeros((n_actions, s_len) + R_MESH.shape)
        # interating over policies
        for i, u in enumerate(P.u_actions):    
            action = ACTION_CACHE[u]
            dK = action["dK"]
            stencils = action["stencils"]
            for local_s, s in enumerate(range(t, s_stop)):
                flows_[i, local_s] = dK if s == t else 0.0
                flows_[i, local_s] += diffuse_cached(flows[s], stencils)
            q_funtion[i] = flows_[i, : min(horizont, s_len)].sum(axis=0)
        idx = np.argmax(q_funtion, axis=0)
        for local_s, s in enumerate(range(t, s_stop)):
            flows[s] = np.take_along_axis(flows_[:, local_s], idx[None, ...], axis=0)[0]
        optimal_policy[t] = np.asarray(P.u_actions)[idx]
        capital[t] = np.take_along_axis(q_funtion, idx[None, ...], axis=0)[0]
    return capital, optimal_policy


def get_optimal_policy_capital_short(horizont):
    horizont = horizon_to_steps(horizont)
    nt = horizon_to_steps
    optimal_policy = np.zeros((nt + 1, ) + R_MESH.shape)
    capital = np.zeros((nt + 1, ) + R_MESH.shape)

    for t in tqdm(range(nt)[::-1]):
        n_actions = len(P.u_actions) 
        q_funtion = np.zeros((n_actions, ) + R_MESH.shape)
        # interating over policies
        for i, u in enumerate(P.u_actions):    
            action = ACTION_CACHE[u]
            cv = diffuse_cached(capital[t + 1], action["stencils"])
            q_funtion[i] = action["dK"] + cv
        idx = np.argmax(q_funtion, axis=0)
        optimal_policy[t] = np.asarray(P.u_actions)[idx]
        capital[t] = np.take_along_axis(q_funtion, idx[None, ...], axis=0)[0]
    return capital, optimal_policy


def get_capital(policy, horizont):
    horizont = horizon_to_steps(horizont)
    nt = P.N
    capital = np.zeros((nt + 1, ) + R_MESH.shape)
    flows = np.zeros((nt + 1, ) + R_MESH.shape)
    for t in tqdm(range(nt)[::-1]):
        dK = stage_dK(R_MESH, Q_MESH, K_MESH, policy[t])
        qn = next_q(Q_MESH, policy[t])
        Kn = K_MESH + dK
        for s in range(t, min(P.N, t + horizont + 1)):
            flows[s] = (s == t) * dK + diffuse(flows[s], qn, Kn)
        capital[t] = flows[t : min(nt + 1, t + horizont)].sum(axis=0)
    return capital

def get_value(policy):
    nt = P.N
    value = np.zeros((nt + 1, ) + R_MESH.shape)
    for t in tqdm(range(nt)[::-1]):
        dK = stage_dK(R_MESH, Q_MESH, K_MESH, policy[t])
        qn = next_q(Q_MESH, policy[t])
        Kn = K_MESH + dK
        value[t] = dK + diffuse(value[t + 1], qn, Kn)
    return value


def get_scenario(rate, q0, K0, policy):
    nt = P.N
    q = np.zeros(nt + 1)
    K = np.zeros(nt + 1)
    q[0] = q0
    K[0] = K0
    for t in range(nt):
        u = interp3_uniform(policy[t], rate[t], q[t], K[t])
        K[t + 1] = K[t] + stage_dK(rate[t], q[t], K[t], u)
        q[t + 1] = next_q(q[t], u)
    return {"rate" : rate, "q" : q, "K" : K}

def main():
    horizonts_years = [0.5, 1, 2, 3]
    metrics = {}
    controls = {}
    deltas = {}
    metrics['value'], controls['value'] = get_optimal_policy()
    for h in horizonts_years:
        metrics[f'capital_{h}'], controls[f'capital_{h}'] = get_optimal_policy_capital(h)
    for k, v in metrics.items():
        deltas[k] = np.gradient(v, R_GRID, axis=1)


if __name__ == "__main__":
    main()
