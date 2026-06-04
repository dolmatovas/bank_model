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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

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
    dt: float = 0.25
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
    u_actions: Tuple[float, ...] = (-40.0, -20.0, 0.0, 20.0, 40.0)

    @property
    def N(self) -> int:
        return int(round(self.T / self.dt))


P = Params()
PAYMENT_TIMES = np.arange(P.dt, P.T + 1e-12, P.dt)
FIXED_RATE = float((1.0 - np.exp(-P.rbar * P.T)) / (P.dt * np.exp(-P.rbar * PAYMENT_TIMES).sum()))

# state grids for value surfaces
R_GRID = np.linspace(-0.03, 0.08, 45)   # 25bp spacing; wide enough for stylized scenarios
Q_GRID = np.linspace(0.0, 140.0, 29)    # 5 notional units
K_GRID = np.linspace(-25.0, 35.0, 25)   # wide enough to avoid clipping in stressed paths
R_MESH, Q_MESH, K_MESH = np.meshgrid(R_GRID, Q_GRID, K_GRID, indexing="ij")

PHI = math.exp(-P.a * P.dt)
BASE_SD = P.sigma * math.sqrt((1.0 - math.exp(-2.0 * P.a * P.dt)) / (2.0 * P.a))
GH_X, GH_W = hermgauss(5)
Z_NODES = np.sqrt(2.0) * GH_X
Z_PROBS = GH_W / np.sqrt(np.pi)

POLICY_ORDER = ["Passive q", "CAR-target", "MtM-band", "0.5y-greedy", "Optimal V"]
SCENARIO_ORDER = ["Base MR", "Early sell-off", "Rally", "Hump vol", "Whipsaw"]
METRIC_ORDER = ["dK_0.5y", "dK_1y", "dK_3y", "MtM", "V"]
METRIC_LABELS = {
    "dK_0.5y": r"$\partial_r \Delta_{0.5y}K$",
    "dK_1y": r"$\partial_r \Delta_{1y}K$",
    "dK_3y": r"$\partial_r \Delta_{3y}K$",
    "MtM": r"$\partial_r MtM$",
    "V": r"$\partial_r V$",
}
MTM_BAND = 3.0

# use numba to accelerate
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
    coupon = qh * (r - FIXED_RATE) * dt
    penalty = (((qh > 1e-8) & ((Kstate / np.maximum(qh, 1e-8)) < P.CAR)).astype(float)) * (dt / P.eps)
    liq = 0.5 * P.lam * (u ** 2) * dt
    return coupon - penalty - liq


def backup_fixed(next_surface: np.ndarray, control_array: np.ndarray, discount=False, sigma_step=BASE_SD) -> np.ndarray:
    dK = stage_dK(R_MESH, Q_MESH, K_MESH, control_array)
    qn = next_q(Q_MESH, control_array)
    Kn = K_MESH + dK
    exp_next = np.zeros_like(R_MESH)
    for z, p in zip(Z_NODES, Z_PROBS):
        rn = P.rbar + PHI * (R_MESH - P.rbar) + sigma_step * z
        exp_next += p * interp3_uniform(next_surface, rn, qn, Kn)
    if discount is True:
        disc = np.exp(-R_MESH * P.dt)
    elif discount is False:
        disc = 1.0
    else:
        disc = discount
    return dK + disc * exp_next


def backup_optimal(next_surface: np.ndarray, discount=False, sigma_step=BASE_SD):
    vals = []
    for u in P.u_actions:
        dK = stage_dK(R_MESH, Q_MESH, K_MESH, u)
        qn = next_q(Q_MESH, u)
        Kn = K_MESH + dK
        exp_next = np.zeros_like(R_MESH)
        for z, p in zip(Z_NODES, Z_PROBS):
            rn = P.rbar + PHI * (R_MESH - P.rbar) + sigma_step * z
            exp_next += p * interp3_uniform(next_surface, rn, qn, Kn)
        if discount is True:
            disc = np.exp(-R_MESH * P.dt)
        elif discount is False:
            disc = 1.0
        else:
            disc = discount
        vals.append(dK + disc * exp_next)
    vals = np.stack(vals, axis=0)
    idx = np.argmax(vals, axis=0)
    best = np.take_along_axis(vals, idx[None, ...], axis=0)[0]
    best_u = np.asarray(P.u_actions)[idx]
    return best, best_u, vals


def compute_optimal_full():
    V = [None] * (P.N + 1)
    U = [None] * P.N
    V[P.N] = np.zeros_like(R_MESH)
    for n in range(P.N - 1, -1, -1):
        V[n], U[n], _ = backup_optimal(V[n + 1], discount=True)
    return V, U


def compute_horizon_optimal_all(H_steps: int):
    vals = [[np.zeros_like(R_MESH) for _ in range(P.N + 1)] for _ in range(H_steps + 1)]
    pols = [[np.zeros_like(R_MESH) for _ in range(P.N)] for _ in range(H_steps + 1)]
    for h in range(1, H_steps + 1):
        prev = vals[h - 1]
        cur = vals[h]
        pol = pols[h]
        for n in range(P.N - 1, -1, -1):
            cur[n], pol[n], _ = backup_optimal(prev[n + 1], discount=False)
    return vals, pols


def compute_policy_metric_surfaces(control_fn: Callable[[int], np.ndarray], H_steps: int, discount=False):
    prev = [np.zeros_like(R_MESH) for _ in range(P.N + 1)]
    for _ in range(1, H_steps + 1):
        cur = [np.zeros_like(R_MESH) for _ in range(P.N + 1)]
        for n in range(P.N - 1, -1, -1):
            cur[n] = backup_fixed(prev[n + 1], control_fn(n), discount=discount)
        prev = cur
    return prev


def compute_delta_surfaces(surfaces: List[np.ndarray]) -> List[np.ndarray]:
    return [np.gradient(surface, R_GRID, axis=0, edge_order=2) for surface in surfaces]


def choose_action_from_surface(r: float, q: float, Kstate: float, next_surface: np.ndarray, discount=False) -> Tuple[float, float]:
    best_u, best_val = 0.0, -1e18
    for u in P.u_actions:
        dK = float(stage_dK(r, q, Kstate, u))
        qn = float(next_q(q, u))
        Kn = Kstate + dK
        exp_next = 0.0
        for z, p in zip(Z_NODES, Z_PROBS):
            rn = P.rbar + PHI * (r - P.rbar) + BASE_SD * z
            exp_next += p * float(interp3_uniform(next_surface, rn, qn, Kn))
        disc = math.exp(-r * P.dt) if discount else 1.0
        val = dK + disc * exp_next
        if val > best_val:
            best_u, best_val = float(u), float(val)
    return best_u, best_val


def scenario_definitions() -> Dict[str, Dict[str, np.ndarray]]:
    t = np.arange(P.N) * P.dt
    scenarios: Dict[str, Dict[str, np.ndarray]] = {}

    scenarios["Base MR"] = {
        "sigma": np.full(P.N, P.sigma),
        "z": np.random.default_rng(1).normal(scale=0.9, size=P.N),
        "desc": "mean reversion with moderate shocks",
    }

    z_up = np.random.default_rng(2).normal(scale=0.5, size=P.N)
    z_up[:6] += 1.4
    scenarios["Early sell-off"] = {
        "sigma": np.full(P.N, P.sigma),
        "z": z_up,
        "desc": "positive shocks over the first 18 months",
    }

    z_down = np.random.default_rng(3).normal(scale=0.5, size=P.N)
    z_down[:6] -= 1.4
    scenarios["Rally"] = {
        "sigma": np.full(P.N, P.sigma),
        "z": z_down,
        "desc": "negative shocks over the first 18 months",
    }

    hump = 1.0 + 1.8 * np.exp(-0.5 * ((t - 1.5) / 0.55) ** 2)
    scenarios["Hump vol"] = {
        "sigma": P.sigma * hump,
        "z": np.random.default_rng(4).normal(scale=0.8, size=P.N),
        "desc": "volatility hump around year 1.5",
    }

    z_whip = 1.0 * ((-1) ** np.arange(P.N)) + np.random.default_rng(5).normal(scale=0.35, size=P.N)
    scenarios["Whipsaw"] = {
        "sigma": np.full(P.N, 1.15 * P.sigma),
        "z": z_whip,
        "desc": "alternating shocks with slightly higher vol",
    }
    return scenarios


def build_policy_controls(Ushort, Ufull):
    def control_passive(_n):
        return np.zeros_like(R_MESH)

    def control_car(_n):
        q_star = np.clip(K_MESH / P.CAR, 0.0, P.q_max)
        return np.clip((q_star - Q_MESH) / P.dt, min(P.u_actions), max(P.u_actions))

    def control_mtm(n):
        v_unit = swap_unit_mtm(n, R_MESH)
        safe_abs_v = np.maximum(np.abs(v_unit), 1e-8)
        q_star = np.where(np.abs(Q_MESH * v_unit) <= MTM_BAND, Q_MESH, np.minimum(Q_MESH, MTM_BAND / safe_abs_v))
        q_star = np.clip(q_star, 0.0, P.q_max)
        return np.clip((q_star - Q_MESH) / P.dt, min(P.u_actions), max(P.u_actions))

    return {
        "Passive q": control_passive,
        "CAR-target": control_car,
        "MtM-band": control_mtm,
        "0.5y-greedy": lambda n: Ushort[n],
        "Optimal V": lambda n: Ufull[n],
    }


def build_policy_action(policy: str, n: int, r: float, q: float, Kstate: float, V1, Vfull) -> float:
    if policy == "Passive q":
        return 0.0
    if policy == "CAR-target":
        q_star = float(np.clip(Kstate / P.CAR, 0.0, P.q_max))
        return float(np.clip((q_star - q) / P.dt, min(P.u_actions), max(P.u_actions)))
    if policy == "MtM-band":
        v_unit = float(swap_unit_mtm(n, np.array([r]))[0])
        if abs(q * v_unit) <= MTM_BAND or abs(v_unit) < 1e-8:
            return 0.0
        q_star = min(q, MTM_BAND / abs(v_unit))
        return float(np.clip((q_star - q) / P.dt, min(P.u_actions), max(P.u_actions)))
    if policy == "0.5y-greedy":
        u, _ = choose_action_from_surface(r, q, Kstate, V1[n + 1], discount=False)
        return u
    if policy == "Optimal V":
        u, _ = choose_action_from_surface(r, q, Kstate, Vfull[n + 1], discount=True)
        return u
    raise KeyError(policy)


def evaluate_surface(surface_list: List[np.ndarray], n: int, r: float, q: float, Kstate: float) -> float:
    return float(interp3_uniform(surface_list[n], r, q, Kstate))


def simulate_policy_on_scenario(
    policy: str,
    scenario_name: str,
    scenario: Dict[str, np.ndarray],
    metrics_surfaces,
    metric_deltas,
    V1,
    Vfull,
    Vfull_delta,
):
    N = P.N
    r = np.zeros(N + 1)
    q = np.zeros(N + 1)
    Kcap = np.zeros(N + 1)
    u = np.zeros(N)
    dK = np.zeros(N)
    coupon = np.zeros(N)
    penalty = np.zeros(N)
    liq = np.zeros(N)
    car_ratio = np.full(N + 1, np.inf)
    mtm = np.zeros(N + 1)
    mtm_delta = np.zeros(N + 1)
    metric_vals = {m: np.zeros(N + 1) for m in METRIC_ORDER}
    metric_deltas_path = {m: np.zeros(N + 1) for m in METRIC_ORDER}

    r[0] = P.rbar
    q[0] = P.q0
    Kcap[0] = P.K0
    car_ratio[0] = Kcap[0] / q[0]
    mtm[0] = q[0] * float(swap_unit_mtm(0, np.array([r[0]]))[0])
    mtm_delta[0] = q[0] * float(swap_unit_delta(0, np.array([r[0]]))[0])

    metric_vals["dK_0.5y"][0] = evaluate_surface(metrics_surfaces[policy]["0.5y"], 0, r[0], q[0], Kcap[0])
    metric_vals["dK_1y"][0] = evaluate_surface(metrics_surfaces[policy]["1y"], 0, r[0], q[0], Kcap[0])
    metric_vals["dK_3y"][0] = evaluate_surface(metrics_surfaces[policy]["3y"], 0, r[0], q[0], Kcap[0])
    metric_vals["MtM"][0] = mtm[0]
    metric_vals["V"][0] = evaluate_surface(Vfull, 0, r[0], q[0], Kcap[0])

    metric_deltas_path["dK_0.5y"][0] = evaluate_surface(metric_deltas[policy]["0.5y"], 0, r[0], q[0], Kcap[0])
    metric_deltas_path["dK_1y"][0] = evaluate_surface(metric_deltas[policy]["1y"], 0, r[0], q[0], Kcap[0])
    metric_deltas_path["dK_3y"][0] = evaluate_surface(metric_deltas[policy]["3y"], 0, r[0], q[0], Kcap[0])
    metric_deltas_path["MtM"][0] = mtm_delta[0]
    metric_deltas_path["V"][0] = evaluate_surface(Vfull_delta, 0, r[0], q[0], Kcap[0])

    for n in range(N):
        u[n] = build_policy_action(policy, n, r[n], q[n], Kcap[n], V1, Vfull)
        q_eff = float(next_q(q[n], u[n]))
        coupon[n] = q_eff * (r[n] - FIXED_RATE) * P.dt
        penalty[n] = (1.0 if ((q_eff > 1e-8) and (Kcap[n] / q_eff < P.CAR)) else 0.0) * (P.dt / P.eps)
        liq[n] = 0.5 * P.lam * u[n] ** 2 * P.dt
        dK[n] = coupon[n] - penalty[n] - liq[n]
        Kcap[n + 1] = Kcap[n] + dK[n]
        q[n + 1] = q_eff
        sigma_n = scenario["sigma"][n]
        sd_n = sigma_n * math.sqrt((1.0 - math.exp(-2.0 * P.a * P.dt)) / (2.0 * P.a))
        r[n + 1] = P.rbar + PHI * (r[n] - P.rbar) + sd_n * scenario["z"][n]
        car_ratio[n + 1] = np.inf if q[n + 1] <= 1e-8 else Kcap[n + 1] / q[n + 1]
        mtm[n + 1] = q[n + 1] * float(swap_unit_mtm(n + 1, np.array([r[n + 1]]))[0])
        mtm_delta[n + 1] = q[n + 1] * float(swap_unit_delta(n + 1, np.array([r[n + 1]]))[0])

        metric_vals["dK_0.5y"][n + 1] = evaluate_surface(metrics_surfaces[policy]["0.5y"], n + 1, r[n + 1], q[n + 1], Kcap[n + 1])
        metric_vals["dK_1y"][n + 1] = evaluate_surface(metrics_surfaces[policy]["1y"], n + 1, r[n + 1], q[n + 1], Kcap[n + 1])
        metric_vals["dK_3y"][n + 1] = evaluate_surface(metrics_surfaces[policy]["3y"], n + 1, r[n + 1], q[n + 1], Kcap[n + 1])
        metric_vals["MtM"][n + 1] = mtm[n + 1]
        metric_vals["V"][n + 1] = evaluate_surface(Vfull, n + 1, r[n + 1], q[n + 1], Kcap[n + 1])

        metric_deltas_path["dK_0.5y"][n + 1] = evaluate_surface(metric_deltas[policy]["0.5y"], n + 1, r[n + 1], q[n + 1], Kcap[n + 1])
        metric_deltas_path["dK_1y"][n + 1] = evaluate_surface(metric_deltas[policy]["1y"], n + 1, r[n + 1], q[n + 1], Kcap[n + 1])
        metric_deltas_path["dK_3y"][n + 1] = evaluate_surface(metric_deltas[policy]["3y"], n + 1, r[n + 1], q[n + 1], Kcap[n + 1])
        metric_deltas_path["MtM"][n + 1] = mtm_delta[n + 1]
        metric_deltas_path["V"][n + 1] = evaluate_surface(Vfull_delta, n + 1, r[n + 1], q[n + 1], Kcap[n + 1])

    return {
        "policy": policy,
        "scenario": scenario_name,
        "r": r,
        "q": q,
        "K": Kcap,
        "u": u,
        "dK": dK,
        "coupon": coupon,
        "penalty": penalty,
        "liq": liq,
        "CAR": car_ratio,
        "mtm": mtm,
        "mtm_delta": mtm_delta,
        "metric_vals": metric_vals,
        "metric_deltas": metric_deltas_path,
    }


def order_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Scenario" in out.columns:
        out["Scenario"] = pd.Categorical(out["Scenario"], categories=SCENARIO_ORDER, ordered=True)
    if "Policy" in out.columns:
        out["Policy"] = pd.Categorical(out["Policy"], categories=POLICY_ORDER, ordered=True)
    return out.sort_values([c for c in ["Scenario", "Policy"] if c in out.columns])


def to_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.2f", caption: str | None = None, label: str | None = None):
    latex = df.to_latex(index=False, escape=False, float_format=lambda x: float_format % x, column_format=None)
    if caption or label:
        latex = latex.replace("\\begin{tabular}", "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}")
        latex = latex.replace("\\end{tabular}", "\\end{tabular}\n")
        if caption:
            latex = latex.replace("\\end{tabular}\n", f"\\end{tabular}\n\\caption{{{caption}}}\n")
        if label:
            latex = latex.replace("\\caption", f"\\label{{{label}}}\n\\caption", 1) if caption else latex.replace("\\end{tabular}\n", f"\\end{tabular}\n\\label{{{label}}}\n")
        latex += "\\end{table}\n"
    path.write_text(latex, encoding="utf-8")


def make_tables(outdir: Path, scenarios, summary_df: pd.DataFrame, delta_df: pd.DataFrame):
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"

    # Parameter table
    param_df = pd.DataFrame(
        [
            [r"$\Delta t$", "0.25y", "Quarterly control / coupon step"],
            [r"$T$", f"{P.T:.0f}y", "Swap maturity"],
            [r"$a$", f"{P.a:.2f}", "Hull-White mean reversion"],
            [r"$\bar r$", f"{100*P.rbar:.2f}\\%", "Long-run short rate and initial flat curve level"],
            [r"$\sigma$", f"{100*P.sigma:.2f}\\%", "Baseline short-rate volatility"],
            [r"$K_{fix}$", f"{100*FIXED_RATE:.3f}\\%", "Par fixed coupon at $t=0$"],
            [r"$q_0$", f"{P.q0:.0f}", "Initial swap notional"],
            [r"$K_0$", f"{P.K0:.1f}", "Initial capital"],
            [r"$CAR$", f"{100*P.CAR:.0f}\\%", "Risk-appetite floor $K_t/q_t$"],
            [r"$\varepsilon$", f"{P.eps:.2f}", r"Breach penalty $1/\varepsilon=4$ per year"],
            [r"$\lambda$", f"{P.lam:.4f}", r"Liquidity cost coefficient"],
            [r"$u_t$", r"$\{-40,-20,0,20,40\}$", r"Optimal-control action set (notional units/year)"],
        ],
        columns=["Параметр", "Значение", "Комментарий"],
    )
    to_latex_table(param_df, tables_dir / "parameters.tex")

    policy_df = pd.DataFrame(
        [
            ["Passive q", r"$u_t=0$", "Пассивно держим исходный номинал."],
            ["CAR-target", r"$q^*=\min(q_{max},K_t/CAR)$", "Подтягиваем номинал к целевому капиталовому плечу."],
            ["MtM-band", r"$|q_t m_t(r_t)|\le M^*$, $M^*=3$", "Сжимаем позицию, когда абсолютный MtM уходит из полосы."],
            ["0.5y-greedy", r"$u_t=\arg\max \Delta_{0.5y}K_t$", "Рецедирующий горизонт: оптимизация только на полгода."],
            ["Optimal V", r"$u_t=\arg\max V_t$", "Полный Беллман на весь остаточный срок."],
        ],
        columns=["Политика", "Правило", "Интерпретация"],
    )
    to_latex_table(policy_df, tables_dir / "policies.tex")

    scenario_df = pd.DataFrame(
        [[name, spec["desc"]] for name, spec in scenarios.items()], columns=["Сценарий", "Конструкция"]
    )
    to_latex_table(scenario_df, tables_dir / "scenarios.tex")

    finalk = summary_df.pivot(index="Policy", columns="Scenario", values="Final K").reindex(index=POLICY_ORDER, columns=SCENARIO_ORDER)
    finalk_rounded = finalk.reset_index().rename(columns={"Policy": "Политика"})
    to_latex_table(finalk_rounded, tables_dir / "finalK.tex")

    stress = summary_df[summary_df["Scenario"].isin(["Rally", "Hump vol"])][
        ["Scenario", "Policy", "Final K", "Penalty", "Liq cost", "Time in breach", "Max |MtM|"]
    ]
    stress = order_df(stress)
    stress = stress.rename(
        columns={
            "Scenario": "Сценарий",
            "Policy": "Политика",
            "Final K": "Финальный K",
            "Penalty": "Штраф",
            "Liq cost": "Ликвидность",
            "Time in breach": "Доля времени в breach",
            "Max |MtM|": "max |MtM|",
        }
    )
    to_latex_table(stress, tables_dir / "stress_table.tex")

    ranking_rows = []
    for scen in SCENARIO_ORDER:
        sub = summary_df[summary_df["Scenario"] == scen].copy()
        best_K = sub.loc[sub["Final K"].idxmax()]
        best_mtm = sub.loc[sub["Max |MtM|"].idxmin()]
        ranking_rows.append([scen, best_K["Policy"], best_K["Final K"], best_mtm["Policy"], best_mtm["Max |MtM|"]])
    ranking_df = pd.DataFrame(
        ranking_rows,
        columns=["Сценарий", "Лучшая политика по K", "Финальный K", "Лучшая по max |MtM|", "min max |MtM|"],
    )
    to_latex_table(ranking_df, tables_dir / "ranking.tex")

    # CSV outputs for transparency / reruns
    order_df(summary_df).to_csv(results_dir / "summary_by_policy_and_scenario.csv", index=False)
    order_df(delta_df).to_csv(results_dir / "delta_summary.csv", index=False)
    param_df.to_csv(results_dir / "parameters.csv", index=False)


def scenario_rate_paths(scenarios) -> Dict[str, np.ndarray]:
    t = np.arange(P.N + 1) * P.dt
    out = {}
    for name, spec in scenarios.items():
        r = np.zeros(P.N + 1)
        r[0] = P.rbar
        for n in range(P.N):
            sd_n = spec["sigma"][n] * math.sqrt((1.0 - math.exp(-2.0 * P.a * P.dt)) / (2.0 * P.a))
            r[n + 1] = P.rbar + PHI * (r[n] - P.rbar) + sd_n * spec["z"][n]
        out[name] = r
    return out


def plot_scenarios(outdir: Path, scenarios):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    t = np.arange(P.N + 1) * P.dt
    for name, r in scenario_rate_paths(scenarios).items():
        ax.plot(t, 100.0 * r, linewidth=2, label=name)
    ax.axhline(100.0 * P.rbar, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Пять stylized сценариев short rate в Hull-White")
    ax.set_xlabel("Годы")
    ax.set_ylabel("Короткая ставка, %")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "scenario_paths.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_hump_policy_paths(outdir: Path, all_sims):
    t = np.arange(P.N + 1) * P.dt
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True)
    for policy in POLICY_ORDER:
        sim = all_sims[("Hump vol", policy)]
        axes[0].plot(t, sim["q"], linewidth=2, label=policy)
        axes[1].plot(t, sim["K"], linewidth=2)
        ratio = np.where(sim["q"] > 1e-8, sim["K"] / sim["q"], np.nan)
        axes[2].plot(t, 100.0 * ratio, linewidth=2)

    axes[0].set_title("Сценарий Hump vol: номинал, капитал и CAR по политикам")
    axes[0].set_ylabel(r"$q_t$")
    axes[1].set_ylabel(r"$K_t$")
    axes[2].set_ylabel(r"$100\cdot K_t/q_t$, %")
    axes[2].set_xlabel("Годы")
    axes[2].axhline(100.0 * P.CAR, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    axes[0].legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "hump_policy_paths.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_normalized_deltas(outdir: Path, all_sims):
    t = np.arange(P.N + 1) * P.dt
    sim = all_sims[("Hump vol", "Optimal V")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for metric in METRIC_ORDER:
        raw = sim["metric_deltas"][metric]
        norm = raw / (abs(raw[0]) + 1e-12)
        ax.plot(t, norm, linewidth=2, label=METRIC_LABELS[metric])
    ax.set_title("Оптимальная политика, сценарий Hump vol: нормированные deltas пяти метрик")
    ax.set_xlabel("Годы")
    ax.set_ylabel("Delta / initial |delta|")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "optimal_hump_normalized_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmaps(outdir: Path, delta_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.8))
    axes = axes.ravel()
    for i, metric in enumerate(METRIC_ORDER):
        ax = axes[i]
        piv = (
            delta_df[delta_df["Metric"] == metric]
            .pivot(index="Policy", columns="Scenario", values="mean_abs_norm_delta")
            .reindex(index=POLICY_ORDER, columns=SCENARIO_ORDER)
        )
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(len(SCENARIO_ORDER)))
        ax.set_xticklabels(SCENARIO_ORDER, rotation=30, ha="right")
        ax.set_yticks(range(len(POLICY_ORDER)))
        ax.set_yticklabels(POLICY_ORDER)
        for rr in range(piv.shape[0]):
            for cc in range(piv.shape[1]):
                ax.text(cc, rr, f"{piv.values[rr,cc]:.2f}", ha="center", va="center", color="white", fontsize=7)
    axes[-1].axis("off")
    cbar = fig.colorbar(im, ax=axes[:-1], shrink=0.85, pad=0.02)
    cbar.set_label(r"$\mathbb{E}|\partial_r M_t| / |\partial_r M_0|$")
    fig.suptitle("Средняя абсолютная нормированная delta по сценариям и политикам", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "figures" / "delta_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rally_decomposition(outdir: Path, all_sims):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(POLICY_ORDER))
    coupons = []
    penalties = []
    liqs = []
    finals = []
    for policy in POLICY_ORDER:
        sim = all_sims[("Rally", policy)]
        coupons.append(sim["coupon"].sum())
        penalties.append(-sim["penalty"].sum())
        liqs.append(-sim["liq"].sum())
        finals.append(sim["K"][-1])
    width = 0.75
    ax.bar(x, coupons, width=width, label="coupon carry")
    ax.bar(x, penalties, width=width, bottom=coupons, label="- breach penalty")
    ax.bar(x, liqs, width=width, bottom=np.array(coupons) + np.array(penalties), label="- liquidity")
    for xi, fk in zip(x, finals):
        ax.text(xi, fk + 0.6 * np.sign(fk if fk != 0 else 1), f"{fk:.1f}", ha="center", va="bottom" if fk >= 0 else "top", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(POLICY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Capital units accumulated over 7Y")
    ax.set_title("Сценарий Rally: разложение накопленного капитала")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "rally_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_finalK_heatmap(outdir: Path, summary_df: pd.DataFrame):
    piv = summary_df.pivot(index="Policy", columns="Scenario", values="Final K").reindex(index=POLICY_ORDER, columns=SCENARIO_ORDER)
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn")
    ax.set_title("Финальный капитал $K_T$ по сценариям и политикам")
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(POLICY_ORDER)))
    ax.set_yticklabels(POLICY_ORDER)
    for rr in range(piv.shape[0]):
        for cc in range(piv.shape[1]):
            ax.text(cc, rr, f"{piv.values[rr, cc]:.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$K_T$")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "finalK_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_optimal_policy_map(outdir: Path, Ufull):
    time_index = 8   # t = 2y
    q_star = 100.0
    q_idx = int(np.argmin(np.abs(Q_GRID - q_star)))
    policy_slice = Ufull[time_index][:, q_idx, :]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    im = ax.imshow(
        policy_slice.T,
        aspect="auto",
        origin="lower",
        extent=[100.0 * R_GRID[0], 100.0 * R_GRID[-1], K_GRID[0], K_GRID[-1]],
        cmap="coolwarm",
        vmin=min(P.u_actions),
        vmax=max(P.u_actions),
    )
    ax.set_title(r"Карта оптимального действия $u^*(t,r,K)$ при $t=2y$, $q=100$")
    ax.set_xlabel("Короткая ставка, %")
    ax.set_ylabel("Капитал K")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$u^*$ (notional / year)")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "optimal_policy_map.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rally_scatter(outdir: Path, all_sims):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for policy in POLICY_ORDER:
        sim = all_sims[("Rally", policy)]
        car = np.where(sim["q"] > 1e-8, sim["K"] / sim["q"], np.nan)
        ax.scatter(sim["q"], 100.0 * car, s=28, alpha=0.85, label=policy)
    ax.axhline(100.0 * P.CAR, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"$q_t$")
    ax.set_ylabel(r"$100\cdot K_t/q_t$, %")
    ax.set_title("Сценарий Rally: траектории в плоскости (q, CAR)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "rally_q_car_scatter.pdf", bbox_inches="tight")
    plt.close(fig)



def years_to_step(years: float) -> int:
    return int(round(years / P.dt))


def collect_retrospective_metric_paths(all_sims) -> pd.DataFrame:
    rows = []
    t = np.arange(P.N + 1) * P.dt
    for scenario in SCENARIO_ORDER:
        for policy in POLICY_ORDER:
            sim = all_sims[(scenario, policy)]
            for metric in METRIC_ORDER:
                raw = np.asarray(sim["metric_deltas"][metric], dtype=float)
                denom = abs(raw[0]) + 1e-12
                norm = raw / denom
                for tt, vv, nn in zip(t, raw, norm):
                    rows.append(
                        {
                            "Scenario": scenario,
                            "Policy": policy,
                            "Metric": metric,
                            "Time": float(tt),
                            "DeltaValue": float(vv),
                            "NormalizedDelta": float(nn),
                        }
                    )
    return order_df(pd.DataFrame(rows))


def collect_state_slice_data(metric_deltas, Vfull_delta, all_sims, reference_scenario="Base MR", reference_policy="Optimal V"):
    metric_to_horizon = {"dK_0.5y": "0.5y", "dK_1y": "1y", "dK_3y": "3y"}
    ref_sim = all_sims[(reference_scenario, reference_policy)]
    rows = []
    ref_rows = []

    for years in (0.0, 1.0, 3.0):
        n = years_to_step(years)
        ref_r = float(ref_sim["r"][n])
        ref_q = float(ref_sim["q"][n])
        ref_K = float(ref_sim["K"][n])
        ref_rows.append(
            {
                "TimeYears": years,
                "TimeLabel": f"t={years:g}y",
                "RefScenario": reference_scenario,
                "RefPolicy": reference_policy,
                "Ref_r": ref_r,
                "Ref_q": ref_q,
                "Ref_K": ref_K,
            }
        )

        for metric, horizon_label in metric_to_horizon.items():
            for policy in POLICY_ORDER:
                surf = metric_deltas[policy][horizon_label][n]
                y_r = interp3_uniform(surf, R_GRID, np.full_like(R_GRID, ref_q), np.full_like(R_GRID, ref_K))
                y_K = interp3_uniform(surf, np.full_like(K_GRID, ref_r), np.full_like(K_GRID, ref_q), K_GRID)
                y_q = interp3_uniform(surf, np.full_like(Q_GRID, ref_r), Q_GRID, np.full_like(Q_GRID, ref_K))
                for x, y in zip(R_GRID, y_r):
                    rows.append(
                        {
                            "TimeYears": years,
                            "TimeLabel": f"t={years:g}y",
                            "Metric": metric,
                            "Policy": policy,
                            "StateVariable": "r",
                            "StateValue": float(x),
                            "StateValuePlot": float(100.0 * x),
                            "DeltaValue": float(y),
                        }
                    )
                for x, y in zip(K_GRID, y_K):
                    rows.append(
                        {
                            "TimeYears": years,
                            "TimeLabel": f"t={years:g}y",
                            "Metric": metric,
                            "Policy": policy,
                            "StateVariable": "K",
                            "StateValue": float(x),
                            "StateValuePlot": float(x),
                            "DeltaValue": float(y),
                        }
                    )
                for x, y in zip(Q_GRID, y_q):
                    rows.append(
                        {
                            "TimeYears": years,
                            "TimeLabel": f"t={years:g}y",
                            "Metric": metric,
                            "Policy": policy,
                            "StateVariable": "q",
                            "StateValue": float(x),
                            "StateValuePlot": float(x),
                            "DeltaValue": float(y),
                        }
                    )

        mtm_r = ref_q * swap_unit_delta(n, R_GRID)
        mtm_q = Q_GRID * float(swap_unit_delta(n, np.array([ref_r]))[0])
        mtm_k = np.full_like(K_GRID, ref_q * float(swap_unit_delta(n, np.array([ref_r]))[0]), dtype=float)
        for x, y in zip(R_GRID, mtm_r):
            rows.append(
                {
                    "TimeYears": years,
                    "TimeLabel": f"t={years:g}y",
                    "Metric": "MtM",
                    "Policy": "State-invariant",
                    "StateVariable": "r",
                    "StateValue": float(x),
                    "StateValuePlot": float(100.0 * x),
                    "DeltaValue": float(y),
                }
            )
        for x, y in zip(K_GRID, mtm_k):
            rows.append(
                {
                    "TimeYears": years,
                    "TimeLabel": f"t={years:g}y",
                    "Metric": "MtM",
                    "Policy": "State-invariant",
                    "StateVariable": "K",
                    "StateValue": float(x),
                    "StateValuePlot": float(x),
                    "DeltaValue": float(y),
                }
            )
        for x, y in zip(Q_GRID, mtm_q):
            rows.append(
                {
                    "TimeYears": years,
                    "TimeLabel": f"t={years:g}y",
                    "Metric": "MtM",
                    "Policy": "State-invariant",
                    "StateVariable": "q",
                    "StateValue": float(x),
                    "StateValuePlot": float(x),
                    "DeltaValue": float(y),
                }
            )

        surf_v = Vfull_delta[n]
        v_r = interp3_uniform(surf_v, R_GRID, np.full_like(R_GRID, ref_q), np.full_like(R_GRID, ref_K))
        v_K = interp3_uniform(surf_v, np.full_like(K_GRID, ref_r), np.full_like(K_GRID, ref_q), K_GRID)
        v_q = interp3_uniform(surf_v, np.full_like(Q_GRID, ref_r), Q_GRID, np.full_like(Q_GRID, ref_K))
        for x, y in zip(R_GRID, v_r):
            rows.append(
                {
                    "TimeYears": years,
                    "TimeLabel": f"t={years:g}y",
                    "Metric": "V",
                    "Policy": "Optimal surface",
                    "StateVariable": "r",
                    "StateValue": float(x),
                    "StateValuePlot": float(100.0 * x),
                    "DeltaValue": float(y),
                }
            )
        for x, y in zip(K_GRID, v_K):
            rows.append(
                {
                    "TimeYears": years,
                    "TimeLabel": f"t={years:g}y",
                    "Metric": "V",
                    "Policy": "Optimal surface",
                    "StateVariable": "K",
                    "StateValue": float(x),
                    "StateValuePlot": float(x),
                    "DeltaValue": float(y),
                }
            )
        for x, y in zip(Q_GRID, v_q):
            rows.append(
                {
                    "TimeYears": years,
                    "TimeLabel": f"t={years:g}y",
                    "Metric": "V",
                    "Policy": "Optimal surface",
                    "StateVariable": "q",
                    "StateValue": float(x),
                    "StateValuePlot": float(x),
                    "DeltaValue": float(y),
                }
            )

    slice_df = pd.DataFrame(rows)
    ref_df = pd.DataFrame(ref_rows)
    return slice_df, ref_df


def plot_retrospective_metric_grid(outdir: Path, retrospective_df: pd.DataFrame):
    fig, axes = plt.subplots(len(SCENARIO_ORDER), len(POLICY_ORDER), figsize=(14.5, 11.4), sharex=True, sharey=True)
    all_norm = retrospective_df["NormalizedDelta"].to_numpy(dtype=float)
    ylim = float(np.nanpercentile(np.abs(all_norm), 97))
    ylim = max(1.4, min(ylim, 6.0))

    for i, scenario in enumerate(SCENARIO_ORDER):
        for j, policy in enumerate(POLICY_ORDER):
            ax = axes[i, j]
            panel = retrospective_df[(retrospective_df["Scenario"] == scenario) & (retrospective_df["Policy"] == policy)]
            for metric in METRIC_ORDER:
                sub = panel[panel["Metric"] == metric]
                ax.plot(
                    sub["Time"].to_numpy(),
                    sub["NormalizedDelta"].to_numpy(),
                    linewidth=1.4,
                    label=METRIC_LABELS[metric] if (i == 0 and j == 0) else None,
                )
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
            ax.axvline(1.0, color="black", linewidth=0.55, linestyle="--", alpha=0.45)
            ax.axvline(3.0, color="black", linewidth=0.55, linestyle="--", alpha=0.45)
            ax.set_xlim(0.0, P.T)
            ax.set_ylim(-1.05 * ylim, 1.05 * ylim)
            if i == 0:
                ax.set_title(policy)
            if j == 0:
                ax.set_ylabel(scenario + "\nnormalised\ndelta")
            if i == len(SCENARIO_ORDER) - 1:
                ax.set_xlabel("Годы")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Ретроспективные траектории всех пяти risk-metric deltas: строки --- сценарии, столбцы --- политики", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(outdir / "figures" / "retrospective_metric_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_state_slice_panels(outdir: Path, slice_df: pd.DataFrame, ref_df: pd.DataFrame, years: float):
    label_map = {"r": "Короткая ставка, %", "K": "Капитал $K$", "q": "Номинал $q$"}
    ref_row = ref_df.loc[np.isclose(ref_df["TimeYears"], years)].iloc[0]
    panel_df = slice_df.loc[np.isclose(slice_df["TimeYears"], years)].copy()

    fig, axes = plt.subplots(len(METRIC_ORDER), 3, figsize=(12.8, 11.7), sharex=False, sharey=False)
    for row, metric in enumerate(METRIC_ORDER):
        for col, state_var in enumerate(["r", "K", "q"]):
            ax = axes[row, col]
            sub = panel_df[(panel_df["Metric"] == metric) & (panel_df["StateVariable"] == state_var)]
            if metric in ("MtM", "V"):
                for policy in sub["Policy"].drop_duplicates():
                    grp = sub[sub["Policy"] == policy]
                    ax.plot(grp["StateValuePlot"], grp["DeltaValue"], linewidth=2.0, label=policy if (row == 0 and col == 0) else None)
                ax.text(
                    0.04,
                    0.95,
                    "policy-invariant\nat fixed state",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            else:
                for policy in POLICY_ORDER:
                    grp = sub[sub["Policy"] == policy]
                    ax.plot(grp["StateValuePlot"], grp["DeltaValue"], linewidth=1.4, label=policy if (row == 0 and col == 0) else None)
            ref_x = {"r": 100.0 * float(ref_row["Ref_r"]), "K": float(ref_row["Ref_K"]), "q": float(ref_row["Ref_q"])}[state_var]
            ax.axvline(ref_x, color="black", linewidth=0.7, linestyle="--", alpha=0.55)
            if row == 0:
                ax.set_title(label_map[state_var])
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if row == len(METRIC_ORDER) - 1:
                ax.set_xlabel(label_map[state_var])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Зависимость risk-metric deltas от состояния агента при t={years:g}y", y=0.995, fontsize=12)
    fig.text(
        0.5,
        0.972,
        (
            f"якорь: Base MR + Optimal V, r={100.0*float(ref_row['Ref_r']):.2f}%, "
            f"q={float(ref_row['Ref_q']):.1f}, K={float(ref_row['Ref_K']):.1f}"
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.952), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    safe = str(years).replace(".", "p")
    fig.savefig(outdir / "figures" / f"state_slices_t{safe}.pdf", bbox_inches="tight")
    plt.close(fig)




def patch_report_layout(tex_path: Path):
    """Post-process generated report.tex to keep tables/figures within page margins."""
    tex = tex_path.read_text(encoding="utf-8")

    tex = tex.replace(
        "\\usepackage{booktabs,longtable,array,multirow}\n\\usepackage{graphicx}\n\\usepackage{caption}\n\\usepackage{subcaption}\n\\usepackage{xcolor}\n",
        "\\usepackage{booktabs,longtable,array,multirow}\n\\usepackage{graphicx}\n\\usepackage{caption}\n\\usepackage{subcaption}\n\\usepackage{tabularx,adjustbox,ragged2e,makecell}\n\\usepackage{xcolor}\n",
    )

    tex = tex.replace(
        "\\setlength{\\parindent}{0pt}\n\\setlength{\\parskip}{5pt}\n",
        "\\setlength{\\parindent}{0pt}\n\\setlength{\\parskip}{5pt}\n"
        "\\captionsetup{font=small}\n"
        "\\newenvironment{fittable}[1][\\small]{%\n"
        "  \\par\\medskip\n"
        "  \\begin{center}\n"
        "  #1\n"
        "  \\setlength{\\tabcolsep}{4.5pt}%\n"
        "  \\renewcommand{\\arraystretch}{1.10}%\n"
        "  \\begin{adjustbox}{max width=\\linewidth}\n"
        "}{%\n"
        "  \\end{adjustbox}\n"
        "  \\end{center}\n"
        "  \\par\\medskip\n"
        "}\n",
    )

    replacements = [
        ("\\input{tables/parameters.tex}",
         "\\begin{fittable}\n\\input{tables/parameters.tex}\n\\end{fittable}"),
        ("Сравниваемые политики:\n\\input{tables/policies.tex}",
         "Сравниваемые политики.\n\n\\begin{fittable}\n\\input{tables/policies.tex}\n\\end{fittable}"),
        ("\\input{tables/scenarios.tex}",
         "\\begin{fittable}\n\\input{tables/scenarios.tex}\n\\end{fittable}"),
        ("Численно финальный капитал выглядит так:\n\\input{tables/finalK.tex}",
         "Численно финальный капитал выглядит так.\n\n\\begin{fittable}\n\\input{tables/finalK.tex}\n\\end{fittable}"),
        ("Две самые показательные stress-ситуации --- Rally и Hump vol --- собраны отдельно:\n\\input{tables/stress_table.tex}",
         "Две самые показательные stress-ситуации --- Rally и Hump vol --- собраны отдельно.\n\n\\begin{fittable}[\\scriptsize]\n\\input{tables/stress_table.tex}\n\\end{fittable}"),
        ("{\\small \\input{tables/validation_checks.tex} }",
         "\\begin{fittable}\n\\input{tables/validation_checks.tex}\n\\end{fittable}"),
        ("\\\\textit{Rally} и \\\\textit{Hump vol}",
         "\\textit{Rally} и \\textit{Hump vol}"),
    ]
    for old, new in replacements:
        tex = tex.replace(old, new)

    atlas_old = (
        "\\begin{landscape}\n"
        "\\begin{figure}[p]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.985\\linewidth]{figures/retrospective_metric_grid.pdf}\n"
        "  \\caption{Ретроспективные траектории нормированных deltas всех пяти метрик. Такой формат позволяет сразу увидеть, какие сочетания сценария и политики порождают наиболее резкие перестройки rate-risk профиля.}\n"
        "\\end{figure}\n"
        "\\end{landscape}"
    )
    atlas_new = (
        "\\clearpage\n"
        "\\begin{landscape}\n"
        "\\thispagestyle{plain}\n"
        "\\begin{center}\n"
        "  \\includegraphics[width=0.95\\linewidth,height=0.78\\textheight,keepaspectratio]{figures/retrospective_metric_grid.pdf}\n\n"
        "  \\captionof{figure}{Ретроспективные траектории нормированных deltas всех пяти метрик. Такой формат позволяет сразу увидеть, какие сочетания сценария и политики порождают наиболее резкие перестройки rate-risk профиля.}\n"
        "\\end{center}\n"
        "\\end{landscape}\n"
        "\\clearpage"
    )
    tex = tex.replace(atlas_old, atlas_new)

    state_caps = {
        "0p0": "State-slices risk-metric deltas в начальном состоянии $t=0$. Штриховая вертикаль отмечает reference-state $(r_0,q_0,K_0)$.",
        "1p0": "State-slices через $1$ год. Видно, что чувствительность horizon-capital метрик к $K$ и $q$ становится более нелинейной, особенно рядом с областью $K/q \\approx CAR$.",
        "3p0": "State-slices через $3$ года. По мере укорачивания свопа рынок теряет duration, а управленческие метрики сильнее отражают оставшуюся опциональность на deleveraging.",
    }
    for key, cap in state_caps.items():
        old = (
            "\\begin{landscape}\n"
            "\\begin{figure}[p]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.985\\linewidth]{{figures/state_slices_t{key}.pdf}}\n"
            f"  \\caption{{{cap}}}\n"
            "\\end{figure}\n"
            "\\end{landscape}"
        )
        new = (
            "\\clearpage\n"
            "\\begin{landscape}\n"
            "\\thispagestyle{plain}\n"
            "\\begin{center}\n"
            f"  \\includegraphics[width=0.95\\linewidth,height=0.80\\textheight,keepaspectratio]{{figures/state_slices_t{key}.pdf}}\n\n"
            f"  \\captionof{{figure}}{{{cap}}}\n"
            "\\end{center}\n"
            "\\end{landscape}\n"
            "\\clearpage"
        )
        tex = tex.replace(old, new)

    tex_path.write_text(tex, encoding="utf-8")


def make_report_tex(outdir: Path):
    tex = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[left=22mm,right=22mm,top=24mm,bottom=24mm]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Liberation Serif}}
\usepackage{{polyglossia}}
\setdefaultlanguage{{russian}}
\setotherlanguage{{english}}
\usepackage{{amsmath,amssymb,mathtools,bm}}
\usepackage{{booktabs,longtable,array,multirow}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{pdflscape}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}}
\usepackage{{enumitem}}
\setlist[itemize]{{topsep=2pt,itemsep=2pt,leftmargin=1.4em}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}

\title{{Игрушечный пример управления 7-летним процентным свопом\\[2mm]
\large Сравнение пяти rate-risk метрик, пяти сценариев и пяти политик}}
\author{{Сгенерировано автоматически на базе Python / LaTeX}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{center}}
\fbox{{\parbox{{0.94\linewidth}}{{
\textbf{{Что именно сделано.}} Построен toy-example в духе \textit{{control under constraints}}: short rate следует one-factor Hull-White, инструмент --- 7Y payer swap, управление идет через скорость изменения номинала $u_t=\dot q_t$, а капитал $K_t$ несет: (i) coupon/carry по свопу, (ii) жесткий штраф за нарушение риск-аппетита $K_t/q_t < CAR$, (iii) квадратичный liquidity cost $\lambda \dot q_t^2/2$. Сравниваются пять метрик риска по ставке --- $\partial_r \Delta_{{0.5y}}K$, $\partial_r \Delta_{{1y}}K$, $\partial_r \Delta_{{3y}}K$, $\partial_r MtM$ и $\partial_r V$ --- на пяти сценариях и пяти политиках.
}}}}
\end{{center}}

\section*{{1. Модель и toy-упрощения}}
Рынок моделируется one-factor Hull-White / Ornstein-Uhlenbeck short-rate динамикой
\[
 dr_t = a(\bar r - r_t)dt + \sigma_t dW_t,
\]
где в базовой калибровке $a={P.a:.2f}$, $\bar r={100*P.rbar:.2f}\%$, $\sigma={100*P.sigma:.2f}\%$. Для построения value surfaces мы используем одну и ту же динамику под real-world и pricing measure: это сознательное toy-упрощение, которое позволяет сконцентрироваться на сравнении метрик и политик, а не на калибровочных деталях.

Инструмент --- 7-летний payer swap с квартальным шагом и исходным номиналом $q_0={P.q0:.0f}$. В toy-оценке оставшегося свопа используется local-flat-curve proxy по текущей short rate $r_t$. Для оставшихся дат $T_i>t$ единичный MtM задается как
\[
 m_t(r) = 1 - e^{{-r(T_N-t)}} - K_{{fix}}\sum_{{i:t<T_i\le T_N}} \Delta t\, e^{{-r(T_i-t)}},
\]
а портфельный MtM равен $MtM_t = q_t m_t(r_t)$. Поэтому delta рынка для этой метрики равна
\[
 \partial_r MtM_t = q_t\Bigg[(T_N-t)e^{{-r_t(T_N-t)}} + K_{{fix}}\sum_{{i:t<T_i\le T_N}} \Delta t\,(T_i-t)e^{{-r_t(T_i-t)}}\Bigg].
\]

Капитал обновляется на дискретной квартальной сетке. В начале шага мы меняем номинал на $q_{{t+}} = q_t + u_t\Delta t$ и затем за шаг получаем
\[
 \Delta K_t = q_{{t+}}(r_t-K_{{fix}})\Delta t
 - \mathbf{{1}}\!\left(\frac{{K_t}}{{q_{{t+}}}}<CAR\right)\frac{{\Delta t}}{{\varepsilon}}
 - \frac{{\lambda}}{{2}}u_t^2\Delta t.
\]
Здесь $CAR=10\%$, $\varepsilon={P.eps:.2f}$, $\lambda={P.lam:.4f}$. Для удобства оптимальный контроль ищется на дискретном множестве скоростей
\[
 u_t \in \{{-40,-20,0,20,40\}}.
\]

\input{{tables/parameters.tex}}

\section*{{2. Метрики риска и политики}}
Для каждой фиксированной политики $\pi$ вычисляются три horizon-метрики накопленного капитала
\[
 \Delta_TK_t^\pi(s) = \mathbb{{E}}^{{RW}}\!\left[\sum_{{j=t}}^{{t+T-\Delta t}} \Delta K_j \mid s_t=s\right], \qquad T\in\{{0.5y,1y,3y\}},
\]
а также рыночный MtM и полная control-aware стоимость
\[
 V_t(s)=\sup_u\; \mathbb{{E}}\left[\sum_{{j=t}}^{{T-\Delta t}} e^{{-r_j\Delta t}}\Delta K_j\mid s_t=s\right], \qquad s=(r_t,q_t,K_t).
\]
Во всех случаях анализируется именно процентный риск через чувствительность по текущей ставке,
\[
 \mathcal{{D}}^M_t = \frac{{\partial M_t}}{{\partial r_t}}.
\]
На практике $\partial_r \Delta_TK$ и $\partial_r V$ получаются из value surfaces через центральную разность по оси $r$.

Сравниваемые политики:
\input{{tables/policies.tex}}

\section*{{3. Сценарии рынка}}
Сценарии --- это не новая модель, а пять осмысленных pathwise реализаций той же Hull-White динамики. Они заданы через разные последовательности шоков $z_n$ и, в одном случае, через hump-shaped $\sigma_t$.

\input{{tables/scenarios.tex}}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.96\linewidth]{{figures/scenario_paths.pdf}}
  \caption{{Пять stylized сценариев short rate. Hump vol отличается именно выпуклой во времени волатильностью; остальные --- разными shock-patterns при постоянной базовой $\sigma$.}}
\end{{figure}}

\section*{{4. Ключевые результаты}}
\subsection*{{4.1. Что делают политики в stress-сценарии с hump-волатильностью}}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.96\linewidth]{{figures/hump_policy_paths.pdf}}
  \caption{{Сценарий Hump vol. Пассивная позиция держит номинал и теряет капитал. MtM-band и full-optimal рано сжимают позицию; optimal делает это мягче, чем CAR-target, и потому сохраняет больше капитала.}}
\end{{figure}}

\subsection*{{4.2. Как ведут себя пять deltas у оптимальной политики}}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.93\linewidth]{{figures/optimal_hump_normalized_deltas.pdf}}
  \caption{{Нормированные deltas пяти метрик вдоль оптимальной траектории в сценарии Hump vol. Сравнение в нормированном виде важно, потому что raw $\partial_r MtM$ на порядок больше horizon-metrics.}}
\end{{figure}}

Содержательно здесь видно следующее.
\begin{{itemize}}
  \item $\partial_r MtM$ убывает почти монотонно: по мере укорачивания swap life портфель теряет duration.
  \item $\partial_r \Delta_{{0.5y}}K$ наиболее зубчатая: это short-horizon metric, чувствительная к ближайшему coupon carry и к локальным сдвигам $q_t$.
  \item $\partial_r \Delta_{{1y}}K$ и особенно $\partial_r \Delta_{{3y}}K$ сглаженнее: они усредняют ожидаемую mean reversion и будущую политику управления.
  \item $\partial_r V$ сидит между long-horizon capital metric и MtM: в ней одновременно видны и дальний carry, и опциональность на будущее сокращение позиции.
\end{{itemize}}

\subsection*{{4.3. Полный retrospective atlas по сценариям и политикам}}
Следующий atlas сводит вместе все $25$ комбинаций $(\text{{сценарий}},\text{{политика}})$. В каждом маленьком окне показаны нормированные траектории пяти risk-metric deltas; вертикальные штрихи соответствуют моментам $t=1y$ и $t=3y$, для которых ниже отдельно показаны state-slices.

\begin{{landscape}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.985\linewidth]{{figures/retrospective_metric_grid.pdf}}
  \caption{{Ретроспективные траектории нормированных deltas всех пяти метрик. Такой формат позволяет сразу увидеть, какие сочетания сценария и политики порождают наиболее резкие перестройки rate-risk профиля.}}
\end{{figure}}
\end{{landscape}}

\subsection*{{4.4. Сводка по всем сценариям и политикам}}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{{figures/finalK_heatmap.pdf}}
  \caption{{Финальный капитал $K_T$. Полный optimal-control выигрывает в \\textit{{Rally}} и \\textit{{Hump vol}}, почти не уступая пассивной позиции в спокойных сценариях.}}
\end{{figure}}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.97\linewidth]{{figures/delta_heatmaps.pdf}}
  \caption{{Средняя абсолютная нормированная delta каждой из пяти метрик. Нормировка на стартовую абсолютную delta делает сценарии и политики сопоставимыми на одной шкале.}}
\end{{figure}}

Численно финальный капитал выглядит так:
\input{{tables/finalK.tex}}

Две самые показательные stress-ситуации --- Rally и Hump vol --- собраны отдельно:
\input{{tables/stress_table.tex}}

\subsection*{{4.5. Почему naive политики ведут себя по-разному}}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{{figures/rally_decomposition.pdf}}
  \caption{{Сценарий Rally: разложение итогового капитала на coupon carry, breach penalty и liquidity cost. Пассивная позиция ловит большой отрицательный carry и штрафы; optimal и MtM-band рано сжимают номинал и тем самым режут левый хвост.}}
\end{{figure}}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.82\linewidth]{{figures/rally_q_car_scatter.pdf}}
  \caption{{Сценарий Rally в координатах $(q_t, K_t/q_t)$. CAR-target механически гонится за целевым плечом, но при ограничении на скорость не успевает вернуться в безопасную зону и накапливает штраф.}}
\end{{figure}}

Здесь главный qualitative takeaway такой:
\begin{{itemize}}
  \item \textbf{{Passive q}} сохраняет upside в sell-off, но берет на себя весь downside в rally.
  \item \textbf{{CAR-target}} интуитивен, но слишком локален: когда капитал уже просел, ограничения на скорость и liquidity cost не дают быстро восстановить отношение $K/q$.
  \item \textbf{{MtM-band}} хорошо стабилизирует рыночную стоимость и почти всегда держит CAR выше порога, но часто преждевременно отрезает profitable exposure.
  \item \textbf{{0.5y-greedy}} полезна в коротком stress-window, но систематически недооценивает дальний эффект будущих штрафов.
  \item \textbf{{Optimal V}} лучше всего балансирует carry, future deleveraging option и penalty avoidance. Именно поэтому она доминирует в Rally и чуть лучше других защитных правил в Hump vol.
\end{{itemize}}

\subsection*{{4.6. Геометрия оптимального контроля}}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.88\linewidth]{{figures/optimal_policy_map.pdf}}
  \caption{{Срез оптимального правила управления при $t=2y$ и $q=100$. Низкий капитал и низкая ставка толкают к отрицательному $u^*$ (де-рискинг); высокий капитал и высокий short rate позволяют удерживать или даже наращивать номинал.}}
\end{{figure}}

\section*{{5. Зависимость risk-metrics от состояния агента}}
Чтобы увидеть, как метрики зависят именно от состояния $s=(r,q,K)$, ниже зафиксированы координатные срезы по reference path \textit{{Base MR + Optimal V}}: в моменты $t=0$, $1y$ и $3y$ две координаты держатся на reference-state, а третья варьируется по сетке. Для horizon-capital метрик показаны все пять политик; для $MtM$ и $V$ линия одна, потому что при фиксированном состоянии это policy-invariant объекты.

\begin{{landscape}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.985\linewidth]{{figures/state_slices_t0p0.pdf}}
  \caption{{State-slices risk-metric deltas в начальном состоянии $t=0$. Штриховая вертикаль отмечает reference-state $(r_0,q_0,K_0)$.}}
\end{{figure}}
\end{{landscape}}

\begin{{landscape}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.985\linewidth]{{figures/state_slices_t1p0.pdf}}
  \caption{{State-slices через $1$ год. Видно, что чувствительность horizon-capital метрик к $K$ и $q$ становится более нелинейной, особенно рядом с областью $K/q \approx CAR$.}}
\end{{figure}}
\end{{landscape}}

\begin{{landscape}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.985\linewidth]{{figures/state_slices_t3p0.pdf}}
  \caption{{State-slices через $3$ года. По мере укорачивания свопа рынок теряет duration, а управленческие метрики сильнее отражают оставшуюся опциональность на deleveraging.}}
\end{{figure}}
\end{{landscape}}

Из этих state-slices удобно читать три общих эффекта.
\begin{{itemize}}
  \item При снижении $r$ horizon-capital deltas быстро становятся более отрицательными для правил, которые aggressively режут $q_t$: desk заранее страхуется от будущего отрицательного carry и от риска попасть в breach-zone.
  \item Чувствительность по $K$ имеет kink около границы $K/q \approx CAR$: это прямой след штрафа за нахождение в forbidden-zone и ограничения на скорость изменения позиции.
  \item Зависимость по $q$ сильнее всего различает политики: passive и short-horizon greedy почти линейны, тогда как CAR-target и full-optimal сильнее сглаживают хвосты за счёт будущего deleveraging.
\end{{itemize}}

\subsection*{{5.1. Интерпретация пяти rate-risk метрик}}
Если смотреть именно на retrospective поведение deltas и их state-slices, то toy-example показывает вполне характерную картину.
\begin{{enumerate}}[label=\arabic*)]
  \item \textbf{{$\partial_r MtM$}} --- самая "рынковая" метрика: она почти не зависит от capital friction напрямую и потому сильнее всего отражает остаточную duration позиции.
  \item \textbf{{$\partial_r \Delta_{{0.5y}}K$}} --- наиболее близка к carry/risk-budget операционному управлению на горизонте ALM desk.
  \item \textbf{{$\partial_r \Delta_{{1y}}K$}} и \textbf{{$\partial_r \Delta_{{3y}}K$}} уже чувствуют структуру будущих решений по $q_t$, так что это не просто "swap delta", а delta контролируемой капитальной траектории.
  \item \textbf{{$\partial_r V$}} --- наиболее содержательная управленческая delta: в ней собран и carry, и риск breach-зоны, и цена ограниченной ликвидности.
  \item Ни одна из метрик не является "лучшей" сама по себе. Выбор зависит от того, что desk действительно минимизирует: мгновенный MtM risk, ближайший capital carry, либо full-horizon franchise value.
\end{{enumerate}}

\section*{{6. Что важно помнить}}
Это \textbf{{toy-example}}, а не production pricing / XVA / ALM engine. В частности:
\begin{{itemize}}
  \item использован один фактор Hull-White и flat-curve proxy для MtM;
  \item параметры под real-world и pricing measure взяты одинаковыми;
  \item управление ограничено простым дискретным набором скоростей;
  \item штраф breach-зоны задан намеренно жестко, чтобы поведение политик было видно на графиках.
\end{{itemize}}
Но даже в таком упрощенном сетапе хорошо видно главное: \textbf{{метрика риска определяет не только то, как мы измеряем позицию, но и то, как именно будет выглядеть оптимальное управление номиналом.}}

{{\small Базовые ориентиры по модели: Hull \& White (1990), Brigo \& Mercurio (2006).}}

\end{{document}}
"""
    (outdir / "report.tex").write_text(tex, encoding="utf-8")
    patch_report_layout(outdir / "report.tex")


def main(outdir: str = "."):
    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "results").mkdir(parents=True, exist_ok=True)

    scenarios = scenario_definitions()

    # 1) optimal value surfaces
    vals2, pols2 = compute_horizon_optimal_all(2)
    V1 = vals2[1]
    Ushort = pols2[2]
    Vfull, Ufull = compute_optimal_full()
    policy_control_fns = build_policy_controls(Ushort, Ufull)

    # 2) metric surfaces for each policy and each horizon
    horizons = {"0.5y": 2, "1y": 4, "3y": 12}
    metrics_surfaces = {}
    for policy, control_fn in policy_control_fns.items():
        metrics_surfaces[policy] = {}
        for horizon_label, h_steps in horizons.items():
            metrics_surfaces[policy][horizon_label] = compute_policy_metric_surfaces(control_fn, h_steps, discount=False)
    metric_deltas = {
        policy: {h: compute_delta_surfaces(surface_list) for h, surface_list in policy_dict.items()}
        for policy, policy_dict in metrics_surfaces.items()
    }
    Vfull_delta = compute_delta_surfaces(Vfull)

    # 3) scenario simulation under each policy
    all_sims = {}
    for scenario_name, scenario in scenarios.items():
        for policy in POLICY_ORDER:
            all_sims[(scenario_name, policy)] = simulate_policy_on_scenario(
                policy,
                scenario_name,
                scenario,
                metrics_surfaces,
                metric_deltas,
                V1,
                Vfull,
                Vfull_delta,
            )

    summary_rows = []
    delta_rows = []
    for (scenario_name, policy), sim in all_sims.items():
        summary_rows.append(
            {
                "Scenario": scenario_name,
                "Policy": policy,
                "Final K": sim["K"][-1],
                "Cum coupon": sim["coupon"].sum(),
                "Penalty": sim["penalty"].sum(),
                "Liq cost": sim["liq"].sum(),
                "Time in breach": np.mean(sim["CAR"][:-1] < P.CAR),
                "Avg q": sim["q"][:-1].mean(),
                "Final q": sim["q"][-1],
                "Min CAR": np.nanmin(np.where(np.isfinite(sim["CAR"]), sim["CAR"], np.nan)),
                "Max |MtM|": np.max(np.abs(sim["mtm"])),
            }
        )
        for metric in METRIC_ORDER:
            raw = sim["metric_deltas"][metric][:-1]
            delta_rows.append(
                {
                    "Scenario": scenario_name,
                    "Policy": policy,
                    "Metric": metric,
                    "mean_abs_delta": float(np.mean(np.abs(raw))),
                    "mean_abs_norm_delta": float(np.mean(np.abs(raw / (abs(raw[0]) + 1e-12)))),
                    "start_delta": float(raw[0]),
                    "end_delta": float(raw[-1]),
                }
            )

    summary_df = order_df(pd.DataFrame(summary_rows))
    delta_df = order_df(pd.DataFrame(delta_rows))
    retrospective_df = collect_retrospective_metric_paths(all_sims)
    state_slice_df, state_ref_df = collect_state_slice_data(metric_deltas, Vfull_delta, all_sims)

    # 4) save tables, results and plots
    make_tables(outdir, scenarios, summary_df, delta_df)
    retrospective_df.to_csv(outdir / "results" / "retrospective_metric_paths.csv", index=False)
    state_slice_df.to_csv(outdir / "results" / "state_slice_data.csv", index=False)
    state_ref_df.to_csv(outdir / "results" / "state_slice_reference.csv", index=False)

    plot_scenarios(outdir, scenarios)
    plot_hump_policy_paths(outdir, all_sims)
    plot_normalized_deltas(outdir, all_sims)
    plot_retrospective_metric_grid(outdir, retrospective_df)
    plot_state_slice_panels(outdir, state_slice_df, state_ref_df, 0.0)
    plot_state_slice_panels(outdir, state_slice_df, state_ref_df, 1.0)
    plot_state_slice_panels(outdir, state_slice_df, state_ref_df, 3.0)
    plot_delta_heatmaps(outdir, delta_df)
    plot_rally_decomposition(outdir, all_sims)
    plot_finalK_heatmap(outdir, summary_df)
    plot_optimal_policy_map(outdir, Ufull)
    plot_rally_scatter(outdir, all_sims)
    make_report_tex(outdir)

    # 5) small textual summary for reproducibility
    readme = f"""Files generated by hw_swap_toy.py

Core outputs:
- report.tex : LaTeX source of the note
- figures/*.pdf : charts used in the note
- tables/*.tex : LaTeX tables used in the note
- results/*.csv : raw summary tables for reruns / audit
- results/retrospective_metric_paths.csv : pathwise retrospective deltas for all scenario/policy/metric combinations
- results/state_slice_data.csv : state-slice values for t=0,1,3y
- results/state_slice_reference.csv : anchor states used for the slices

Model summary:
- Hull-White parameters: a={P.a}, rbar={P.rbar}, sigma={P.sigma}
- Swap maturity: {P.T} years, dt={P.dt}
- Initial state: r0={P.rbar}, q0={P.q0}, K0={P.K0}
- Fixed coupon: {FIXED_RATE:.8f}
"""
    (outdir / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()
    main(args.outdir)
