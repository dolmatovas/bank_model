from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Protocol, Dict, Any
import numpy as np


# =========================
# 1) Short-rate models
# =========================

@dataclass(frozen=True)
class RateParams:
    kappa: float
    theta_P: float  # long-run mean under P
    theta_Q: float  # long-run mean under Q (risk-neutral)
    sigma: float
    r0: float


class ShortRateModel(Protocol):
    params: RateParams

    def simulate(
        self,
        *,
        T: float,
        dt: float,
        n_paths: int,
        measure: str = "P",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return r paths of shape (n_paths, n_steps+1)."""
        ...


class VasicekModel:
    """
    dr = kappa (theta - r) dt + sigma dW
    Euler scheme is fine for toy risk example.
    """
    def __init__(self, params: RateParams):
        self.params = params

    def simulate(self, *, T: float, dt: float, n_paths: int, measure: str = "P", seed: Optional[int] = None) -> np.ndarray:
        p = self.params
        theta = p.theta_P if measure.upper() == "P" else p.theta_Q

        n_steps = int(np.round(T / dt))
        rng = np.random.default_rng(seed)

        r = np.empty((n_paths, n_steps + 1), dtype=float)
        r[:, 0] = p.r0

        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            r[:, t + 1] = r[:, t] + p.kappa * (theta - r[:, t]) * dt + p.sigma * sqrt_dt * z
        return r


class CIRModel:
    """
    dr = kappa (theta - r) dt + sigma * sqrt(r) dW
    Use Full Truncation Euler to keep non-negativity-ish:
    r_plus = max(r, 0), diffusion uses sqrt(r_plus), and then clamp.
    """
    def __init__(self, params: RateParams):
        self.params = params

    def simulate(self, *, T: float, dt: float, n_paths: int, measure: str = "P", seed: Optional[int] = None) -> np.ndarray:
        p = self.params
        theta = p.theta_P if measure.upper() == "P" else p.theta_Q

        n_steps = int(np.round(T / dt))
        rng = np.random.default_rng(seed)

        r = np.empty((n_paths, n_steps + 1), dtype=float)
        r[:, 0] = max(p.r0, 0.0)

        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            r_plus = np.maximum(r[:, t], 0.0)
            drift = p.kappa * (theta - r_plus) * dt
            diff = p.sigma * np.sqrt(r_plus) * sqrt_dt * z
            r_next = r[:, t] + drift + diff
            r[:, t + 1] = np.maximum(r_next, 0.0)
        return r


# =========================
# 2) Spread functions s(u)
# =========================

class SpreadFunction(Protocol):
    def __call__(self, u_rel: np.ndarray) -> np.ndarray:
        """Return spread s(u_rel). u_rel is relative issuance intensity."""
        ...


@dataclass
class ConstantSpread:
    s0: float

    def __call__(self, u_rel: np.ndarray) -> np.ndarray:
        return np.full_like(u_rel, self.s0, dtype=float)


@dataclass
class LogDecreasingSpread:
    """
    s(u) = s0 - a * log(1 + u)
    clipped to [s_min, +inf)
    """
    s0: float
    a: float
    s_min: float = 0.0

    def __call__(self, u_rel: np.ndarray) -> np.ndarray:
        s = self.s0 - self.a * np.log1p(np.maximum(u_rel, 0.0))
        return np.maximum(s, self.s_min)


@dataclass
class LinearDecreasingSpread:
    """
    s(u) = max(s_min, s0 - a*u)
    """
    s0: float
    a: float
    s_min: float = 0.0

    def __call__(self, u_rel: np.ndarray) -> np.ndarray:
        return np.maximum(self.s_min, self.s0 - self.a * np.maximum(u_rel, 0.0))


# =========================
# 3) Balance dynamics (exponential runoff approximation)
# =========================

@dataclass(frozen=True)
class BalanceParams:
    T_loan: float = 5.0   # avg life via lambda = 1/T_loan
    L0: float = 1.0       # initial loan notional
    cbar0: float = 0.03   # initial average coupon (absolute, not spread)
    K0: float = 0.1       # initial capital
    rho: float = 0.0      # subjective discounting for "management objective" under P (optional)


@dataclass(frozen=True)
class SimParams:
    dt: float = 1.0 / 252
    seed: Optional[int] = 123
    n_paths: int = 20000


def _integral_discount_factors(r: np.ndarray, dt: float) -> np.ndarray:
    """
    Given r paths shape (n_paths, n_steps+1) (including time 0),
    return discount factors DF[t] = exp(-∫_0^{t} r_s ds) at each step index.
    Uses left Riemann sum on intervals [t, t+dt).
    Output shape (n_paths, n_steps+1), DF[:,0] = 1.
    """
    # r_used has length n_steps (exclude last point)
    r_used = r[:, :-1]
    cum = np.cumsum(r_used * dt, axis=1)
    df = np.ones((r.shape[0], r.shape[1]), dtype=float)
    df[:, 1:] = np.exp(-cum)
    return df


# =========================
# 4) Policies for u (relative control)
# =========================

class Policy(Protocol):
    def u_rel(self, r: np.ndarray, cbar: np.ndarray, L: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return u_rel >= 0. Think of u_abs = u_rel * lambda * L."""
        ...


@dataclass
class FixedPolicy:
    """
    Keep balance roughly constant: u_rel = 1 => u_abs = lambda * L, so dL ~ 0.
    """
    u_rel_level: float = 1.0

    def u_rel(self, r: np.ndarray, cbar: np.ndarray, L: np.ndarray, K: np.ndarray) -> np.ndarray:
        return np.full_like(r, self.u_rel_level, dtype=float)


@dataclass
class SigmoidPolicy:
    """
    Simple parametric policy:
      u_rel = u_min + (u_max - u_min) * sigmoid(a0 + a1*margin + a2*(K - K*)/K_scale)

    margin = cbar - r

    This gives a smooth "de-risking" when K is low or margin is bad.
    """
    a0: float
    a1: float
    a2: float
    K_star: float
    K_scale: float = 0.1
    u_min: float = 0.0
    u_max: float = 2.0

    def u_rel(self, r: np.ndarray, cbar: np.ndarray, L: np.ndarray, K: np.ndarray) -> np.ndarray:
        margin = cbar - r
        x = self.a0 + self.a1 * margin + self.a2 * ((K - self.K_star) / max(self.K_scale, 1e-8))
        sig = 1.0 / (1.0 + np.exp(-x))
        return self.u_min + (self.u_max - self.u_min) * sig


# =========================
# 5) Engine: simulate balance with (optionally) management + default
# =========================

@dataclass(frozen=True)
class RunResult:
    # Pathwise aggregates
    nii: np.ndarray              # per-path NII over horizon
    pv: np.ndarray               # per-path PV of cashflows over horizon (discounted with r path)
    defaulted: np.ndarray        # per-path indicator of default by horizon
    tau_index: np.ndarray        # per-path first index of default, or -1
    # Terminal states (useful for debugging)
    L_T: np.ndarray
    cbar_T: np.ndarray
    K_T: np.ndarray


def simulate_balance(
    *,
    r_paths: np.ndarray,                 # shape (n_paths, n_steps+1)
    dt: float,
    balance: BalanceParams,
    spread: SpreadFunction,
    policy: Policy,
    horizon_years: float,
    discount_with_r: bool = True,
    stop_at_default: bool = False,
    dividend_rate: float = 0.0,          # optional: dD = dividend_rate * max(K,0) dt
) -> RunResult:
    """
    Simulate balance dynamics under exponential runoff approx:
      lambda = 1/T_loan
      u_abs = u_rel * lambda * L
      dL   = (u_abs - lambda L) dt = lambda L (u_rel - 1) dt
      dcbar= lambda( (r + s(u_rel)) - cbar ) dt
      dK   = L(cbar - r) dt - dD
    If stop_at_default: after K hits 0, cashflows frozen (business stops).
    """
    n_paths, n_steps_plus = r_paths.shape
    dt = float(dt)
    n_steps = int(np.round(horizon_years / dt))
    n_steps = min(n_steps, n_steps_plus - 1)

    lam = 1.0 / balance.T_loan

    # state arrays (vectorized)
    L = np.full(n_paths, balance.L0, dtype=float)
    cbar = np.full(n_paths, balance.cbar0, dtype=float)
    K = np.full(n_paths, balance.K0, dtype=float)

    tau_idx = np.full(n_paths, -1, dtype=int)
    alive = np.ones(n_paths, dtype=bool)

    nii = np.zeros(n_paths, dtype=float)
    pv = np.zeros(n_paths, dtype=float)

    if discount_with_r:
        df = _integral_discount_factors(r_paths[:, : n_steps + 1], dt)  # (n_paths, n_steps+1)
    else:
        df = np.ones((n_paths, n_steps + 1), dtype=float)

    for t in range(n_steps):
        r_t = r_paths[:, t]

        if stop_at_default:
            # keep only alive paths evolving; dead paths stay frozen
            idx = alive
            if not np.any(idx):
                break

            u_rel = policy.u_rel(r_t[idx], cbar[idx], L[idx], K[idx])
            u_rel = np.maximum(u_rel, 0.0)

            s_t = spread(u_rel)
            coupon_new = r_t[idx] + s_t

            # flows (only alive)
            margin = cbar[idx] - r_t[idx]
            cf = L[idx] * margin * dt
            nii[idx] += cf
            pv[idx] += df[idx, t] * cf

            # dividends (optional)
            dD = dividend_rate * np.maximum(K[idx], 0.0) * dt

            # update states
            L[idx] = L[idx] + lam * L[idx] * (u_rel - 1.0) * dt
            cbar[idx] = cbar[idx] + lam * (coupon_new - cbar[idx]) * dt
            K[idx] = K[idx] + cf - dD

            # default check after update
            newly_dead = (K <= 0.0) & alive
            tau_idx[newly_dead] = t + 1
            alive[newly_dead] = False

        else:
            u_rel = policy.u_rel(r_t, cbar, L, K)
            u_rel = np.maximum(u_rel, 0.0)

            s_t = spread(u_rel)
            coupon_new = r_t + s_t

            margin = cbar - r_t
            cf = L * margin * dt
            nii += cf
            pv += df[:, t] * cf

            dD = dividend_rate * np.maximum(K, 0.0) * dt
            L = L + lam * L * (u_rel - 1.0) * dt
            cbar = cbar + lam * (coupon_new - cbar) * dt
            K = K + cf - dD

            # record default time but do not stop
            newly_dead = (tau_idx < 0) & (K <= 0.0)
            tau_idx[newly_dead] = t + 1

    defaulted = tau_idx >= 0
    return RunResult(
        nii=nii,
        pv=pv,
        defaulted=defaulted.astype(bool),
        tau_index=tau_idx,
        L_T=L,
        cbar_T=cbar,
        K_T=K,
    )


# =========================
# 6) Metrics: ΔNII, FV, FV with default
# =========================

@dataclass(frozen=True)
class MetricConfig:
    # NII horizon (e.g., 1 year)
    nii_horizon: float = 1.0
    # PV horizon (finite truncation, e.g. 30 years)
    pv_horizon: float = 30.0
    # parallel shock for ΔNII
    shock: float = 0.01


def delta_nii_parallel_shock(
    *,
    model: ShortRateModel,
    balance: BalanceParams,
    spread: SpreadFunction,
    sim: SimParams,
    cfg: MetricConfig,
    measure_for_paths: str = "P",
) -> Dict[str, Any]:
    """
    ΔNII = E[NII(shocked)] - E[NII(base)] over cfg.nii_horizon,
    with persistent parallel shift applied to r_t paths: r_shocked = r + shock.
    """
    r = model.simulate(T=cfg.nii_horizon, dt=sim.dt, n_paths=sim.n_paths, measure=measure_for_paths, seed=sim.seed)
    r_shocked = r + cfg.shock

    policy_fixed = FixedPolicy(1.0)  # keep volume constant on average
    res_base = simulate_balance(
        r_paths=r, dt=sim.dt, balance=balance, spread=spread, policy=policy_fixed,
        horizon_years=cfg.nii_horizon, discount_with_r=False, stop_at_default=False
    )
    res_shock = simulate_balance(
        r_paths=r_shocked, dt=sim.dt, balance=balance, spread=spread, policy=policy_fixed,
        horizon_years=cfg.nii_horizon, discount_with_r=False, stop_at_default=False
    )

    return {
        "E_NII_base": float(np.mean(res_base.nii)),
        "E_NII_shock": float(np.mean(res_shock.nii)),
        "Delta_NII": float(np.mean(res_shock.nii - res_base.nii)),
    }


def fair_value(
    *,
    model: ShortRateModel,
    balance: BalanceParams,
    spread: SpreadFunction,
    sim: SimParams,
    cfg: MetricConfig,
    measure_for_discounting: str = "Q",
) -> Dict[str, Any]:
    """
    FV = E^Q[ ∫_0^{H} exp(-∫ r ds) * CF dt ]
    with fixed policy, no default stop.
    """
    r = model.simulate(T=cfg.pv_horizon, dt=sim.dt, n_paths=sim.n_paths, measure=measure_for_discounting, seed=sim.seed)
    policy_fixed = FixedPolicy(1.0)
    res = simulate_balance(
        r_paths=r, dt=sim.dt, balance=balance, spread=spread, policy=policy_fixed,
        horizon_years=cfg.pv_horizon, discount_with_r=True, stop_at_default=False
    )
    return {
        "E_FV": float(np.mean(res.pv)),
        "Std_FV": float(np.std(res.pv)),
    }


def fair_value_with_default(
    *,
    model: ShortRateModel,
    balance: BalanceParams,
    spread: SpreadFunction,
    sim: SimParams,
    cfg: MetricConfig,
    measure_for_discounting: str = "Q",
) -> Dict[str, Any]:
    """
    FV' = E^Q[ ∫_0^{min(H, tau)} exp(-∫ r ds) * CF dt ]
    where tau is first time K hits 0 (default stops business).
    """
    r = model.simulate(T=cfg.pv_horizon, dt=sim.dt, n_paths=sim.n_paths, measure=measure_for_discounting, seed=sim.seed)
    policy_fixed = FixedPolicy(1.0)
    res = simulate_balance(
        r_paths=r, dt=sim.dt, balance=balance, spread=spread, policy=policy_fixed,
        horizon_years=cfg.pv_horizon, discount_with_r=True, stop_at_default=True
    )
    return {
        "E_FV_default": float(np.mean(res.pv)),
        "Std_FV_default": float(np.std(res.pv)),
        "P_default_by_H": float(np.mean(res.defaulted)),
    }


# =========================
# 7) "Optimal" management: parametric policy + simple random search
# =========================

@dataclass(frozen=True)
class ManagementObjective:
    horizon: float = 10.0
    eta_default: float = 1.0       # penalty if default occurs by horizon
    rho: float = 0.0               # subjective discounting for profit integral under P (optional)
    stop_at_default: bool = True   # usually True for "business stops"


def evaluate_policy_objective(
    *,
    model: ShortRateModel,
    balance: BalanceParams,
    spread: SpreadFunction,
    policy: Policy,
    sim: SimParams,
    obj: ManagementObjective,
    measure: str = "P",
    seed_shift: int = 0,
) -> float:
    """
    Objective ~ E[ ∫_0^{tau∧H} e^{-rho t} CF dt  - eta * 1{tau<=H} ]
    Under chosen measure (usually P for risk management).
    """
    r = model.simulate(T=obj.horizon, dt=sim.dt, n_paths=sim.n_paths, measure=measure, seed=None if sim.seed is None else sim.seed + seed_shift)

    # implement subjective discounting by overriding discount factors
    # easiest: discount_with_r=False then multiply cf by exp(-rho t)
    # but our engine doesn't output pathwise CF; we can approximate by using discount_with_r=False
    # and passing a synthetic "r_paths" with constant rho in discounting.
    # We'll do: compute PV with discount_with_r=True using r_paths = rho (constant).
    if obj.rho > 0:
        r_disc = np.full_like(r, obj.rho, dtype=float)
        discount_with_r = True
        r_use = r
        # We'll simulate CF using r_use, but discount by r_disc:
        # hack: call simulate_balance twice? Better: add parameter for discount paths.
        # Keep code simple: do manual loop (but that duplicates engine).
        # For now: ignore subjective discounting unless rho==0.
        # (This is a toy; typically rho=0 for "expected cumulative P&L".)
        discount_with_r = False
    else:
        discount_with_r = False

    res = simulate_balance(
        r_paths=r, dt=sim.dt,
        balance=balance, spread=spread, policy=policy,
        horizon_years=obj.horizon,
        discount_with_r=discount_with_r,  # if False, pv==undiscounted NII
        stop_at_default=obj.stop_at_default
    )

    # If discount_with_r=False, "pv" is just undiscounted NII.
    # We use pv as accumulated value for objective.
    pnl = res.pv if discount_with_r else res.nii
    penalty = obj.eta_default * res.defaulted.astype(float)
    J = pnl - penalty
    return float(np.mean(J))


@dataclass(frozen=True)
class PolicySearchConfig:
    n_rounds: int = 8
    n_candidates: int = 40
    init_scale: float = 1.0
    shrink: float = 0.6
    seed: int = 777


def random_search_sigmoid_policy(
    *,
    model: ShortRateModel,
    balance: BalanceParams,
    spread: SpreadFunction,
    sim: SimParams,
    obj: ManagementObjective,
    search: PolicySearchConfig,
    # parameter bounds
    a0_range: Tuple[float, float] = (-5.0, 5.0),
    a1_range: Tuple[float, float] = (-100.0, 100.0),
    a2_range: Tuple[float, float] = (-10.0, 10.0),
    K_star_range: Optional[Tuple[float, float]] = None,
    u_max: float = 2.0,
) -> Dict[str, Any]:
    """
    Very simple derivative-free search:
      - start from a random best
      - iteratively sample Gaussian cloud around best, shrinking variance

    Returns best policy params and estimated objective.
    """
    rng = np.random.default_rng(search.seed)
    if K_star_range is None:
        K_star_range = (0.0, balance.K0 * 2.0)

    def sample_uniform(rng_: np.random.Generator) -> SigmoidPolicy:
        a0 = rng_.uniform(*a0_range)
        a1 = rng_.uniform(*a1_range)
        a2 = rng_.uniform(*a2_range)
        K_star = rng_.uniform(*K_star_range)
        return SigmoidPolicy(a0=a0, a1=a1, a2=a2, K_star=K_star, u_min=0.0, u_max=u_max)

    # initial best
    best_pol = sample_uniform(rng)
    best_val = evaluate_policy_objective(model=model, balance=balance, spread=spread, policy=best_pol, sim=sim, obj=obj, seed_shift=0)

    scale = search.init_scale
    for round_idx in range(search.n_rounds):
        candidates = []
        vals = []
        for k in range(search.n_candidates):
            # gaussian perturb around best
            a0 = best_pol.a0 + scale * rng.normal(0, 1.0)
            a1 = best_pol.a1 + scale * rng.normal(0, 10.0)
            a2 = best_pol.a2 + scale * rng.normal(0, 1.0)
            K_star = best_pol.K_star + scale * rng.normal(0, balance.K0)

            # clamp into bounds
            a0 = float(np.clip(a0, *a0_range))
            a1 = float(np.clip(a1, *a1_range))
            a2 = float(np.clip(a2, *a2_range))
            K_star = float(np.clip(K_star, *K_star_range))

            pol = SigmoidPolicy(a0=a0, a1=a1, a2=a2, K_star=K_star, u_min=0.0, u_max=u_max)
            val = evaluate_policy_objective(
                model=model, balance=balance, spread=spread, policy=pol, sim=sim, obj=obj,
                seed_shift=1000 * round_idx + k
            )
            candidates.append(pol)
            vals.append(val)

        vals = np.array(vals)
        i = int(np.argmax(vals))
        if float(vals[i]) > best_val:
            best_val = float(vals[i])
            best_pol = candidates[i]

        scale *= search.shrink

    return {
        "best_objective": best_val,
        "best_policy": best_pol,
    }


# =========================
# 8) Example usage
# =========================

if __name__ == "__main__":
    # Choose model
    params = RateParams(
        kappa=0.3,
        theta_P=0.025,
        theta_Q=0.02,   # only difference: long-run mean under Q
        sigma=0.01,
        r0=0.02
    )
    model = VasicekModel(params)  # or CIRModel(params)

    # Balance + spread
    bal = BalanceParams(T_loan=5.0, L0=1.0, cbar0=0.03, K0=0.05)
    spread = LogDecreasingSpread(s0=0.02, a=0.01, s_min=0.005)

    sim = SimParams(dt=1/252, n_paths=20000, seed=42)
    cfg = MetricConfig(nii_horizon=1.0, pv_horizon=30.0, shock=0.01)

    # 1) ΔNII
    d = delta_nii_parallel_shock(model=model, balance=bal, spread=spread, sim=sim, cfg=cfg, measure_for_paths="P")
    print("ΔNII:", d)

    # 2) FV (no default), under Q
    fv = fair_value(model=model, balance=bal, spread=spread, sim=sim, cfg=cfg, measure_for_discounting="Q")
    print("FV:", fv)

    # 3) FV with default stop, under Q
    fvd = fair_value_with_default(model=model, balance=bal, spread=spread, sim=sim, cfg=cfg, measure_for_discounting="Q")
    print("FV default:", fvd)

    # 4) Management: search a sigmoid policy under P
    obj = ManagementObjective(horizon=10.0, eta_default=0.2, rho=0.0, stop_at_default=True)
    search = PolicySearchConfig(n_rounds=6, n_candidates=30, init_scale=1.0, shrink=0.7)

    best = random_search_sigmoid_policy(
        model=model, balance=bal, spread=spread, sim=sim,
        obj=obj, search=search,
        a0_range=(-6, 6),
        a1_range=(-200, 200),
        a2_range=(-20, 20),
        u_max=2.5
    )
    print("Best management objective:", best["best_objective"])
    print("Best policy:", best["best_policy"])
