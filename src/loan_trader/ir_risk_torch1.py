# torch_policy_training.py
# Differentiable training of parametric policy u_theta with:
#  - sampling initial states X0 ~ nu
#  - mini-batch Monte Carlo paths
#  - smooth default proxy for gradients
#  - optional hard-default evaluation (no grad)

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1) Short rate: Vasicek
# =========================

@dataclass(frozen=True)
class RateParams:
    kappa: float
    theta_P: float
    theta_Q: float
    sigma: float


def simulate_vasicek_paths(
    rate: RateParams,
    r0: torch.Tensor,                 # [B]
    *,
    T: float,
    dt: float,
    measure: Literal["P", "Q"] = "P",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Simulate Vasicek paths per batch element.
    r: [B, steps+1]
    """
    assert r0.ndim == 1
    B = r0.shape[0]
    steps = int(round(T / dt))
    theta = rate.theta_P if measure == "P" else rate.theta_Q

    gen = None
    if seed is not None:
        gen = torch.Generator(device=r0.device)
        gen.manual_seed(seed)

    z = torch.randn(B, steps, device=r0.device, dtype=r0.dtype, generator=gen)
    r = torch.empty(B, steps + 1, device=r0.device, dtype=r0.dtype)
    r[:, 0] = r0

    sqrt_dt = math.sqrt(dt)
    for t in range(steps):
        rt = r[:, t]
        r[:, t + 1] = rt + rate.kappa * (theta - rt) * dt + rate.sigma * sqrt_dt * z[:, t]
    return r


def discount_factors(r: torch.Tensor, dt: float) -> torch.Tensor:
    """DF: [B, steps+1], DF[:,0]=1"""
    cum = torch.cumsum(r[:, :-1] * dt, dim=1)
    df = torch.ones_like(r)
    df[:, 1:] = torch.exp(-cum)
    return df


# =========================
# 2) Spread s(u): differentiable
# =========================

class Spread(nn.Module):
    def forward(self, u_rel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LogDecreasingSpread(Spread):
    """
    s(u) = max(s_min, s0 - a*log(1+u))
    """
    def __init__(self, s0: float, a: float, s_min: float = 0.0):
        super().__init__()
        self.s0 = float(s0)
        self.a = float(a)
        self.s_min = float(s_min)

    def forward(self, u_rel: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u_rel, min=0.0)
        s = self.s0 - self.a * torch.log1p(u)
        return torch.clamp(s, min=self.s_min)


# =========================
# 3) Policy u_theta
# =========================

class SigmoidPolicy(nn.Module):
    """
    u_rel = u_min + (u_max-u_min) * sigmoid( a0 + a1*margin + a2*(ell-ell*) )
    margin = cbar - r
    ell = K/L  (capital ratio)
    """
    def __init__(
        self,
        *,
        u_min: float = 0.0,
        u_max: float = 2.5,
        init: Tuple[float, float, float, float] = (0.0, 0.0, 5.0, 0.05),  # a0,a1,a2,ell*
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        a0, a1, a2, ell_star = init
        self.a0 = nn.Parameter(torch.tensor(a0, device=device, dtype=dtype))
        self.a1 = nn.Parameter(torch.tensor(a1, device=device, dtype=dtype))
        self.a2 = nn.Parameter(torch.tensor(a2, device=device, dtype=dtype))
        # enforce ell_star > 0 via softplus
        self.ell_star_raw = nn.Parameter(torch.tensor(math.log(math.exp(ell_star) - 1.0), device=device, dtype=dtype))

    @property
    def ell_star(self) -> torch.Tensor:
        return F.softplus(self.ell_star_raw)

    def forward(self, r: torch.Tensor, cbar: torch.Tensor, L: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        margin = cbar - r
        ell = K / torch.clamp(L, min=1e-8)
        x = self.a0 + self.a1 * margin + self.a2 * (ell - self.ell_star)
        sig = torch.sigmoid(x)
        return self.u_min + (self.u_max - self.u_min) * sig


# =========================
# 4) Balance + objective (smooth default proxy)
# =========================

@dataclass(frozen=True)
class BalanceParams:
    T_loan: float = 5.0   # lambda=1/T
    # no fixed L0,K0,cbar0 here: we sample them


@dataclass(frozen=True)
class ObjectiveParams:
    horizon: float = 5.0
    rho: float = 0.0                 # optional discount of profit (under P)
    use_r_discount: bool = False     # if True multiply by DF from r_t
    # smooth "default/runoff" knobs
    k_surv: float = 0.01             # survival sigmoid smoothness in currency units of K
    k_pen: float = 0.2               # weight for capital shortfall penalty
    k_pen_scale: float = 0.01        # softplus scale for -K
    # regularization on issuance (optional)
    u_reg: float = 0.0               # weight for E[u^2]


def simulate_objective_smooth(
    r: torch.Tensor,                  # [B, steps+1]
    *,
    dt: float,
    bal: BalanceParams,
    spread: Spread,
    policy: SigmoidPolicy,
    obj: ObjectiveParams,
    L0: torch.Tensor,                 # [B]
    cbar0: torch.Tensor,              # [B]
    K0: torch.Tensor,                 # [B]
) -> torch.Tensor:
    """
    Differentiable objective J(theta) = E[ sum_t disc*( w*CF - k_pen*softplus(-K) - u_reg*u^2 ) dt ].
    Dynamics:
      dL   = lambda*L*(u_rel-1) dt
      dcbar= u_rel*lambda*(r + s(u_rel) - cbar) dt   <-- corrected dependence on u_rel
      dK   = w*L*(cbar-r) dt                         <-- accrue mainly while "alive"
    """
    B, steps_plus = r.shape
    steps = steps_plus - 1
    lam = 1.0 / bal.T_loan

    L = L0
    cbar = cbar0
    K = K0

    if obj.use_r_discount:
        df = discount_factors(r, dt)  # [B, steps+1]
    else:
        df = torch.ones_like(r)

    if obj.rho > 0:
        tgrid = torch.arange(steps, device=r.device, dtype=r.dtype) * dt
        df_rho = torch.exp(-obj.rho * tgrid)  # [steps]
    else:
        df_rho = torch.ones((steps,), device=r.device, dtype=r.dtype)

    total = torch.zeros((), device=r.device, dtype=r.dtype)

    for t in range(steps):
        rt = r[:, t]
        u_rel = policy(rt, cbar, L, K)
        u_rel = torch.clamp(u_rel, min=0.0)

        s = spread(u_rel)
        c_new = rt + s

        cf = L * (cbar - rt)  # instantaneous NII rate

        # smooth survival: ~1 when K>0, ~0 when K<0
        w = torch.sigmoid(K / obj.k_surv)

        pen = F.softplus((-K) / obj.k_pen_scale)  # ~0 if K>>0

        disc = df[:, t] * df_rho[t]
        step = disc * (w * cf - obj.k_pen * pen - obj.u_reg * (u_rel ** 2)) * dt
        total = total + step.mean()

        # state updates
        L = L + lam * L * (u_rel - 1.0) * dt
        cbar = cbar + (u_rel * lam) * (c_new - cbar) * dt   # <-- important corrected formula
        K = K + (w * cf) * dt

    return total


# =========================
# 5) Initial state sampler X0 ~ nu
# =========================

@dataclass(frozen=True)
class InitDist:
    # We recommend normalizing L0=1 and sampling ell0 = K0/L0.
    L0: float = 1.0

    # r0 distribution (truncated normal)
    r_mu: float = 0.02
    r_sigma: float = 0.01
    r_min: float = -0.01
    r_max: float = 0.10

    # margin m0 = cbar0 - r0 distribution
    m_min: float = 0.005
    m_max: float = 0.03

    # capital ratio ell0 = K0/L0 distribution
    ell_min: float = 0.01
    ell_max: float = 0.12

    # stress mixture
    w_stress: float = 0.25
    # stress tweaks
    stress_r_shift: float = 0.02     # higher starting rate
    stress_m_low: float = 0.002      # squeezed margin upper bound in stress
    stress_ell_max: float = 0.04     # low capital ratio in stress


def sample_initial_states(
    B: int,
    dist: InitDist,
    *,
    device="cpu",
    dtype=torch.float32,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)

    # stress indicator
    u = torch.rand(B, device=device, dtype=dtype, generator=g)
    is_stress = (u < dist.w_stress)

    # r0 ~ truncated normal
    r0 = dist.r_mu + dist.r_sigma * torch.randn(B, device=device, dtype=dtype, generator=g)
    r0 = torch.clamp(r0, min=dist.r_min, max=dist.r_max)
    # shift r0 up in stress
    r0 = r0 + is_stress.to(dtype) * dist.stress_r_shift

    # margin m0 uniform; in stress, squeeze margin
    m_hi = torch.full((B,), dist.m_max, device=device, dtype=dtype)
    m_hi = torch.where(is_stress, torch.full((B,), dist.stress_m_low, device=device, dtype=dtype), m_hi)
    m0 = dist.m_min + (m_hi - dist.m_min) * torch.rand(B, device=device, dtype=dtype, generator=g)
    cbar0 = r0 + m0

    # ell0 uniform; in stress, lower ell
    ell_hi = torch.full((B,), dist.ell_max, device=device, dtype=dtype)
    ell_hi = torch.where(is_stress, torch.full((B,), dist.stress_ell_max, device=device, dtype=dtype), ell_hi)
    ell0 = dist.ell_min + (ell_hi - dist.ell_min) * torch.rand(B, device=device, dtype=dtype, generator=g)

    L0 = torch.full((B,), dist.L0, device=device, dtype=dtype)
    K0 = ell0 * L0

    return {"r0": r0, "L0": L0, "cbar0": cbar0, "K0": K0, "is_stress": is_stress}


# =========================
# 6) Hard-default evaluation (no grad)
# =========================

@torch.no_grad()
def evaluate_hard_default(
    rate: RateParams,
    bal: BalanceParams,
    spread: Spread,
    policy: SigmoidPolicy,
    dist: InitDist,
    *,
    dt: float,
    horizon: float,
    B: int,
    n_mc: int = 1,                   # repeat with different rate noise seeds
    base_seed: int = 10,
    device="cpu",
    dtype=torch.float32,
) -> Dict[str, float]:
    steps = int(round(horizon / dt))
    lam = 1.0 / bal.T_loan

    total_J = 0.0
    total_pdef = 0.0

    for k in range(n_mc):
        X0 = sample_initial_states(B, dist, device=device, dtype=dtype, seed=base_seed + 1000 * k)
        r = simulate_vasicek_paths(rate, X0["r0"], T=horizon, dt=dt, measure="P", seed=base_seed + 2000 * k)

        L = X0["L0"].clone()
        cbar = X0["cbar0"].clone()
        K = X0["K0"].clone()

        defaulted = torch.zeros(B, device=device, dtype=torch.bool)
        alive = torch.ones(B, device=device, dtype=torch.bool)

        J = torch.zeros(B, device=device, dtype=dtype)

        for t in range(steps):
            rt = r[:, t]
            # after default: u=0 (runoff)
            u_rel = torch.zeros(B, device=device, dtype=dtype)
            idx = alive
            if idx.any():
                u_rel[idx] = torch.clamp(policy(rt[idx], cbar[idx], L[idx], K[idx]), min=0.0)

            s = spread(u_rel)
            c_new = rt + s
            cf = L * (cbar - rt)

            J = J + cf * dt

            # update states: if defaulted => u=0 => dL=-lam L dt and dcbar=0
            # we implement via masking
            # L
            L = torch.where(defaulted, L + (-lam * L) * dt, L + lam * L * (u_rel - 1.0) * dt)
            # cbar
            cbar = torch.where(defaulted, cbar, cbar + (u_rel * lam) * (c_new - cbar) * dt)
            # K
            K = K + cf * dt

            newly = (K <= 0.0) & alive
            defaulted = defaulted | newly
            alive = alive & (~newly)

        total_J += float(J.mean().item())
        total_pdef += float(defaulted.float().mean().item())

    return {"E_J": total_J / n_mc, "P_default": total_pdef / n_mc}


# =========================
# 7) Training loop (SGD/Adam)
# =========================

@dataclass(frozen=True)
class TrainConfig:
    dt: float = 1.0 / 252
    horizon: float = 5.0
    batch_size: int = 2048
    lr: float = 5e-2
    steps: int = 2000
    grad_clip: float = 5.0
    # Monte Carlo within batch: for simplicity 1 path per initial state;
    # increase by repeating simulation with different noise seeds and averaging.
    n_mc_per_step: int = 1
    # logging / eval
    log_every: int = 200
    eval_every: int = 500
    eval_batch: int = 5000


def train_policy(
    *,
    rate: RateParams,
    bal: BalanceParams,
    spread: Spread,
    policy: SigmoidPolicy,
    init_dist: InitDist,
    obj: ObjectiveParams,
    cfg: TrainConfig,
    device="cpu",
    dtype=torch.float32,
    seed: int = 0,
) -> None:
    policy.to(device=device, dtype=dtype)
    spread.to(device=device, dtype=dtype)
    policy.train()

    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)

        # Monte Carlo average over n_mc_per_step different rate noise seeds
        J_total = 0.0
        for m in range(cfg.n_mc_per_step):
            X0 = sample_initial_states(
                cfg.batch_size,
                init_dist,
                device=device,
                dtype=dtype,
                seed=seed + 10_000 * step + 1_000 * m,
            )
            r = simulate_vasicek_paths(
                rate,
                X0["r0"],
                T=cfg.horizon,
                dt=cfg.dt,
                measure="P",
                seed=seed + 20_000 * step + 2_000 * m,
            )
            J = simulate_objective_smooth(
                r,
                dt=cfg.dt,
                bal=bal,
                spread=spread,
                policy=policy,
                obj=obj,
                L0=X0["L0"],
                cbar0=X0["cbar0"],
                K0=X0["K0"],
            )
            J_total = J_total + J

        J_mean = J_total / float(cfg.n_mc_per_step)
        loss = -J_mean
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)

        opt.step()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                p = {n: float(v.detach().cpu().item()) for n, v in policy.named_parameters()}
                p["ell_star"] = float(policy.ell_star.detach().cpu().item())
                print(f"[step {step:5d}] J_smooth={float(J_mean.item()):.6f}  params={p}")

        if step % cfg.eval_every == 0:
            stats = evaluate_hard_default(
                rate, bal, spread, policy, init_dist,
                dt=cfg.dt, horizon=cfg.horizon,
                B=cfg.eval_batch, n_mc=1,
                base_seed=seed + 999 + step,
                device=device, dtype=dtype,
            )
            print(f"  eval(hard): E[J]={stats['E_J']:.6f}  P(def)={stats['P_default']:.4f}")

    print("Training done.")


# =========================
# 8) Example run
# =========================

if __name__ == "__main__":
    device = "cpu"
    dtype = torch.float32

    rate = RateParams(kappa=0.3, theta_P=0.025, theta_Q=0.02, sigma=0.015)
    bal = BalanceParams(T_loan=5.0)

    spread = LogDecreasingSpread(s0=0.02, a=0.01, s_min=0.005)

    policy = SigmoidPolicy(
        u_min=0.0, u_max=2.5,
        init=(0.0, 0.0, 8.0, 0.05),
        device=device, dtype=dtype,
    )

    # distribution of starting states X0
    init_dist = InitDist(
        L0=1.0,
        r_mu=0.02, r_sigma=0.01, r_min=-0.01, r_max=0.10,
        m_min=0.005, m_max=0.03,
        ell_min=0.01, ell_max=0.12,
        w_stress=0.25,
        stress_r_shift=0.02,
        stress_m_low=0.004,
        stress_ell_max=0.04,
    )

    obj = ObjectiveParams(
        horizon=5.0,
        rho=0.0,
        use_r_discount=False,  # under P objective, often leave False
        k_surv=0.01,
        k_pen=0.25,
        k_pen_scale=0.01,
        u_reg=0.0,
    )

    cfg = TrainConfig(
        dt=1/252,
        horizon=5.0,
        batch_size=2048,
        lr=5e-2,
        steps=2000,
        grad_clip=5.0,
        n_mc_per_step=1,
        log_every=200,
        eval_every=500,
        eval_batch=5000,
    )

    train_policy(
        rate=rate,
        bal=bal,
        spread=spread,
        policy=policy,
        init_dist=init_dist,
        obj=obj,
        cfg=cfg,
        device=device,
        dtype=dtype,
        seed=123,
    )

    # Final evaluation
    final_stats = evaluate_hard_default(
        rate, bal, spread, policy, init_dist,
        dt=cfg.dt, horizon=cfg.horizon, B=20000, n_mc=2,
        base_seed=2024, device=device, dtype=dtype
    )
    print("Final hard eval:", final_stats)
