import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ir_risk

# -----------------------
# 2) Spread s(u): differentiable
# -----------------------

class Spread(nn.Module):
    def forward(self, u_rel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class LinearDecrisingSpread(Spread):
    """
    s(u) = max(s_min, s0 - a*log(1+u))
    """
    def __init__(self, s_min, s_max):
        super().__init__()
        self.s_min = float(s_min)
        self.s_max = float(s_max)

    def forward(self, u_rel: torch.Tensor) -> torch.Tensor:
        u_pos = torch.clamp(u_rel, min=0.0, max=2.0)
        s = self.s_max - (self.s_max - self.s_min) * u_pos / 2.0
        return s


# -----------------------
# 3) Policy u_theta: simple sigmoid parametric
# -----------------------

class SigmoidPolicy(nn.Module):
    """
    u_rel = u_min + (u_max-u_min)*sigmoid(a0 + a1*margin + a2*(K-K*)/K_scale)
    margin = cbar - r
    Parameters are trainable.
    """
    def __init__(
        self,
        u_min: float = 0.0,
        u_max: float = 2.5,
        K_scale: float = 0.05,
        init: Optional[Tuple[float, float, float, float]] = None,  # (a0,a1,a2,K*)
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.K_scale = float(K_scale)

        if init is None:
            init = (0.0, 0.0, 0.0, 0.05)

        a0, a1, a2, K_star = init
        self.a0 = nn.Parameter(torch.tensor(a0, device=device, dtype=dtype))
        self.a1 = nn.Parameter(torch.tensor(a1, device=device, dtype=dtype))
        self.a2 = nn.Parameter(torch.tensor(a2, device=device, dtype=dtype))
        # K_star should be positive-ish; keep it unconstrained but you may want softplus
        self.K_star = nn.Parameter(torch.tensor(K_star, device=device, dtype=dtype))

    def forward(self, r: torch.Tensor, cbar: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        margin = cbar - r
        x = self.a0 + self.a1 * margin + self.a2 * ((K - self.K_star) / max(self.K_scale, 1e-8))
        sig = torch.sigmoid(x)
        return self.u_min + (self.u_max - self.u_min) * sig


# -----------------------
# 4) Balance sim: differentiable objective
# -----------------------

@dataclass(frozen=True)
class BalanceParams:
    T_loan: float = 5.0      # lambda = 1/T
    L0: float = 1.0
    cbar0: float = 0.03
    K0: float = 0.05


@dataclass(frozen=True)
class ObjectiveParams:
    horizon: float = 5.0
    rho: float = 0.0                 # optional extra discounting of profits
    k_surv: float = 0.01             # smoothness for survival sigmoid
    k_pen: float = 0.15              # weight of capital shortfall penalty
    k_pen_scale: float = 0.01        # smoothness for softplus(-K/scale)
    use_r_discount: bool = False     # if True: discount by DF(r); else only rho discount


def simulate_and_objective_torch(
    r: torch.Tensor,                 # [n_paths, n_steps+1], exogenous
    dt: float,
    bal: BalanceParams,
    spread: Spread,
    policy: SigmoidPolicy,
    obj: ObjectiveParams,
) -> torch.Tensor:
    """
    Differentiable objective:
      J = E[ sum_t disc_t * w_t * CF_t  - k_pen * softplus(-K_t/k_pen_scale)*dt ]
    where w_t = sigmoid(K_t/k_surv) approximates "alive".

    NOTE: We *do not* hard-stop at default to keep differentiability.
    """
    device = r.device
    dtype = r.dtype
    n_paths, n_steps_plus = r.shape
    n_steps = n_steps_plus - 1

    lam = 1.0 / bal.T_loan

    # states (require_grad False, but depend on policy outputs => autograd flows through)
    L = torch.full((n_paths,), bal.L0, device=device, dtype=dtype)
    cbar = torch.full((n_paths,), bal.cbar0, device=device, dtype=dtype)
    K = torch.full((n_paths,), bal.K0, device=device, dtype=dtype)

    # discounting
    if obj.use_r_discount:
        df_r = discount_factors_from_r(r, dt)  # [n, n_steps+1]
    else:
        df_r = torch.ones_like(r)

    if obj.rho > 0:
        t_grid = torch.arange(n_steps, device=device, dtype=dtype) * dt
        df_rho = torch.exp(-obj.rho * t_grid)  # [n_steps]
    else:
        df_rho = torch.ones((n_steps,), device=device, dtype=dtype)

    total = torch.zeros((), device=device, dtype=dtype)

    for t in range(n_steps):
        r_t = r[:, t]

        # control
        u_rel = policy(r_t, cbar, K)
        u_rel = torch.clamp(u_rel, min=0.0)

        # new coupon and spread
        s_t = spread(u_rel)
        c_new = r_t + s_t

        # cashflow and smooth survival
        cf = L * (cbar - r_t)  # instantaneous NII rate
        w = torch.sigmoid(K / obj.k_surv)  # ~1 if K>0, ~0 if K<0
        disc = df_r[:, t] * df_rho[t]

        # penalty for capital shortfall (smooth)
        pen = F.softplus((-K) / obj.k_pen_scale)  # ~0 if K>>0
        step_value = disc * (w * cf - obj.k_pen * pen) * dt

        total = total + step_value.mean()

        # state updates (Euler)
        L = L + lam * L * (u_rel - 1.0) * dt
        cbar = cbar + lam * (c_new - cbar) * dt
        K = K + (cf * w) * dt  # optionally only accrue profit while "alive"

    return total


# -----------------------
# 5) Training loop (Adam or LBFGS)
# -----------------------

def train_policy_adam(
    rate_params: RateParams,
    bal: BalanceParams,
    spread: Spread,
    policy: SigmoidPolicy,
    *,
    T: float,
    dt: float,
    n_paths: int,
    measure: Literal["P", "Q"] = "P",
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    obj: ObjectiveParams = ObjectiveParams(),
    lr: float = 5e-2,
    steps: int = 300,
    grad_clip: float = 10.0,
) -> None:
    policy.train()
    spread.train()

    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # фиксируем сценарии ставок для стабильного градиента (common random numbers)
    r = simulate_vasicek_torch(rate_params, T=T, dt=dt, n_paths=n_paths, measure=measure, seed=seed, device=device, dtype=dtype)

    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        J = simulate_and_objective_torch(r, dt, bal, spread, policy, obj)
        loss = -J  # maximize J
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)

        opt.step()

        if (k + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():
                p = {n: v.item() for n, v in policy.named_parameters()}
                print(f"step {k+1:4d}/{steps}  J={J.item():.6f}  params={p}")


def train_policy_lbfgs(
    rate_params: RateParams,
    bal: BalanceParams,
    spread: Spread,
    policy: SigmoidPolicy,
    *,
    T: float,
    dt: float,
    n_paths: int,
    measure: Literal["P", "Q"] = "P",
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    obj: ObjectiveParams = ObjectiveParams(),
    max_iter: int = 80,
) -> None:
    policy.train()
    spread.train()

    r = simulate_vasicek_torch(rate_params, T=T, dt=dt, n_paths=n_paths, measure=measure, seed=seed, device=device, dtype=dtype)

    opt = torch.optim.LBFGS(policy.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        J = simulate_and_objective_torch(r, dt, bal, spread, policy, obj)
        loss = -J
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        J = simulate_and_objective_torch(r, dt, bal, spread, policy, obj)
        p = {n: v.item() for n, v in policy.named_parameters()}
        print(f"LBFGS done  J={J.item():.6f}  params={p}")


# -----------------------
# 6) Example usage
# -----------------------

if __name__ == "__main__":
    device = "cpu"
    dtype = torch.float32

    rate = RateParams(kappa=0.3, theta_P=0.025, theta_Q=0.02, sigma=0.015, r0=0.02)
    bal = BalanceParams(T_loan=5.0, L0=1.0, cbar0=0.03, K0=0.05)

    spread = LogDecreasingSpread(s0=0.02, a=0.01, s_min=0.005)

    policy = SigmoidPolicy(
        u_min=0.0, u_max=2.5, K_scale=0.05,
        init=(0.0, 0.0, 0.0, 0.05),
        device=device, dtype=dtype
    ).to(device=device, dtype=dtype)

    obj = ObjectiveParams(
        horizon=5.0,
        rho=0.0,
        k_surv=0.01,
        k_pen=0.15,
        k_pen_scale=0.01,
        use_r_discount=False,
    )

    # Adam (robust)
    train_policy_adam(
        rate_params=rate, bal=bal, spread=spread, policy=policy,
        T=obj.horizon, dt=1/252, n_paths=4000, measure="P",
        seed=123, device=device, dtype=dtype, obj=obj,
        lr=5e-2, steps=300
    )

    # Optionally: polish with LBFGS afterwards
    train_policy_lbfgs(
        rate_params=rate, bal=bal, spread=spread, policy=policy,
        T=obj.horizon, dt=1/252, n_paths=4000, measure="P",
        seed=123, device=device, dtype=dtype, obj=obj,
        max_iter=60
    )
