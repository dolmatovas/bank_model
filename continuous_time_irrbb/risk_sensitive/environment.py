"""IRRBB environment with (r,F,Y,E) plus an optional pathwise loan balance.

Both loan types amortize at the same exponential rate. Y stores fixed coupons
and floating spreads locked at origination, so no vintage ladder is needed.
The extra balance is needed when issuance responds to the short rate; the
stationary solver divides it out after each step and stays four-dimensional.
This module deliberately does not alter the existing v2 environment.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Parameters:
    loans: float = 100.0
    deposits: float = 90.0
    # Zero keeps checkpoints produced before the growing-book extension valid.
    loan_growth: float = 0.0
    # Zero sensitivity preserves the deterministic growth used by old checkpoints.
    growth_reference_rate: float = 0.08
    growth_rate_sensitivity: float = 0.0
    amortization: float = 1 / 3
    kappa_p: float = 0.5
    mean_rate_p: float = 0.08
    kappa_q: float = 0.5
    mean_rate_q: float = 0.08
    rate_vol: float = 0.08
    fixed_spread: float = 0.015
    floating_spread_0: float = 0.015
    floating_demand_slope: float = 0.005
    deposit_beta: float = 0.35
    deposit_spread: float = 0.005
    operating_cost: float = 0.01
    funding_spread: float = 0.02
    dividend_fraction: float = 0.5
    # In the perpetual model shareholders receive profits and fund losses.
    signed_owner_flow: bool = False
    quadrature_nodes: int = 64

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in ("quadrature_nodes", "signed_owner_flow"):
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.loans, self.deposits, self.amortization,
               self.kappa_p, self.kappa_q) <= 0:
            raise ValueError("Volumes, amortization and mean-reversion speeds must be positive")
        if not 0 <= self.deposit_beta <= 1 or not 0 <= self.dividend_fraction <= 1:
            raise ValueError("Deposit beta and dividend fraction must be in [0,1]")
        if not isinstance(self.signed_owner_flow, bool):
            raise ValueError("signed_owner_flow must be a boolean")
        if self.floating_demand_slope > self.floating_spread_0:
            raise ValueError("Floating spread must remain nonnegative for all actions")
        if not isinstance(self.quadrature_nodes, int) or not 8 <= self.quadrature_nodes <= 128:
            raise ValueError("quadrature_nodes must be an integer from 8 to 128")


@dataclass
class Step:
    state: Tensor
    discount: Tensor
    dividend_pv: Tensor
    dividend_rate: Tensor
    midpoint_equity: Tensor
    ex_cash_profit_mid: Tensor
    next_loans: Tensor
    midpoint_loans: Tensor
    book_growth: Tensor


class Environment(torch.nn.Module):
    """Amounts in one currency unit, rates per year; action f is fixed share."""

    def __init__(self, parameters: Parameters | None = None):
        super().__init__()
        self.p = parameters or Parameters()
        nodes, weights = np.polynomial.laguerre.laggauss(self.p.quadrature_nodes)
        self.register_buffer("laguerre_nodes", torch.tensor(nodes, dtype=torch.float64))
        self.register_buffer("laguerre_weights", torch.tensor(weights, dtype=torch.float64))

    def loans_at(self, time: float) -> float:
        """Book size by time only when growth is deterministic (or at time zero)."""
        if not math.isfinite(time) or time < 0:
            raise ValueError("Time must be finite and nonnegative")
        if self.p.growth_rate_sensitivity and time != 0:
            raise ValueError("Rate-dependent growth requires the pathwise loan balance")
        return self.p.loans * math.exp(self.p.loan_growth * time)

    def deposits_at(self, time: float) -> float:
        if self.p.growth_rate_sensitivity and time != 0:
            raise ValueError("Rate-dependent growth requires the pathwise loan balance")
        return self.p.deposits * math.exp(self.p.loan_growth * time)

    def growth_rate(self, rate: Tensor) -> Tensor:
        """Net book growth implied by nonnegative rate-sensitive new issuance."""
        p = self.p
        if p.growth_rate_sensitivity == 0:
            return torch.full_like(rate, p.loan_growth)
        issuance_per_loan = (p.amortization + p.loan_growth) * torch.exp(
            -p.growth_rate_sensitivity * (rate - p.growth_reference_rate))
        return issuance_per_loan - p.amortization

    def _loan_balance(self, x: Tensor, time: float, loans: Tensor | None) -> Tensor:
        if loans is None:
            return torch.full_like(x[..., 0], self.loans_at(time))
        if loans.shape != x.shape[:-1] or not torch.isfinite(loans).all() or (loans <= 0).any():
            raise ValueError("Pathwise loan balance must be positive, finite and match state batch")
        return loans

    def validate_state(self, x: Tensor, *, time: float = 0.0,
                       loans: Tensor | None = None) -> None:
        if x.ndim < 1 or x.shape[-1] != 4 or not x.is_floating_point():
            raise ValueError("State must have shape (...,4), ordered (r,F,Y,E)")
        if not torch.isfinite(x).all():
            raise ValueError("State must be finite")
        balance = self._loan_balance(x, time, loans)
        if ((x[..., 0] < 0) | (x[..., 1] < 0) | (x[..., 1] > balance + 1e-6)
                | (x[..., 2] < 0)).any():
            raise ValueError("Require r>=0, F in [0,L], Y>=0; E may be negative")

    def bond_q(self, rate: Tensor, maturity: Tensor) -> Tensor:
        """CIR Q zero-coupon price, including deterministic-rate limit."""
        p = self.p
        if p.rate_vol == 0:
            integral = (p.mean_rate_q * maturity
                        + (rate - p.mean_rate_q) * (-torch.expm1(-p.kappa_q * maturity)) / p.kappa_q)
            return torch.exp(-integral)
        g = math.sqrt(p.kappa_q**2 + 2 * p.rate_vol**2)
        decay = torch.exp(-g * maturity)
        denominator = g + p.kappa_q + (g - p.kappa_q) * decay
        b = 2 * (-torch.expm1(-g * maturity)) / denominator
        log_a = (2 * p.kappa_q * p.mean_rate_q / p.rate_vol**2) * (
            math.log(2 * g) + (p.kappa_q - g) * maturity / 2 - torch.log(denominator))
        return torch.exp(log_a - b * rate)

    def annuity_q(self, rate: Tensor) -> Tensor:
        p = self.p
        nodes = self.laguerre_nodes.to(rate)
        weights = self.laguerre_weights.to(rate)
        return (self.bond_q(rate.unsqueeze(-1), nodes / p.amortization) * weights).sum(-1) / p.amortization

    def new_fixed_coupon(self, rate: Tensor) -> Tensor:
        return self.annuity_q(rate).reciprocal() - self.p.amortization + self.p.fixed_spread

    def new_floating_spread(self, floating_share: Tensor) -> Tensor:
        return self.p.floating_spread_0 - self.p.floating_demand_slope * floating_share

    def cash(self, x: Tensor, time: float = 0.0,
             loans: Tensor | None = None) -> Tensor:
        balance = self._loan_balance(x, time, loans)
        return x[..., 3] + (self.p.deposits / self.p.loans - 1) * balance

    def funding_cost(self, x: Tensor, time: float = 0.0,
                     loans: Tensor | None = None) -> Tensor:
        return self.p.funding_spread * (-self.cash(x, time, loans)).clamp_min(0)

    def profit(self, x: Tensor, time: float = 0.0,
               loans: Tensor | None = None) -> Tensor:
        p = self.p
        r, f, y, _ = x.unbind(-1)
        balance = self._loan_balance(x, time, loans)
        deposits = balance * (p.deposits / p.loans)
        return (y + r * (balance - f) + r * self.cash(x, time, balance)
                - (p.deposit_beta * r + p.deposit_spread) * deposits
                - self.funding_cost(x, time, balance) - p.operating_cost * deposits)

    def dividend_rate(self, x: Tensor, time: float = 0.0,
                      loans: Tensor | None = None) -> Tensor:
        profit = self.profit(x, time, loans)
        return self.p.dividend_fraction * (profit if self.p.signed_owner_flow
                                           else profit.clamp_min(0))

    def loan_pv(self, x: Tensor, time: float = 0.0,
                loans: Tensor | None = None) -> Tensor:
        return (self._loan_balance(x, time, loans) - x[..., 1]
                + (x[..., 2] + self.p.amortization * x[..., 1]) * self.annuity_q(x[..., 0]))

    def terminal_value(self, x: Tensor, time: float = 0.0,
                       loans: Tensor | None = None) -> Tensor:
        """Signed liquidation: loans at Q market value, deposits at face."""
        balance = self._loan_balance(x, time, loans)
        return (self.cash(x, time, balance) + self.loan_pv(x, time, balance)
                - balance * (self.p.deposits / self.p.loans))

    def drift(self, x: Tensor, fixed_share: Tensor, time: float = 0.0,
              loans: Tensor | None = None) -> Tensor:
        p = self.p
        r, f, y, _ = x.unbind(-1)
        q = 1 - fixed_share
        balance = self._loan_balance(x, time, loans)
        issuance = (p.amortization + self.growth_rate(r)) * balance
        margin = fixed_share * self.new_fixed_coupon(r) + q * self.new_floating_spread(q)
        return torch.stack((p.kappa_p * (p.mean_rate_p - r),
                            issuance * fixed_share - p.amortization * f,
                            issuance * margin - p.amortization * y,
                            self.profit(x, time, balance)
                            - self.dividend_rate(x, time, balance)), -1)

    @staticmethod
    def exp_integral(rate: Tensor, dt: float) -> Tensor:
        z = rate * dt
        small = z.abs() < 1e-5
        safe = torch.where(small, torch.ones_like(z), z)
        ratio = torch.expm1(z) / safe
        series = 1 + z / 2 + z.square() / 6 + z**3 / 24
        return dt * torch.where(small, series, ratio)

    def step(self, x: Tensor, fixed_share: Tensor, noise: Tensor, dt: float,
             *, time: float = 0.0, validate: bool = True,
             fixed_coupon: Tensor | None = None,
             loans: Tensor | None = None) -> Step:
        """Frozen-rate exponential midpoint scheme; noise is standard normal."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if fixed_share.shape != x.shape[:-1] or noise.shape != x.shape[:-1]:
            raise ValueError("Action and noise must match the state batch shape")
        if fixed_coupon is not None and fixed_coupon.shape != x.shape[:-1]:
            raise ValueError("Cached fixed coupon must match the state batch shape")
        if validate:
            self.validate_state(x, time=time, loans=loans)
            if (not torch.isfinite(fixed_share).all() or not torch.isfinite(noise).all()
                    or ((fixed_share < 0) | (fixed_share > 1)).any()):
                raise ValueError("Require finite action in [0,1] and finite noise")
            if fixed_coupon is not None and not torch.isfinite(fixed_coupon).all():
                raise ValueError("Cached fixed coupon must be finite")
        p = self.p
        r, f, y, e = x.unbind(-1)
        balance = self._loan_balance(x, time, loans)
        growth = self.growth_rate(r)
        half_balance = balance * torch.exp(growth * dt / 2)
        next_balance = balance * torch.exp(growth * dt)
        q = 1 - fixed_share
        coupon = self.new_fixed_coupon(r) if fixed_coupon is None else fixed_coupon
        margin = fixed_share * coupon + q * self.new_floating_spread(q)
        half_decay = math.exp(-p.amortization * dt / 2)
        half_issuance = half_balance - half_decay * balance
        mid_f = half_decay * f + half_issuance * fixed_share
        mid_y = half_decay * y + half_issuance * margin
        forcing = self.profit(x, time, balance) - r * e - self.dividend_rate(x, time, balance)
        mid_e = torch.exp(r * dt / 2) * e + self.exp_integral(r, dt / 2) * forcing
        mid = torch.stack((r, mid_f, mid_y, mid_e), -1)
        dividend = self.dividend_rate(mid, time + dt / 2, half_balance)
        ex_cash_profit = self.profit(mid, time + dt / 2, half_balance) - r * mid_e
        next_e = torch.exp(r * dt) * e + self.exp_integral(r, dt) * (ex_cash_profit - dividend)
        decay = half_decay**2
        issuance = next_balance - decay * balance
        next_f = decay * f + issuance * fixed_share
        next_y = decay * y + issuance * margin
        if p.rate_vol == 0:
            next_r = p.mean_rate_p + (r - p.mean_rate_p) * math.exp(-p.kappa_p * dt)
        else:
            next_r = (r + p.kappa_p * (p.mean_rate_p - r) * dt
                      + p.rate_vol * torch.sqrt(r.clamp_min(0) * dt) * noise).clamp_min(0)
        state = torch.stack((next_r, next_f, next_y, next_e), -1)
        return Step(state, torch.exp(-r * dt),
                    dividend * self.exp_integral(-r, dt), dividend, mid_e,
                    ex_cash_profit, next_balance, half_balance,
                    next_balance / balance)
