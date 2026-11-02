"""One-step recursive entropic Bellman operator for the proposed environment."""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .environment import Environment


@dataclass
class BellmanResult:
    action: Tensor
    value: Tensor
    action_grid: Tensor
    q_values: Tensor


class BellmanOperator(torch.nn.Module):
    """Compute one-step values using Gaussian quadrature over the rate shock.

    The discrete objective is dividend_pv + exp(-r*dt) CE_gamma(V_next).
    Discount is outside CE: this is a recursive preference in time-t units,
    not one exponential utility of the whole time-zero-discounted cash flow.
    The Euler rate step is projected at zero; convergence near zero needs care.
    """

    def __init__(self, environment: Environment, risk_aversion: float,
                 quadrature_nodes: int = 9):
        super().__init__()
        if not math.isfinite(risk_aversion) or risk_aversion < 0:
            raise ValueError("Risk aversion must be finite and nonnegative")
        if not isinstance(quadrature_nodes, int) or not 2 <= quadrature_nodes <= 64:
            raise ValueError("quadrature_nodes must be an integer from 2 to 64")
        self.environment = environment
        self.risk_aversion = risk_aversion
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
        weights = weights / math.sqrt(math.pi)
        weights = weights / weights.sum()
        self.register_buffer("normal_nodes", torch.tensor(nodes * math.sqrt(2), dtype=torch.float64))
        self.register_buffer("normal_weights", torch.tensor(weights, dtype=torch.float64))

    def evaluate(self, time: float, state: Tensor, fixed_share: Tensor, dt: float,
                 next_value: Callable[[float, Tensor], Tensor],
                 *, fixed_coupon: Tensor | None = None) -> Tensor:
        """Bellman Q(t,x,f); action and state have matching batch shapes.

        `next_value(t+dt, next_state)` must return one scalar per next state.
        This method is differentiable through the transition and continuation.
        """
        if not math.isfinite(time):
            raise ValueError("time must be finite")
        self.environment.validate_state(state, time=time)
        if fixed_share.shape != state.shape[:-1]:
            raise ValueError("fixed_share must match the state batch shape")
        if not torch.isfinite(fixed_share).all() or ((fixed_share < 0) | (fixed_share > 1)).any():
            raise ValueError("fixed_share must be finite and in [0,1]")
        if fixed_coupon is not None and fixed_coupon.shape != state.shape[:-1]:
            raise ValueError("Cached fixed coupon must match the state batch shape")
        n = self.normal_nodes.numel()
        batch = state.shape[:-1]
        expanded_state = state.unsqueeze(-2).expand(*batch, n, 4)
        expanded_action = fixed_share.unsqueeze(-1).expand(*batch, n)
        shocks = self.normal_nodes.to(state).view(*((1,) * len(batch)), n).expand(*batch, n)
        coupon = self.environment.new_fixed_coupon(state[..., 0]) if fixed_coupon is None else fixed_coupon
        step = self.environment.step(expanded_state, expanded_action, shocks, dt,
                                     time=time, validate=False,
                                     fixed_coupon=coupon.unsqueeze(-1).expand(*batch, n))
        if self.environment.p.growth_rate_sensitivity:
            future = next_value(time + dt, step.state,
                                book_growth=step.book_growth)
        else:
            future = next_value(time + dt, step.state)
        if future.shape != (*batch, n) or not torch.isfinite(future).all():
            raise ValueError("next_value must return finite values with state batch shape")
        weights = self.normal_weights.to(state).view(*((1,) * len(batch)), n)
        if self.risk_aversion == 0:
            certainty_equivalent = (weights * future).sum(-1)
        else:
            log_weights = weights.log()
            certainty_equivalent = -torch.logsumexp(
                log_weights - self.risk_aversion * future, dim=-1) / self.risk_aversion
        return step.dividend_pv[..., 0] + step.discount[..., 0] * certainty_equivalent

    def maximize(self, time: float, state: Tensor, dt: float,
                 next_value: Callable[[float, Tensor], Tensor],
                 *, action_count: int = 65) -> BellmanResult:
        """Vectorized bounded search on a uniform action grid, endpoints included.

        This returns the maximum over the specified grid, not a continuous
        global-optimality certificate. Increase action_count to check stability.
        """
        if not isinstance(action_count, int) or action_count < 2:
            raise ValueError("action_count must be an integer >= 2")
        self.environment.validate_state(state, time=time)
        batch = state.shape[:-1]
        grid = torch.linspace(0, 1, action_count, dtype=state.dtype, device=state.device)
        actions = grid.view(*((1,) * len(batch)), action_count).expand(*batch, action_count)
        states = state.unsqueeze(-2).expand(*batch, action_count, 4)
        coupon = self.environment.new_fixed_coupon(state[..., 0])
        q_values = self.evaluate(time, states, actions, dt, next_value,
                                 fixed_coupon=coupon.unsqueeze(-1).expand(*batch, action_count))
        index = q_values.argmax(-1, keepdim=True)
        best_action = torch.take_along_dim(actions, index, dim=-1).squeeze(-1)
        best_value = torch.take_along_dim(q_values, index, dim=-1).squeeze(-1)
        return BellmanResult(best_action, best_value, grid, q_values)
