"""Fitted Bellman fixed point for the perpetual, growing-book IRRBB model.

The network represents value in currency units for a reference bank of size L0.
With rate-dependent growth, the absolute value also requires the current
pathwise loan balance: V(L,x) = (L/L0) * v(rescale(x,L0/L)).
Risk aversion in currency units is gamma_bar / L(t), so gamma_bar is constant
per unit of the loan book. No terminal liquidation payoff enters this solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .bellman import BellmanOperator
from .environment import Environment, Parameters
from .solver import bellman_targets, sample_initial_states


def rescale_amounts(state: Tensor, factor: float | Tensor) -> Tensor:
    """Scale F, Y and E, leaving the rate unchanged."""
    amount_factor = factor.unsqueeze(-1) if isinstance(factor, Tensor) else factor
    return torch.cat((state[..., :1], state[..., 1:] * amount_factor), dim=-1)


class StationaryValueNetwork(nn.Module):
    def __init__(self, loans: float, width: int = 64, depth: int = 2):
        super().__init__()
        self.loans = loans
        layers: list[nn.Module] = []
        size = 4
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        last = nn.Linear(size, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        z = torch.stack(((x[..., 0] - .08) / .08,
                         (x[..., 1] / self.loans - .5) * 2,
                         x[..., 2] / (self.loans * .1) - 1,
                         x[..., 3] / (self.loans * .1) - 1), -1)
        return self.loans * self.net(z).squeeze(-1)


@dataclass(frozen=True)
class StationaryConfig:
    dt: float = .25
    risk_aversion_per_loan: float = 10.0
    quadrature_nodes: int = 7
    action_count: int = 33
    train_states: int = 2048
    validation_states: int = 512
    exploration_steps: int = 12
    target_batch: int = 128
    train_batch: int = 256
    iterations: int = 100
    epochs_per_iteration: int = 20
    patience: int = 6
    learning_rate: float = .001
    width: int = 64
    depth: int = 2
    residual_tolerance: float = .05
    seed: int = 42

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(self.risk_aversion_per_loan) or self.risk_aversion_per_loan < 0:
            raise ValueError("risk_aversion_per_loan must be finite and nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be positive and finite")
        positive = ("quadrature_nodes", "action_count", "train_states", "validation_states",
                    "exploration_steps", "target_batch", "train_batch", "iterations",
                    "epochs_per_iteration", "patience", "width", "depth")
        if any(not isinstance(getattr(self, name), int) or getattr(self, name) < 1
               for name in positive):
            raise ValueError("All count parameters must be positive integers")
        if not 2 <= self.quadrature_nodes <= 64 or self.action_count < 2:
            raise ValueError("Need 2..64 quadrature nodes and at least two actions")


@torch.no_grad()
def generate_stationary_cloud(env: Environment, n: int, seed: int,
                              exploration_steps: int, *, device: torch.device,
                              dtype: torch.dtype) -> Tensor:
    """Explore persistent portfolio regimes, then return states at the same L0."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = sample_initial_states(n, env.p, gen, device=device, dtype=dtype)
    depths = torch.randint(0, exploration_steps + 1, (n,), generator=gen).to(device)
    regimes = torch.rand(n, generator=gen, dtype=dtype).to(device)
    cloud = x.clone()
    for k in range(exploration_steps):
        # A quarter-year exploration step is independent of the Bellman time step.
        dt = .25
        random_action = torch.rand(n, generator=gen, dtype=dtype).to(device)
        action = torch.where(regimes < .25, torch.zeros_like(random_action),
                             torch.where(regimes < .5, torch.ones_like(random_action),
                                         torch.where(regimes < .75,
                                                     torch.full_like(random_action, .5),
                                                     random_action)))
        noise = torch.randn(n, generator=gen, dtype=dtype).to(device)
        step = env.step(x, action, noise, dt, time=0, validate=False)
        x = rescale_amounts(step.state, step.book_growth.reciprocal())
        selected = depths == k + 1
        cloud[selected] = x[selected]
    return cloud


def stationary_continuation(env: Environment, net: StationaryValueNetwork):
    def continuation(time: float, next_state: Tensor,
                     *, book_growth: Tensor | None = None) -> Tensor:
        if book_growth is None:
            growth = math.exp(env.p.loan_growth * time)
            return growth * net(rescale_amounts(next_state, 1 / growth))
        return book_growth * net(rescale_amounts(next_state, book_growth.reciprocal()))
    return continuation


def make_operator(env: Environment, config: StationaryConfig,
                  state: Tensor) -> BellmanOperator:
    return BellmanOperator(env, config.risk_aversion_per_loan / env.p.loans,
                           config.quadrature_nodes).to(state)


def fit_projection(previous: StationaryValueNetwork, train_x: Tensor, train_y: Tensor,
                   valid_x: Tensor, valid_y: Tensor, config: StationaryConfig
                   ) -> tuple[StationaryValueNetwork, dict]:
    net = StationaryValueNetwork(previous.loans, config.width, config.depth).to(train_x)
    net.load_state_dict(previous.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    best_loss = math.inf
    best_state = None
    stale = 0
    for epoch in range(config.epochs_per_iteration):
        order = torch.randperm(len(train_x), device=train_x.device)
        for ids in order.split(config.train_batch):
            loss = ((net(train_x[ids]) - train_y[ids]) / net.loans).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            valid_loss = ((net(valid_x) - valid_y) / net.loans).square().mean().item()
        if valid_loss < best_loss - 1e-9:
            best_loss = valid_loss
            best_state = {name: value.detach().clone()
                          for name, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, {"projection_rmse": math.sqrt(best_loss) * net.loans,
                 "epochs": epoch + 1}


@dataclass
class StationarySolution:
    environment: Environment
    config: StationaryConfig
    value_net: StationaryValueNetwork

    def __post_init__(self) -> None:
        self.operator = make_operator(self.environment, self.config,
                                      next(self.value_net.parameters()))

    def _book_scale(self, state: Tensor, time: float,
                    loans: Tensor | None) -> float | Tensor:
        if loans is not None:
            if (loans.shape != state.shape[:-1] or not torch.isfinite(loans).all()
                    or (loans <= 0).any()):
                raise ValueError("loans must be positive, finite and match the state batch")
            return loans / self.environment.p.loans
        if self.environment.p.growth_rate_sensitivity and time != 0:
            raise ValueError("Pass pathwise loans when growth depends on the rate")
        return math.exp(self.environment.p.loan_growth * time)

    def value(self, state: Tensor, *, time: float = 0.0,
              loans: Tensor | None = None) -> Tensor:
        growth = self._book_scale(state, time, loans)
        return growth * self.value_net(rescale_amounts(state, 1 / growth))

    @torch.no_grad()
    def action(self, state: Tensor, *, time: float = 0.0,
               action_count: int | None = None,
               loans: Tensor | None = None) -> Tensor:
        growth = self._book_scale(state, time, loans)
        x = rescale_amounts(state, 1 / growth)
        return self.operator.maximize(0, x, self.config.dt,
                                      stationary_continuation(self.environment, self.value_net),
                                      action_count=action_count or self.config.action_count).action

    def save(self, path: str | Path, diagnostics: list[dict]) -> None:
        torch.save({"parameters": asdict(self.environment.p),
                    "stationary_config": asdict(self.config),
                    "network": {k: v.detach().cpu() for k, v in self.value_net.state_dict().items()},
                    "diagnostics": diagnostics}, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu",
             dtype: torch.dtype = torch.float32) -> "StationarySolution":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        env = Environment(Parameters(**payload["parameters"])).to(device=device, dtype=dtype)
        config = StationaryConfig(**payload["stationary_config"])
        net = StationaryValueNetwork(env.p.loans, config.width, config.depth).to(
            device=device, dtype=dtype)
        net.load_state_dict(payload["network"])
        net.eval()
        return cls(env, config, net)


def solve_stationary(env: Environment, config: StationaryConfig,
                     *, device: torch.device, dtype: torch.dtype,
                     progress=None,
                     initial_net: StationaryValueNetwork | None = None
                     ) -> tuple[StationarySolution, list[dict], bool]:
    if not env.p.signed_owner_flow:
        raise ValueError("Stationary solver requires signed_owner_flow=True")
    train_x = generate_stationary_cloud(env, config.train_states, config.seed + 1,
                                         config.exploration_steps, device=device, dtype=dtype)
    valid_x = generate_stationary_cloud(env, config.validation_states, config.seed + 2,
                                         config.exploration_steps, device=device, dtype=dtype)
    net = StationaryValueNetwork(env.p.loans, config.width, config.depth).to(
        device=device, dtype=dtype)
    if initial_net is not None:
        net.load_state_dict(initial_net.state_dict())
    operator = make_operator(env, config, train_x)
    diagnostics: list[dict] = []
    converged = False
    for iteration in range(1, config.iterations + 1):
        continuation = stationary_continuation(env, net)
        train_y = bellman_targets(operator, 0, train_x, config.dt, continuation,
                                   action_count=config.action_count,
                                   batch_size=config.target_batch)
        valid_y = bellman_targets(operator, 0, valid_x, config.dt, continuation,
                                   action_count=config.action_count,
                                   batch_size=config.target_batch)
        new_net, fit = fit_projection(net, train_x, train_y, valid_x, valid_y, config)
        new_continuation = stationary_continuation(env, new_net)
        fixed_target = bellman_targets(operator, 0, valid_x, config.dt,
                                        new_continuation, action_count=config.action_count,
                                        batch_size=config.target_batch)
        with torch.no_grad():
            residual = (new_net(valid_x) - fixed_target).abs()
            action_old = operator.maximize(0, valid_x[:64], config.dt, continuation,
                                           action_count=config.action_count).action
            action_new = operator.maximize(0, valid_x[:64], config.dt, new_continuation,
                                           action_count=config.action_count).action
            action_change = (action_new - action_old).abs().mean().item()
        info = {"iteration": iteration, **fit,
                "bellman_rmse": residual.square().mean().sqrt().item(),
                "bellman_q95_abs": torch.quantile(residual, .95).item(),
                "mean_action_change": action_change}
        diagnostics.append(info)
        if progress is not None:
            progress(info)
        net = new_net
        if (info["bellman_rmse"] <= config.residual_tolerance
                and info["bellman_q95_abs"] <= 3 * config.residual_tolerance
                and action_change <= 1 / (config.action_count - 1)):
            converged = True
            break
    return StationarySolution(env, config, net), diagnostics, converged
