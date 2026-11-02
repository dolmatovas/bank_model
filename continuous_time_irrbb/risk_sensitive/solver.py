"""Backward fitted Bellman solution for the finite-horizon risk-sensitive model.

Targets are computed from a frozen next-time value network. A separate value
network is fitted at every time layer. This is an approximate dynamic-programming
solver; its held-out Bellman errors are diagnostics, not an optimality proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from .bellman import BellmanOperator
from .environment import Environment, Parameters


class ValueNetwork(nn.Module):
    """Predict shareholder value as equity plus a scaled franchise adjustment."""

    def __init__(self, loans: float, width: int = 64, depth: int = 2):
        super().__init__()
        if width < 4 or depth < 1:
            raise ValueError("width must be >=4 and depth >=1")
        self.loans = loans
        self.adjustment_scale = loans / 10
        layers: list[nn.Module] = []
        size = 4
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        layers.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        z = torch.stack(((x[..., 0] - .08) / .08,
                         (x[..., 1] / self.loans - .5) * 2,
                         x[..., 2] / (self.loans * .1) - 1,
                         x[..., 3] / (self.loans * .1) - 1), -1)
        return x[..., 3] + self.adjustment_scale * self.net(z).squeeze(-1)


@dataclass(frozen=True)
class SolverConfig:
    horizon: float = 5.0
    steps: int = 60
    risk_aversion: float = 0.1
    quadrature_nodes: int = 7
    action_count: int = 33
    train_paths: int = 2048
    validation_paths: int = 512
    target_batch: int = 128
    train_batch: int = 256
    epochs: int = 100
    terminal_epochs: int = 250
    patience: int = 12
    learning_rate: float = 0.001
    width: int = 64
    depth: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        positive_ints = ("steps", "quadrature_nodes", "action_count", "train_paths",
                         "validation_paths", "target_batch", "train_batch", "epochs",
                         "terminal_epochs",
                         "patience", "width", "depth")
        for name in positive_ints:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.quadrature_nodes <= 64 or self.action_count < 2 or self.width < 4:
            raise ValueError("Need 2..64 quadrature nodes, >=2 actions and width >=4")
        if not math.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("horizon must be positive and finite")
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0:
            raise ValueError("risk_aversion must be finite and nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

    @property
    def dt(self) -> float:
        return self.horizon / self.steps


def sample_initial_states(n: int, p: Parameters, generator: torch.Generator,
                          *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Broad support plus a neighborhood of the reported reference bank."""
    u = torch.rand((n, 5), generator=generator, dtype=dtype)
    r = .02 + .16 * u[:, 0]
    f = p.loans * (.1 + .8 * u[:, 1])
    fixed_coupon = .03 + .13 * u[:, 2]
    old_float_spread = .005 + .02 * u[:, 3]
    y = f * fixed_coupon + (p.loans - f) * old_float_spread
    e = p.loans * (-.02 + .27 * u[:, 4])
    states = torch.stack((r, f, y, e), -1)
    anchor = n // 5
    if anchor:
        z = torch.randn((anchor, 4), generator=generator, dtype=dtype)
        anchor_r = (.08 + .015 * z[:, 0]).clamp_min(0)
        anchor_f = (p.loans * (.5 + .1 * z[:, 1])).clamp(0, p.loans)
        # Old contracts belong to the initial bank, not to the new pricing
        # curve being swept. Keep their coupon/spread mix fixed across runs.
        anchor_y = (anchor_f * .1 + (p.loans - anchor_f) * .0125
                    + .5 * z[:, 2]).clamp_min(0)
        anchor_e = p.loans * .12 + 2 * z[:, 3]
        states[:anchor] = torch.stack((anchor_r, anchor_f, anchor_y, anchor_e), -1)
    return states.to(device)


@torch.no_grad()
def generate_state_cloud(env: Environment, config: SolverConfig, n: int,
                         seed: int, *, device: torch.device,
                         dtype: torch.dtype) -> list[Tensor]:
    """Forward exploration with persistent fixed, floating and mixed regimes.

    Persistent regimes are essential: a path that chooses f≈1 for years reaches
    a very different part of state space from a path with i.i.d. random f.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = sample_initial_states(n, env.p, gen, device=device, dtype=dtype)
    noise = torch.randn((config.steps, n), generator=gen, dtype=dtype).to(device)
    uniforms = torch.rand((config.steps, n), generator=gen, dtype=dtype).to(device)
    regimes = torch.rand(n, generator=gen, dtype=dtype).to(device)
    anchor = n // 5
    if anchor:
        # Keep reference-like banks under each persistent portfolio regime.
        regimes[:anchor] = torch.linspace(0, .6, anchor, dtype=dtype, device=device)
    clouds = [x]
    for k in range(config.steps):
        action = torch.where(regimes < .2, torch.zeros_like(uniforms[k]),
                             torch.where(regimes < .4, torch.full_like(uniforms[k], .5),
                                         torch.where(regimes < .6, torch.ones_like(uniforms[k]),
                                                     uniforms[k])))
        x = env.step(x, action, noise[k], config.dt,
                     time=k * config.dt, validate=False).state
        clouds.append(x)
    return clouds


def _continuation(env: Environment, next_net: ValueNetwork | None
                  ) -> Callable[[float, Tensor], Tensor]:
    if next_net is None:
        return lambda t, x: env.terminal_value(x, time=t)
    return lambda _t, x: next_net(x)


@torch.no_grad()
def bellman_targets(operator: BellmanOperator, time: float, states: Tensor,
                    dt: float, next_value: Callable[[float, Tensor], Tensor],
                    *, action_count: int, batch_size: int,
                    fixed_action: float | None = None,
                    action_policy: Callable[[Tensor], Tensor] | None = None) -> Tensor:
    if fixed_action is not None and action_policy is not None:
        raise ValueError("Choose either a fixed action or an action policy")
    values: list[Tensor] = []
    for part in states.split(batch_size):
        if fixed_action is None and action_policy is None:
            target = operator.maximize(time, part, dt, next_value,
                                       action_count=action_count).value
        else:
            action = (action_policy(part) if action_policy is not None else
                      torch.full(part.shape[:-1], fixed_action,
                                 dtype=part.dtype, device=part.device))
            target = operator.evaluate(time, part, action, dt, next_value)
        values.append(target)
    return torch.cat(values)


def fit_value_network_for_environment(env: Environment, train_x: Tensor, train_y: Tensor,
                                      valid_x: Tensor, valid_y: Tensor,
                                      config: SolverConfig,
                                      *, previous: ValueNetwork | None,
                                      epoch_limit: int | None = None
                                      ) -> tuple[ValueNetwork, dict]:
    net = ValueNetwork(env.p.loans, config.width, config.depth).to(train_x)
    if previous is not None:
        net.load_state_dict(previous.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    best_state: dict[str, Tensor] | None = None
    best_mse = math.inf
    best_epoch = 0
    stale = 0
    limit = config.epochs if epoch_limit is None else epoch_limit
    if limit < 1:
        raise ValueError("epoch_limit must be positive")
    scale = net.adjustment_scale
    for epoch in range(limit):
        net.train()
        order = torch.randperm(len(train_x), device=train_x.device)
        for ids in order.split(config.train_batch):
            prediction = net(train_x[ids])
            loss = ((prediction - train_y[ids]) / scale).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            valid_mse = ((net(valid_x) - valid_y) / scale).square().mean().item()
        if valid_mse < best_mse - 1e-7:
            best_mse = valid_mse
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
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
    with torch.no_grad():
        errors = (net(valid_x) - valid_y).abs()
    diagnostics = {
        "epochs": best_epoch,
        "validation_rmse": math.sqrt(best_mse) * scale,
        "validation_mae": errors.mean().item(),
        "validation_q95_abs": torch.quantile(errors, .95).item(),
        "validation_max_abs": errors.max().item(),
    }
    return net, diagnostics


@dataclass
class FittedSolution:
    environment: Environment
    config: SolverConfig
    value_nets: list[ValueNetwork]

    def __post_init__(self) -> None:
        if len(self.value_nets) != self.config.steps:
            raise ValueError("Need one fitted value network per nonterminal time layer")
        self.operator = BellmanOperator(self.environment, self.config.risk_aversion,
                                        self.config.quadrature_nodes).to(next(self.value_nets[0].parameters()))

    def value(self, layer: int, state: Tensor) -> Tensor:
        if layer == self.config.steps:
            return self.environment.terminal_value(state, time=self.config.horizon)
        return self.value_nets[layer](state)

    def action(self, layer: int, state: Tensor, *, action_count: int | None = None) -> Tensor:
        if not 0 <= layer < self.config.steps:
            raise ValueError("layer must be a nonterminal time index")
        next_value = _continuation(self.environment,
                                   self.value_nets[layer + 1] if layer + 1 < self.config.steps else None)
        return self.operator.maximize(layer * self.config.dt, state, self.config.dt,
                                      next_value,
                                      action_count=action_count or self.config.action_count).action

    def save(self, path: str | Path, layer_diagnostics: list[dict]) -> None:
        torch.save({"parameters": asdict(self.environment.p),
                    "solver_config": asdict(self.config),
                    "networks": [{name: t.detach().cpu() for name, t in net.state_dict().items()}
                                 for net in self.value_nets],
                    "layer_diagnostics": layer_diagnostics}, path)

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu",
             dtype: torch.dtype = torch.float32) -> "FittedSolution":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        env = Environment(Parameters(**payload["parameters"])).to(device=device, dtype=dtype)
        config = SolverConfig(**payload["solver_config"])
        nets = []
        for state in payload["networks"]:
            net = ValueNetwork(env.p.loans, config.width, config.depth).to(device=device, dtype=dtype)
            net.load_state_dict(state)
            net.eval()
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            nets.append(net)
        return cls(env, config, nets)


def solve(env: Environment, config: SolverConfig,
          train_cloud: list[Tensor], valid_cloud: list[Tensor],
          *, fixed_action: float | None = None, epoch_limit: int | None = None,
          policy: Callable[[int, Tensor], Tensor] | None = None,
          progress: Callable[[int, dict], None] | None = None
          ) -> tuple[FittedSolution, list[dict]]:
    """Backward fitted dynamic programming on independent state clouds."""
    if fixed_action is not None and not 0 <= fixed_action <= 1:
        raise ValueError("fixed_action must be in [0,1]")
    if fixed_action is not None and policy is not None:
        raise ValueError("Choose either fixed_action or policy")
    if len(train_cloud) != config.steps + 1 or len(valid_cloud) != config.steps + 1:
        raise ValueError("Clouds must have one state array per time layer")
    operator = BellmanOperator(env, config.risk_aversion, config.quadrature_nodes).to(train_cloud[0])
    nets: list[ValueNetwork | None] = [None] * config.steps
    diagnostics: list[dict] = [{} for _ in range(config.steps)]
    for layer in range(config.steps - 1, -1, -1):
        continuation = _continuation(env, nets[layer + 1] if layer + 1 < config.steps else None)
        t = layer * config.dt
        layer_policy = ((lambda x, k=layer: policy(k, x)) if policy is not None else None)
        train_target = bellman_targets(operator, t, train_cloud[layer], config.dt,
                                       continuation, action_count=config.action_count,
                                       batch_size=config.target_batch, fixed_action=fixed_action,
                                       action_policy=layer_policy)
        valid_target = bellman_targets(operator, t, valid_cloud[layer], config.dt,
                                       continuation, action_count=config.action_count,
                                       batch_size=config.target_batch, fixed_action=fixed_action,
                                       action_policy=layer_policy)
        net, info = fit_value_network_for_environment(
            env, train_cloud[layer], train_target, valid_cloud[layer], valid_target,
            config, previous=nets[layer + 1] if layer + 1 < config.steps else None,
            epoch_limit=(epoch_limit if epoch_limit is not None else
                         config.terminal_epochs if layer == config.steps - 1 else config.epochs))
        nets[layer] = net
        diagnostics[layer] = {"layer": layer, "time": t, **info}
        if progress is not None:
            progress(layer, diagnostics[layer])
    return FittedSolution(env, config, nets), diagnostics  # type: ignore[arg-type]


@torch.no_grad()
def rollout_financial_values(solution: FittedSolution, initial: Tensor,
                             noises: Tensor, *, action_count: int | None = None,
                             fixed_action: float | None = None,
                             action_batch: int = 128) -> dict[str, float]:
    """Independent MC of cash payoffs; this is NOT recursive entropic value."""
    env, cfg = solution.environment, solution.config
    if noises.shape != (cfg.steps, initial.shape[0]):
        raise ValueError("noises must have shape (steps, paths)")
    x = initial
    discount = torch.ones(initial.shape[0], dtype=initial.dtype, device=initial.device)
    payoff = torch.zeros_like(discount)
    action_total = torch.zeros_like(discount)
    minimum_car = x[..., 3] / env.loans_at(0.0)
    for layer in range(cfg.steps):
        if fixed_action is None:
            action = torch.cat([solution.action(layer, part, action_count=action_count)
                                for part in x.split(action_batch)])
        else:
            action = torch.full_like(discount, fixed_action)
        step = env.step(x, action, noises[layer], cfg.dt,
                        time=layer * cfg.dt, validate=False)
        payoff = payoff + discount * step.dividend_pv
        discount = discount * step.discount
        action_total = action_total + action
        x = step.state
        minimum_car = torch.minimum(minimum_car,
                                    x[..., 3] / env.loans_at((layer + 1) * cfg.dt))
    terminal_car = x[..., 3] / env.loans_at(cfg.horizon)
    payoff = payoff + discount * env.terminal_value(x, time=cfg.horizon)
    return {"mean_payoff": payoff.mean().item(),
            "standard_error": (payoff.std(unbiased=True) / math.sqrt(len(payoff))).item(),
            "q05_payoff": torch.quantile(payoff, .05).item(),
            "q95_payoff": torch.quantile(payoff, .95).item(),
            "mean_fixed_share": (action_total / cfg.steps).mean().item(),
            "mean_terminal_car": terminal_car.mean().item(),
            "q05_terminal_car": torch.quantile(terminal_car, .05).item(),
            "mean_path_min_car": minimum_car.mean().item()}
