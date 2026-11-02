"""Evaluate a permanently fixed issuance mix against a stationary IRRBB policy.

The frozen policy keeps the *new-issuance* fixed share f0 chosen in an unshocked
reference state for every future date and path. Existing loans still amortize,
the book still grows, and all cash flows use the original recursive entropic
Bellman objective. This is not contractual run-off MtM.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "continuous_time_irrbb.risk_sensitive"

from .solver import bellman_targets
from .stationary_solver import (StationarySolution, fit_projection,
                                generate_stationary_cloud, make_operator,
                                rescale_amounts, stationary_continuation)
from .train import choose_device


def local_cloud(env, n, seed, reference, shock, action, steps):
    """Independent normalized rollouts from base and shocked reference states."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    device, dtype = reference.device, reference.dtype
    start = reference.expand(n, -1).clone()
    start[n // 2:] = start[n // 2:] + torch.tensor(
        [shock, 0., 0., 0.], device=device, dtype=dtype)
    depth = torch.randint(0, steps + 1, (n,), generator=gen).to(device)
    cloud = start.clone()
    state = start
    dt = .25
    for k in range(steps):
        noise = torch.randn(n, generator=gen, dtype=dtype).to(device)
        fixed = torch.full((n,), action, device=device, dtype=dtype)
        transition = env.step(state, fixed, noise, dt, time=0,
                              validate=False)
        state = rescale_amounts(transition.state,
                                transition.book_growth.reciprocal())
        selected = depth == k + 1
        cloud[selected] = state[selected]
    return cloud


def cloud_pair(env, n, seed, reference, shock, action, steps, device, dtype):
    broad = generate_stationary_cloud(env, n // 2, seed, steps,
                                      device=device, dtype=dtype)
    local = local_cloud(env, n - n // 2, seed + 100, reference, shock,
                        action, steps)
    return torch.cat((broad, local), dim=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--reference-state", type=float, nargs=4, metavar=("R", "F", "Y", "E"))
    p.add_argument("--shock", type=float, default=.01)
    p.add_argument("--action-count", type=int, default=129)
    p.add_argument("--train-states", type=int, default=4096)
    p.add_argument("--validation-states", type=int, default=1024)
    p.add_argument("--exploration-steps", type=int, default=24)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--epochs-per-iteration", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--residual-tolerance", type=float, default=.003)
    p.add_argument("--seed", type=int, default=1943)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.shock <= 0 or args.action_count < 2:
        raise ValueError("Need positive shock and at least two actions")
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    solution = StationarySolution.load(args.checkpoint, device=device, dtype=dtype)
    env = solution.environment
    source_report = args.checkpoint.with_name("report.json")
    if args.reference_state is not None:
        row = args.reference_state
    elif source_report.exists():
        row = json.loads(source_report.read_text(encoding="utf-8"))["reference_state"]
    else:
        raise ValueError("Pass --reference-state when the checkpoint has no report.json")
    base = torch.tensor([row], device=device, dtype=dtype)
    bumped = base + torch.tensor([args.shock, 0., 0., 0.], device=device, dtype=dtype)
    env.validate_state(base)
    env.validate_state(bumped)
    with torch.no_grad():
        f0 = solution.action(base, action_count=args.action_count).item()
    config = replace(solution.config, train_states=args.train_states,
                     validation_states=args.validation_states,
                     exploration_steps=args.exploration_steps,
                     iterations=args.iterations,
                     epochs_per_iteration=args.epochs_per_iteration,
                     patience=args.patience, learning_rate=args.learning_rate,
                     residual_tolerance=args.residual_tolerance, seed=args.seed)
    if args.smoke:
        config = replace(config, train_states=128, validation_states=64,
                         exploration_steps=3, iterations=3,
                         epochs_per_iteration=3)
    train_x = cloud_pair(env, config.train_states, config.seed + 1,
                         base, args.shock, f0, config.exploration_steps,
                         device, dtype)
    valid_x = cloud_pair(env, config.validation_states, config.seed + 2,
                         base, args.shock, f0, config.exploration_steps,
                         device, dtype)
    anchors = torch.cat((base, bumped))
    net = solution.value_net
    operator = make_operator(env, config, base)
    diagnostics = []
    converged = False
    for iteration in range(1, config.iterations + 1):
        continuation = stationary_continuation(env, net)
        train_y = bellman_targets(operator, 0, train_x, config.dt, continuation,
                                  action_count=2, batch_size=config.target_batch,
                                  fixed_action=f0)
        valid_y = bellman_targets(operator, 0, valid_x, config.dt, continuation,
                                  action_count=2, batch_size=config.target_batch,
                                  fixed_action=f0)
        new_net, fit = fit_projection(net, train_x, train_y,
                                      valid_x, valid_y, config)
        new_continuation = stationary_continuation(env, new_net)
        fixed_y = bellman_targets(operator, 0, valid_x, config.dt,
                                  new_continuation, action_count=2,
                                  batch_size=config.target_batch,
                                  fixed_action=f0)
        anchor_y = bellman_targets(operator, 0, anchors, config.dt,
                                   new_continuation, action_count=2,
                                   batch_size=2, fixed_action=f0)
        with torch.no_grad():
            residual = new_net(valid_x) - fixed_y
            anchor_residual = new_net(anchors) - anchor_y
            anchor_values = new_net(anchors)
        info = {"iteration": iteration, **fit,
                "bellman_rmse": residual.square().mean().sqrt().item(),
                "bellman_q95_abs": torch.quantile(residual.abs(), .95).item(),
                "anchor_max_abs": anchor_residual.abs().max().item(),
                "anchor_value_change": (anchor_values - net(anchors)).abs().max().item()}
        diagnostics.append(info)
        if iteration == 1 or iteration % 10 == 0 or iteration == config.iterations:
            print(f"iter={iteration:04d} bellman_rmse={info['bellman_rmse']:.6g} "
                  f"q95={info['bellman_q95_abs']:.6g} "
                  f"anchor_max={info['anchor_max_abs']:.6g} "
                  f"anchor_change={info['anchor_value_change']:.6g}", flush=True)
        net = new_net
        if (info["bellman_rmse"] <= config.residual_tolerance
                and info["bellman_q95_abs"] <= 3 * config.residual_tolerance
                and info["anchor_max_abs"] <= config.residual_tolerance
                and info["anchor_value_change"] <= config.residual_tolerance):
            converged = True
            break
    fixed_solution = StationarySolution(env, config, net)
    with torch.no_grad():
        optimal = solution.value(anchors).cpu().tolist()
        frozen = fixed_solution.value(anchors).cpu().tolist()
        adaptive_delta = optimal[1] - optimal[0]
        frozen_delta = frozen[1] - frozen[0]
        premium_base = optimal[0] - frozen[0]
        premium_shock = optimal[1] - frozen[1]
    report = {"checkpoint": str(args.checkpoint), "parameters": asdict(env.p),
              "config": asdict(config), "dtype": args.dtype, "reference_state": row,
              "rate_shock": args.shock, "frozen_fixed_share": f0,
              "adaptive_value_base": optimal[0], "adaptive_value_shock": optimal[1],
              "frozen_value_base": frozen[0], "frozen_value_shock": frozen[1],
              "adaptive_delta_v": adaptive_delta, "frozen_delta_v": frozen_delta,
              "premium_base": premium_base, "premium_shock": premium_shock,
              "change_in_adaptation_premium": premium_shock - premium_base,
              "converged_on_validation_cloud": converged,
              "diagnostics": diagnostics,
              "warning": "Fitted infinite-horizon policy evaluation; Bellman residuals are diagnostics, not certified error bounds. Frozen f means the same issuance mix at every future date, not contractual run-off."}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({"parameters": asdict(env.p), "stationary_config": asdict(config),
                "network": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "fixed_action": f0}, args.output / "frozen_value.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(f"saved {args.output}; converged={converged}; f0={f0:.4f}; "
          f"adaptive ΔV={adaptive_delta:.6f}; frozen ΔV={frozen_delta:.6f}; "
          f"change in adaptation premium={premium_shock-premium_base:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
