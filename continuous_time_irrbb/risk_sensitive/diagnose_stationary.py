"""Inspect a stationary checkpoint's actions and one-step Bellman values.

Example: python diagnose_stationary.py runs/stationary_refined/stationary_solution.pt --device cuda
"""

import argparse
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "continuous_time_irrbb.risk_sensitive"

from .stationary_solver import (StationarySolution, generate_stationary_cloud,
                                make_operator, stationary_continuation)
from .train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--action-count", type=int, default=None)
    args = parser.parse_args()
    if args.states < 2 or args.paths < 2 or args.years <= 0:
        raise ValueError("Require at least two states and paths, and positive years")
    device = choose_device(args.device)
    solution = StationarySolution.load(args.checkpoint, device=device)
    env, cfg = solution.environment, solution.config
    dtype = next(solution.value_net.parameters()).dtype
    action_count = args.action_count or cfg.action_count
    if action_count < 2:
        raise ValueError("--action-count must be at least 2")
    reference = torch.tensor([[.08, .5 * env.p.loans, .05625 * env.p.loans,
                               .12 * env.p.loans]], dtype=dtype, device=device)
    cloud = generate_stationary_cloud(env, args.states, cfg.seed + 10000,
                                       cfg.exploration_steps, device=device, dtype=dtype)
    continuation = stationary_continuation(env, solution.value_net)
    from dataclasses import replace
    neutral_operator = make_operator(env, replace(cfg, risk_aversion_per_loan=0), reference)
    with torch.no_grad():
        reference_result = solution.operator.maximize(
            0, reference, cfg.dt, continuation, action_count=action_count)
        cloud_result = solution.operator.maximize(
            0, cloud, cfg.dt, continuation, action_count=action_count)
        top2 = cloud_result.q_values.topk(2, dim=-1).values
        gaps = top2[..., 0] - top2[..., 1]
        print("parameters: s_F={:.4%}, s_V^0={:.4%}, eta={:.4%}, gamma_bar={:g}".format(
            env.p.fixed_spread, env.p.floating_spread_0,
            env.p.floating_demand_slope, cfg.risk_aversion_per_loan))
        print("floating spread: f=0 -> {:.4%}; f=0.5 -> {:.4%}".format(
            env.p.floating_spread_0 - env.p.floating_demand_slope,
            env.p.floating_spread_0 - .5 * env.p.floating_demand_slope))
        print("reference: f*={:.4f}, V={:.5f}".format(
            reference_result.action.item(), solution.value(reference).item()))
        for f in (0., .5, 1.):
            action = torch.full((1,), f, dtype=dtype, device=device)
            q = solution.operator.evaluate(0, reference, action, cfg.dt, continuation).item()
            q_neutral = neutral_operator.evaluate(0, reference, action, cfg.dt,
                                                  continuation).item()
            print("  f={:.1f}: Q_risk={:.6f}, Q_same_V_no_risk={:.6f}, local_risk_cost={:.6f}".format(
                f, q, q_neutral, q_neutral - q))
        actions = cloud_result.action
        print("exploration cloud ({} states): f=0 {:.1%}, 0<f<1 {:.1%}, f=1 {:.1%}".format(
            args.states, (actions == 0).float().mean().item(),
            ((actions > 0) & (actions < 1)).float().mean().item(),
            (actions == 1).float().mean().item()))
        print("Q gap to second-best grid action: median={:.6g}, q10={:.6g}".format(
            gaps.median().item(), torch.quantile(gaps, .1).item()))
        x = reference.expand(args.paths, -1).clone()
        loans = torch.full((args.paths,), env.p.loans, dtype=dtype, device=device)
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 20000)
        steps = max(1, round(args.years / cfg.dt))
        checkpoints = {0, steps, *[min(steps, round(y / cfg.dt)) for y in (1., 3.)]}
        for k in range(steps + 1):
            action = solution.action(x, time=k * cfg.dt, loans=loans,
                                     action_count=action_count)
            if k in checkpoints:
                print("on-policy t={:.2f}: mean f={:.4f}, f=0 {:.1%}, mean CAR={:.2%}".format(
                    k * cfg.dt, action.mean().item(), (action == 0).float().mean().item(),
                    (x[..., 3] / loans).mean().item()))
            if k < steps:
                noise = torch.randn(args.paths, generator=generator, dtype=dtype).to(device)
                transition = env.step(x, action, noise, cfg.dt, time=k * cfg.dt,
                                      loans=loans, validate=False)
                x, loans = transition.state, transition.next_loans
    report_path = args.checkpoint.with_name("report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("diagnostics"):
            info = report["diagnostics"][-1]
            print("saved validation Bellman RMSE={:.6g}; converged={}".format(
                info["bellman_rmse"], report.get("converged_on_validation_cloud")))
    print("The cloud is exploratory, not the stationary distribution under the learned policy.")
    print("The no-risk column keeps the risk-sensitive continuation fixed; it is not a risk-neutral solution.")


if __name__ == "__main__":
    main()
