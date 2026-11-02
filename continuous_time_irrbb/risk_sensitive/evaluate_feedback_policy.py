"""Re-evaluate a learned stationary feedback policy against a frozen-f run.

The policy is greedy with respect to the saved optimal value approximation,
but its value is fitted independently under the same Bellman objective and
state cloud as the frozen-policy evaluator. This avoids comparing two value
functions with unrelated fixed-point level errors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "continuous_time_irrbb.risk_sensitive"

from .evaluate_frozen_policy import cloud_pair
from .stationary_solver import (StationarySolution, fit_projection,
                                make_operator, stationary_continuation)
from .train import choose_device


@torch.no_grad()
def policy_targets(operator, states, actions, dt, continuation, batch_size):
    values = []
    for batch, action in zip(states.split(batch_size), actions.split(batch_size)):
        values.append(operator.evaluate(0, batch, action, dt, continuation))
    return torch.cat(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("frozen_run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--action-count", type=int, default=129)
    args = p.parse_args()
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    frozen_report = json.loads((args.frozen_run / "report.json").read_text(encoding="utf-8"))
    config = StationarySolution.load(args.frozen_run / "frozen_value.pt",
                                     device=device, dtype=dtype).config
    source = StationarySolution.load(args.checkpoint, device=device, dtype=dtype)
    frozen = StationarySolution.load(args.frozen_run / "frozen_value.pt",
                                     device=device, dtype=dtype)
    if frozen.environment.p != source.environment.p or frozen.config.dt != source.config.dt:
        raise ValueError("The feedback and frozen policies use different environments")
    env = source.environment
    base = torch.tensor([frozen_report["reference_state"]],
                        device=device, dtype=dtype)
    shocked = base + torch.tensor([frozen_report["rate_shock"], 0., 0., 0.],
                                  device=device, dtype=dtype)
    anchors = torch.cat((base, shocked))
    f0 = frozen_report["frozen_fixed_share"]
    train_x = cloud_pair(env, config.train_states, config.seed + 1,
                         base, frozen_report["rate_shock"], f0,
                         config.exploration_steps, device, dtype)
    valid_x = cloud_pair(env, config.validation_states, config.seed + 2,
                         base, frozen_report["rate_shock"], f0,
                         config.exploration_steps, device, dtype)
    with torch.no_grad():
        train_a = torch.cat([source.action(part, action_count=args.action_count)
                             for part in train_x.split(config.target_batch)])
        valid_a = torch.cat([source.action(part, action_count=args.action_count)
                             for part in valid_x.split(config.target_batch)])
        anchor_a = source.action(anchors, action_count=args.action_count)
    net = source.value_net
    operator = make_operator(env, config, base)
    diagnostics = []
    converged = False
    for iteration in range(1, config.iterations + 1):
        continuation = stationary_continuation(env, net)
        train_y = policy_targets(operator, train_x, train_a, config.dt,
                                 continuation, config.target_batch)
        valid_y = policy_targets(operator, valid_x, valid_a, config.dt,
                                 continuation, config.target_batch)
        new_net, fit = fit_projection(net, train_x, train_y,
                                      valid_x, valid_y, config)
        new_cont = stationary_continuation(env, new_net)
        valid_target = policy_targets(operator, valid_x, valid_a, config.dt,
                                      new_cont, config.target_batch)
        anchor_target = operator.evaluate(0, anchors, anchor_a,
                                          config.dt, new_cont)
        with torch.no_grad():
            residual = new_net(valid_x) - valid_target
            anchor_residual = new_net(anchors) - anchor_target
            anchor_change = (new_net(anchors) - net(anchors)).abs().max().item()
        info = {"iteration": iteration, **fit,
                "bellman_rmse": residual.square().mean().sqrt().item(),
                "bellman_q95_abs": torch.quantile(residual.abs(), .95).item(),
                "anchor_max_abs": anchor_residual.abs().max().item(),
                "anchor_value_change": anchor_change}
        diagnostics.append(info)
        if iteration == 1 or iteration % 10 == 0 or iteration == config.iterations:
            print(f"iter={iteration:04d} bellman_rmse={info['bellman_rmse']:.6g} "
                  f"q95={info['bellman_q95_abs']:.6g} "
                  f"anchor_max={info['anchor_max_abs']:.6g} "
                  f"anchor_change={anchor_change:.6g}", flush=True)
        net = new_net
        if (info["bellman_rmse"] <= config.residual_tolerance
                and info["bellman_q95_abs"] <= 3 * config.residual_tolerance
                and info["anchor_max_abs"] <= config.residual_tolerance
                and anchor_change <= config.residual_tolerance):
            converged = True
            break
    evaluated = StationarySolution(env, config, net)
    with torch.no_grad():
        feedback_values = evaluated.value(anchors).cpu().tolist()
        frozen_values = frozen.value(anchors).cpu().tolist()
        source_values = source.value(anchors).cpu().tolist()
    report = {"source_checkpoint": str(args.checkpoint),
              "frozen_run": str(args.frozen_run),
              "config": asdict(config), "dtype": args.dtype,
              "reference_state": frozen_report["reference_state"],
              "rate_shock": frozen_report["rate_shock"],
              "frozen_fixed_share": f0,
              "feedback_action_count": args.action_count,
              "feedback_value_base": feedback_values[0],
              "feedback_value_shock": feedback_values[1],
              "frozen_value_base": frozen_values[0],
              "frozen_value_shock": frozen_values[1],
              "feedback_delta_v": feedback_values[1]-feedback_values[0],
              "frozen_delta_v": frozen_values[1]-frozen_values[0],
              "premium_base": feedback_values[0]-frozen_values[0],
              "premium_shock": feedback_values[1]-frozen_values[1],
              "change_in_adaptation_premium":
                  (feedback_values[1]-frozen_values[1])
                  -(feedback_values[0]-frozen_values[0]),
              "source_delta_v": source_values[1]-source_values[0],
              "converged_on_validation_cloud": converged,
              "diagnostics": diagnostics,
              "warning": "Both values are fitted infinite-horizon policy evaluations, not certified exact values. Feedback is frozen as a state-dependent mapping from the source checkpoint."}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({"parameters": asdict(env.p), "stationary_config": asdict(config),
                "network": {k: v.detach().cpu() for k, v in net.state_dict().items()}},
               args.output / "feedback_value.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(f"saved {args.output}; converged={converged}; "
          f"feedback ΔV={report['feedback_delta_v']:.6f}; "
          f"frozen ΔV={report['frozen_delta_v']:.6f}; "
          f"change in adaptation premium={report['change_in_adaptation_premium']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
