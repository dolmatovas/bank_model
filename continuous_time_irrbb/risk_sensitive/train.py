"""Train a full-horizon fitted Bellman solution for the proposed IRRBB model.

Run from the repository root:
    python -m continuous_time_irrbb.risk_sensitive.train --output runs/example
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import torch

# Support both `python train.py` here and `python -m ...train` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "continuous_time_irrbb.risk_sensitive"

from .environment import Environment, Parameters
from .solver import (SolverConfig, bellman_targets, generate_state_cloud,
                     rollout_financial_values, solve)


def choose_device(request: str) -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            request = "cuda"
        elif torch.backends.mps.is_available():
            request = "mps"
        else:
            request = "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(request)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--risk-aversion", type=float, default=.1)
    parser.add_argument("--quadrature-nodes", type=int, default=7)
    parser.add_argument("--action-count", type=int, default=33)
    parser.add_argument("--train-paths", type=int, default=2048)
    parser.add_argument("--validation-paths", type=int, default=512)
    parser.add_argument("--target-batch", type=int, default=128)
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--terminal-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-paths", type=int, default=256)
    parser.add_argument("--eval-action-count", type=int, default=17)
    parser.add_argument("--baseline-risk", action="store_true",
                        help="Also fit recursive entropic values for constant f=0,0.5,1")
    parser.add_argument("--skip-policy-risk", action="store_true",
                        help="Skip independent recursive evaluation of the learned policy")
    parser.add_argument("--policy-eval-epochs", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--floating-demand-slope", type=float, default=.005)
    parser.add_argument("--fixed-spread", type=float, default=None,
                        help="Fixed spread; defaults to --floating-spread")
    parser.add_argument("--floating-spread", type=float, default=.015)
    parser.add_argument("--deposit-beta", type=float, default=.35)
    parser.add_argument("--deposits", type=float, default=90.0)
    parser.add_argument("--loan-growth", type=float, default=.05,
                        help="Deterministic annual growth rate of loans and deposits")
    parser.add_argument("--dividend-fraction", type=float, default=.9,
                        help="Fraction of positive profit paid as dividends")
    parser.add_argument("--amortization", type=float, default=1 / 3)
    parser.add_argument("--funding-spread", type=float, default=.02)
    parser.add_argument("--rate-vol", type=float, default=.08)
    parser.add_argument("--kappa-p", type=float, default=.5)
    parser.add_argument("--mean-rate-p", type=float, default=.08)
    parser.add_argument("--kappa-q", type=float, default=.5)
    parser.add_argument("--mean-rate-q", type=float, default=.08)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny two-step run for checking installation and data flow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = SolverConfig(
        horizon=args.horizon, steps=args.steps, risk_aversion=args.risk_aversion,
        quadrature_nodes=args.quadrature_nodes, action_count=args.action_count,
        train_paths=args.train_paths, validation_paths=args.validation_paths,
        target_batch=args.target_batch, train_batch=args.train_batch,
        epochs=args.epochs, terminal_epochs=args.terminal_epochs,
        patience=args.patience, learning_rate=args.learning_rate,
        width=args.width, depth=args.depth, seed=args.seed)
    if args.smoke:
        config = replace(config, horizon=.25, steps=2, quadrature_nodes=5,
                         action_count=9, train_paths=128, validation_paths=64,
                         target_batch=64, train_batch=64, epochs=10,
                         terminal_epochs=10, patience=4,
                         width=32)
    p = Parameters(floating_demand_slope=args.floating_demand_slope,
                   fixed_spread=(args.floating_spread if args.fixed_spread is None
                                 else args.fixed_spread),
                   floating_spread_0=args.floating_spread,
                   deposit_beta=args.deposit_beta,
                   deposits=args.deposits,
                   loan_growth=args.loan_growth,
                   dividend_fraction=args.dividend_fraction,
                   amortization=args.amortization,
                   funding_spread=args.funding_spread,
                   rate_vol=args.rate_vol, kappa_p=args.kappa_p,
                   mean_rate_p=args.mean_rate_p,
                   kappa_q=args.kappa_q, mean_rate_q=args.mean_rate_q)
    if args.eval_paths < 2 or args.eval_action_count < 2:
        raise ValueError("eval-paths and eval-action-count must be >=2")
    if args.baseline_epochs is not None and args.baseline_epochs < 1:
        raise ValueError("baseline-epochs must be positive")
    if args.policy_eval_epochs is not None and args.policy_eval_epochs < 1:
        raise ValueError("policy-eval-epochs must be positive")
    env = Environment(p).to(device=device, dtype=dtype)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    print(f"device={device}, dtype={dtype}, layers={config.steps}, dt={config.dt:.6g}", flush=True)
    train_cloud = generate_state_cloud(env, config, config.train_paths, config.seed + 1,
                                       device=device, dtype=dtype)
    valid_cloud = generate_state_cloud(env, config, config.validation_paths, config.seed + 2,
                                       device=device, dtype=dtype)
    def progress(layer: int, info: dict) -> None:
        print(f"layer={layer:03d}/{config.steps - 1:03d} "
              f"val_rmse={info['validation_rmse']:.5g} "
              f"q95={info['validation_q95_abs']:.5g} "
              f"epochs={info['epochs']}", flush=True)
    solution, diagnostics = solve(env, config, train_cloud, valid_cloud, progress=progress)
    solution.save(args.output / "solution.pt", diagnostics)

    # Compare parameter points on exactly the same legacy book. Old coupons
    # must not change when the pricing of *new* loans changes.
    reference_y = .05625 * p.loans
    reference = torch.tensor([[.08, .5 * p.loans, reference_y, .12 * p.loans]],
                             dtype=dtype, device=device)
    with torch.no_grad():
        start_action = solution.action(0, reference).item()
        start_value = solution.value(0, reference).item()
        next_value = (lambda _t, x: solution.value(1, x))
        start_bellman = bellman_targets(
            solution.operator, 0, reference, config.dt, next_value,
            action_count=config.action_count, batch_size=1).item()

    eval_paths = 64 if args.smoke else args.eval_paths
    gen = torch.Generator(device="cpu").manual_seed(config.seed + 100)
    noises = torch.randn((config.steps, eval_paths), generator=gen,
                         dtype=dtype).to(device)
    starts = reference.expand(eval_paths, -1).clone()
    eval_actions = min(args.eval_action_count, config.action_count)
    financial = {
        "learned_policy": rollout_financial_values(solution, starts, noises,
                                                    action_count=eval_actions),
    }
    for action in (0.0, 0.5, 1.0):
        financial[f"constant_{action:g}"] = rollout_financial_values(
            solution, starts, noises, fixed_action=action)

    policy_risk = None
    if not args.skip_policy_risk:
        print("fitting independent recursive value of learned policy", flush=True)
        risk_train = generate_state_cloud(env, config, config.train_paths, config.seed + 3,
                                          device=device, dtype=dtype)
        risk_valid = generate_state_cloud(env, config, config.validation_paths, config.seed + 4,
                                          device=device, dtype=dtype)
        policy_eval, policy_diag = solve(
            env, config, risk_train, risk_valid,
            policy=lambda layer, x: solution.action(layer, x),
            epoch_limit=args.policy_eval_epochs)
        policy_eval.save(args.output / "policy_evaluation.pt", policy_diag)
        with torch.no_grad():
            evaluated_value = policy_eval.value(0, reference).item()
        policy_risk = {
            "independent_fitted_value_0": evaluated_value,
            "optimal_fit_minus_policy_evaluation": start_value - evaluated_value,
            "validation_rmse_0": policy_diag[0]["validation_rmse"],
            "note": "Separate fitted recursion on independent state clouds; not a rigorous bound.",
        }

    risk_baselines = {}
    best_constant_minus_optimum = None
    if args.baseline_risk:
        for action in (0.0, 0.5, 1.0):
            print(f"fitting recursive value of constant f={action:g}", flush=True)
            baseline, base_diag = solve(
                env, config, train_cloud, valid_cloud, fixed_action=action,
                epoch_limit=args.baseline_epochs)
            with torch.no_grad():
                risk_baselines[f"constant_{action:g}"] = {
                    "fitted_value_0": baseline.value(0, reference).item(),
                    "validation_rmse_0": base_diag[0]["validation_rmse"],
                }

    quality_flags = []
    if diagnostics[-1]["validation_rmse"] > .2:
        quality_flags.append("Terminal-layer fit is coarse; increase paths/epochs/network capacity.")
    if policy_risk is not None:
        if abs(policy_risk["optimal_fit_minus_policy_evaluation"]) > .05:
            quality_flags.append("Optimal and independently evaluated policy values differ materially.")
        mc = financial["learned_policy"]
        if policy_risk["independent_fitted_value_0"] > mc["mean_payoff"] + 3 * mc["standard_error"] + .1:
            quality_flags.append("Fitted risk value exceeds the Monte Carlo mean payoff beyond sampling tolerance.")
    if risk_baselines:
        best_baseline = max(item["fitted_value_0"] for item in risk_baselines.values())
        best_constant_minus_optimum = best_baseline - start_value
        if best_constant_minus_optimum > .02:
            quality_flags.append(
                f"Separately fitted constant-policy baseline exceeds fitted optimum by "
                f"{best_constant_minus_optimum:.4g}; refine before claiming superiority.")
        if policy_risk is not None and best_baseline - policy_risk["independent_fitted_value_0"] > .02:
            quality_flags.append("A constant-policy baseline exceeds independent learned-policy evaluation.")

    report = {
        "device": str(device), "dtype": args.dtype,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": asdict(p), "solver_config": asdict(config),
        "reference_state": reference[0].tolist(),
        "reference_fixed_share_0": start_action,
        "reference_fitted_value_0": start_value,
        "reference_bellman_target_0": start_bellman,
        "reference_bellman_residual_0": start_value - start_bellman,
        "layer_diagnostics": diagnostics,
        "financial_monte_carlo": financial,
        "recursive_policy_evaluation": policy_risk,
        "fitted_recursive_baselines": risk_baselines,
        "best_constant_minus_fitted_optimum": best_constant_minus_optimum,
        "quality_flags": quality_flags,
        "warning": "Financial Monte Carlo payoffs are not recursive entropic values. "
                   "Held-out Bellman errors are diagnostics, not a global optimality certificate.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.output / 'solution.pt'} and {args.output / 'report.json'}", flush=True)
    print(f"reference: f0={start_action:.4f}, V0={start_value:.5f}, "
          f"Bellman residual={start_value - start_bellman:.5g}", flush=True)
    for flag in quality_flags:
        print(f"QUALITY WARNING: {flag}", flush=True)


if __name__ == "__main__":
    main()
