"""Train a perpetual, time-independent IRRBB policy by fitted Bellman iteration.

From this directory: python train_stationary.py --output runs/stationary_001
From the repository root: python -m continuous_time_irrbb.risk_sensitive.train_stationary ...
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "continuous_time_irrbb.risk_sensitive"

from .environment import Environment, Parameters
from .stationary_solver import StationaryConfig, StationarySolution, solve_stationary
from .train import choose_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--dt", type=float, default=.25)
    p.add_argument("--risk-aversion-per-loan", type=float, default=10.0,
                   help="Dimensionless gamma on value per loan; 10 equals old gamma=.1 at L0=100")
    p.add_argument("--loan-growth", type=float, default=.05)
    p.add_argument("--growth-reference-rate", type=float, default=None,
                   help="r0; defaults to --mean-rate-p (0.08 by default)")
    p.add_argument("--growth-rate-sensitivity", type=float, default=5.0,
                   help="chi; 5 makes a +100 bp move cut new issuance by 4.88%%")
    p.add_argument("--dividend-fraction", type=float, default=.9)
    p.add_argument("--fixed-spread", type=float, default=None)
    p.add_argument("--floating-spread", type=float, default=.015)
    p.add_argument("--floating-demand-slope", type=float, default=.005)
    p.add_argument("--deposits", type=float, default=90.0)
    p.add_argument("--deposit-beta", type=float, default=.35)
    p.add_argument("--amortization", type=float, default=1 / 3)
    p.add_argument("--rate-vol", type=float, default=.08)
    p.add_argument("--kappa-p", type=float, default=.5)
    p.add_argument("--mean-rate-p", type=float, default=.08)
    p.add_argument("--kappa-q", type=float, default=.5)
    p.add_argument("--mean-rate-q", type=float, default=.08)
    p.add_argument("--funding-spread", type=float, default=.02)
    p.add_argument("--quadrature-nodes", type=int, default=7)
    p.add_argument("--action-count", type=int, default=33)
    p.add_argument("--train-states", type=int, default=2048)
    p.add_argument("--validation-states", type=int, default=512)
    p.add_argument("--exploration-steps", type=int, default=12)
    p.add_argument("--target-batch", type=int, default=128)
    p.add_argument("--train-batch", type=int, default=256)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--epochs-per-iteration", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--residual-tolerance", type=float, default=.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial-solution", type=Path, default=None,
                   help="Warm start from a stationary checkpoint, possibly with a different dt")
    p.add_argument("--smoke", action="store_true", help="Small integration check; no convergence expected")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    env = Environment(Parameters(
        deposits=args.deposits, loan_growth=args.loan_growth,
        growth_reference_rate=(args.mean_rate_p if args.growth_reference_rate is None
                               else args.growth_reference_rate),
        growth_rate_sensitivity=args.growth_rate_sensitivity,
        amortization=args.amortization, kappa_p=args.kappa_p,
        mean_rate_p=args.mean_rate_p, kappa_q=args.kappa_q,
        mean_rate_q=args.mean_rate_q, rate_vol=args.rate_vol,
        fixed_spread=(args.floating_spread if args.fixed_spread is None
                      else args.fixed_spread),
        floating_spread_0=args.floating_spread,
        floating_demand_slope=args.floating_demand_slope,
        deposit_beta=args.deposit_beta, funding_spread=args.funding_spread,
        dividend_fraction=args.dividend_fraction,
        signed_owner_flow=True)).to(device=device, dtype=dtype)
    config = StationaryConfig(
        dt=args.dt, risk_aversion_per_loan=args.risk_aversion_per_loan,
        quadrature_nodes=args.quadrature_nodes, action_count=args.action_count,
        train_states=args.train_states, validation_states=args.validation_states,
        exploration_steps=args.exploration_steps, target_batch=args.target_batch,
        train_batch=args.train_batch, iterations=args.iterations,
        epochs_per_iteration=args.epochs_per_iteration, patience=args.patience,
        learning_rate=args.learning_rate, width=args.width, depth=args.depth,
        residual_tolerance=args.residual_tolerance, seed=args.seed)
    if args.smoke:
        config = replace(config, action_count=9, quadrature_nodes=5,
                         train_states=128, validation_states=64,
                         exploration_steps=3, target_batch=64, train_batch=64,
                         iterations=3, epochs_per_iteration=5, width=32)
    initial_net = None
    if args.initial_solution is not None:
        previous = StationarySolution.load(args.initial_solution, device=device, dtype=dtype)
        old = previous.environment.p
        comparable = replace(env.p, loan_growth=old.loan_growth,
                             growth_reference_rate=old.growth_reference_rate,
                             growth_rate_sensitivity=old.growth_rate_sensitivity)
        if old != comparable:
            raise ValueError("Initial solution has different non-growth economic parameters")
        if (previous.config.risk_aversion_per_loan != config.risk_aversion_per_loan
                or previous.config.width != config.width
                or previous.config.depth != config.depth):
            raise ValueError("Initial solution has incompatible risk aversion or network architecture")
        initial_net = previous.value_net
    args.output.mkdir(parents=True, exist_ok=False)

    def progress(info: dict) -> None:
        print(f"iter={info['iteration']:03d} projection_rmse={info['projection_rmse']:.4g} "
              f"bellman_rmse={info['bellman_rmse']:.4g} "
              f"q95={info['bellman_q95_abs']:.4g} "
              f"mean_delta_f={info['mean_action_change']:.4g}", flush=True)

    solution, diagnostics, converged = solve_stationary(
        env, config, device=device, dtype=dtype, progress=progress,
        initial_net=initial_net)
    solution.save(args.output / "stationary_solution.pt", diagnostics)
    reference = torch.tensor([[.08, .5 * env.p.loans, .05625 * env.p.loans,
                               .12 * env.p.loans]], device=device, dtype=dtype)
    with torch.no_grad():
        value = solution.value(reference).item()
        action = solution.action(reference).item()
    report = {"parameters": asdict(env.p), "stationary_config": asdict(config),
              "device": str(device), "dtype": args.dtype,
              "reference_state": reference[0].tolist(),
              "initial_solution": (str(args.initial_solution) if args.initial_solution else None),
              "reference_value": value, "reference_fixed_share": action,
              "converged_on_validation_cloud": converged,
              "diagnostics": diagnostics,
              "warning": "Held-out Bellman residual is diagnostic, not a global optimality bound. "
                         "If converged is false, do not treat the saved policy as solved."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.output}; converged={converged}; "
          f"reference V={value:.5g}, f={action:.4g}", flush=True)


if __name__ == "__main__":
    main()
