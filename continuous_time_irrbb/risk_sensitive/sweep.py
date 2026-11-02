"""Run a reproducible Cartesian parameter sweep of full-horizon solutions.

The same seed is passed to every run. This couples initial sampling and rate
shocks across parameter points where possible. Each run is isolated in its own
process and output directory; summary.csv is refreshed after every run.

Example (from repository root):
    python -m continuous_time_irrbb.risk_sensitive.sweep \
      --output continuous_time_irrbb/runs/sweep_001 \
      --gammas 0 0.1 0.5 1 --etas 0 0.005 0.01 \
      --spread-gaps 0 0.005 --device cuda
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Point:
    gamma: float
    eta: float
    spread_gap: float
    deposit_beta: float
    rate_vol: float
    amortization: float
    deposits: float


FIELDS = (
    "run_id", "status", "gamma", "eta", "spread_gap", "fixed_spread",
    "floating_spread_0", "deposit_beta", "rate_vol", "amortization",
    "deposits", "f0", "mean_fixed_share", "fitted_optimal_value_0",
    "independent_policy_value_0", "best_constant_value_0", "mean_cash_payoff",
    "terminal_validation_rmse", "quality_flags", "elapsed_seconds", "output",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New sweep directory")
    p.add_argument("--gammas", type=float, nargs="+", required=True)
    p.add_argument("--etas", type=float, nargs="+", required=True)
    p.add_argument("--spread-gaps", type=float, nargs="+", default=[0.0],
                   help="delta=s_F-s_V^0; default 0 (equal additive spreads)")
    p.add_argument("--floating-spread", type=float, default=.015)
    p.add_argument("--deposit-betas", type=float, nargs="+", default=[.35])
    p.add_argument("--rate-vols", type=float, nargs="+", default=[.08])
    p.add_argument("--amortizations", type=float, nargs="+", default=[1 / 3])
    p.add_argument("--deposits", type=float, nargs="+", default=[90.0])
    p.add_argument("--loan-growth", type=float, default=.05)
    p.add_argument("--dividend-fraction", type=float, default=.9)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--horizon", type=float, default=5.0)
    p.add_argument("--train-paths", type=int, default=2048)
    p.add_argument("--validation-paths", type=int, default=512)
    p.add_argument("--target-batch", type=int, default=128)
    p.add_argument("--train-batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--terminal-epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--action-count", type=int, default=33)
    p.add_argument("--quadrature-nodes", type=int, default=7)
    p.add_argument("--eval-paths", type=int, default=256)
    p.add_argument("--eval-action-count", type=int, default=17)
    p.add_argument("--policy-eval-epochs", type=int, default=None)
    p.add_argument("--baseline-epochs", type=int, default=None)
    p.add_argument("--baseline-risk", action="store_true",
                   help="Fit recursive values of f=0,0.5,1 at every grid point")
    p.add_argument("--skip-policy-risk", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Two-step integration check only")
    p.add_argument("--dry-run", action="store_true", help="Print grid without training")
    p.add_argument("--resume", action="store_true", help="Reuse completed run directories")
    p.add_argument("--max-runs", type=int, default=100,
                   help="Guard against an accidentally huge Cartesian grid")
    return p.parse_args()


def build_points(args: argparse.Namespace) -> list[Point]:
    if not math.isfinite(args.floating_spread) or args.floating_spread < 0:
        raise ValueError("floating-spread must be finite and nonnegative")
    points = [Point(*values) for values in itertools.product(
        args.gammas, args.etas, args.spread_gaps, args.deposit_betas,
        args.rate_vols, args.amortizations, args.deposits)]
    if len(points) > args.max_runs or args.max_runs < 1:
        raise ValueError(f"Grid has {len(points)} runs; increase --max-runs if intended")
    for point in points:
        if not all(math.isfinite(value) for value in asdict(point).values()):
            raise ValueError("All grid values must be finite")
        if point.gamma < 0 or not 0 <= point.eta <= args.floating_spread:
            raise ValueError("Require gamma>=0 and 0<=eta<=floating-spread")
        if args.floating_spread + point.spread_gap < 0:
            raise ValueError("Fixed spread s_V^0+spread_gap must be nonnegative")
        if not 0 <= point.deposit_beta <= 1 or point.rate_vol < 0:
            raise ValueError("Require beta in [0,1] and sigma>=0")
        if point.amortization <= 0 or point.deposits <= 0:
            raise ValueError("Amortization and deposits must be positive")
    return points


def train_command(args: argparse.Namespace, point: Point, output: Path) -> list[str]:
    cmd = [sys.executable, "-m", "continuous_time_irrbb.risk_sensitive.train",
           "--output", str(output), "--device", args.device, "--dtype", args.dtype,
           "--seed", str(args.seed), "--steps", str(args.steps),
           "--horizon", str(args.horizon),
           "--train-paths", str(args.train_paths),
           "--validation-paths", str(args.validation_paths),
           "--target-batch", str(args.target_batch),
           "--train-batch", str(args.train_batch),
           "--epochs", str(args.epochs),
           "--terminal-epochs", str(args.terminal_epochs),
           "--patience", str(args.patience),
           "--learning-rate", str(args.learning_rate),
           "--width", str(args.width), "--depth", str(args.depth),
           "--action-count", str(args.action_count),
           "--quadrature-nodes", str(args.quadrature_nodes),
           "--eval-paths", str(args.eval_paths),
           "--eval-action-count", str(args.eval_action_count),
           "--risk-aversion", str(point.gamma),
           "--floating-demand-slope", str(point.eta),
           "--floating-spread", str(args.floating_spread),
           "--fixed-spread", str(args.floating_spread + point.spread_gap),
           "--deposit-beta", str(point.deposit_beta),
           "--rate-vol", str(point.rate_vol),
           "--amortization", str(point.amortization),
           "--deposits", str(point.deposits)]
    cmd += ["--loan-growth", str(args.loan_growth),
            "--dividend-fraction", str(args.dividend_fraction)]
    if args.policy_eval_epochs is not None:
        cmd += ["--policy-eval-epochs", str(args.policy_eval_epochs)]
    if args.baseline_epochs is not None:
        cmd += ["--baseline-epochs", str(args.baseline_epochs)]
    if args.baseline_risk:
        cmd.append("--baseline-risk")
    if args.skip_policy_risk:
        cmd.append("--skip-policy-risk")
    if args.smoke:
        cmd.append("--smoke")
    return cmd


def report_row(run_id: str, point: Point, args: argparse.Namespace,
               run_dir: Path, status: str, report: dict | None = None) -> dict:
    row = {name: "" for name in FIELDS}
    row.update({"run_id": run_id, "status": status,
                "gamma": point.gamma, "eta": point.eta,
                "spread_gap": point.spread_gap,
                "fixed_spread": args.floating_spread + point.spread_gap,
                "floating_spread_0": args.floating_spread,
                "deposit_beta": point.deposit_beta,
                "rate_vol": point.rate_vol,
                "amortization": point.amortization,
                "deposits": point.deposits,
                "output": str(run_dir)})
    if report is not None:
        financial = report["financial_monte_carlo"]["learned_policy"]
        policy_eval = report.get("recursive_policy_evaluation") or {}
        baselines = report.get("fitted_recursive_baselines") or {}
        row.update({"f0": report["reference_fixed_share_0"],
                    "mean_fixed_share": financial["mean_fixed_share"],
                    "fitted_optimal_value_0": report["reference_fitted_value_0"],
                    "independent_policy_value_0": policy_eval.get("independent_fitted_value_0", ""),
                    "best_constant_value_0": (max(v["fitted_value_0"] for v in baselines.values())
                                               if baselines else ""),
                    "mean_cash_payoff": financial["mean_payoff"],
                    "terminal_validation_rmse": report["layer_diagnostics"][-1]["validation_rmse"],
                    "quality_flags": " | ".join(report.get("quality_flags", [])),
                    "elapsed_seconds": report["elapsed_seconds"]})
    return row


def write_summary(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    points = build_points(args)
    print(f"{len(points)} parameter points; same seed={args.seed} for every run", flush=True)
    if args.dry_run:
        for index, point in enumerate(points):
            print(f"run_{index:04d}: {point}")
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Sweep directory already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items() if key not in ("resume", "dry_run", "max_runs")}
    config_path = args.output / "sweep_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume parameters differ from the saved sweep configuration")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for index, point in enumerate(points):
        run_id = f"run_{index:04d}"
        run_dir = args.output / run_id
        report_path = run_dir / "report.json"
        if run_dir.exists() and not report_path.exists():
            suffix = 1
            while (args.output / f"{run_id}.incomplete_{suffix}").exists():
                suffix += 1
            backup = args.output / f"{run_id}.incomplete_{suffix}"
            run_dir.rename(backup)
            old_log = args.output / f"{run_id}.log"
            if old_log.exists():
                old_log.rename(args.output / f"{run_id}.incomplete_{suffix}.log")
            print(f"{run_id}: preserved incomplete run as {backup.name}; retrying", flush=True)
        if report_path.exists():
            print(f"{run_id}: reading completed run", flush=True)
            row = report_row(run_id, point, args, run_dir, "complete",
                             json.loads(report_path.read_text(encoding="utf-8")))
        else:
            print(f"{run_id}: gamma={point.gamma:g}, eta={point.eta:g}, "
                  f"gap={point.spread_gap:g}, beta={point.deposit_beta:g}, "
                  f"sigma={point.rate_vol:g}", flush=True)
            log_path = args.output / f"{run_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(train_command(args, point, run_dir),
                                           stdout=log, stderr=subprocess.STDOUT,
                                           check=False)
            if completed.returncode == 0 and report_path.exists():
                row = report_row(run_id, point, args, run_dir, "complete",
                                 json.loads(report_path.read_text(encoding="utf-8")))
                print(f"  f0={row['f0']:.3f}, mean f={row['mean_fixed_share']:.3f}, "
                      f"flags={bool(row['quality_flags'])}", flush=True)
            else:
                row = report_row(run_id, point, args, run_dir, "failed")
                print(f"  failed (exit={completed.returncode}); see {log_path}", flush=True)
        rows.append(row)
        write_summary(args.output / "summary.csv", rows)
    print(f"saved {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
