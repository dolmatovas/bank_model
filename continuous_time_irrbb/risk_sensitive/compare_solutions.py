"""Compare two solver resolutions for the same economic model and horizon.

This is an empirical stability check, not a certified optimality bound.
Run from the repository root with --coarse and --fine pointing to run folders.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from .solver import FittedSolution, sample_initial_states


def load_run(path: Path, device: str, dtype: torch.dtype):
    solution = FittedSolution.load(path / "solution.pt", device=device, dtype=dtype)
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    return solution, report


@torch.no_grad()
def evaluate(solution: FittedSolution, states: torch.Tensor, layer: int,
             action_count: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, actions = [], []
    for part in states.split(batch):
        values.append(solution.value(layer, part).cpu())
        actions.append(solution.action(layer, part, action_count=action_count).cpu())
    return torch.cat(values), torch.cat(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--action-count", type=int, default=129)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if args.states < 2 or args.action_count < 2 or args.batch < 1:
        parser.error("Require states>=2, action-count>=2 and batch>=1")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    coarse, coarse_report = load_run(args.coarse, args.device, dtype)
    fine, fine_report = load_run(args.fine, args.device, dtype)
    if vars(coarse.environment.p) != vars(fine.environment.p):
        parser.error("Economic parameters differ; this is not a resolution comparison")
    if not math.isclose(coarse.config.horizon, fine.config.horizon):
        parser.error("Horizons differ")
    generator = torch.Generator(device="cpu").manual_seed(20261016)
    states = sample_initial_states(args.states, coarse.environment.p, generator,
                                   device=torch.device(args.device), dtype=dtype)
    reference = torch.tensor(coarse_report["reference_state"], device=args.device, dtype=dtype)
    if coarse_report["reference_state"] != fine_report["reference_state"]:
        parser.error("Reference states differ")
    states[0] = reference

    print(f"coarse: {coarse.config.steps} steps, {coarse.config.action_count} actions, "
          f"{coarse.config.quadrature_nodes} quadrature nodes")
    print(f"fine:   {fine.config.steps} steps, {fine.config.action_count} actions, "
          f"{fine.config.quadrature_nodes} quadrature nodes")
    for fraction in (0, .5, .9):
        c_layer = min(round(fraction * coarse.config.steps), coarse.config.steps - 1)
        f_layer = min(round(fraction * fine.config.steps), fine.config.steps - 1)
        c_time, f_time = c_layer * coarse.config.dt, f_layer * fine.config.dt
        if not math.isclose(c_time, f_time, abs_tol=1e-8):
            print(f"t≈{fraction * coarse.config.horizon:.2f}: skipped; "
                  f"grid times differ ({c_time:.4f} versus {f_time:.4f})")
            continue
        c_val, c_action = evaluate(coarse, states, c_layer, args.action_count, args.batch)
        f_val, f_action = evaluate(fine, states, f_layer, args.action_count, args.batch)
        vd, ad = (c_val - f_val).abs(), (c_action - f_action).abs()
        print(f"t={c_time:.3f}: reference f={c_action[0]:.3f} -> {f_action[0]:.3f}, "
              f"V={c_val[0]:.4f} -> {f_val[0]:.4f}; "
              f"sample mean|Δf|={ad.mean():.3f}, q95|Δf|={torch.quantile(ad, .95):.3f}, "
              f"mean|ΔV|={vd.mean():.4f}, q95|ΔV|={torch.quantile(vd, .95):.4f}")
    for label, report in (("coarse", coarse_report), ("fine", fine_report)):
        errors = [item["validation_rmse"] for item in report["layer_diagnostics"]]
        independent = report.get("recursive_policy_evaluation") or {}
        print(f"{label}: max layer RMSE={max(errors):.4f}, "
              f"mean layer RMSE={sum(errors) / len(errors):.4f}, "
              f"Bellman residual at reference={report['reference_bellman_residual_0']:.4f}, "
              f"optimal fit - independent policy value="
              f"{independent.get('optimal_fit_minus_policy_evaluation', 'unavailable')}")
        for flag in report.get("quality_flags", []):
            print(f"  QUALITY WARNING: {flag}")


if __name__ == "__main__":
    main()
