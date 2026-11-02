"""Sweep gamma/eta while inheriting the exact environment from a checkpoint.

Each point starts from the same template weights, not from the previous point.
This is numerical initialization; policies are optimized anew for each objective.
Works as `python sweep_stationary.py ...` from risk_sensitive/ or as a module.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import torch

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='continuous_time_irrbb.risk_sensitive'

from .environment import Environment
from .stationary_solver import StationarySolution, solve_stationary
from .train import choose_device

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-solution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gammas',type=float,nargs='+',required=True)
    parser.add_argument('--etas',type=float,nargs='+',default=None,
                        help='Default: eta from template (one-dimensional gamma sweep)')
    parser.add_argument('--growth-rate-sensitivity',type=float,default=None,
                        help='Default: chi from template')
    parser.add_argument('--operating-cost',type=float,default=None,
                        help='Default: omega from template; 0 disables operating expenses')
    parser.add_argument('--device',choices=('auto','cpu','cuda','mps'),default='auto')
    parser.add_argument('--dtype',choices=('float32','float64'),default='float32')
    for name,kind in (('iterations',int),('train-states',int),('validation-states',int),
                      ('epochs-per-iteration',int),('learning-rate',float),
                      ('residual-tolerance',float),('seed',int)):
        parser.add_argument('--'+name,type=kind,default=None,help='Default: from template')
    parser.add_argument('--resume',action='store_true',help='Skip saved points, including unconverged ones')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Small integration run; no convergence expected')
    parser.add_argument('--max-runs',type=int,default=64)
    return parser.parse_args()

def main():
    args=parse_args()
    device=choose_device('cpu' if args.dry_run else args.device)
    dtype=torch.float64 if args.dtype=='float64' else torch.float32
    template=StationarySolution.load(args.template_solution,device=device,dtype=dtype)
    environment=template.environment.p
    if args.growth_rate_sensitivity is not None:
        environment=replace(environment,growth_rate_sensitivity=args.growth_rate_sensitivity)
    if args.operating_cost is not None:
        environment=replace(environment,operating_cost=args.operating_cost)
    gammas=list(dict.fromkeys(args.gammas))
    etas=list(dict.fromkeys(args.etas if args.etas is not None else [environment.floating_demand_slope]))
    if any(not math.isfinite(g) or g<0 for g in gammas):raise ValueError('Require finite gamma>=0')
    if any(not math.isfinite(e) or not 0<=e<=environment.floating_spread_0 for e in etas):
        raise ValueError('Require 0<=eta<=sV0')
    points=list(itertools.product(gammas,etas))
    if not 1<=len(points)<=args.max_runs:raise ValueError('Grid exceeds --max-runs')
    overrides={name:getattr(args,name) for name in ('iterations','train_states','validation_states',
                 'epochs_per_iteration','learning_rate','residual_tolerance','seed')
               if getattr(args,name) is not None}
    config=replace(template.config,**overrides)
    if args.smoke:
        config=replace(config,iterations=2,train_states=64,validation_states=32,
                       epochs_per_iteration=2,exploration_steps=2,action_count=9,quadrature_nodes=3)
    signature={'version':1,'template_sha256':hashlib.sha256(args.template_solution.read_bytes()).hexdigest(),
               'parameters':asdict(environment),'config':asdict(config),
               'gammas':gammas,'etas':etas,'dtype':args.dtype,'smoke':args.smoke}
    print(f'{len(points)} points; chi={environment.growth_rate_sensitivity:g}; '
          f'g0={environment.loan_growth:g}; alpha={environment.dividend_fraction:g}; '
          f'omega={environment.operating_cost:g}; '
          f'dt={config.dt:g}; iterations={config.iterations}',flush=True)
    for g,e in points:print(f'  gamma_bar={g:g}, eta={e:g}',flush=True)
    if args.dry_run:return
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'{args.output} exists; use a new output or --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'sweep_config.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=signature:raise ValueError('Resume settings differ from saved sweep')
    else:manifest.write_text(json.dumps(signature,ensure_ascii=False,indent=2))
    rows=[]
    for gamma,eta in points:
        run_id=f'g{gamma:g}_e{eta:g}'
        folder=args.output/run_id
        report_path=folder/'report.json'
        params=replace(environment,floating_demand_slope=eta)
        run_config=replace(config,risk_aversion_per_loan=gamma)
        if folder.exists():
            if not (report_path.exists() and (folder/'stationary_solution.pt').exists()):
                raise FileExistsError(f'Incomplete run: {folder}; move it aside before resuming')
            report=json.loads(report_path.read_text())
            if report['parameters']!=asdict(params) or report['stationary_config']!=asdict(run_config):
                raise ValueError(f'Saved point differs: {folder}')
            print('reuse',run_id,'converged=',report['converged_on_validation_cloud'],flush=True)
        else:
            torch.manual_seed(run_config.seed)
            env=Environment(params).to(device=device,dtype=dtype)
            def progress(info):
                if info['iteration']==1 or info['iteration']%25==0:
                    print(f"{run_id} iter={info['iteration']}: RMSE={info['bellman_rmse']:.6g}, "
                          f"q95={info['bellman_q95_abs']:.6g}",flush=True)
            solution,diagnostics,converged=solve_stationary(
                env,run_config,device=device,dtype=dtype,progress=progress,
                initial_net=template.value_net)
            # Same initial contracts at every parameter point; no eta-dependent old spread.
            source_report=args.template_solution.with_name('report.json')
            if source_report.exists():
                reference=json.loads(source_report.read_text())['reference_state']
            else:reference=[.08,.5*params.loans,.05625*params.loans,.12*params.loans]
            state=torch.tensor([reference],device=device,dtype=dtype)
            with torch.no_grad():
                value=solution.value(state).item();action=solution.action(state).item()
            report={'parameters':asdict(params),'stationary_config':asdict(run_config),
                    'reference_state':reference,'reference_value':value,'reference_fixed_share':action,
                    'converged_on_validation_cloud':converged,'diagnostics':diagnostics,
                    'initial_solution':str(args.template_solution),'device':str(device),'dtype':args.dtype,
                    'warning':'Residual convergence is diagnostic, not a global optimality bound.'}
            folder.mkdir(exist_ok=False)
            solution.save(folder/'stationary_solution.pt',diagnostics)
            report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print('saved',run_id,'converged=',converged,flush=True)
        rows.append({'run':run_id,'gamma':gamma,'eta':eta,'chi':params.growth_rate_sensitivity,
                     'reference_value':report['reference_value'],'f_reference':report['reference_fixed_share'],
                     'bellman_rmse':report['diagnostics'][-1]['bellman_rmse'],
                     'converged':report['converged_on_validation_cloud']})
        temporary=args.output/'summary.csv.tmp'
        with temporary.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        temporary.replace(args.output/'summary.csv')

if __name__=='__main__':main()
