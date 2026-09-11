"""Small reproducible parameter experiments for Question 3."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import replace
from simulation import SimulationConfig, simulate
from verify_problem3 import verify_solution

def run_experiments(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Ablate the forecasts actually supplied by the problem.  This measures the
    # value of using later publications without inventing unavailable forecasts.
    schedules = ((), (6,), (12,), (18,), (6, 12), (6, 18),
                 (12, 18), (6, 12, 18))
    for revision_hours in schedules:
        cfg = SimulationConfig(attachment_dir=args.attachment_dir, horizon_steps=6,
            max_steps=args.max_steps or 144, n_scenarios=args.n_scenarios or 2,
            seed=args.seed, lookback_days=args.lookback_days, time_limit=args.time_limit,
            mip_rel_gap=args.mip_rel_gap, deterministic=args.deterministic,
            initial_soc=args.initial_soc, revision_hours=revision_hours,
            terminal_penalty=args.terminal_penalty if args.terminal_penalty is not None else .1,
            cvar_weight=args.cvar_weight if args.cvar_weight is not None else 0.0)
        result = simulate(cfg, args.date_start, args.date_end or args.date_start)
        check = verify_solution(result)
        rows.append(dict(revision_hours=list(revision_hours), costs=result.costs,
                         total_cost=result.total_cost,
                         emergency_energy=float(result.emergency[result.executed].sum()),
                         emergency_intervals=int((result.emergency[result.executed] > 1e-8).sum()),
                         minimum_soc=float(result.soc_after[result.executed].min()),
                         executed_steps=int(result.executed.sum()),
                         verified=check["passed"]))
    path = out / "experiments.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode":"experiments", "outputs":{"experiments":str(path)}, "runs":len(rows)}, ensure_ascii=False))
    return 0
