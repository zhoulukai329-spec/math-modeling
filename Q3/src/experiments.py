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
    # Keep the experiment bounded and reproducible while exercising the real rolling MILP.
    for penalty, cvar in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.2)):
        cfg = SimulationConfig(attachment_dir=args.attachment_dir, horizon_steps=6,
            max_steps=args.max_steps or 37, n_scenarios=args.n_scenarios or 2,
            seed=args.seed, lookback_days=args.lookback_days, time_limit=args.time_limit,
            mip_rel_gap=args.mip_rel_gap, deterministic=args.deterministic,
            initial_soc=args.initial_soc, terminal_penalty=penalty, cvar_weight=cvar)
        result = simulate(cfg, args.date_start, args.date_end or args.date_start)
        check = verify_solution(result)
        rows.append(dict(terminal_penalty=penalty, cvar_weight=cvar,
                         total_cost=result.total_cost, executed_steps=int(result.executed.sum()),
                         verified=check["passed"]))
    path = out / "experiments.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode":"experiments", "outputs":{"experiments":str(path)}, "runs":len(rows)}, ensure_ascii=False))
    return 0
