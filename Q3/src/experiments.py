"""Small reproducible parameter experiments for Question 3."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from dataclasses import replace
from simulation import SimulationConfig, simulate
from verify_problem3 import verify_solution


def build_experiment_config(args, revision_hours):
    """Use the same maintained model as the formal event-policy run."""
    return SimulationConfig(
        attachment_dir=args.attachment_dir,
        horizon_steps=144,
        max_steps=args.max_steps if args.max_steps is not None else 144,
        n_scenarios=args.n_scenarios if args.n_scenarios is not None else 12,
        seed=args.seed,
        lookback_days=args.lookback_days,
        time_limit=args.time_limit,
        mip_rel_gap=args.mip_rel_gap,
        deterministic=args.deterministic,
        initial_soc=args.initial_soc,
        revision_hours=revision_hours,
        terminal_penalty=args.terminal_penalty if args.terminal_penalty is not None else 0.5,
        cvar_weight=args.cvar_weight if args.cvar_weight is not None else 0.0,
        backend="event-policy",
    )


def run_experiments(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Ablate the forecasts actually supplied by the problem.  This measures the
    # value of using later publications without inventing unavailable forecasts.
    schedules = ((), (6,), (12,), (18,), (6, 12), (6, 18),
                 (12, 18), (6, 12, 18))
    for day in args.experiment_dates:
        for revision_hours in schedules:
            cfg = build_experiment_config(args, revision_hours)
            result = simulate(cfg, day, day)
            check = verify_solution(result)
            if not check["passed"]:
                raise RuntimeError(
                    f"revision experiment failed verification for {day} {revision_hours}: "
                    f"{check['errors']}"
                )
            rows.append(dict(date=day, revision_hours=list(revision_hours), costs=result.costs,
                             total_cost=result.total_cost,
                             emergency_energy=float(result.emergency[result.executed].sum()),
                             emergency_intervals=int((result.emergency[result.executed] > 1e-8).sum()),
                             minimum_soc=float(result.soc_after[result.executed].min()),
                             end_soc=float(result.soc_after[result.executed][-1]),
                             elapsed_seconds=result.elapsed_seconds,
                             solve_count=len(result.solve_log),
                             executed_steps=int(result.executed.sum()),
                             verified=check["passed"]))
    path = out / "experiments.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = out / "experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ("date", "revision_hours", "total_cost", "emergency_energy",
                  "emergency_intervals", "minimum_soc", "end_soc", "elapsed_seconds",
                  "solve_count", "executed_steps", "verified")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("+".join(map(str, row[key])) if key == "revision_hours" else row[key])
                             for key in fields})
    print(json.dumps({"mode":"experiments", "outputs":{"json":str(path), "csv":str(csv_path)},
                      "runs":len(rows)}, ensure_ascii=False))
    return 0
