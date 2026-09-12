"""Compare Q3 rolling MILP and event policy on bounded intervals."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from dispatch_core.backend_comparison import comparison_row, write_comparisons
from dispatch_core.horizon_sensitivity import summarize_result
from data_io import load_inputs
from simulation import SimulationConfig, simulate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dates", nargs="+", default=["2025-08-01"])
    p.add_argument("--horizon-hours", type=int, default=12, choices=(12, 18, 24))
    p.add_argument("--steps", type=int, default=72)
    p.add_argument("--scenarios", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=ROOT / "Q3" / "analysis")
    args = p.parse_args(argv)
    data, rows = load_inputs(), []
    for day in args.dates:
        summaries = {}
        for backend in ("rolling-milp", "event-policy"):
            result = simulate(SimulationConfig(data=data, backend=backend,
                horizon_steps=args.horizon_hours * 6, max_steps=args.steps,
                n_scenarios=args.scenarios, terminal_penalty=.5), day, day)
            summaries[backend] = summarize_result(result, day)
        rows.append(comparison_row("Q3", day, args.horizon_hours,
                                   summaries["rolling-milp"], summaries["event-policy"]))
        print(f"Q3 backend comparison complete: {day}", flush=True)
    print(write_comparisons(rows, args.output_dir))
    return 0


if __name__ == "__main__": raise SystemExit(main())

