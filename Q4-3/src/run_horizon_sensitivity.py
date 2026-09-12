"""Run bounded Q4-3 12/18/24-hour representative-day experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dispatch_core.horizon_sensitivity import (
    HORIZON_HOURS, REPRESENTATIVE_DATES, hours_to_steps, summarize_result, write_records,
)
from dispatch_core.plotting import plot_horizon_sensitivity
from data_io import ATTACHMENT_DIR, load_inputs
from simulation import SimulationConfig, simulate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dates", nargs="+", default=list(REPRESENTATIVE_DATES))
    parser.add_argument("--attachment-dir", type=Path, default=ATTACHMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Q4-3" / "analysis")
    args = parser.parse_args(argv)
    data = load_inputs(args.attachment_dir)
    records = []
    for day in args.dates:
        for hours in HORIZON_HOURS:
            result = simulate(SimulationConfig(
                data=data, horizon_steps=hours_to_steps(hours), max_steps=144,
                n_scenarios=args.scenarios, seed=args.seed, backend="event-policy",
                terminal_penalty=0.5, terminal_soc=6000.0, cvar_weight=0.0,
            ), day, day)
            records.append({"question": "Q4-3", "date": day, "horizon_hours": hours,
                **summarize_result(result, day), "backend": "event-policy",
                "n_scenarios": args.scenarios})
            print(f"Q4-3 {day} H={hours}h complete", flush=True)
    csv_path, json_path = write_records(records, args.output_dir)
    plot_horizon_sensitivity(records, args.output_dir, "最小前瞻长度敏感性（仍覆盖当日剩余时段）")
    print(f"saved: {csv_path}\n{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
