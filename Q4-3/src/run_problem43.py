"""Question 4-3 dynamic-price rolling/event MILP command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_io import ATTACHMENT_DIR
from make_results import save_result43, trim_result
from simulation import SimulationConfig, simulate
from verify_problem43 import verify_solution43


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--date-start", default="2025-02-01")
    parser.add_argument("--date-end")
    parser.add_argument("--horizon-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--scenarios", "--n-scenarios", type=int, dest="n_scenarios")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lookback-days", type=int, default=28)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--terminal-penalty", type=float, default=0.5)
    parser.add_argument("--terminal-soc", type=float, default=6000.0)
    parser.add_argument("--cvar-weight", type=float, default=0.0)
    parser.add_argument("--cvar-alpha", type=float, default=0.9)
    parser.add_argument("--initial-soc", type=float, default=6000.0)
    parser.add_argument("--revision-hours", type=int, nargs="+", default=[6, 12, 18])
    parser.add_argument("--backend", choices=("event-policy", "rolling-milp"), default="event-policy")
    parser.add_argument("--attachment-dir", type=Path, default=ATTACHMENT_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    full = args.mode == "full"
    config = SimulationConfig(
        attachment_dir=args.attachment_dir,
        horizon_steps=args.horizon_steps if args.horizon_steps is not None else (12 if not full else 144),
        max_steps=args.max_steps if args.max_steps is not None else (6 if not full else None),
        n_scenarios=args.n_scenarios if args.n_scenarios is not None else (1 if not full else 12),
        seed=args.seed, lookback_days=args.lookback_days, time_limit=args.time_limit,
        mip_rel_gap=args.mip_rel_gap, deterministic=args.deterministic,
        initial_soc=args.initial_soc, revision_hours=tuple(args.revision_hours),
        backend=args.backend, terminal_penalty=args.terminal_penalty,
        terminal_soc=args.terminal_soc, cvar_weight=args.cvar_weight, cvar_alpha=args.cvar_alpha,
    )
    end = args.date_end or ("2025-12-31" if full else args.date_start)
    # Full mode starts in January so the February SOC and residual library are causal.
    start = "2025-01-01" if full else args.date_start
    from dispatch_core.workbook_contract import preflight_template
    preflight_template(args.attachment_dir / "附件5/result4-3.xlsx", args.output_dir / "result4-3.xlsx")
    result = simulate(config, start, end)
    if full:
        result = trim_result(result, args.date_start)
    report = verify_solution43(result)
    if not report["passed"]:
        raise RuntimeError(f"Q4-3 simulation verification failed: {report['errors']}")
    paths = save_result43(result, args.output_dir, prefix=args.mode,
                          template_path=args.attachment_dir / "附件5" / "result4-3.xlsx")
    report = verify_solution43(result, workbook_path=paths["workbook"],
                               template_path=args.attachment_dir / "附件5" / "result4-3.xlsx")
    verification_path = args.output_dir / f"{args.mode}_verification.json"
    verification_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"Q4-3 workbook verification failed: {report['errors']}")
    print(json.dumps({
        "mode": args.mode, "backend": args.backend,
        "executed_steps": int(result.executed.sum()), "solve_count": len(result.solve_log),
        "costs": result.costs, "total_cost": result.total_cost,
        "elapsed_seconds": result.elapsed_seconds,
        "verification_passed": True,
        "outputs": {key: str(value) for key, value in paths.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
