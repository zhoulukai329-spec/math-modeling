"""Question 3 real-attachment rolling MILP command line."""
from __future__ import annotations

import argparse
from datetime import date
import importlib.util
import json
from pathlib import Path

from data_io import ATTACHMENT_DIR
from make_results import save_result, trim_result
from simulation import SimulationConfig, simulate
from verify_problem3 import verify_solution


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full", "experiments"), default="smoke")
    parser.add_argument("--date-start", default="2025-02-01")
    parser.add_argument("--date-end")
    parser.add_argument("--horizon-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--scenarios", "--n-scenarios", type=int, dest="n_scenarios")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lookback-days", type=int, default=28)
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--terminal-penalty", type=float)
    parser.add_argument("--terminal-soc", type=float)
    parser.add_argument("--cvar-weight", type=float)
    parser.add_argument("--cvar-alpha", type=float)
    parser.add_argument("--initial-soc", type=float, default=6000)
    parser.add_argument("--revision-hours", type=int, nargs="+", default=[6, 12, 18])
    parser.add_argument("--experiment-dates", nargs="+",
                        default=["2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01"],
                        help="代表日修订时刻消融；仅在 experiments 模式使用")
    parser.add_argument("--backend", choices=("event-policy", "rolling-milp"), default="event-policy",
                        help="event-policy用于快速全年计算；rolling-milp仅建议作代表日精度对照")
    parser.add_argument("--attachment-dir", type=Path, default=ATTACHMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--calibration", type=Path, help="Frozen January-only calibration JSON required for full mode")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "smoke" and args.calibration is not None:
        parser.error("calibration is only supported by full/experiments modes; smoke uses explicit parameters")
    if args.mode == "experiments":
        # Task 6 owns this interface; an absent implementation fails explicitly.
        if importlib.util.find_spec("experiments") is None:
            parser.error("experiments infrastructure is not implemented yet (Task 6)")
        from experiments import run_experiments
        return run_experiments(args)
    tuning = {}
    if args.mode == "full":
        if args.calibration is None:
            parser.error("full mode requires --calibration with frozen January-only parameters (Task 6)")
        try:
            record = json.loads(args.calibration.read_text(encoding="utf-8"))
            if record.get("frozen") is not True or record.get("training_start") != "2025-01-01" or record.get("training_end") != "2025-01-31":
                raise ValueError("calibration must be frozen and trained on 2025-01-01 through 2025-01-31")
            tuning = record["parameters"]
            allowed = {"terminal_penalty", "terminal_soc", "cvar_weight", "cvar_alpha"}
            if set(tuning) - allowed or not {"terminal_penalty", "terminal_soc", "cvar_weight"} <= set(tuning):
                raise ValueError("calibration requires terminal_penalty, terminal_soc, cvar_weight; optional cvar_alpha")
            if date.fromisoformat(args.date_start) <= date(2025, 1, 31):
                raise ValueError("full evaluation must begin after January calibration")
            if any(getattr(args, key) is not None for key in allowed):
                raise ValueError("full mode cannot override frozen calibration parameters")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"invalid calibration: {exc}")
    else:
        tuning = {key: getattr(args, key) for key in ("terminal_penalty", "terminal_soc", "cvar_weight", "cvar_alpha")
                  if getattr(args, key) is not None}
    config = SimulationConfig(attachment_dir=args.attachment_dir,
        horizon_steps=args.horizon_steps if args.horizon_steps is not None else (6 if args.mode == "smoke" else 144),
        max_steps=args.max_steps if args.max_steps is not None else (37 if args.mode == "smoke" else None),
        n_scenarios=args.n_scenarios if args.n_scenarios is not None else (2 if args.mode == "smoke" else 12),
        seed=args.seed, lookback_days=args.lookback_days, time_limit=args.time_limit,
        mip_rel_gap=args.mip_rel_gap, deterministic=args.deterministic, initial_soc=args.initial_soc,
        revision_hours=tuple(args.revision_hours), backend=args.backend, **tuning)
    end = args.date_end or (args.date_start if args.mode == "smoke" else "2025-12-31")
    sim_start = "2025-01-01" if args.mode == "full" else args.date_start
    from dispatch_core.workbook_contract import preflight_template
    preflight_template(args.attachment_dir / "附件5/result3.xlsx", args.output_dir / "result3.xlsx")
    result = simulate(config, sim_start, end)
    if args.mode == "full":
        result = trim_result(result, args.date_start)
    report = verify_solution(result)
    if not report["passed"]:
        raise RuntimeError(f"simulation failed independent verification: {report['errors']}")
    paths = save_result(result, args.output_dir, prefix=args.mode,
                        template_path=args.attachment_dir / "附件5/result3.xlsx")
    report = verify_solution(result, workbook_path=paths["workbook"], template_path=args.attachment_dir / "附件5/result3.xlsx")
    report_path = args.output_dir / f"{args.mode}_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"workbook failed independent verification: {report['errors']}")
    if args.calibration is not None:
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        metrics["calibration"] = record
        metrics["calibration_source"] = str(args.calibration.resolve())
        paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(mode=args.mode, executed_steps=int(result.executed.sum()), complete=bool(result.executed.all()),
        solve_count=len(result.solve_log), costs=result.costs, total_cost=result.total_cost,
        elapsed_seconds=result.elapsed_seconds, verification_passed=report["passed"],
        outputs={key: str(path) for key, path in paths.items()}), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
