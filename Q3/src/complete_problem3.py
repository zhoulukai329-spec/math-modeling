"""Reproducible January calibration and complete annual Q3 evaluation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from calibration import calibrate
from data_io import ATTACHMENT_DIR, load_inputs
from make_results import save_result, trim_result
from simulation import SimulationConfig, simulate
from verify_problem3 import verify_solution


def build_complete_config(data, attachment_dir=ATTACHMENT_DIR):
    """Return the maintained annual configuration for the legacy entry point."""
    return SimulationConfig(data=data, attachment_dir=attachment_dir,
                            horizon_steps=144, deterministic=False, n_scenarios=12,
                            max_steps=None, time_limit=30, backend="event-policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = "startup"

    def progress(row):
        message = dict(phase=phase, **row)
        print(json.dumps(message, ensure_ascii=False), flush=True)
        (args.output_dir / "progress.json").write_text(
            json.dumps(message, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    progress(dict(event="started"))
    from dispatch_core.workbook_contract import preflight_template
    preflight_template(ATTACHMENT_DIR / "附件5/result3.xlsx", args.output_dir / "result3.xlsx")
    config = build_complete_config(load_inputs())
    if args.calibration:
        phase = "calibration"
        progress(dict(event="loading_frozen_calibration", path=str(args.calibration)))
        record = json.loads(args.calibration.read_text(encoding="utf-8"))
    else:
        phase = "calibration"
        progress(dict(event="started"))
        record, _ = calibrate(config, args.output_dir / "calibration", progress=progress)
    progress(dict(event="completed", parameters=record["parameters"]))
    config = replace(config, **record["parameters"])
    phase = "annual_simulation"
    progress(dict(event="started", date_start="2025-01-01", date_end="2025-12-31"))
    result = simulate(config, "2025-01-01", "2025-12-31", progress=progress)
    progress(dict(event="completed", executed_steps=int(result.executed.sum()),
                  elapsed_seconds=result.elapsed_seconds))
    result = trim_result(result, "2025-02-01")
    phase = "result_verification"
    progress(dict(event="started", target="simulation_result"))
    check = verify_solution(result)
    if not check["passed"] or not result.executed.all():
        raise RuntimeError(f"annual run failed: {check}")
    progress(dict(event="completed", target="simulation_result", passed=True))
    phase = "saving_outputs"
    progress(dict(event="started"))
    paths = save_result(result, args.output_dir, prefix="full")
    progress(dict(event="completed", outputs={key: str(value) for key, value in paths.items()}))
    phase = "result_verification"
    progress(dict(event="started", target="workbook"))
    check = verify_solution(result, workbook_path=paths["workbook"])
    (args.output_dir / "full_verification.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    if not check["passed"]:
        raise RuntimeError(f"annual workbook failed: {check}")
    progress(dict(event="completed", target="workbook", passed=True))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    metrics["calibration"] = record
    metrics["evaluation_scope"] = "January warm-up; saved submission and cash costs cover February-December."
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    phase = "complete"
    progress(dict(event="completed", executed_steps=int(result.executed.sum()),
                  total_cost=result.total_cost, verification_passed=True))
    print(json.dumps(dict(complete=True, executed_steps=int(result.executed.sum()),
                         total_cost=result.total_cost, verification_passed=True)), flush=True)


if __name__ == "__main__":
    main()
