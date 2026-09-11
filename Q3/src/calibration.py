"""Evaluate terminal weights using January only; save verifiable evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from make_results import save_result
from simulation import simulate
from verify_problem3 import verify_solution


def calibrate(config, output_dir, progress=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for penalty in (0.0, 0.1, 0.5):
        cfg = replace(config, max_steps=None, deterministic=True, cvar_weight=0.0,
                      terminal_soc=6000.0, terminal_penalty=penalty, initial_soc=6000.0)
        result = simulate(cfg, "2025-01-01", "2025-01-31", progress=progress)
        check = verify_solution(result)
        if not check["passed"] or not result.executed.all():
            raise RuntimeError(f"January calibration failed verification: {check}")
        paths = save_result(result, output / f"terminal_{penalty:g}", prefix="january")
        # A fixed salvage convention prevents choosing a parameter solely by
        # exhausting the end-of-training battery. It is not a cash expense.
        salvage_price = float(result.price.min()) * config.discharge_efficiency
        score = result.total_cost - salvage_price * float(result.soc_after[-1, -1])
        trials.append(dict(parameters=dict(terminal_penalty=penalty, terminal_soc=6000.0,
                                          cvar_weight=0.0, cvar_alpha=0.9),
                           cash_cost=result.total_cost, score=score,
                           final_soc=float(result.soc_after[-1, -1]), executed_steps=int(result.executed.sum()),
                           verification=check, solution=str(paths["solution"].resolve()),
                           sha256=hashlib.sha256(paths["solution"].read_bytes()).hexdigest()))
    winner = min(trials, key=lambda row: row["score"])
    record = dict(frozen=True, training_start="2025-01-01", training_end="2025-01-31",
                  parameters=winner["parameters"], trials=trials, horizon_steps=config.horizon_steps,
                  selection="Minimum January cash cost less final SOC valued at minimum tariff times discharge efficiency.",
                  scope="Only terminal penalty is tuned. terminal_soc=6000, cvar_weight=0 and cvar_alpha=.9 are fixed design choices.")
    path = output / "frozen_calibration.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record, path
