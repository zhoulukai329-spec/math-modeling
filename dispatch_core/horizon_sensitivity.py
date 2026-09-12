"""Common metrics and file output for bounded horizon experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HORIZON_HOURS = (12, 18, 24)
REPRESENTATIVE_DATES = ("2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01")
FIELDS = ("question", "date", "horizon_hours", "total_cost", "emergency_kwh",
          "end_soc_kwh", "first_commitment_kwh", "first_discharge_kwh",
          "solve_seconds", "max_residual", "backend", "n_scenarios")


def hours_to_steps(hours: int) -> int:
    if hours not in HORIZON_HOURS:
        raise ValueError("horizon must be 12, 18, or 24 hours")
    return hours * 6


def summarize_result(result, operating_day=None) -> dict[str, float]:
    mask = np.asarray(result.executed, dtype=bool)
    if not mask.any():
        raise ValueError("sensitivity result contains no executed interval")
    balance = (result.final_commitment[mask] + result.pv_energy[mask]
               + result.discharge[mask] + result.emergency[mask]
               - result.load_energy[mask] - result.charge[mask] - result.spill[mask])
    first = tuple(np.argwhere(mask)[0])
    total_cost = float(sum(result.costs.values()))
    if operating_day is not None and hasattr(result, "versions"):
        day_text = str(operating_day)
        commitment_cost = sum(
            float(version.baseline_cost + version.up_cost + version.down_cost)
            for version in result.versions
            if str(version.target_times[0].date()) == day_text
        )
        emergency_cost = float(np.sum(5.0 * result.price[mask] * result.emergency[mask]))
        total_cost = commitment_cost + emergency_cost
    return {
        "total_cost": total_cost,
        "emergency_kwh": float(np.sum(result.emergency[mask])),
        "end_soc_kwh": float(result.soc_after[mask][-1]),
        "first_commitment_kwh": float(result.final_commitment[first]),
        "first_discharge_kwh": float(result.discharge[first]),
        "solve_seconds": float(sum(float(row.get("elapsed_seconds", 0.0)) for row in result.solve_log)),
        "max_residual": float(np.max(np.abs(balance))),
    }


def write_records(records, output_dir: str | Path, stem: str = "horizon_sensitivity"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean = [{field: row[field] for field in FIELDS} for row in records]
    csv_path, json_path = output / f"{stem}.csv", output / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(clean)
    json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    return csv_path, json_path
