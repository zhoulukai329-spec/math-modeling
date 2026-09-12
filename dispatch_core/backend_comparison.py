"""Reference-versus-fast backend comparison records."""
from __future__ import annotations

import csv
import json
from pathlib import Path


FIELDS = ("question", "date", "horizon_hours", "rolling_cost", "event_cost",
          "cost_difference", "rolling_emergency_kwh", "event_emergency_kwh",
          "emergency_difference_kwh", "rolling_end_soc_kwh", "event_end_soc_kwh",
          "end_soc_difference_kwh", "rolling_solve_seconds", "event_solve_seconds",
          "speedup", "rolling_max_residual", "event_max_residual")


def comparison_row(question, day, horizon_hours, rolling, event):
    event_seconds = float(event["solve_seconds"])
    return {
        "question": question, "date": day, "horizon_hours": horizon_hours,
        "rolling_cost": float(rolling["total_cost"]), "event_cost": float(event["total_cost"]),
        "cost_difference": float(event["total_cost"] - rolling["total_cost"]),
        "rolling_emergency_kwh": float(rolling["emergency_kwh"]),
        "event_emergency_kwh": float(event["emergency_kwh"]),
        "emergency_difference_kwh": float(event["emergency_kwh"] - rolling["emergency_kwh"]),
        "rolling_end_soc_kwh": float(rolling["end_soc_kwh"]),
        "event_end_soc_kwh": float(event["end_soc_kwh"]),
        "end_soc_difference_kwh": float(event["end_soc_kwh"] - rolling["end_soc_kwh"]),
        "rolling_solve_seconds": float(rolling["solve_seconds"]),
        "event_solve_seconds": event_seconds,
        "speedup": float(rolling["solve_seconds"] / event_seconds) if event_seconds > 0 else float("inf"),
        "rolling_max_residual": float(rolling["max_residual"]),
        "event_max_residual": float(event["max_residual"]),
    }


def write_comparisons(rows, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output / "backend_comparison.csv", output / "backend_comparison.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path

