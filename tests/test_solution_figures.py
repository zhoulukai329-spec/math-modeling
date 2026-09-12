from __future__ import annotations

from pathlib import Path

import csv

import pytest

from dispatch_core.plotting import plot_horizon_sensitivity, plot_revision_sensitivity


def test_horizon_plot_exports_png_svg_pdf(tmp_path):
    records = [
        {"question": "Q3", "date": day, "horizon_hours": horizon,
         "total_cost": 1000 + horizon + index, "emergency_kwh": 20 - index,
         "solve_seconds": horizon / 10}
        for index, day in enumerate(("2025-02-01", "2025-05-01"))
        for horizon in (12, 18, 24)
    ]
    paths = plot_horizon_sensitivity(records, tmp_path, title="Q3前瞻长度敏感性")
    assert {path.suffix for path in paths} == {".png", ".svg", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_revision_plot_exports_paired_raw_points_and_all_formats(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    schedules = ("", "6", "12", "18", "6+12", "6+18", "12+18", "6+12+18")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "date", "revision_hours", "total_cost", "emergency_energy", "verified"
        ))
        writer.writeheader()
        for day_index, day in enumerate(("2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01")):
            for schedule_index, schedule in enumerate(schedules):
                writer.writerow({
                    "date": day,
                    "revision_hours": schedule,
                    "total_cost": 30000 + day_index * 1000 - schedule_index * 100,
                    "emergency_energy": 1500 + day_index * 100 - schedule_index * 20,
                    "verified": "True",
                })
    paths = plot_revision_sensitivity(csv_path, tmp_path)
    suffixes = {path.suffix for path in paths}
    assert {".png", ".svg", ".pdf"} <= suffixes
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert (tmp_path / "_qa" / "fig7_revision_schedule_sensitivity_grayscale.png").exists()


def test_revision_plot_rejects_duplicate_date_schedule_rows(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "date", "revision_hours", "total_cost", "emergency_energy", "verified"
        ))
        writer.writeheader()
        row = {
            "date": "2025-02-01", "revision_hours": "", "total_cost": 1,
            "emergency_energy": 1, "verified": "True",
        }
        writer.writerow(row)
        writer.writerow(row)
    with pytest.raises(ValueError, match="duplicate"):
        plot_revision_sensitivity(csv_path, tmp_path)
