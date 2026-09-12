from __future__ import annotations

from pathlib import Path

from dispatch_core.plotting import plot_horizon_sensitivity


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

