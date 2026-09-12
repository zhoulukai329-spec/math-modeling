from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from dispatch_core.horizon_sensitivity import HORIZON_HOURS, hours_to_steps, summarize_result


def test_only_12_18_24_hour_horizons_are_accepted():
    assert HORIZON_HOURS == (12, 18, 24)
    assert [hours_to_steps(value) for value in HORIZON_HOURS] == [72, 108, 144]
    with pytest.raises(ValueError, match="12, 18, or 24"):
        hours_to_steps(48)


def test_summary_recomputes_selected_day_metrics_without_next_day_baseline():
    mask = np.array([[True, True, False]])
    result = SimpleNamespace(
        dates=("2025-02-01",), executed=mask,
        emergency=np.array([[1.0, 2.0, np.nan]]),
        soc_after=np.array([[5000.0, 4900.0, np.nan]]),
        final_commitment=np.array([[4.0, 5.0, 6.0]]),
        pv_energy=np.array([[1.0, 1.0, np.nan]]),
        load_energy=np.array([[5.0, 7.0, np.nan]]),
        charge=np.array([[0.0, 0.0, np.nan]]),
        discharge=np.array([[0.0, 0.0, np.nan]]),
        spill=np.array([[1.0, 1.0, np.nan]]),
        solve_log=[{"elapsed_seconds": 0.2}, {"elapsed_seconds": 0.3}],
        costs={"baseline": 20.0, "revision_up": 2.0, "revision_down": 1.0, "emergency": 9.0},
    )
    summary = summarize_result(result)
    assert summary["emergency_kwh"] == 3.0
    assert summary["end_soc_kwh"] == 4900.0
    assert summary["solve_seconds"] == pytest.approx(0.5)
    assert summary["max_residual"] == pytest.approx(0.0)
