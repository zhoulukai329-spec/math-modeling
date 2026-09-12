from dispatch_core.backend_comparison import comparison_row


def test_backend_comparison_reports_signed_accuracy_differences_and_speedup():
    rolling = {"total_cost": 100.0, "emergency_kwh": 20.0, "end_soc_kwh": 5000.0,
               "solve_seconds": 10.0, "max_residual": 1e-10}
    fast = {"total_cost": 102.0, "emergency_kwh": 18.0, "end_soc_kwh": 5100.0,
            "solve_seconds": 2.0, "max_residual": 2e-10}
    row = comparison_row("Q3", "2025-08-01", 12, rolling, fast)
    assert row["cost_difference"] == 2.0
    assert row["emergency_difference_kwh"] == -2.0
    assert row["end_soc_difference_kwh"] == 100.0
    assert row["speedup"] == 5.0

