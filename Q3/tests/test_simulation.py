"""Integration tests execute the real sparse MILP, with tiny battery fixtures."""
from dataclasses import replace
from datetime import date, datetime, timedelta
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from data_io import InputData, template_datetimes


def test_simulation_interface_exists():
    assert (SRC / "simulation.py").exists(), "Task 3 rolling simulator is missing"


@pytest.fixture
def sim():
    return importlib.import_module("simulation")


def inputs(days=3, first=date(2025, 2, 1)):
    dates = tuple(first + timedelta(days=i) for i in range(days))
    load = np.full((days, 144), 2.0)
    return InputData(dates, load, np.zeros_like(load), np.ones_like(load), {
        (day, minute): np.zeros(24) for day in dates for minute in (0, 360, 720, 1080)
    })


def config(sim, data, **overrides):
    values = dict(data=data, horizon_steps=3, max_steps=3, n_scenarios=2,
                  initial_soc=0.0, soc_min=0.0, soc_max=10.0,
                  charge_limit=0.0, discharge_limit=0.0,
                  terminal_soc=0.0, terminal_penalty=0.0)
    values.update(overrides)
    return sim.SimulationConfig(**values)


def test_ledger_settles_adjacent_versions_and_preserves_baseline(sim):
    times = template_datetimes(date(2025, 2, 1))[:3]
    ledger = sim.CommitmentLedger(times, np.array([1.0, 2.0, 3.0]))
    ledger.revise(datetime(2025, 2, 1), [10, 10, 10], kind="baseline")
    ledger.revise(datetime(2025, 2, 1, 0, 1), [10, 14, 8])
    ledger.revise(datetime(2025, 2, 1, 0, 2), [10, 12, 11])
    np.testing.assert_allclose(ledger.baseline, [10, 10, 10])
    np.testing.assert_allclose(ledger.commitment, [10, 12, 11])
    assert ledger.costs == pytest.approx(dict(baseline=60, revision_up=25.5, revision_down=5))
    np.testing.assert_allclose(ledger.versions[-1].revision_up, [0, 0, 3])
    np.testing.assert_allclose(ledger.versions[-1].revision_down, [0, 2, 0])


def test_ledger_rejects_frozen_past_and_duplicate_baseline_atomically(sim):
    times = template_datetimes(date(2025, 2, 1))[:2]
    ledger = sim.CommitmentLedger(times, np.ones(2))
    ledger.revise(datetime(2025, 2, 1), [2, 2], kind="baseline")
    ledger.mark_executed(times[0])
    with pytest.raises(ValueError, match="frozen"):
        ledger.revise(times[0], [3, 2])
    with pytest.raises(ValueError, match="baseline"):
        ledger.revise(times[0], [2, 2], kind="baseline")
    assert len(ledger.versions) == 1
    np.testing.assert_allclose(ledger.commitment, [2, 2])


def test_every_step_executes_real_first_action_and_four_costs(sim):
    data = inputs()
    data.load_energy[1, :3] = [4, 5, 3]
    result = sim.simulate(config(sim, data), data.dates[1], data.dates[1])
    assert result.executed.sum() == 3
    assert len(result.solve_log) == 4  # one full baseline, three execution solves
    np.testing.assert_allclose(result.baseline[0, :143], 2.0)
    # With only one prior row, its final cell is not yet revealed at midnight.
    assert result.baseline[0, 143] == 0.0
    np.testing.assert_allclose(result.emergency[0, :3], [2, 3, 1])
    np.testing.assert_allclose(result.charge[0, :3], 0)
    assert np.isnan(result.emergency[0, 3:]).all()
    assert result.costs == pytest.approx(dict(baseline=286, revision_up=0, revision_down=0, emergency=30))
    assert result.total_cost == pytest.approx(316)
    assert all(row["n_binary"] >= 1 for row in result.solve_log)
    assert all(row["current_action_disagreement"] < 1e-6 for row in result.solve_log if row["kind"] == "execution")


def test_soc_is_carried_between_solves_including_midnight(sim):
    data = inputs()
    data.load_energy[1:] = 3.0
    result = sim.simulate(config(sim, data, max_steps=146, initial_soc=5.0,
                                 discharge_limit=1.0), data.dates[1], data.dates[2])
    mask = result.executed
    before, after = result.soc_before[mask], result.soc_after[mask]
    np.testing.assert_allclose(before[1:], after[:-1], atol=1e-6)
    np.testing.assert_allclose(after, before - result.discharge[mask] / .9, atol=1e-6)
    assert result.timestamps[0, -1] == datetime(2025, 2, 3)
    assert result.timestamps[1, 0] == datetime(2025, 2, 3, 0, 10)
    assert result.executed[0, -1] and result.executed[1, 0]


def test_four_update_clocks_and_midnight_baseline_do_not_revise_old_tail(sim):
    data = inputs()
    data.load_energy[1] = 4.0
    result = sim.simulate(config(sim, data, max_steps=145), data.dates[1], data.dates[2])
    versions = result.versions
    assert [(v.issued_at, v.kind) for v in versions] == [
        (datetime(2025, 2, 2), "baseline"),
        (datetime(2025, 2, 2, 6), "revision"),
        (datetime(2025, 2, 2, 12), "revision"),
        (datetime(2025, 2, 2, 18), "revision"),
        (datetime(2025, 2, 3), "baseline"),
    ]
    assert len(versions[-1].target_times) == 144
    assert versions[-1].target_times[0] == datetime(2025, 2, 3, 0, 10)
    assert result.final_commitment[0, -1] == versions[-2].commitment[-1]
    assert result.costs["baseline"] == pytest.approx(sum(v.baseline_cost for v in versions))
    assert result.costs["revision_up"] == pytest.approx(sum(v.up_cost for v in versions))


def test_future_actuals_cannot_change_smoke_decisions_or_costs(sim):
    data = inputs()
    baseline = sim.simulate(config(sim, data), data.dates[1], data.dates[1])
    altered = replace(data, load_energy=data.load_energy.copy(), pv_energy=data.pv_energy.copy())
    altered.load_energy[1, 3:] = 9999
    altered.pv_energy[1, 3:] = 7777
    altered.load_energy[2] = 8888
    altered.pv_energy[2] = 6666
    changed = sim.simulate(config(sim, altered), data.dates[1], data.dates[1])
    for name in ("baseline", "final_commitment", "charge", "discharge", "emergency", "soc_after"):
        np.testing.assert_allclose(getattr(baseline, name), getattr(changed, name), equal_nan=True)
    assert baseline.costs == changed.costs


def test_january_warmup_is_deterministic_and_february_uses_k(sim):
    data = inputs(first=date(2025, 1, 31))
    january = sim.simulate(config(sim, data, n_scenarios=5), data.dates[0], data.dates[0])
    february = sim.simulate(config(sim, data, n_scenarios=5), data.dates[1], data.dates[1])
    assert {row["scenario_count"] for row in january.solve_log} == {1}
    assert {row["scenario_count"] for row in february.solve_log} == {5}


def test_residual_panel_uses_published_forecasts_and_only_revealed_cells(sim):
    data = inputs()
    data.pv_energy[:] = 3
    panel = sim.build_historical_residuals(data, datetime(2025, 2, 2))
    np.testing.assert_allclose(panel[0, :143], 3)
    assert np.isnan(panel[0, 143])
    assert np.isnan(panel[1:]).all()


def test_recorded_actions_match_real_solver_current_common_action(sim, monkeypatch):
    data = inputs(days=4)
    data.pv_energy[1] = 2
    data.pv_energy[2] = 4
    data.load_energy[3, :3] = [3, 4, 5]
    real_solve = sim.solve_mpc
    captured = []
    def capture(problem):
        solution = real_solve(problem)
        captured.append((problem, solution))
        return solution
    monkeypatch.setattr(sim, "solve_mpc", capture)
    result = sim.simulate(config(sim, data, n_scenarios=3), data.dates[3], data.dates[3])
    # The observer wraps the real optimizer; no decisions or solver data are fabricated.
    for step, (problem, solution) in enumerate(captured[1:]):
        assert problem.load_energy[0] == data.load_energy[3, step]
        np.testing.assert_allclose(problem.pv_energy[:, 0], data.pv_energy[3, step])
        for name in ("charge", "discharge", "emergency", "spill", "mode"):
            np.testing.assert_allclose(getattr(solution, name)[:, 0], getattr(result, name)[0, step])
        np.testing.assert_allclose(problem.fixed_commitment, solution.commitment)
    assert any(np.ptp(problem.pv_energy[:, 1]) > 0 for problem, _ in captured[1:])
    assert any(np.ptp(solution.spill[:, 1]) > 0 for _, solution in captured[1:])
    assert result.costs["emergency"] == pytest.approx(5 * np.nansum(result.emergency))


def test_failed_solver_stops_without_rule_fallback(sim, monkeypatch):
    from model import MPCSolution
    monkeypatch.setattr(sim, "solve_mpc", lambda problem: MPCSolution(False, False, "infeasible"))
    data = inputs()
    with pytest.raises(sim.SimulationSolveError, match="baseline.*infeasible"):
        sim.simulate(config(sim, data), data.dates[1], data.dates[1])


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"horizon_steps": 0}, {"n_scenarios": 0}])
def test_invalid_simulation_sizes_fail_early(sim, kwargs):
    with pytest.raises(ValueError, match="positive"):
        config(sim, inputs(), **kwargs)


def test_soc_roundoff_is_clipped_but_real_bound_violation_is_rejected(sim):
    assert sim._clip_soc_roundoff(1199.9999999999998, 1200.0, 10800.0) == 1200.0
    assert sim._clip_soc_roundoff(10800.000000000002, 1200.0, 10800.0) == 10800.0
    with pytest.raises(sim.SimulationSolveError, match="outside bounds"):
        sim._clip_soc_roundoff(1199.99, 1200.0, 10800.0)


def test_event_policy_solves_only_at_commitment_events_and_executes_every_step(sim):
    data = inputs()
    result = sim.simulate(
        config(sim, data, backend="event-policy", max_steps=37, deterministic=True),
        data.dates[1], data.dates[1],
    )
    assert result.executed.sum() == 37
    assert [row["kind"] for row in result.solve_log] == ["baseline", "revision"]
    assert all(row["backend"] == "event-policy" for row in result.solve_log)
    np.testing.assert_allclose(result.emergency[0, :37], 0.0)
    assert np.isfinite(result.discharge_reference[0, :37]).all()


def test_event_policy_does_not_use_unrevealed_future_actuals(sim):
    data = inputs()
    baseline = sim.simulate(
        config(sim, data, backend="event-policy", max_steps=3),
        data.dates[1], data.dates[1],
    )
    changed_data = replace(data, load_energy=data.load_energy.copy(), pv_energy=data.pv_energy.copy())
    changed_data.load_energy[1, 3:] = 9999
    changed_data.pv_energy[1, 3:] = 9999
    changed = sim.simulate(
        config(sim, changed_data, backend="event-policy", max_steps=3),
        data.dates[1], data.dates[1],
    )
    for name in ("baseline", "charge", "discharge", "emergency", "soc_after"):
        np.testing.assert_allclose(getattr(baseline, name), getattr(changed, name), equal_nan=True)


def test_backend_name_is_validated(sim):
    with pytest.raises(ValueError, match="backend"):
        config(sim, inputs(), backend="unknown")


def test_independent_verifier_accepts_event_log_without_fake_execution_solves(sim):
    verifier = importlib.import_module("verify_problem3")
    data = inputs()
    result = sim.simulate(
        config(sim, data, backend="event-policy", max_steps=3),
        data.dates[1], data.dates[1],
    )
    report = verifier.verify_solution(result)
    assert report["passed"], report["errors"]
