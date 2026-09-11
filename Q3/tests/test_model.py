"""Small physical and economic oracles for the scenario MILP."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


def test_model_interface_exists():
    assert importlib.util.find_spec("model") is not None, "Task 2 MILP module is missing"


@pytest.fixture
def model():
    import model
    return model


def problem(model, load, pv=None, **kwargs):
    load = np.asarray(load, dtype=float)
    values = dict(load_energy=load, pv_energy=np.zeros_like(load) if pv is None else pv,
                  price=np.ones(len(load)), initial_soc=0.0, soc_min=0.0, soc_max=10.0,
                  charge_limit=2.0, discharge_limit=2.0,
                  charge_efficiency=0.9, discharge_efficiency=0.9,
                  terminal_soc=0.0, terminal_penalty=0.0, throughput_penalty=0.0)
    values.update(kwargs)
    return model.MPCProblem(**values)


def assert_physical(p, s):
    assert s.has_solution and s.optimal
    np.testing.assert_allclose(s.commitment + p.pv_energy + s.emergency + s.discharge,
                               p.load_energy + s.charge + s.spill, atol=1e-7)
    np.testing.assert_allclose(np.diff(s.soc, axis=1),
                               p.charge_efficiency * s.charge - s.discharge / p.discharge_efficiency,
                               atol=1e-7)
    assert np.all(s.soc >= p.soc_min - 1e-7)
    assert np.all(s.soc <= p.soc_max + 1e-7)
    assert np.max(s.charge * s.discharge) < 1e-7
    assert np.max(s.charge * s.emergency) < 1e-7


def test_energy_not_converted_again_and_baseline_cost(model):
    p = problem(model, [6.0], fixed_commitment=[0.0], charge_limit=0.0, discharge_limit=0.0)
    s = model.solve_mpc(p)
    assert_physical(p, s)
    assert s.emergency[0, 0] == pytest.approx(6.0)
    assert s.expected_cost == pytest.approx(30.0)
    baseline = model.solve_mpc(replace(p, fixed_commitment=None))
    assert baseline.commitment[0] == pytest.approx(6.0)
    assert baseline.expected_cost == pytest.approx(6.0)


def test_soc_efficiency_and_charge_discharge_exclusion(model):
    p = problem(model, [0.0, 1.62], pv=[2.0, 0.0], fixed_commitment=[0.0, 0.0])
    s = model.solve_mpc(p)
    assert_physical(p, s)
    np.testing.assert_allclose(s.charge, [[2.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(s.discharge, [[0.0, 1.62]], atol=1e-7)
    assert s.soc[0, 1] == pytest.approx(1.8)


def test_charge_cannot_use_emergency_even_with_large_terminal_penalty(model):
    p = problem(model, [1.0], fixed_commitment=[0.0], terminal_soc=5.0, terminal_penalty=100.0)
    s = model.solve_mpc(p)
    assert_physical(p, s)
    assert s.charge[0, 0] == pytest.approx(0.0)
    assert s.emergency[0, 0] == pytest.approx(1.0)
    assert s.terminal_cost == pytest.approx(500.0)


def test_discharge_and_emergency_can_coexist(model):
    p = problem(model, [5.0], initial_soc=2.0, fixed_commitment=[0.0])
    s = model.solve_mpc(p)
    assert_physical(p, s)
    assert s.discharge[0, 0] == pytest.approx(1.8)
    assert s.emergency[0, 0] == pytest.approx(3.2)


def test_current_actions_common_future_recourse_and_weighted_cost(model):
    p = problem(model, [0.0, 4.0], pv=[[0.0, 0.0], [0.0, 4.0]],
                fixed_commitment=[0.0, 0.0], scenario_probabilities=[0.25, 0.75])
    s = model.solve_mpc(p)
    assert_physical(p, s)
    for name in ("charge", "discharge", "emergency", "spill", "mode"):
        values = getattr(s, name)
        assert values[0, 0] == pytest.approx(values[1, 0])
    np.testing.assert_allclose(s.emergency[:, 1], [4.0, 0.0], atol=1e-7)
    assert s.expected_cost == pytest.approx(5.0)


def test_nonanticipativity_forces_shared_battery_action(model):
    # With independent first actions the sunny scenario could empty its battery
    # now while the dark scenario would save it for the more costly second step.
    p = problem(model, [1.0, 1.0], pv=[[0.0, 0.0], [0.0, 1.0]], initial_soc=1.0,
                price=[1.0, 10.0], fixed_commitment=[0.0, 0.0],
                charge_efficiency=1.0, discharge_efficiency=1.0)
    s = model.solve_mpc(p)
    assert_physical(p, s)
    np.testing.assert_allclose(s.discharge[:, 0], [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(s.emergency[:, 0], [1.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(s.discharge[:, 1], [1.0, 0.0], atol=1e-7)


def test_terminal_soc_is_soft_absolute_deviation(model):
    p = problem(model, [0.0], fixed_commitment=[0.0], terminal_soc=5.0, terminal_penalty=2.0)
    s = model.solve_mpc(p)
    assert_physical(p, s)
    assert s.terminal_cost == pytest.approx(10.0)
    assert s.objective == pytest.approx(10.0)
    over = model.solve_mpc(replace(p, initial_soc=8.0, discharge_limit=0.0))
    assert over.terminal_cost == pytest.approx(6.0)


def test_revision_cost_uses_adjacent_reference_and_fixed_mask(model):
    p = problem(model, [2.0, 4.0], previous_commitment=[3.0, 2.0],
                fixed_commitment=[2.0, np.nan], charge_limit=0.0, discharge_limit=0.0)
    s = model.solve_mpc(p)
    assert_physical(p, s)
    np.testing.assert_allclose(s.commitment, [2.0, 4.0], atol=1e-7)
    np.testing.assert_allclose(s.revision_up, [0.0, 2.0], atol=1e-7)
    np.testing.assert_allclose(s.revision_down, [1.0, 0.0], atol=1e-7)
    assert s.procurement_cost == pytest.approx(3.5)


def test_cvar_matches_weighted_tail_and_adds_eta_excess_variables(model):
    p = problem(model, [0.0, 4.0], pv=[[0.0, 0.0], [0.0, 4.0]],
                fixed_commitment=[0.0, 0.0], scenario_probabilities=[0.05, 0.95],
                charge_limit=0.0, discharge_limit=0.0)
    neutral = model.solve_mpc(p)
    risk = model.solve_mpc(replace(p, cvar_weight=2.0))
    assert_physical(p, risk)
    assert risk.cvar == pytest.approx(10.0)
    assert risk.objective == pytest.approx(21.0)
    assert risk.cvar_eta is not None and risk.cvar_excess.shape == (2,)
    assert risk.diagnostics["n_variables"] == neutral.diagnostics["n_variables"] + 3
    assert risk.diagnostics["n_constraints"] == neutral.diagnostics["n_constraints"] + 2
    assert risk.diagnostics["n_binary"] == 4
    assert risk.diagnostics["matrix_format"] == "csc"


def test_cvar_changes_common_procurement_against_rare_emergency(model):
    p = problem(model, [0.0, 4.0], pv=[[0.0, 0.0], [0.0, 4.0]],
                scenario_probabilities=[0.05, 0.95], charge_limit=0.0, discharge_limit=0.0)
    neutral = model.solve_mpc(p)
    risk = model.solve_mpc(replace(p, cvar_weight=2.0))
    assert neutral.commitment[1] == pytest.approx(0.0)
    assert risk.commitment[1] == pytest.approx(4.0)


def test_surplus_fixed_commitment_remains_feasible(model):
    p = problem(model, [0.0], pv=[10.0], fixed_commitment=[100.0], charge_limit=0.0)
    s = model.solve_mpc(p)
    assert_physical(p, s)
    assert s.spill[0, 0] == pytest.approx(110.0)
    assert s.diagnostics["emergency_upper_max"] == 0.0


@pytest.mark.parametrize("changes, match", [
    ({"pv_energy": [[0.0, 0.0], [1.0, 0.0]]}, "current"),
    ({"pv_energy": [[0.0, 0.0], [0.0, 0.0]], "scenario_probabilities": [0.2, 0.2]}, "probabilities"),
    ({"load_energy": [np.nan, 1.0]}, "finite"),
    ({"price": [-1.0, 1.0]}, "nonnegative"),
    ({"initial_soc": 11.0}, "initial_soc"),
    ({"charge_efficiency": 0.0}, "efficiency"),
    ({"fixed_commitment": [np.inf, 0.0]}, "fixed_commitment"),
    ({"cvar_alpha": 1.0}, "cvar_alpha"),
    ({"time_limit": 0.0}, "time_limit"),
])
def test_invalid_problem_rejected_at_construction(model, changes, match):
    with pytest.raises(ValueError, match=match):
        problem(model, [1.0, 1.0], **changes) if "load_energy" not in changes else replace(
            problem(model, [1.0, 1.0]), **changes)


def test_time_limit_without_incumbent_is_explicit(model, monkeypatch):
    # A solver deadline is nondeterministic; substitute only its documented result.
    monkeypatch.setattr(model, "milp", lambda **kwargs: SimpleNamespace(
        status=1, message="Time limit reached", x=None, fun=None))
    s = model.solve_mpc(problem(model, [1.0]))
    assert not s.has_solution and not s.optimal
    assert s.commitment is None and s.charge is None
    assert s.diagnostics["timed_out"] is True
    assert s.diagnostics["solver_status"] == 1


def test_time_limit_with_real_feasible_incumbent_can_be_executed(model, monkeypatch):
    real_milp = model.milp
    def limited(**kwargs):
        result = real_milp(**kwargs)
        result.status = 1
        result.message = "Time limit reached"
        return result
    monkeypatch.setattr(model, "milp", limited)
    s = model.solve_mpc(problem(model, [1.0]))
    assert s.has_solution and not s.optimal
    assert s.diagnostics["timed_out"] is True
    assert s.diagnostics["max_constraint_violation"] < 1e-6
    assert s.commitment[0] == pytest.approx(1.0)
