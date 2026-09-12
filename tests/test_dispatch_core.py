from __future__ import annotations

import numpy as np
import pytest

from dispatch_core import BatteryLimits, dispatch_step, deduplicate_scenarios


LIMITS = BatteryLimits(
    soc_min=1200.0,
    soc_max=10800.0,
    charge_limit=5000.0 / 6.0,
    discharge_limit=5000.0 / 6.0,
    charge_efficiency=0.9,
    discharge_efficiency=0.9,
)


def test_surplus_charges_before_spilling_and_respects_soc_ceiling():
    step = dispatch_step(
        load_energy=100.0,
        pv_energy=200.0,
        commitment=1000.0,
        soc=10700.0,
        discharge_reference=0.0,
        limits=LIMITS,
    )
    assert step.charge == pytest.approx(100.0 / 0.9)
    assert step.spill == pytest.approx(1100.0 - 100.0 / 0.9)
    assert step.soc_after == pytest.approx(10800.0)
    assert step.discharge == step.emergency == 0.0


def test_charge_reference_preserves_capacity_when_latest_milp_plans_to_spill():
    step = dispatch_step(
        load_energy=100.0, pv_energy=200.0, commitment=300.0, soc=6000.0,
        charge_reference=40.0, discharge_reference=0.0, limits=LIMITS,
    )
    assert step.charge == pytest.approx(40.0)
    assert step.spill == pytest.approx(360.0)


def test_deficit_uses_only_reference_discharge_and_emergency_covers_rest():
    step = dispatch_step(
        load_energy=900.0,
        pv_energy=0.0,
        commitment=0.0,
        soc=6000.0,
        discharge_reference=300.0,
        limits=LIMITS,
    )
    assert step.discharge == pytest.approx(300.0)
    assert step.emergency == pytest.approx(600.0)
    assert step.soc_after == pytest.approx(6000.0 - 300.0 / 0.9)
    assert step.charge == step.spill == 0.0


def test_reference_above_deficit_cannot_create_charging_or_negative_emergency():
    step = dispatch_step(
        load_energy=80.0,
        pv_energy=10.0,
        commitment=20.0,
        soc=6000.0,
        discharge_reference=500.0,
        limits=LIMITS,
    )
    assert step.discharge == pytest.approx(50.0)
    assert step.emergency == 0.0
    assert step.energy_balance_residual == pytest.approx(0.0)


def test_boundary_roundoff_is_clipped_but_material_violation_fails():
    clipped = dispatch_step(
        load_energy=0.0,
        pv_energy=0.0,
        commitment=0.0,
        soc=1199.999999,
        discharge_reference=0.0,
        limits=LIMITS,
    )
    assert clipped.soc_before == 1200.0
    with pytest.raises(ValueError, match="outside"):
        dispatch_step(
            load_energy=0.0,
            pv_energy=0.0,
            commitment=0.0,
            soc=1199.99,
            discharge_reference=0.0,
            limits=LIMITS,
        )


def test_duplicate_scenarios_merge_probabilities_in_first_seen_order():
    values = np.array([[1.0, 2.0], [4.0, 5.0], [1.0, 2.0]])
    unique, probability, inverse = deduplicate_scenarios(values, [0.2, 0.3, 0.5])
    np.testing.assert_array_equal(unique, [[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_allclose(probability, [0.7, 0.3])
    np.testing.assert_array_equal(inverse, [0, 1, 0])


def test_scenario_probabilities_are_validated():
    with pytest.raises(ValueError, match="probabilities"):
        deduplicate_scenarios(np.ones((2, 3)), [0.2, 0.2])


def test_small_positive_charge_still_sets_charge_mode():
    step = dispatch_step(
        load_energy=0.0,
        pv_energy=0.0,
        commitment=5e-6,
        soc=6000.0,
        charge_reference=5e-6,
        discharge_reference=0.0,
        limits=LIMITS,
    )
    assert step.charge == 5e-6
    assert step.mode == 1
