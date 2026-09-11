import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_io as dio
import optimization as opt


class DispatchPolicyTests(unittest.TestCase):
    def test_first_stage_never_uses_emergency_while_charging(self):
        import forecast as fc

        price, typical_load_kw, typical_pv_kw = dio.read_price_typical()
        _dates, _net, load, pv = dio.read_actual_data()
        f, _r, _lh, _ph, load_resid, pv_resid = fc.build_causal_forecasts(
            load, pv, typical_load_kw * dio.DT, typical_pv_kw * dio.DT
        )
        scenarios = fc.scenarios_for_day(
            31, f, load_resid, pv_resid,
            n_scenarios=12, lookback=28, seed=2025,
        )

        _g, result = opt.build_first_stage(
            price, scenarios, dio.E0_START, dio.terminal_value(price)
        )

        self.assertEqual(result["model_type"], "two_stage_stochastic_milp")
        self.assertEqual(result["stats"]["emergency_charge_overlap"], 0)

    def test_reference_policy_may_preserve_battery_and_buy_current_emergency(self):
        net = np.zeros(dio.N)
        plan = np.zeros(dio.N)
        reference = np.zeros(dio.N)
        net[0] = 100.0

        c, d, w, e, energy = opt.causal_dispatch(
            net, plan, dio.E0_START, discharge_reference=reference
        )

        self.assertAlmostEqual(d[0], 0.0)
        self.assertAlmostEqual(e[0], 100.0)
        self.assertAlmostEqual(c[0], 0.0)
        self.assertAlmostEqual(w[0], 0.0)
        self.assertAlmostEqual(energy[1], dio.E0_START)

    def test_emergency_equals_gap_after_selected_discharge_and_never_charges(self):
        net = np.zeros(dio.N)
        plan = np.zeros(dio.N)
        reference = np.zeros(dio.N)
        net[:3] = [100.0, 100.0, -50.0]
        reference[:3] = [30.0, 200.0, 0.0]

        c, d, w, e, _energy = opt.causal_dispatch(
            net, plan, dio.E0_START, discharge_reference=reference
        )

        np.testing.assert_allclose(d[:2], [30.0, 100.0])
        np.testing.assert_allclose(e[:2], [70.0, 0.0])
        self.assertTrue(np.all(c[e > 1e-9] == 0.0))
        self.assertTrue(np.all(w[e > 1e-9] == 0.0))
        expected = np.maximum(0.0, net - plan - d)
        np.testing.assert_allclose(e, expected, atol=1e-12)

    def test_output_stat_accepts_already_cropped_output_arrays(self):
        import robustness as rb

        planned = np.ones(334)
        emergency = np.full(334, 2.0)
        emergency_energy = np.ones((334, dio.N))
        full_year_mask = np.arange(365) >= 31

        result = rb.output_stat(
            planned, emergency, emergency_energy, full_year_mask
        )

        self.assertEqual(result["planned"], 334.0)
        self.assertEqual(result["emergency"], 668.0)
        self.assertEqual(result["emergency_days"], 334)


if __name__ == "__main__":
    unittest.main()
