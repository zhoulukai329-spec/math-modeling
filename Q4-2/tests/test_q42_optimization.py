import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import data_io as dio
import optimization as opt


class DynamicPriceOptimizationTests(unittest.TestCase):
    def test_objective_uses_each_scenario_price(self):
        net = np.zeros((2, dio.N))
        price = np.vstack([np.ones(dio.N), np.full(dio.N, 3.0)])
        g, result = opt.build_first_stage(
            price, net, dio.E0_START, v_terminal=0.0
        )

        self.assertEqual(g.shape, (dio.N,))
        self.assertEqual(result["model_type"], "joint_price_netload_stochastic_milp")
        self.assertAlmostEqual(result["stats"]["mean_plan_price"][0], 2.0)

    def test_soc_roundoff_near_lower_bound_is_clipped(self):
        net = np.zeros(dio.N)
        plan = np.zeros(dio.N)
        c, d, w, e, energy = opt.causal_dispatch(
            net, plan, 1199.999999, discharge_reference=np.zeros(dio.N)
        )

        self.assertEqual(energy[0], dio.E_MIN)
        self.assertGreaterEqual(energy.min(), dio.E_MIN)
        self.assertLessEqual(energy.max(), dio.E_MAX)
        np.testing.assert_allclose(c + d + w + e, 0.0)

    def test_material_soc_violation_is_not_hidden(self):
        with self.assertRaises(ValueError):
            opt.causal_dispatch(
                np.zeros(dio.N), np.zeros(dio.N), dio.E_MIN - 0.1,
                discharge_reference=np.zeros(dio.N),
            )


if __name__ == "__main__":
    unittest.main()
