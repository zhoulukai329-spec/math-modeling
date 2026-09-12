import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import data_io as dio
from run_problem42 import simulate_days


class DynamicPriceSimulationTests(unittest.TestCase):
    def test_settlement_uses_realized_daily_price(self):
        days = 2
        net = np.zeros((days, dio.N))
        net[:, 0] = 20.0
        net_forecast = net.copy()
        price_forecast = np.ones((days, dio.N))
        price_actual = np.ones((days, dio.N))
        price_actual[1] *= 7.0
        zeros = np.zeros_like(net)

        sim = simulate_days(
            net, price_actual, net_forecast, price_forecast,
            zeros, zeros, n_scenarios=1, lookback=1, seed=2,
            E_start=dio.E0_START, start_day=0, end_day=1,
        )

        expected_plan = np.sum(price_actual * sim["g"], axis=1)
        expected_emergency = np.sum(
            dio.EMERGENCY_MULT * price_actual * sim["e"], axis=1
        )
        np.testing.assert_allclose(sim["planned_cost"], expected_plan)
        np.testing.assert_allclose(sim["emergency_cost"], expected_emergency)
        self.assertEqual(sim["scenario_source_days"].shape, (days, 1))
        self.assertEqual(sim["scenario_source_days"][0, 0], -1)
        self.assertLess(sim["scenario_source_days"][1, 0], 1)


if __name__ == "__main__":
    unittest.main()
