import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import data_io as dio
import forecast as fc


class DynamicPriceDataTests(unittest.TestCase):
    def test_attachment4_is_cross_day_aligned_without_row_roll(self):
        raw = np.arange(3 * dio.N, dtype=float).reshape(3, dio.N)
        aligned = dio.align_price_cross_day(raw, first_price=9.5)

        self.assertEqual(aligned[0, 0], 9.5)
        np.testing.assert_array_equal(aligned[0, 1:], raw[0, :-1])
        self.assertEqual(aligned[1, 0], raw[0, -1])
        np.testing.assert_array_equal(aligned[1, 1:], raw[1, :-1])

    def test_next_day_forecasts_use_last_week_and_published_midnight_quote(self):
        net_forecast = np.arange(10 * dio.N, dtype=float).reshape(10, dio.N)
        price_actual = 1000.0 + net_forecast
        net_next, price_next = fc.next_day_point_forecasts(
            net_forecast, price_actual, next_midnight_price=7.25
        )
        np.testing.assert_array_equal(net_next, net_forecast[-7])
        np.testing.assert_array_equal(price_next[1:], price_actual[-7, 1:])
        self.assertEqual(price_next[0], 7.25)

    def test_price_forecast_is_causal_and_uses_weekly_seasonality(self):
        price = np.arange(10 * dio.N, dtype=float).reshape(10, dio.N)
        forecast, residual = fc.build_causal_price_forecasts(price)

        self.assertEqual(forecast[0, 0], price[0, 0])
        np.testing.assert_array_equal(forecast[0], np.full(dio.N, price[0, 0]))
        self.assertEqual(forecast[4, 0], price[4, 0])
        np.testing.assert_array_equal(forecast[4, 1:], price[:4, 1:].mean(axis=0))
        self.assertEqual(forecast[8, 0], price[8, 0])
        np.testing.assert_array_equal(forecast[8, 1:], price[1, 1:])
        np.testing.assert_array_equal(residual, price - forecast)

        changed = price.copy()
        changed[8:] += 1_000_000.0
        changed_forecast, _ = fc.build_causal_price_forecasts(changed)
        np.testing.assert_array_equal(changed_forecast[:8], forecast[:8])
        self.assertEqual(changed_forecast[8, 0], changed[8, 0])
        np.testing.assert_array_equal(changed_forecast[8, 1:], forecast[8, 1:])

    def test_q42_has_no_attachment1_input(self):
        self.assertFalse(hasattr(dio, "ATTACH1"))
        self.assertFalse(hasattr(dio, "read_price_typical"))

    def test_joint_scenarios_share_the_same_historical_day(self):
        d = 5
        net_forecast = np.zeros((6, dio.N))
        price_forecast = np.full((6, dio.N), 10.0)
        net_residual = np.stack([np.full(dio.N, i) for i in range(6)])
        price_residual = np.stack([np.full(dio.N, 10 * i) for i in range(6)])

        net_s, price_s, source_days = fc.joint_scenarios_for_day(
            d, net_forecast, price_forecast, net_residual, price_residual,
            n_scenarios=4, lookback=5, seed=11,
        )

        for s, source in enumerate(source_days):
            np.testing.assert_allclose(net_s[s], source)
            np.testing.assert_allclose(price_s[s], 10.0 + 10.0 * source)
        self.assertTrue(np.all(source_days < d))


if __name__ == "__main__":
    unittest.main()
