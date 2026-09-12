import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_io as dio
from make_results import build_plan_rows
from forecast import next_day_point_forecast


class PlanCrossDayTests(unittest.TestCase):
    def test_each_template_row_ends_with_following_midnight(self):
        dates = np.array([45689.0, 45690.0])
        g = np.vstack([np.arange(dio.N), 500.0 + np.arange(dio.N)]).astype(float)
        price = np.arange(1, dio.N + 1, dtype=float)
        rows = build_plan_rows(dates, g, price, boundary_g0=999.0,
                               boundary_price0=price[0])

        self.assertEqual(rows[1][1:144], list(g[0, 1:]))
        self.assertEqual(rows[1][144], g[1, 0])
        self.assertEqual(rows[2][144], 999.0)
        self.assertEqual(rows[1][145], sum(rows[1][1:145]))
        self.assertEqual(rows[1][146], np.dot(price[1:], g[0, 1:]) + price[0] * g[1, 0])

    def test_next_day_point_forecast_uses_last_week_without_future_actuals(self):
        history = np.arange(10 * dio.N, dtype=float).reshape(10, dio.N)
        np.testing.assert_array_equal(next_day_point_forecast(history), history[-7])


if __name__ == "__main__":
    unittest.main()
