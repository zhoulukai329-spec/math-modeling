import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import data_io as dio
import make_results as mr


class Q42ResultTests(unittest.TestCase):
    def test_plan_rows_use_next_midnight_and_recompute_row_total_and_cost(self):
        dates = np.array([45689.0, 45690.0])
        g = np.vstack([np.arange(dio.N, dtype=float),
                       1000.0 + np.arange(dio.N, dtype=float)])
        price = np.vstack([np.full(dio.N, 2.0), np.full(dio.N, 3.0)])
        rows = mr.build_plan_rows(dates, g, price, boundary_g0=777.0,
                                  boundary_price0=5.0)

        self.assertEqual(len(rows[0]), 147)
        self.assertEqual(rows[1][1], 1.0)
        self.assertEqual(rows[1][144], g[1, 0])
        self.assertEqual(rows[1][145], g[0, 1:].sum() + g[1, 0])
        self.assertEqual(rows[1][146], 2.0 * g[0, 1:].sum() + 3.0 * g[1, 0])
        self.assertEqual(rows[2][144], 777.0)
        self.assertEqual(rows[2][146], 3.0 * g[1, 1:].sum() + 5.0 * 777.0)

    def test_writer_targets_q42_template_without_modifying_it(self):
        original = dio.TEMPLATE42.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result4-2.xlsx"
            rows = [
                mr.build_plan_rows(np.array([45689.0]), np.zeros((1, dio.N)),
                                   np.ones((1, dio.N)), 0.0, 1.0),
                mr.build_charge_rows(np.array([45689.0]), np.zeros((1, dio.N)),
                                     np.zeros((1, dio.N)), np.full((1, dio.N+1), 6000.0)),
                mr.build_emergency_rows(np.array([45689.0]), np.zeros((1, dio.N))),
            ]
            mr.write_result_xlsx(rows, output_path=out)
            self.assertTrue(out.exists())
        self.assertEqual(dio.TEMPLATE42.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
