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
    def test_plan_rows_accept_daily_price_costs_and_keep_template_order(self):
        dates = np.array([45689.0])
        g = np.arange(dio.N, dtype=float)[None, :]
        costs = np.array([321.5])
        rows = mr.build_plan_rows(dates, g, costs)

        self.assertEqual(len(rows[0]), 147)
        self.assertEqual(rows[1][1], 1.0)
        self.assertEqual(rows[1][144], 0.0)
        self.assertEqual(rows[1][145], g.sum())
        self.assertEqual(rows[1][146], costs[0])

    def test_writer_targets_q42_template_without_modifying_it(self):
        original = dio.TEMPLATE42.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result4-2.xlsx"
            rows = [
                mr.build_plan_rows(np.array([45689.0]), np.zeros((1, dio.N)), np.zeros(1)),
                mr.build_charge_rows(np.array([45689.0]), np.zeros((1, dio.N)),
                                     np.zeros((1, dio.N)), np.full((1, dio.N+1), 6000.0)),
                mr.build_emergency_rows(np.array([45689.0]), np.zeros((1, dio.N))),
            ]
            mr.write_result_xlsx(rows, output_path=out)
            self.assertTrue(out.exists())
        self.assertEqual(dio.TEMPLATE42.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
