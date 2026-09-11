import unittest

import numpy as np

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_io


class CrossDayAlignmentTests(unittest.TestCase):
    def test_midnight_uses_previous_row_tail(self):
        raw = np.arange(3 * data_io.N, dtype=float).reshape(3, data_io.N)
        aligned = data_io._align_cross_day(raw, first_0=-1.0)

        self.assertEqual(aligned[0, 0], -1.0)
        self.assertTrue(np.array_equal(aligned[0, 1:], raw[0, : data_io.N - 1]))
        self.assertEqual(aligned[1, 0], raw[0, data_io.N - 1])
        self.assertTrue(np.array_equal(aligned[1, 1:], raw[1, : data_io.N - 1]))
        self.assertEqual(aligned[2, 0], raw[1, data_io.N - 1])
        self.assertNotEqual(aligned[1, 0], raw[1, data_io.N - 1])


if __name__ == "__main__":
    unittest.main()
