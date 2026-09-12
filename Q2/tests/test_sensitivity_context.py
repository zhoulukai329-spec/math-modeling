import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import robustness


class SensitivityContextTests(unittest.TestCase):
    def test_formal_start_soc_maps_calendar_days_to_saved_output_rows(self):
        energy = np.array([
            [3100.0, 3200.0],
            [4100.0, 4200.0],
            [5100.0, 5200.0],
        ])

        actual = robustness.formal_start_soc(
            energy, np.array([31, 33]), output_start_day=31
        )

        np.testing.assert_array_equal(actual, np.array([3100.0, 5100.0]))

    def test_formal_start_soc_rejects_days_outside_saved_period(self):
        with self.assertRaises(ValueError):
            robustness.formal_start_soc(
                np.zeros((3, 2)), np.array([30]), output_start_day=31
            )


if __name__ == "__main__":
    unittest.main()
