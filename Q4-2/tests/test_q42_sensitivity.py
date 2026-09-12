import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import sensitivity


class Q42SensitivityTests(unittest.TestCase):
    def test_formal_start_soc_uses_the_saved_full_run_state(self):
        energy = np.array([
            [6000.0, 6100.0],
            [7000.0, 7100.0],
            [8000.0, 8100.0],
        ])

        actual = sensitivity.formal_start_soc(
            energy, np.array([31, 32, 33]), output_start_day=31
        )

        np.testing.assert_array_equal(actual, np.array([6000.0, 7000.0, 8000.0]))

    def test_formal_start_soc_rejects_nonformal_days(self):
        with self.assertRaises(ValueError):
            sensitivity.formal_start_soc(
                np.zeros((2, 3)), np.array([30]), output_start_day=31
            )


if __name__ == "__main__":
    unittest.main()
