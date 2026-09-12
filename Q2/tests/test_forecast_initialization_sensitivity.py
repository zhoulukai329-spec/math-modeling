import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from forecast_initialization_sensitivity import build_priors


class ForecastInitializationSensitivityTests(unittest.TestCase):
    def test_builds_zero_and_flat_current_load_priors(self):
        load = np.zeros((2, 144))
        load[0, 0] = 575.0
        priors = build_priors(load)

        np.testing.assert_array_equal(priors["zero_prior"][0], np.zeros(144))
        np.testing.assert_array_equal(priors["zero_prior"][1], np.zeros(144))
        np.testing.assert_array_equal(
            priors["flat_current_load_prior"][0], np.full(144, 575.0)
        )
        np.testing.assert_array_equal(
            priors["flat_current_load_prior"][1], np.zeros(144)
        )


if __name__ == "__main__":
    unittest.main()
