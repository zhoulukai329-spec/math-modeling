from datetime import date, datetime, timedelta

import numpy as np
import pytest

from dispatch_core.pv_forecast import PVForecastTimeline


@pytest.mark.parametrize('hour', [0, 6, 12, 18])
def test_previous_release_until_first_knot_including_midnight(hour):
    now = datetime(2025, 2, 2, hour)
    previous = now - timedelta(hours=6)
    timeline = PVForecastTimeline({
        (previous.date(), previous.hour * 60): np.arange(1, 25) * 6.,
        (now.date(), hour * 60): np.arange(101, 125) * 6.,
        (now.date(), (hour + 6) * 60): np.full(24, 9999.),
    })
    targets = [now + timedelta(minutes=m) for m in [0, 10, 50, 60, 70]]
    expected = [6, 6 + 1/6, 6 + 5/6, 101, 101 + 1/6]
    np.testing.assert_allclose(timeline.energy(targets, now), expected)
    # Historical residuals select precisely the same effective forecast.
    np.testing.assert_allclose([timeline.energy([t], t)[0] for t in targets], expected)


def test_startup_prior_and_no_unpublished_information():
    now = datetime(2025, 1, 1)
    timeline = PVForecastTimeline({(date(2025, 1, 1), 0): np.full(24, 600.)})
    np.testing.assert_allclose(timeline.energy([now, now + timedelta(minutes=50),
                                              now + timedelta(hours=1)], now), [0, 0, 100])
    np.testing.assert_allclose(timeline.energy([now + timedelta(hours=2)],
                                              now - timedelta(minutes=1)), [0])
