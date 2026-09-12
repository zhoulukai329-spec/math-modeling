"""PV publications become usable at their first knot: release + one hour."""
from bisect import bisect_right
from datetime import datetime, time, timedelta

import numpy as np


class PVForecastTimeline:
    def __init__(self, publications):
        rows = sorted((datetime.combine(day, time()) + timedelta(minutes=minute),
                       np.asarray(values, dtype=float))
                      for (day, minute), values in publications.items())
        self.releases = [row[0] for row in rows]
        self.values = [row[1] for row in rows]

    def energy(self, targets, known_at):
        """Use only known publications whose first hourly knot has arrived.

        Before the first available publication becomes effective, use an
        explicit zero-PV startup prior. Beyond 24h retain the existing terminal
        persistence convention for virtual lookahead; never extrapolate left.
        """
        result = []
        for target in targets:
            latest = min(known_at, target - timedelta(hours=1))
            index = bisect_right(self.releases, latest) - 1
            if index < 0:
                result.append(0.0)
                continue
            offset = (target - self.releases[index]).total_seconds() / 3600
            result.append(float(np.interp(offset, np.arange(1, 25),
                                          self.values[index], left=np.nan)) / 6)
        return np.asarray(result)
