"""Causal point forecasts and residual PV scenarios for Question 3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Sequence

import numpy as np

from data_io import InputData, coerce_date, parse_clock_minutes, power_kw_to_energy, source_datetimes


@dataclass(frozen=True)
class InformationForecast:
    """Ten-minute energy forecasts available at one release instant."""

    issue_time: datetime
    target_times: np.ndarray
    load_energy: np.ndarray
    pv_energy: np.ndarray

    @property
    def net_energy(self) -> np.ndarray:
        return self.load_energy - self.pv_energy


def target_datetime(
    operating_day: date | datetime | str,
    release: time | datetime | str | int | float,
    horizon: int | float,
) -> datetime:
    """Return release + horizon hours with calendar rollover, never wrapping a row."""
    base = datetime.combine(coerce_date(operating_day), time())
    return base + timedelta(minutes=parse_clock_minutes(release), hours=float(horizon))


def _source_column(target: datetime) -> int:
    """Find the left-endpoint source column for a wall-clock target time."""
    minute = target.hour * 60 + target.minute
    if minute == 0:
        return 143  # 00:00 is the preceding operating row's final left endpoint.
    if minute % 10:
        raise ValueError("forecast targets must lie on the ten-minute grid")
    return minute // 10 - 1


def _causal_load_profile(data: InputData, issue_time: datetime) -> np.ndarray:
    """Mean only cells whose real left endpoint precedes ``issue_time``.

    A row label alone is insufficient: its final ``00:00+1`` cell belongs to
    the next calendar day and is unavailable at that next day's midnight.
    """
    available = source_datetimes(list(data.dates)) < issue_time
    count = available.sum(axis=0)
    total = np.where(available, data.load_energy, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.zeros(144, dtype=float), where=count > 0)


def _interpolate_pv_energy(hourly_kw: np.ndarray, issue_time: datetime, targets: Sequence[datetime]) -> np.ndarray:
    """Linearly interpolate hourly PV kW and convert to kWh once.

    Attachment 3's first value is for one hour after issue.  Before that
    point we hold that first published value; after 24 hours we hold the last
    published value.  Neither operation rotates data across a day boundary.
    """
    knots = np.arange(1, 25, dtype=float)
    offsets = np.asarray([(target - issue_time).total_seconds() / 3600.0 for target in targets])
    interpolated_kw = np.interp(offsets, knots, hourly_kw, left=hourly_kw[0], right=hourly_kw[-1])
    return power_kw_to_energy(interpolated_kw)


def build_information_forecast(
    data: InputData,
    operating_day: date | datetime | str,
    release: time | datetime | str | int | float,
    horizon_steps: int = 144,
) -> InformationForecast:
    """Build an energy forecast from information available at ``operating_day/release``.

    Actual load history is truncated strictly before the issue date.  PV uses
    the forecast issued at this release; it never reads a target-day actual.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    issue_day = coerce_date(operating_day)
    release_minutes = parse_clock_minutes(release)
    if not 0 <= release_minutes < 24 * 60:
        raise ValueError("release must be a same-day clock time")
    issue_time = target_datetime(issue_day, release_minutes, 0)
    hourly_kw = data.pv_hourly_forecasts.get((issue_day, release_minutes))
    if hourly_kw is None:
        raise KeyError(f"missing PV forecast for {issue_day.isoformat()} {release_minutes // 60:02d}:{release_minutes % 60:02d}")

    targets = tuple(issue_time + timedelta(minutes=10 * i) for i in range(horizon_steps))
    profile = _causal_load_profile(data, issue_time)
    load_energy = np.asarray([profile[_source_column(target)] for target in targets])
    pv_energy = _interpolate_pv_energy(hourly_kw, issue_time, targets)
    return InformationForecast(issue_time, np.asarray(targets, dtype=object), load_energy, pv_energy)


def _historical_path(
    residuals: np.ndarray,
    first_day: int,
    start_step: int,
    horizon_steps: int,
) -> np.ndarray:
    """Take chronologically adjacent residual values; never use ``np.roll``."""
    pieces: list[np.ndarray] = []
    day, step, remaining = first_day, start_step, horizon_steps
    while remaining:
        if day >= residuals.shape[0]:
            raise ValueError("insufficient historical residuals for requested horizon")
        take = min(144 - step, remaining)
        pieces.append(residuals[day, step: step + take])
        remaining -= take
        day += 1
        step = 0
    return np.concatenate(pieces)


def build_pv_scenarios(
    point_pv_energy: np.ndarray,
    pv_residuals: np.ndarray,
    issue_day_index: int,
    start_step: int,
    horizon_steps: int,
    n_scenarios: int = 12,
    lookback_days: int = 28,
    seed: int = 2025,
    operating_dates: Sequence[date | datetime | str] | None = None,
    issue_time: datetime | None = None,
) -> np.ndarray:
    """Add only pre-issue PV residual paths to a point forecast.

    Candidates must finish before ``issue_day_index``.  This makes the
    cutoff explicit even for horizons crossing a source-row boundary.
    """
    for name, value in (("horizon_steps", horizon_steps), ("n_scenarios", n_scenarios),
                        ("lookback_days", lookback_days)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    point = np.asarray(point_pv_energy, dtype=float)
    residuals = np.asarray(pv_residuals, dtype=float)
    if point.shape != (horizon_steps,) or residuals.ndim != 2 or residuals.shape[1] != 144:
        raise ValueError("invalid point forecast or residual matrix shape")
    if not (0 <= start_step < 144 and 0 <= issue_day_index <= residuals.shape[0]):
        raise ValueError("invalid issue day or source-row step")
    if operating_dates is None:
        origin = date(2000, 1, 1)
        dates = tuple(origin + timedelta(days=i) for i in range(residuals.shape[0]))
    else:
        dates = tuple(coerce_date(day) for day in operating_dates)
        if len(dates) != residuals.shape[0]:
            raise ValueError("operating_dates must match the residual matrix rows")
    if issue_time is None:
        if issue_day_index < len(dates):
            cutoff = datetime.combine(dates[issue_day_index], time())
        else:
            cutoff = datetime.combine(dates[-1] + timedelta(days=1), time())
    else:
        cutoff = issue_time

    first = max(0, issue_day_index - lookback_days)
    timestamps = source_datetimes(list(dates))
    paths: list[np.ndarray] = []
    for day in range(first, issue_day_index):
        try:
            path = _historical_path(residuals, day, start_step, horizon_steps)
            path_times = _historical_path(timestamps, day, start_step, horizon_steps)
        except ValueError:
            continue
        if np.all(path_times < cutoff):
            paths.append(path)
    if not paths:
        paths = [np.zeros(horizon_steps, dtype=float)]
    rng = np.random.default_rng(seed + issue_day_index)
    chosen = rng.choice(len(paths), size=n_scenarios, replace=len(paths) < n_scenarios)
    return np.maximum(0.0, np.stack([point + paths[int(i)] for i in chosen]))
