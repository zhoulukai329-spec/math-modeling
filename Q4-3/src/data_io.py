"""Exact, timestamp-safe readers for Question 4-3 input workbooks.

The workbooks use a non-natural operating-day order: each row is labelled by
left endpoints ``00:10, 00:20, ..., 00:00+1``.  This module deliberately
preserves that order.  Values leaving this module are interval energies in
kWh (except price), so downstream optimisation must not multiply by ``DT``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np
from openpyxl import load_workbook


DT_HOURS = 1.0 / 6.0
PERIODS_PER_DAY = 144
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
ATTACHMENT_DIR = REPO_ROOT / "attachment"


@dataclass(frozen=True)
class InputData:
    """All Q4-3 inputs, with actual load/PV converted to kWh/interval."""

    dates: tuple[date, ...]
    load_energy: np.ndarray
    pv_energy: np.ndarray
    price: np.ndarray
    pv_hourly_forecasts: Mapping[tuple[date, int], np.ndarray]
    template_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n_days = len(self.dates)
        expected = (n_days, PERIODS_PER_DAY)
        for name in ("load_energy", "pv_energy", "price"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            object.__setattr__(self, name, value)
        cleaned = {
            (coerce_date(day), int(release)): np.asarray(values, dtype=float)
            for (day, release), values in self.pv_hourly_forecasts.items()
        }
        for key, values in cleaned.items():
            if values.shape != (24,):
                raise ValueError(f"PV forecast {key} must contain 24 hourly kW values")
        object.__setattr__(self, "pv_hourly_forecasts", cleaned)

    @property
    def date_to_index(self) -> dict[date, int]:
        return {d: i for i, d in enumerate(self.dates)}


def coerce_date(value: date | datetime | np.datetime64 | str) -> date:
    """Normalise an input day without changing its calendar date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return datetime.fromisoformat(str(value)[:10]).date()
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        year, month, day = text.split("-", 2)
        return date(int(year), int(month), int(day))


def parse_clock_minutes(value: time | datetime | str | int | float) -> int:
    """Parse a workbook clock cell; ``0:00+1`` is explicitly 1440 minutes."""
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    if isinstance(value, (int, float)):
        # Excel time cells are fractions of a day.  Integral minute input is
        # intentionally accepted too, which makes the public helper usable in tests.
        raw = float(value)
        return int(round(raw * 1440)) if abs(raw) <= 1 else int(round(raw))
    text = str(value).strip()
    # Result-template cells use interval labels rather than bare clocks.  The
    # final label is often written ``0:00-0:10+1``: its left endpoint is the
    # following midnight, while e.g. ``23:50-0:00+1`` remains 23:50 today.
    if "-" in text:
        left, right = text.split("-", 1)
        if "+" not in left and left in {"0:00", "00:00"} and "+" in right:
            text = left + "+" + right.split("+", 1)[1]
        else:
            text = left
    suffix = 0
    if "+" in text:
        text, offset = text.split("+", 1)
        suffix = int(offset) * 1440
    hour, minute = text.split(":", 1)
    return int(hour) * 60 + int(minute) + suffix


def template_left_endpoint_minutes() -> np.ndarray:
    """Return the attachment/template sequence; never rotate it to midnight."""
    return np.arange(10, 24 * 60 + 10, 10, dtype=int)


def template_datetimes(operating_day: date | datetime | str) -> tuple[datetime, ...]:
    """Map one source row to its actual left-endpoint datetimes.

    The last element is midnight of the following date, as required by the
    source row's ``0:00+1`` label.
    """
    start = datetime.combine(coerce_date(operating_day), time())
    return tuple(start + timedelta(minutes=int(m)) for m in template_left_endpoint_minutes())


def source_datetimes(operating_days: tuple[date, ...] | list[date]) -> np.ndarray:
    """Return each source cell's true calendar left endpoint, shape ``(D, 144)``.

    The final column in operating row ``d`` is midnight of calendar day
    ``d + 1``.  Consumers must use this matrix for information cutoffs instead
    of treating every cell in a row as belonging to its row label.
    """
    return np.asarray([template_datetimes(day) for day in operating_days], dtype=object)


def power_kw_to_energy(power_kw: np.ndarray, dt_hours: float = DT_HOURS) -> np.ndarray:
    """Perform the sole kW -> kWh interval conversion at an input boundary."""
    return np.asarray(power_kw, dtype=float) * float(dt_hours)


def _require_template_order(header: tuple[object, ...] | list[object]) -> tuple[str, ...]:
    if len(header) < PERIODS_PER_DAY + 1:
        raise ValueError(f"expected a date column and {PERIODS_PER_DAY} intervals")
    actual = np.array([parse_clock_minutes(v) for v in header[1: PERIODS_PER_DAY + 1]], dtype=int)
    expected = template_left_endpoint_minutes()
    if not np.array_equal(actual, expected):
        raise ValueError("input interval columns are not in the template left-endpoint order")
    return tuple(str(v) for v in header[1: PERIODS_PER_DAY + 1])


def _read_actuals(path: Path) -> tuple[tuple[date, ...], np.ndarray, np.ndarray]:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        if len(book.worksheets) < 2:
            raise ValueError("附件2 must contain load and PV sheets")
        sheets = book.worksheets[:2]
        headers = [next(ws.iter_rows(values_only=True)) for ws in sheets]
        _require_template_order(headers[0])
        _require_template_order(headers[1])

        matrices: list[np.ndarray] = []
        all_dates: list[date] | None = None
        for ws in sheets:
            rows = list(ws.iter_rows(min_row=2, values_only=True))
            days = [coerce_date(row[0]) for row in rows]
            values = np.asarray([row[1: PERIODS_PER_DAY + 1] for row in rows], dtype=float)
            if values.shape != (len(days), PERIODS_PER_DAY):
                raise ValueError("actual-data matrix shape does not match its date rows")
            if all_dates is None:
                all_dates = days
            elif days != all_dates:
                raise ValueError("load and PV date rows do not match")
            matrices.append(power_kw_to_energy(values))
        assert all_dates is not None
        return tuple(all_dates), matrices[0], matrices[1]
    finally:
        book.close()


def _read_price(path: Path, expected_dates: tuple[date, ...]) -> np.ndarray:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = book.worksheets[0]
        _require_template_order(next(ws.iter_rows(values_only=True)))
        rows = list(ws.iter_rows(min_row=2, values_only=True))
        dates = tuple(coerce_date(row[0]) for row in rows)
        if dates != expected_dates:
            raise ValueError("price dates do not match actual-data dates")
        return np.asarray([row[1: PERIODS_PER_DAY + 1] for row in rows], dtype=float)
    finally:
        book.close()


def _read_pv_forecasts(path: Path) -> dict[tuple[date, int], np.ndarray]:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = book.worksheets[0]
        out: dict[tuple[date, int], np.ndarray] = {}
        current_day: date | None = None
        current_releases: list[int] = []
        last_day: date | None = None
        expected_releases = [0, 6 * 60, 12 * 60, 18 * 60]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] not in (None, ""):
                next_day = coerce_date(row[0])
                if current_day is not None and current_releases != expected_releases:
                    raise ValueError("each PV forecast day must contain releases 0:00, 6:00, 12:00, 18:00 in order")
                if last_day is not None and next_day <= last_day:
                    raise ValueError("PV forecast dates must be strictly increasing")
                current_day = next_day
                last_day = next_day
                current_releases = []
            if current_day is None or row[1] in (None, ""):
                raise ValueError("PV forecast row is missing its operating date or release time")
            if len(row) != 26:
                raise ValueError("PV forecast row must have exactly 24 hourly columns")
            release = parse_clock_minutes(row[1])
            if len(current_releases) >= len(expected_releases) or release != expected_releases[len(current_releases)]:
                raise ValueError("PV forecast releases must be 0:00, 6:00, 12:00, 18:00 in order")
            values = np.asarray(row[2:], dtype=float)
            if values.shape != (24,):
                raise ValueError("PV forecast row must have 24 hourly columns")
            key = (current_day, release)
            if key in out:
                raise ValueError(f"duplicate PV forecast release {key}")
            out[key] = values
            current_releases.append(release)
        if current_day is None or current_releases != expected_releases:
            raise ValueError("each PV forecast day must contain releases 0:00, 6:00, 12:00, 18:00 in order")
        return out
    finally:
        book.close()


def _read_template_labels(path: Path) -> tuple[str, ...]:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        header = next(book.worksheets[0].iter_rows(values_only=True))
        _require_template_order(header)
        return tuple(str(value) for value in header[1: PERIODS_PER_DAY + 1])
    finally:
        book.close()


def load_inputs(
    attachment_dir: Path | str = ATTACHMENT_DIR,
) -> InputData:
    """Read Q4-3 attachments exactly, preserving their original row/column order.

    No array in this function is rolled, rotated, or otherwise realigned.  In
    particular, the last source column remains the following midnight.
    """
    root = Path(attachment_dir)
    dates, load_energy, pv_energy = _read_actuals(root / "附件2.xlsx")
    price = _read_price(root / "附件4.xlsx", dates)
    forecasts = _read_pv_forecasts(root / "附件3.xlsx")
    if {day for day, _release in forecasts} != set(dates):
        raise ValueError("PV forecast dates do not match actual-data dates")
    labels = _read_template_labels(root / "附件5" / "result4-3.xlsx")
    return InputData(dates, load_energy, pv_energy, price, forecasts, labels)
