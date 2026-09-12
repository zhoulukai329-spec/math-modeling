from __future__ import annotations

from datetime import date, datetime, time, timedelta
import importlib
from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _modules():
    for name in ("data_io", "price_forecast"):
        sys.modules.pop(name, None)
    return importlib.import_module("data_io"), importlib.import_module("price_forecast")


def _calendar(days=10):
    first = date(2025, 1, 1)
    dates = tuple(first + timedelta(days=i) for i in range(days))
    stamps = np.asarray([
        [datetime.combine(day, time()) + timedelta(minutes=10 * (j + 1)) for j in range(144)]
        for day in dates
    ], dtype=object)
    return dates, stamps


def test_q43_reads_attachment4_in_original_left_endpoint_order():
    dio, _ = _modules()
    data = dio.load_inputs(Path(__file__).resolve().parents[2] / "attachment")
    assert data.price.shape == (365, 144)
    assert data.template_labels[-1].startswith("0:00")
    assert not hasattr(dio, "ATTACHMENT1")


def test_price_forecast_uses_revealed_current_price_and_week_lag_for_future():
    _, pf = _modules()
    _, stamps = _calendar()
    price = np.arange(10 * 144, dtype=float).reshape(10, 144) + 1
    issue = datetime(2025, 1, 9)
    targets = np.array([issue, issue + timedelta(minutes=10), issue + timedelta(hours=1)], dtype=object)
    point = pf.build_causal_price_forecast(price, stamps, issue, targets)
    assert point[0] == price[7, -1]
    assert point[1] == price[1, 0]
    assert point[2] == price[1, 5]


def test_unrevealed_future_price_changes_cannot_change_the_forecast():
    _, pf = _modules()
    _, stamps = _calendar()
    price = np.arange(10 * 144, dtype=float).reshape(10, 144) + 1
    issue = datetime(2025, 1, 5, 6)
    targets = stamps[4, 36:48]
    before = pf.build_causal_price_forecast(price, stamps, issue, targets)
    changed = price.copy()
    changed[stamps >= issue] += 1_000_000
    after = pf.build_causal_price_forecast(changed, stamps, issue, targets)
    np.testing.assert_array_equal(after, before)


def test_insufficient_history_uses_only_revealed_same_slot_values():
    _, pf = _modules()
    _, stamps = _calendar(days=4)
    price = np.arange(4 * 144, dtype=float).reshape(4, 144) + 10
    issue = datetime(2025, 1, 4)
    target = np.array([datetime(2025, 1, 4, 0, 10)], dtype=object)
    point = pf.build_causal_price_forecast(price, stamps, issue, target)
    assert point[0] == np.mean(price[:3, 0])


def test_price_scenarios_use_explicit_historical_source_rows():
    _, pf = _modules()
    _, stamps = _calendar(days=10)
    price = np.tile(np.arange(144, dtype=float), (10, 1)) + np.arange(10)[:, None] * 10
    issue = datetime(2025, 1, 10)
    targets = stamps[9, :3]
    point = pf.build_causal_price_forecast(price, stamps, issue, targets)
    scenarios = pf.build_price_scenarios(
        point, price, stamps, targets, issue, source_days=np.array([1, 3])
    )
    assert scenarios.shape == (2, 3)
    assert np.isfinite(scenarios).all() and np.all(scenarios >= 0)
    assert not np.array_equal(scenarios[0], scenarios[1])


def test_price_scenario_path_continues_forward_across_source_midnight():
    _, pf = _modules()
    _, stamps = _calendar(days=10)
    price = np.arange(10 * 144, dtype=float).reshape(10, 144) + 1
    issue = datetime(2025, 1, 9)
    targets = np.array([issue, issue + timedelta(minutes=10), issue + timedelta(minutes=20)], dtype=object)
    point = pf.build_causal_price_forecast(price, stamps, issue, targets)
    scenarios = pf.build_price_scenarios(point, price, stamps, targets, issue, source_days=[2])
    assert scenarios.shape == (1, 3)
    source_targets = np.array([stamps[2, -1], stamps[3, 0], stamps[3, 1]], dtype=object)
    historical_point = pf.build_causal_price_forecast(price, stamps, source_targets[0], source_targets)
    source_actual = np.array([price[2, -1], price[3, 0], price[3, 1]])
    np.testing.assert_allclose(scenarios[0], np.maximum(0, point + source_actual - historical_point))
