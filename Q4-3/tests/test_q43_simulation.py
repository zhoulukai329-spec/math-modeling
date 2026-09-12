from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import importlib
from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def modules():
    for name in ("data_io", "forecast", "simulation", "price_forecast"):
        sys.modules.pop(name, None)
    dio = importlib.import_module("data_io")
    sim = importlib.import_module("simulation")
    return dio, sim


def inputs(dio, days=4):
    dates = tuple(date(2025, 2, 1) + timedelta(days=i) for i in range(days))
    load = np.full((days, 144), 2.0)
    pv = np.zeros_like(load)
    price = 1.0 + np.arange(days)[:, None] + np.arange(144)[None, :] / 1000
    forecasts = {(day, minute): np.zeros(24) for day in dates for minute in (0, 360, 720, 1080)}
    return dio.InputData(dates, load, pv, price, forecasts)


def config(sim, data, **changes):
    values = dict(data=data, horizon_steps=3, max_steps=3, n_scenarios=2,
                  initial_soc=0.0, soc_min=0.0, soc_max=10.0,
                  charge_limit=0.0, discharge_limit=0.0,
                  terminal_soc=0.0, terminal_penalty=0.0,
                  backend="event-policy", january_warmup=False)
    values.update(changes)
    return sim.SimulationConfig(**values)


def test_event_backend_uses_dynamic_forecasts_but_settles_at_actual_target_prices():
    dio, sim = modules()
    data = inputs(dio)
    result = sim.simulate(config(sim, data), data.dates[2], data.dates[2])
    assert result.executed.sum() == 3
    assert [row["kind"] for row in result.solve_log] == ["baseline"]
    assert result.solve_log[0]["price_information"] == "causal-forecast"
    np.testing.assert_allclose(result.baseline[0, :3], 2.0)
    expected = float(data.price[2] @ result.baseline[0])
    assert result.costs["baseline"] == expected
    assert result.emergency[0, :3].sum() == 0
    assert np.isfinite(result.price_forecast[0, :3]).all()


def test_future_actual_prices_cannot_change_q43_commitments_or_actions():
    dio, sim = modules()
    data = inputs(dio)
    first = sim.simulate(config(sim, data), data.dates[2], data.dates[2])
    changed_price = data.price.copy()
    changed_price[2, 3:] += 100000
    changed_price[3] += 100000
    changed = sim.simulate(config(sim, replace(data, price=changed_price)), data.dates[2], data.dates[2])
    for name in ("baseline", "final_commitment", "charge", "discharge", "emergency", "soc_after"):
        np.testing.assert_allclose(getattr(first, name), getattr(changed, name), equal_nan=True)


def test_q43_reference_backend_remains_selectable():
    dio, sim = modules()
    data = inputs(dio)
    result = sim.simulate(config(sim, data, backend="rolling-milp", max_steps=2), data.dates[2], data.dates[2])
    assert [row["kind"] for row in result.solve_log] == ["baseline", "execution", "execution"]
    assert all(row["backend"] == "rolling-milp" for row in result.solve_log)

