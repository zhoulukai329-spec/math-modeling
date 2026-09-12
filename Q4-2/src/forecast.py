# -*- coding: utf-8 -*-
"""Causal net-load and price forecasts for Q4-2."""
import sys
from pathlib import Path

import numpy as np


Q2_SRC = Path(__file__).resolve().parents[2] / "Q2" / "src"
if str(Q2_SRC) not in sys.path:
    sys.path.append(str(Q2_SRC))
import importlib.util


def _load_q2_forecast():
    spec = importlib.util.spec_from_file_location("_q2_forecast", Q2_SRC / "forecast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_q2_forecast = _load_q2_forecast()
build_causal_forecasts = _q2_forecast.build_causal_forecasts


def next_day_point_forecasts(net_actual, price_actual, next_midnight_price):
    """Build the post-period day forecast from information known at midnight."""
    net_actual = np.asarray(net_actual, dtype=float)
    price_actual = np.asarray(price_actual, dtype=float)
    if net_actual.ndim != 2 or price_actual.shape != net_actual.shape or len(net_actual) < 7:
        raise ValueError("跨年边界预测需要形状一致且至少7天的历史")
    net_next = net_actual[-7].copy()
    price_next = price_actual[-7].copy()
    price_next[0] = float(next_midnight_price)
    return net_next, price_next


def build_causal_price_forecasts(price_actual):
    """Forecast each day using only prices observed before that day's 00:00."""
    price_actual = np.asarray(price_actual, dtype=float)
    if price_actual.ndim != 2:
        raise ValueError("电价实际矩阵形状错误")
    forecast = np.empty_like(price_actual)
    for d in range(price_actual.shape[0]):
        if d == 0:
            # No earlier price history exists. Persistence of the quote known
            # at 00:00 is the only data-causal prior available in Attachment 4.
            forecast[d] = price_actual[d, 0]
        elif d < 7:
            forecast[d] = price_actual[:d].mean(axis=0)
        else:
            forecast[d] = price_actual[d - 7]
        # At the 00:00 decision instant the current 00:00-00:10 real-time
        # price has been published. Later intervals remain forecasts.
        forecast[d, 0] = price_actual[d, 0]
    return forecast, price_actual - forecast


def joint_scenarios_for_day(d, net_forecast, price_forecast,
                            net_residual, price_residual,
                            n_scenarios=12, lookback=28, seed=2025):
    """Sample matched historical residual days for net load and price."""
    if d < 0 or d >= len(net_forecast):
        raise IndexError("目标日期索引越界")
    rng = np.random.default_rng(seed + d)
    available = np.arange(max(0, d - lookback), d, dtype=int)
    if len(available) == 0:
        source = np.full(n_scenarios, -1, dtype=int)
        net_s = np.repeat(net_forecast[d][None, :], n_scenarios, axis=0)
        price_s = np.repeat(price_forecast[d][None, :], n_scenarios, axis=0)
    else:
        source = rng.choice(
            available, size=n_scenarios, replace=len(available) < n_scenarios
        )
        net_s = net_forecast[d][None, :] + net_residual[source]
        price_s = price_forecast[d][None, :] + price_residual[source]
    # Actual Attachment 4 is nonnegative. Residual recombination can cross zero;
    # zero is the conservative physical lower bound for this problem statement.
    price_s = np.maximum(price_s, 0.0)
    return net_s, price_s, source
