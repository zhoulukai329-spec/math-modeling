"""Causal dynamic-price point forecasts and matched historical scenarios."""
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np


def _validated(price_actual, timestamps):
    price = np.asarray(price_actual, dtype=float)
    times = np.asarray(timestamps, dtype=object)
    if price.ndim != 2 or price.shape != times.shape or price.shape[1] != 144:
        raise ValueError("price_actual and timestamps must have shape (D,144)")
    if not np.isfinite(price).all() or np.any(price < 0):
        raise ValueError("actual prices must be finite and nonnegative")
    return price, times


def build_causal_price_forecast(price_actual, timestamps, issue_time, target_times):
    """Forecast targets using current/revealed quotes, week lag, then slot mean.

    The single 2025-01-01 boundary with no earlier quote uses the first
    Attachment-4 quote as a documented initialization prior.  Formal reported
    results start after January, so this cell is warm-up only.
    """
    price, times = _validated(price_actual, timestamps)
    issue = issue_time if isinstance(issue_time, datetime) else datetime.fromisoformat(str(issue_time))
    targets = np.asarray(target_times, dtype=object)
    flat_time, flat_price = times.ravel(), price.ravel()
    lookup = {stamp: float(value) for stamp, value in zip(flat_time, flat_price)}
    revealed = flat_time <= issue
    last_known = float(flat_price[np.flatnonzero(revealed)[-1]]) if revealed.any() else float(flat_price[0])
    out = np.empty(len(targets), dtype=float)
    for i, target in enumerate(targets):
        if target <= issue and target in lookup:
            out[i] = lookup[target]
            continue
        lag = target - timedelta(days=7)
        if lag < issue and lag in lookup:
            out[i] = lookup[lag]
            continue
        same_slot = np.array([
            value for stamp, value in zip(flat_time[revealed], flat_price[revealed])
            if stamp.time() == target.time()
        ], dtype=float)
        out[i] = float(same_slot.mean()) if same_slot.size else last_known
    return out


def build_price_scenarios(point_forecast, price_actual, timestamps, target_times,
                          issue_time, source_days):
    """Add price errors from explicitly paired, fully historical source rows."""
    price, times = _validated(price_actual, timestamps)
    point = np.asarray(point_forecast, dtype=float)
    targets = np.asarray(target_times, dtype=object)
    sources = np.asarray(source_days, dtype=int)
    if point.shape != (len(targets),) or sources.ndim != 1 or sources.size == 0:
        raise ValueError("invalid point forecast, targets, or source days")
    issue = issue_time if isinstance(issue_time, datetime) else datetime.fromisoformat(str(issue_time))
    scenarios = []
    target_offsets = np.array([(target.hour * 60 + target.minute) // 10 - 1 for target in targets])
    for source in sources:
        if source < 0 or source >= len(price):
            residual = np.zeros(len(targets))
        else:
            source_targets = times[source, np.mod(target_offsets, 144)]
            if np.any(source_targets >= issue):
                raise ValueError("price scenario source is not fully historical")
            historical_point = build_causal_price_forecast(price, times,
                datetime.combine(source_targets[0].date(), datetime.min.time()), source_targets)
            residual = price[source, np.mod(target_offsets, 144)] - historical_point
        scenarios.append(np.maximum(0.0, point + residual))
    return np.asarray(scenarios, dtype=float)

