"""Exact scenario reduction helpers."""
from __future__ import annotations

import numpy as np


def deduplicate_scenarios(values, probabilities=None):
    """Merge byte-identical scenario rows without changing their total weight."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError("scenarios must be a nonempty finite matrix")
    count = array.shape[0]
    probability = (np.full(count, 1.0 / count) if probabilities is None
                   else np.asarray(probabilities, dtype=float))
    if (probability.shape != (count,) or not np.isfinite(probability).all()
            or np.any(probability <= 0)
            or not np.isclose(probability.sum(), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("probabilities must be positive and sum to one")

    positions: dict[bytes, int] = {}
    rows: list[np.ndarray] = []
    weights: list[float] = []
    inverse = np.empty(count, dtype=int)
    contiguous = np.ascontiguousarray(array)
    for source, row in enumerate(contiguous):
        key = row.tobytes()
        target = positions.get(key)
        if target is None:
            target = len(rows)
            positions[key] = target
            rows.append(row.copy())
            weights.append(0.0)
        weights[target] += float(probability[source])
        inverse[source] = target
    return np.asarray(rows, dtype=float), np.asarray(weights, dtype=float), inverse

