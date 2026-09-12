"""Validate the extra physical interval retained when warm-up rows are removed."""
from datetime import datetime, time, timedelta
import numpy as np


def boundary_errors(result, tolerance=1e-6):
    boundary = getattr(result, "calendar_boundary", None)
    trimmed = any(row.get("trimmed_warmup_boundary") for row in result.solve_log)
    if boundary is None:
        return ["calendar boundary missing after warm-up trim"] if trimmed else []
    try:
        stamp = datetime.fromisoformat(boundary["timestamp"])
        c, d, before, after = [float(boundary[k]) for k in
                              ("charge", "discharge", "soc_before", "soc_after")]
        cfg = result.config
        valid = (
            stamp == datetime.combine(result.dates[0], time())
            and np.isfinite([c, d, before, after]).all()
            and -tolerance <= c <= cfg.charge_limit + tolerance
            and -tolerance <= d <= cfg.discharge_limit + tolerance
            and not (c > tolerance and d > tolerance)
            and cfg.soc_min - tolerance <= before <= cfg.soc_max + tolerance
            and cfg.soc_min - tolerance <= after <= cfg.soc_max + tolerance
            and abs(after - before - cfg.charge_efficiency*c + d/cfg.discharge_efficiency) <= tolerance
            and bool(result.executed[0, 0])
            and result.timestamps[0, 0] == stamp + timedelta(minutes=10)
            and abs(after - result.soc_before[0, 0]) <= tolerance
            and abs(after - cfg.initial_soc) <= tolerance
        )
        return [] if valid else ["calendar boundary physics or continuity"]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return ["calendar boundary malformed"]
