"""Fast causal execution between forecast/commitment events.

All flow arguments are interval energies in kWh.  The function sees only the
current interval's actual load/PV and the latest already-issued commitment and
discharge reference; it has no route by which emergency energy can charge the
battery.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BatteryLimits:
    soc_min: float
    soc_max: float
    charge_limit: float
    discharge_limit: float
    charge_efficiency: float
    discharge_efficiency: float
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        finite = (
            self.soc_min, self.soc_max, self.charge_limit, self.discharge_limit,
            self.charge_efficiency, self.discharge_efficiency, self.tolerance,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("battery limits must be finite")
        if self.soc_min < 0 or self.soc_max < self.soc_min:
            raise ValueError("invalid SOC bounds")
        if self.charge_limit < 0 or self.discharge_limit < 0:
            raise ValueError("power limits must be nonnegative")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("efficiencies must be in (0, 1]")
        if self.tolerance < 0:
            raise ValueError("tolerance must be nonnegative")


@dataclass(frozen=True)
class DispatchStep:
    charge: float
    discharge: float
    emergency: float
    spill: float
    mode: int
    soc_before: float
    soc_after: float
    energy_balance_residual: float


def _clip_boundary(value: float, lower: float, upper: float, tolerance: float) -> float:
    if value < lower - tolerance or value > upper + tolerance:
        raise ValueError(f"SOC {value} outside [{lower}, {upper}]")
    return float(min(max(value, lower), upper))


def dispatch_step(
    *,
    load_energy: float,
    pv_energy: float,
    commitment: float,
    soc: float,
    charge_reference: float | None = None,
    discharge_reference: float,
    limits: BatteryLimits,
) -> DispatchStep:
    """Execute one observed interval while preserving the latest battery plan.

    A positive surplus charges first and then spills.  A deficit discharges no
    more than the latest scenario-weighted reference; emergency supply covers
    the remainder.  Consequently the optimizer may intentionally retain SOC.
    """
    raw = (load_energy, pv_energy, commitment, soc, discharge_reference)
    if charge_reference is not None:
        raw += (charge_reference,)
    if not all(math.isfinite(float(value)) for value in raw):
        raise ValueError("dispatch inputs must be finite")
    if min(load_energy, pv_energy, commitment, discharge_reference,
           0.0 if charge_reference is None else charge_reference) < -limits.tolerance:
        raise ValueError("energy flows and references must be nonnegative")
    load_energy = max(0.0, float(load_energy))
    pv_energy = max(0.0, float(pv_energy))
    commitment = max(0.0, float(commitment))
    discharge_reference = max(0.0, float(discharge_reference))
    charge_cap = float("inf") if charge_reference is None else max(0.0, float(charge_reference))
    soc_before = _clip_boundary(float(soc), limits.soc_min, limits.soc_max, limits.tolerance)

    surplus = commitment + pv_energy - load_energy
    charge = discharge = emergency = spill = 0.0
    if surplus >= 0:
        capacity_input = (limits.soc_max - soc_before) / limits.charge_efficiency
        charge = min(surplus, charge_cap, limits.charge_limit, max(0.0, capacity_input))
        spill = surplus - charge
        # ``mode`` is a semantic binary indicator, not a material-flow filter:
        # even a tiny positive charge must be represented by the charge mode.
        mode = 1 if charge > 0.0 else 0
    else:
        deficit = -surplus
        available_output = (soc_before - limits.soc_min) * limits.discharge_efficiency
        discharge = min(deficit, discharge_reference, limits.discharge_limit,
                        max(0.0, available_output))
        emergency = deficit - discharge
        mode = 0

    soc_after = soc_before + limits.charge_efficiency * charge - discharge / limits.discharge_efficiency
    soc_after = _clip_boundary(soc_after, limits.soc_min, limits.soc_max, limits.tolerance)
    residual = commitment + pv_energy + discharge + emergency - load_energy - charge - spill
    if abs(residual) > limits.tolerance:
        raise ValueError(f"energy balance residual {residual} exceeds tolerance")
    return DispatchStep(float(charge), float(discharge), float(emergency), float(spill), mode,
                        soc_before, soc_after, float(residual))
