"""Causal ten-minute MPC execution and version-to-version cash settlement.

Rows retain the attachment's 00:10 ... next-day 00:00 left endpoints.
At midnight the new row's baseline is planned while the old row's tail stays
fixed. Beyond-row purchases in an MPC horizon are virtual 1x-price lookahead;
they enter the ledger only when that row's actual baseline is issued.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from data_io import InputData, coerce_date, load_inputs, source_datetimes
from forecast import build_information_forecast, build_pv_scenarios
from model import MPCProblem, MPCSolution, solve_mpc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dispatch_core import BatteryLimits, dispatch_step


@dataclass(frozen=True)
class CommitmentVersion:
    issued_at: datetime
    kind: str
    target_times: tuple[datetime, ...]
    commitment: np.ndarray
    revision_up: np.ndarray
    revision_down: np.ndarray
    baseline_cost: float
    up_cost: float
    down_cost: float


class CommitmentLedger:
    """One business row; mutation is atomic and executed/past cells are frozen."""

    def __init__(self, target_times: Sequence[datetime], price: np.ndarray):
        self.target_times = tuple(target_times)
        self.price = np.array(price, dtype=float, copy=True)
        if (not self.target_times or self.price.shape != (len(self.target_times),)
                or not np.isfinite(self.price).all() or np.any(self.price < 0)
                or any(b <= a for a, b in zip(self.target_times, self.target_times[1:]))):
            raise ValueError("ledger requires ordered unique times and nonnegative finite prices")
        self.baseline = np.full(len(self.target_times), np.nan)
        self.commitment = self.baseline.copy()
        self.executed = np.zeros(len(self.target_times), dtype=bool)
        self.versions: list[CommitmentVersion] = []

    @property
    def costs(self) -> dict[str, float]:
        return dict(baseline=sum(v.baseline_cost for v in self.versions),
                    revision_up=sum(v.up_cost for v in self.versions),
                    revision_down=sum(v.down_cost for v in self.versions))

    def revise(self, issued_at: datetime, commitment: Sequence[float],
               *, kind: str = "revision") -> CommitmentVersion:
        candidate = np.array(commitment, dtype=float, copy=True)
        if candidate.shape != self.price.shape or not np.isfinite(candidate).all() or np.any(candidate < -1e-7):
            raise ValueError("commitment must be a nonnegative finite ledger-sized vector")
        candidate = np.maximum(candidate, 0.0)
        if kind not in ("baseline", "revision"):
            raise ValueError("kind must be baseline or revision")
        if self.versions and issued_at < self.versions[-1].issued_at:
            raise ValueError("versions must be chronological")
        if kind == "baseline":
            if self.versions:
                raise ValueError("baseline already exists")
            if any(target < issued_at for target in self.target_times):
                raise ValueError("baseline cannot purchase past intervals")
            up = down = np.zeros_like(candidate)
            baseline_cost = float(self.price @ candidate)
        else:
            if not self.versions:
                raise ValueError("baseline required before revision")
            frozen = self.executed | (np.asarray(self.target_times, dtype=object) < issued_at)
            if np.any(np.abs(candidate[frozen] - self.commitment[frozen]) > 1e-7):
                raise ValueError("past or executed commitment is frozen")
            candidate[frozen] = self.commitment[frozen]
            up = np.maximum(candidate - self.commitment, 0.0)
            down = np.maximum(self.commitment - candidate, 0.0)
            baseline_cost = 0.0
        version = CommitmentVersion(issued_at, kind, self.target_times, candidate.copy(),
                                    up.copy(), down.copy(), baseline_cost,
                                    float(1.5 * self.price @ up), float(.5 * self.price @ down))
        # Version arrays are immutable snapshots, independent of live state.
        for array in (version.commitment, version.revision_up, version.revision_down):
            array.setflags(write=False)
        if kind == "baseline":
            self.baseline = candidate.copy()
        self.commitment = candidate
        self.versions.append(version)
        return version

    def mark_executed(self, timestamp: datetime) -> None:
        if not self.versions:
            raise ValueError("baseline required before execution")
        index = self.target_times.index(timestamp)
        if self.executed[index]:
            raise ValueError("interval already executed")
        self.executed[index] = True


@dataclass(frozen=True)
class SimulationConfig:
    data: InputData | None = field(default=None, repr=False)
    attachment_dir: str | Path | None = None
    horizon_steps: int = 144
    max_steps: int | None = None
    n_scenarios: int = 12
    lookback_days: int = 28
    seed: int = 2025
    deterministic: bool = False
    january_warmup: bool = True
    revision_hours: tuple[int, ...] = (6, 12, 18)
    initial_soc: float = 6000.0
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    charge_limit: float = 5000.0 / 6.0
    discharge_limit: float = 5000.0 / 6.0
    charge_efficiency: float = .9
    discharge_efficiency: float = .9
    terminal_soc: float = 6000.0
    terminal_penalty: float = .1
    throughput_penalty: float = 1e-6
    cvar_weight: float = 0.0
    cvar_alpha: float = .9
    time_limit: float = 30.0
    mip_rel_gap: float = 1e-4
    backend: str = "rolling-milp"

    def __post_init__(self):
        for name in ("horizon_steps", "max_steps", "n_scenarios", "lookback_days"):
            value = getattr(self, name)
            if name == "max_steps" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon_steps > 144:
            raise ValueError("horizon_steps cannot exceed the published 24-hour horizon")
        if self.backend not in {"rolling-milp", "event-policy"}:
            raise ValueError("backend must be 'rolling-milp' or 'event-policy'")
        if (len(set(self.revision_hours)) != len(self.revision_hours)
                or any(isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= 23
                       for h in self.revision_hours)):
            raise ValueError("revision_hours must be unique integer hours in 1..23")

    def model_parameters(self) -> dict[str, Any]:
        names = ("soc_min", "soc_max", "charge_limit", "discharge_limit", "charge_efficiency",
                 "discharge_efficiency", "terminal_soc", "terminal_penalty", "throughput_penalty",
                 "cvar_weight", "cvar_alpha", "time_limit", "mip_rel_gap")
        return {name: getattr(self, name) for name in names}


@dataclass
class SimulationResult:
    """All matrices are (selected operating days,144), in source column order.

    Unexecuted physical values and unavailable future baselines are NaN, never
    zero-filled. Cash costs include full issued commitments even in a partial
    smoke run; only emergency cost is conditional on actual execution.
    """
    dates: tuple[date, ...]
    timestamps: np.ndarray
    executed: np.ndarray
    baseline: np.ndarray
    final_commitment: np.ndarray
    load_energy: np.ndarray
    pv_energy: np.ndarray
    price: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray
    mode: np.ndarray
    discharge_reference: np.ndarray
    soc_before: np.ndarray
    soc_after: np.ndarray
    versions: tuple[CommitmentVersion, ...]
    costs: dict[str, float]
    solve_log: list[dict[str, Any]]
    config: SimulationConfig
    elapsed_seconds: float

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs.values()))


class SimulationSolveError(RuntimeError):
    """No valid incumbent: stop rather than substitute a dispatch rule."""


def _clip_soc_roundoff(value: float, lower: float, upper: float,
                       tolerance: float = 1e-6) -> float:
    """Clip solver-sized floating error at SOC bounds; reject real violations."""
    if value < lower - tolerance or value > upper + tolerance:
        raise SimulationSolveError(
            f"executed SOC {value} outside bounds [{lower}, {upper}]")
    return float(np.clip(value, lower, upper))


def _release_catalog(data: InputData):
    releases = sorted((datetime.combine(day, time()) + timedelta(minutes=minute), key)
                      for key in data.pv_hourly_forecasts for day, minute in [key])
    return [pair[0] for pair in releases], [pair[1] for pair in releases]


class _ResidualHistory:
    """Reveal residual cells only when their actual left endpoint is in the past."""

    def __init__(self, data: InputData):
        self.data = data
        self.times = source_datetimes(list(data.dates))
        self.panel = np.full_like(data.pv_energy, np.nan)
        self.releases, self.keys = _release_catalog(data)
        self.position = 0

    def reveal_before(self, cutoff: datetime) -> np.ndarray:
        times = self.times.ravel()
        while self.position < len(times) and times[self.position] < cutoff:
            target = times[self.position]
            release_index = bisect_right(self.releases, target) - 1
            if release_index < 0:
                raise ValueError(f"no published PV forecast available for historical {target}")
            release = self.releases[release_index]
            hourly = self.data.pv_hourly_forecasts[self.keys[release_index]]
            offset = (target - release).total_seconds() / 3600
            point = np.interp(offset, np.arange(1, 25), hourly) / 6
            # This is the only actual PV read by historical scenario building.
            self.panel.flat[self.position] = self.data.pv_energy.flat[self.position] - point
            self.position += 1
        return self.panel


def build_historical_residuals(data: InputData, issue_time: datetime) -> np.ndarray:
    """Actual minus latest-at-target published PV forecast; unknown cells are NaN.

    Scenario paths sample these errors in chronological order. Their historical
    forecasts are the latest release at each historical target, so this is a
    rolling-forecast error library, not a fixed-lead residual calibration.
    """
    return _ResidualHistory(data).reveal_before(issue_time).copy()


def simulate(config: SimulationConfig, date_start: date | str,
             date_end: date | str, *, progress=None) -> SimulationResult:
    """Run all selected rows (inclusive), or stop after ``max_steps`` executions.

    ``horizon_steps`` bounds execution lookahead. Issued baseline/revision solves
    additionally cover the entire affected row so a short smoke horizon never
    leaves real commitments undefined. Every execution calls ``solve_mpc`` with
    all already issued commitments fixed and a common current observation.
    """
    started = perf_counter()
    data = config.data if config.data is not None else (
        load_inputs() if config.attachment_dir is None else load_inputs(config.attachment_dir))
    start, end = coerce_date(date_start), coerce_date(date_end)
    if start > end or start not in data.date_to_index or end not in data.date_to_index:
        raise ValueError("date range must be ordered and present in the input")
    if any(b - a != timedelta(days=1) for a, b in zip(data.dates, data.dates[1:])):
        raise ValueError("simulation requires consecutive input operating dates")
    source_rows = [i for i, day in enumerate(data.dates) if start <= day <= end]
    dates = tuple(data.dates[i] for i in source_rows)
    all_times = source_datetimes(list(data.dates))
    flat_times = all_times.ravel()
    time_index = {timestamp: i for i, timestamp in enumerate(flat_times)}
    shape = (len(dates), 144)
    observed = {name: np.full(shape, np.nan) for name in
                ("load_energy", "pv_energy", "charge", "discharge", "emergency", "spill", "mode",
                 "discharge_reference",
                 "soc_before", "soc_after")}
    executed = np.zeros(shape, dtype=bool)
    ledgers: dict[date, CommitmentLedger] = {}
    versions: list[CommitmentVersion] = []
    solve_log: list[dict[str, Any]] = []
    residuals = _ResidualHistory(data)
    releases, release_keys = _release_catalog(data)
    forecast_cache: dict[tuple[datetime, int], Any] = {}
    reference_by_time: dict[datetime, float] = {}
    soc = config.initial_soc
    limits = BatteryLimits(config.soc_min, config.soc_max, config.charge_limit,
                           config.discharge_limit, config.charge_efficiency,
                           config.discharge_efficiency, tolerance=1e-5)

    def solve_at(now: datetime, first: int, length: int, kind: str,
                 mutable_day: date | None = None) -> MPCSolution:
        length = min(length, len(flat_times) - first)
        targets = flat_times[first:first + length]
        release_index = bisect_right(releases, now) - 1
        if release_index < 0:
            raise ValueError(f"missing forecast at {now}")
        release = releases[release_index]
        issue_day, minute = release_keys[release_index]
        offset = int((targets[0] - release).total_seconds() // 600)
        # A publication's forecast never changes between decisions. Construct
        # the longest usable slice once instead of rebuilding history 144x/day.
        cache_key = release, 289
        if cache_key not in forecast_cache:
            forecast_cache.clear()
            forecast_cache[cache_key] = build_information_forecast(data, issue_day, minute, 289)
        forecast = forecast_cache[cache_key]
        load = forecast.load_energy[offset:offset + length].copy()
        point = forecast.pv_energy[offset:offset + length].copy()
        deterministic = config.deterministic or (config.january_warmup and now.month == 1)
        k = 1 if deterministic else config.n_scenarios
        historical = residuals.reveal_before(now)
        if deterministic:
            pv = point[None, :]
        else:
            issue_index = int(np.searchsorted(np.asarray(data.dates, dtype=object), now.date()))
            pv = build_pv_scenarios(point, historical, issue_day_index=issue_index,
                                   start_step=first % 144, horizon_steps=length, n_scenarios=k,
                                   lookback_days=config.lookback_days, seed=config.seed,
                                   operating_dates=data.dates, issue_time=now)
        current_observed = targets[0] == now
        if current_observed:
            load[0] = data.load_energy.flat[first]
            pv[:, 0] = data.pv_energy.flat[first]
        else:
            # Initial baseline begins at 00:10, which is future at 00:00.
            # The current-action equality uses a common point, not its actual.
            pv[:, 0] = point[0]
        previous = np.zeros(length)
        fixed = np.full(length, np.nan)
        baseline_mask = np.ones(length, dtype=bool)
        for step, source_index in enumerate(range(first, first + length)):
            day = data.dates[source_index // 144]
            ledger = ledgers.get(day)
            if ledger is not None and ledger.versions:
                previous[step] = ledger.commitment[source_index % 144]
                baseline_mask[step] = False
                if day != mutable_day or kind != "revision":
                    fixed[step] = previous[step]
        problem = MPCProblem(load, pv, data.price.ravel()[first:first + length], soc,
                             previous_commitment=previous, fixed_commitment=fixed,
                             baseline_mask=baseline_mask, **config.model_parameters())
        solution = solve_mpc(problem)
        entry = dict(solution.diagnostics, timestamp=now.isoformat(), kind=kind,
                     backend=config.backend,
                     scenario_count=k, horizon_steps=length, status=solution.status,
                     optimal=solution.optimal, has_solution=solution.has_solution,
                     forecast_issue=release.isoformat(), observed_current=current_observed,
                     initial_soc=float(soc), virtual_baseline_count=int(baseline_mask.sum()))
        solve_log.append(entry)
        if not solution.has_solution:
            raise SimulationSolveError(f"{kind} solve at {now.isoformat()}: {solution.status}; {solution.diagnostics}")
        if config.backend == "event-policy" and kind in {"baseline", "revision"}:
            reference = problem.scenario_probabilities @ solution.discharge
            for target, value in zip(targets, reference):
                reference_by_time[target] = float(max(0.0, value))
        if kind == "execution":
            disagreement = max(float(np.ptp(getattr(solution, name)[:, 0]))
                               for name in ("charge", "discharge", "emergency", "spill", "mode"))
            entry["current_action_disagreement"] = disagreement
            if disagreement > 1e-5:
                raise SimulationSolveError(f"current actions disagree at {now}")
        return solution

    def issue_baseline(day: date, now: datetime) -> None:
        row = data.date_to_index[day]
        row_first = row * 144
        # At a later midnight include the old row's fixed current interval in
        # planning, so the new baseline does not skip its impact on battery SOC.
        first = time_index.get(now, row_first)
        if first < source_rows[0] * 144:
            first = row_first
        length = max(config.horizon_steps, row_first + 144 - first)
        solution = solve_at(now, first, length, "baseline", day)
        ledger = CommitmentLedger(all_times[row], data.price[row])
        left = row_first - first
        versions.append(ledger.revise(now, solution.commitment[left:left + 144], kind="baseline"))
        ledgers[day] = ledger

    issue_baseline(start, datetime.combine(start, time()))
    count = 0
    for result_row, source_row in enumerate(source_rows):
        for column, now in enumerate(all_times[source_row]):
            if config.max_steps is not None and count >= config.max_steps:
                break
            if now.time() == time() and now.date() in dates and now.date() not in ledgers:
                issue_baseline(now.date(), now)
            first = source_row * 144 + column
            operating_day = data.dates[source_row]
            if now.minute == 0 and now.hour in config.revision_hours and now.date() == operating_day:
                length = max(config.horizon_steps, 144 - column)
                solution = solve_at(now, first, length, "revision", operating_day)
                revised = ledgers[operating_day].commitment.copy()
                revised[column:] = solution.commitment[:144 - column]
                versions.append(ledgers[operating_day].revise(now, revised))
            ix = result_row, column
            if config.backend == "rolling-milp":
                solution = solve_at(now, first, config.horizon_steps, "execution")
                for name in ("charge", "discharge", "emergency", "spill", "mode"):
                    observed[name][ix] = getattr(solution, name)[0, 0]
                observed["discharge_reference"][ix] = observed["discharge"][ix]
                observed["soc_before"][ix] = soc
                soc = _clip_soc_roundoff(
                    soc + config.charge_efficiency * observed["charge"][ix]
                    - observed["discharge"][ix] / config.discharge_efficiency,
                    config.soc_min, config.soc_max)
                if abs(soc - solution.soc[0, 1]) > 1e-5:
                    raise SimulationSolveError(f"executed SOC disagrees with MILP at {now}")
            else:
                reference = reference_by_time.get(now, 0.0)
                action = dispatch_step(
                    load_energy=data.load_energy[source_row, column],
                    pv_energy=data.pv_energy[source_row, column],
                    commitment=ledgers[operating_day].commitment[column],
                    soc=soc,
                    discharge_reference=reference,
                    limits=limits,
                )
                for name in ("charge", "discharge", "emergency", "spill", "mode"):
                    observed[name][ix] = getattr(action, name)
                observed["discharge_reference"][ix] = reference
                observed["soc_before"][ix] = action.soc_before
                soc = action.soc_after
            observed["soc_after"][ix] = soc
            observed["load_energy"][ix] = data.load_energy[source_row, column]
            observed["pv_energy"][ix] = data.pv_energy[source_row, column]
            executed[ix] = True
            ledgers[operating_day].mark_executed(now)
            count += 1
        if config.max_steps is not None and count >= config.max_steps:
            break
        if progress is not None:
            progress(dict(day=str(data.dates[source_row]), executed_steps=count,
                          solve_count=len(solve_log), soc=float(soc),
                          elapsed_seconds=perf_counter() - started))
    baseline, final = np.full(shape, np.nan), np.full(shape, np.nan)
    costs = dict(baseline=0.0, revision_up=0.0, revision_down=0.0, emergency=0.0)
    for row, day in enumerate(dates):
        if day in ledgers:
            baseline[row], final[row] = ledgers[day].baseline, ledgers[day].commitment
            for name, value in ledgers[day].costs.items():
                costs[name] += value
    prices = data.price[source_rows].copy()
    costs["emergency"] = float(np.sum(5 * prices[executed] * observed["emergency"][executed]))
    return SimulationResult(dates, all_times[source_rows].copy(), executed, baseline, final,
                            price=prices, versions=tuple(versions), costs=costs, solve_log=solve_log,
                            config=config, elapsed_seconds=perf_counter() - started, **observed)
