"""Sparse two-stage scenario MPC-MILP, with interval energies already in kWh.

Procurement is common for the whole horizon. Current operation is common,
while future operation is scenario recourse. CVaR concerns procurement plus
emergency monetary cost; terminal and throughput penalties are regularizers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class MPCProblem:
    """One planning/execution horizon.

    ``load_energy``/``price`` are (T,), ``pv_energy`` is (K,T) or (T,).
    All scenarios must have the same current PV observation. The caller owns
    information cutoffs. No input here is converted from kW.

    ``fixed_commitment`` is (T,), with NaN for free entries and a nonnegative
    value for frozen entries. ``previous_commitment=None`` purchases a baseline;
    otherwise only adjacent-version upward/downward changes are charged.
    With a previous vector, optional boolean ``baseline_mask`` marks virtual
    uncommitted cells charged at 1x price instead of revision deltas. These
    mixed-horizon quantities have zero returned revision_up/revision_down.
    Battery limits are interval kWh and SOC is stored kWh, not a fraction.
    """
    load_energy: np.ndarray
    pv_energy: np.ndarray
    price: np.ndarray
    initial_soc: float
    previous_commitment: np.ndarray | None = None
    fixed_commitment: np.ndarray | None = None
    scenario_probabilities: np.ndarray | None = None
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    charge_limit: float = 5000.0 / 6.0
    discharge_limit: float = 5000.0 / 6.0
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    terminal_soc: float = 6000.0
    terminal_penalty: float = 0.1
    throughput_penalty: float = 1e-6
    cvar_weight: float = 0.0
    cvar_alpha: float = 0.9
    time_limit: float = 30.0
    mip_rel_gap: float = 1e-4
    baseline_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        load = np.array(self.load_energy, dtype=float, copy=True)
        if load.ndim != 1 or load.size == 0:
            raise ValueError("load_energy must be a nonempty vector")
        horizon = load.size
        pv = np.array(self.pv_energy, dtype=float, copy=True)
        if pv.ndim == 1:
            pv = pv[None, :]
        if pv.ndim != 2 or pv.shape[0] == 0 or pv.shape[1] != horizon:
            raise ValueError("pv_energy must have shape (K,T) or (T,)")
        price = np.array(self.price, dtype=float, copy=True)
        if price.shape != (horizon,):
            raise ValueError("price must have shape (T,)")
        for name, value in (("load_energy", load), ("pv_energy", pv), ("price", price)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if not np.allclose(pv[:, 0], pv[0, 0], atol=1e-9, rtol=0):
            raise ValueError("current pv_energy must agree across scenarios; supply the common observation")
        probabilities = (np.full(pv.shape[0], 1.0 / pv.shape[0])
                         if self.scenario_probabilities is None
                         else np.array(self.scenario_probabilities, dtype=float, copy=True))
        if (probabilities.shape != (pv.shape[0],) or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-10, rtol=0)):
            raise ValueError("scenario_probabilities must be positive and sum to one")
        object.__setattr__(self, "scenario_probabilities", probabilities)
        for name in ("previous_commitment", "fixed_commitment"):
            original = getattr(self, name)
            if original is None:
                continue
            value = np.array(original, dtype=float, copy=True)
            valid = np.isfinite(value) | (np.isnan(value) if name == "fixed_commitment" else False)
            if value.shape != (horizon,) or not valid.all() or np.any(value < 0):
                raise ValueError(f"{name} must be a nonnegative (T,) vector; only fixed_commitment allows NaN")
            object.__setattr__(self, name, value)
        if self.baseline_mask is not None:
            mask = np.asarray(self.baseline_mask)
            if mask.shape != (horizon,) or mask.dtype.kind != "b":
                raise ValueError("baseline_mask must be a boolean (T,) vector")
            if self.previous_commitment is None:
                raise ValueError("baseline_mask requires previous_commitment")
            object.__setattr__(self, "baseline_mask", mask.copy())
        for name in ("initial_soc", "soc_min", "soc_max", "terminal_soc", "charge_limit",
                     "discharge_limit", "terminal_penalty", "throughput_penalty", "cvar_weight", "mip_rel_gap"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.soc_max < self.soc_min:
            raise ValueError("soc_max must be at least soc_min")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must lie within SOC bounds")
        if not self.soc_min <= self.terminal_soc <= self.soc_max:
            raise ValueError("terminal_soc must lie within SOC bounds")
        for name in ("charge_efficiency", "discharge_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be in (0,1)")
        if not np.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("time_limit must be finite and positive")


@dataclass
class MPCSolution:
    """Decoded feasible incumbent, or explicit missing arrays on solver failure.

    A limit-stopped solve can have ``has_solution=True, optimal=False``. Such
    an incumbent is returned only after its binary modes are rounded and the
    full rounded vector passes bounds, rows and integrality at absolute 1e-5.
    Monetary costs, CVaR and objective are reconstructed from returned decisions;
    diagnostics preserve the raw solver objective, status and gap separately.
    """
    has_solution: bool
    optimal: bool
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    commitment: np.ndarray | None = None
    revision_up: np.ndarray | None = None
    revision_down: np.ndarray | None = None
    charge: np.ndarray | None = None
    discharge: np.ndarray | None = None
    emergency: np.ndarray | None = None
    spill: np.ndarray | None = None
    mode: np.ndarray | None = None
    soc: np.ndarray | None = None
    scenario_cost: np.ndarray | None = None
    procurement_cost: float | None = None
    expected_cost: float | None = None
    terminal_cost: float | None = None
    throughput_cost: float | None = None
    objective: float | None = None
    cvar: float | None = None
    cvar_eta: float | None = None
    cvar_excess: np.ndarray | None = None


def solve_mpc(problem: MPCProblem) -> MPCSolution:
    """Build and solve one SciPy/HiGHS MILP; no execution or ledger mutation."""
    started = perf_counter()
    p = problem
    k, t = p.pv_energy.shape
    probability = p.scenario_probabilities
    indices: dict[str, np.ndarray] = {}
    size = 0

    def allocate(name: str, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal size
        count = int(np.prod(shape))
        result = np.arange(size, size + count).reshape(shape)
        indices[name] = result
        size += count
        return result

    q = allocate("commitment", (t,))
    c = allocate("charge", (k, t))
    d = allocate("discharge", (k, t))
    e = allocate("emergency", (k, t))
    w = allocate("spill", (k, t))
    z = allocate("mode", (k, t))
    soc = allocate("soc", (k, t + 1))
    deviation = allocate("terminal_deviation", (k,))
    revision = p.previous_commitment is not None
    # Mixed horizons: previously committed cells use adjacent-version deltas;
    # virtual next-day purchases use the ordinary 1x baseline price.
    baseline_mask = (np.ones(t, dtype=bool) if not revision else
                     np.zeros(t, dtype=bool) if p.baseline_mask is None else p.baseline_mask)
    if revision:
        up = allocate("revision_up", (t,))
        down = allocate("revision_down", (t,))
    risk = p.cvar_weight > 0
    if risk:
        eta = allocate("eta", (1,))[0]
        excess = allocate("excess", (k,))

    lb = np.zeros(size)
    ub = np.full(size, np.inf)
    objective = np.zeros(size)
    procurement_objective = np.zeros(size)
    integrality = np.zeros(size, dtype=np.uint8)
    net = p.load_energy[None, :] - p.pv_energy
    # More procurement than maximum net demand plus charging capacity is only
    # useful when retaining an already purchased commitment; include that case.
    q_upper = np.maximum(0.0, net.max(axis=0) + p.charge_limit)
    if revision:
        q_upper = np.maximum(q_upper, p.previous_commitment)
    fixed = np.full(t, np.nan) if p.fixed_commitment is None else p.fixed_commitment
    frozen = np.isfinite(fixed)
    q_upper[frozen] = np.maximum(q_upper[frozen], fixed[frozen])
    ub[q] = q_upper
    lb[q[frozen]] = fixed[frozen]
    ub[q[frozen]] = fixed[frozen]
    # Emergency cannot charge. With g,d >= 0 it need never exceed the remaining
    # nonnegative net demand after guaranteed procurement. No arbitrary big-M.
    emergency_upper = np.maximum(0.0, net - np.where(frozen, fixed, 0.0)[None, :])
    ub[c], ub[d], ub[e], ub[z] = p.charge_limit, p.discharge_limit, emergency_upper, 1.0
    ub[w] = q_upper[None, :] + p.pv_energy + p.discharge_limit + emergency_upper
    lb[soc], ub[soc] = p.soc_min, p.soc_max
    lb[soc[:, 0]], ub[soc[:, 0]] = p.initial_soc, p.initial_soc
    integrality[z] = 1
    if revision:
        ub[up], ub[down] = q_upper, p.previous_commitment
        ub[up[baseline_mask]], ub[down[baseline_mask]] = 0.0, 0.0
        procurement_objective[up] = 1.5 * p.price * ~baseline_mask
        procurement_objective[down] = 0.5 * p.price * ~baseline_mask
    procurement_objective[q[baseline_mask]] = p.price[baseline_mask]
    objective += procurement_objective
    objective[e] = probability[:, None] * 5.0 * p.price
    objective[c] = probability[:, None] * p.throughput_penalty
    objective[d] = probability[:, None] * p.throughput_penalty
    objective[deviation] = probability * p.terminal_penalty
    if risk:
        objective[eta] = p.cvar_weight
        objective[excess] = p.cvar_weight * probability / (1.0 - p.cvar_alpha)

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    row_lb: list[float] = []
    row_ub: list[float] = []

    def row(terms: list[tuple[int, float]], lower: float, upper: float) -> None:
        index = len(row_lb)
        for column, value in terms:
            if value != 0:
                rows.append(index)
                columns.append(int(column))
                coefficients.append(float(value))
        row_lb.append(float(lower))
        row_ub.append(float(upper))

    for scenario in range(k):
        for step in range(t):
            ix = scenario, step
            row([(q[step], 1), (e[ix], 1), (d[ix], 1), (c[ix], -1), (w[ix], -1)],
                net[ix], net[ix])
            row([(soc[scenario, step + 1], 1), (soc[ix], -1),
                 (c[ix], -p.charge_efficiency), (d[ix], 1.0 / p.discharge_efficiency)], 0, 0)
            row([(c[ix], 1), (z[ix], -p.charge_limit)], -np.inf, 0)
            row([(d[ix], 1), (z[ix], p.discharge_limit)], -np.inf, p.discharge_limit)
            row([(e[ix], 1), (z[ix], emergency_upper[ix])], -np.inf, emergency_upper[ix])
        row([(soc[scenario, -1], 1), (deviation[scenario], -1)], -np.inf, p.terminal_soc)
        row([(soc[scenario, -1], -1), (deviation[scenario], -1)], -np.inf, -p.terminal_soc)
    for scenario in range(1, k):
        for variable in (c, d, e, w, z):
            row([(variable[scenario, 0], 1), (variable[0, 0], -1)], 0, 0)
    if revision:
        for step in range(t):
            if baseline_mask[step]:
                continue
            row([(q[step], 1), (up[step], -1), (down[step], 1)],
                p.previous_commitment[step], p.previous_commitment[step])
    if risk:
        procurement_terms = [(index, procurement_objective[index])
                             for index in np.flatnonzero(procurement_objective)]
        for scenario in range(k):
            row(procurement_terms + [(e[scenario, step], 5.0 * p.price[step]) for step in range(t)]
                + [(eta, -1), (excess[scenario], -1)], -np.inf, 0)

    matrix = coo_matrix((coefficients, (rows, columns)), shape=(len(row_lb), size)).tocsc()
    constraint_lower, constraint_upper = np.asarray(row_lb), np.asarray(row_ub)
    # HiGHS' default 1e-6 integer tolerance can admit 8e-4 kWh through
    # the 833 kWh mode bounds. Tighten it instead of weakening the audit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*", category=RuntimeWarning)
        result = milp(c=objective, integrality=integrality, bounds=Bounds(lb, ub),
                      constraints=LinearConstraint(matrix, constraint_lower, constraint_upper),
                      options={"time_limit": p.time_limit, "mip_rel_gap": p.mip_rel_gap,
                               "presolve": True, "mip_feasibility_tolerance": 1e-9})
    solver_status = int(result.status)
    message = str(result.message)
    status = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}.get(
        solver_status, "solver_error")
    diagnostics: dict[str, Any] = {
        "solver_status": solver_status, "message": message,
        "timed_out": solver_status == 1 and "time" in message.lower(),
        "limit_reached": solver_status == 1, "elapsed_seconds": perf_counter() - started,
        "n_variables": size, "n_binary": int(integrality.sum()), "n_constraints": len(row_lb),
        "matrix_nnz": int(matrix.nnz), "matrix_format": matrix.format,
        "emergency_upper_max": float(emergency_upper.max()),
        "mip_gap": getattr(result, "mip_gap", None),
        "mip_dual_bound": getattr(result, "mip_dual_bound", None),
        "mip_node_count": getattr(result, "mip_node_count", None),
    }
    x = getattr(result, "x", None)
    has_solution = solver_status in (0, 1) and x is not None
    if has_solution:
        x = np.array(x, dtype=float, copy=True)
        has_solution = x.shape == (size,) and bool(np.isfinite(x).all())
    if has_solution:
        diagnostics["solver_objective"] = float(objective @ x)
        raw_integer_violation = float(np.max(np.abs(x[z] - np.rint(x[z]))))
        x[z] = np.rint(x[z])
        # A near-integer mode can permit material c/d/e through a large bound.
        # Validate the actual rounded mode with all unchanged continuous actions.
        activity = matrix @ x
        bound_violation = max(0.0, float(np.max(lb - x)), float(np.max(x - ub)))
        constraint_violation = max(0.0, float(np.max(constraint_lower - activity)),
                                   float(np.max(activity - constraint_upper)))
        integer_violation = float(np.max(np.abs(x[z] - np.rint(x[z]))))
        diagnostics.update(max_bound_violation=bound_violation,
                           max_constraint_violation=constraint_violation,
                           max_integrality_violation=integer_violation,
                           raw_max_integrality_violation=raw_integer_violation)
        has_solution = max(bound_violation, constraint_violation,
                           integer_violation, raw_integer_violation) <= 1e-5
    diagnostics["incumbent_accepted"] = bool(has_solution)
    solution = MPCSolution(bool(has_solution), bool(has_solution and solver_status == 0), status, diagnostics)
    if not has_solution:
        return solution
    for name in ("commitment", "charge", "discharge", "emergency", "spill", "soc", "mode"):
        setattr(solution, name, x[indices[name]].copy())
    solution.mode = solution.mode.astype(int)
    solution.revision_up = np.maximum(solution.commitment - p.previous_commitment, 0) if revision else np.zeros(t)
    solution.revision_down = np.maximum(p.previous_commitment - solution.commitment, 0) if revision else np.zeros(t)
    solution.revision_up[baseline_mask] = 0.0
    solution.revision_down[baseline_mask] = 0.0
    solution.procurement_cost = float(p.price @ (1.5 * solution.revision_up + 0.5 * solution.revision_down
                                                + baseline_mask * solution.commitment))
    solution.scenario_cost = solution.procurement_cost + np.sum(solution.emergency * (5.0 * p.price), axis=1)
    solution.expected_cost = float(probability @ solution.scenario_cost)
    solution.terminal_cost = float(p.terminal_penalty * (probability @ np.abs(solution.soc[:, -1] - p.terminal_soc)))
    solution.throughput_cost = float(p.throughput_penalty * np.sum(probability[:, None] * (solution.charge + solution.discharge)))
    if risk:
        # Tighten the CVaR representation of the authoritative economic losses.
        # A time-limited incumbent's auxiliary up/down, eta and excess variables
        # need not be tight, even when its returned physical decision is feasible.
        order = np.argsort(solution.scenario_cost)
        quantile_index = min(int(np.searchsorted(np.cumsum(probability[order]), p.cvar_alpha)), k - 1)
        solution.cvar_eta = float(solution.scenario_cost[order[quantile_index]])
        solution.cvar_excess = np.maximum(solution.scenario_cost - solution.cvar_eta, 0.0)
        solution.cvar = float(solution.cvar_eta + (probability @ solution.cvar_excess) / (1.0 - p.cvar_alpha))
    solution.objective = (solution.expected_cost + solution.terminal_cost + solution.throughput_cost
                          + (p.cvar_weight * solution.cvar if risk else 0.0))
    return solution
