# Task 2 — sparse deterministic and scenario MILP

## Outcome and commit

Implemented `Q3/src/model.py` and `Q3/tests/test_model.py` on
`codex/q3-full-milp` in `D:\-`.

Implementation commit: `89c85faf15be1ed1f02a09d6288dbe8bde214a27`
(`feat: implement sparse Q3 scenario MPC MILP`). This report is committed
separately so it can reference the immutable implementation commit.

## TDD evidence

Tests were written before the production module. The first focused red command
was `python -m pytest Q3/tests/test_model.py::test_model_interface_exists -q`:

```text
AssertionError: Task 2 MILP module is missing
1 failed in 0.21s
```

The full pre-implementation command `python -m pytest Q3/tests/test_model.py -q`
then reported `1 failed, 22 errors in 0.35s`; the 22 fixture errors were all
`ModuleNotFoundError: No module named 'model'`, not optimization failures.

After implementation, the same focused suite returned:

```text
23 passed in 0.66s
```

Final integration regression, `python -m pytest Q3/tests -q`:

```text
44 passed in 1.15s
```

`git diff --check` and staged diff checks returned no errors. The installed TDD
skill refers to `writing-good-tests.md`, but that supporting file is absent
from its skill directory; its available `SKILL.md` was read and followed.

## Public interface and units

`solve_mpc(problem: MPCProblem) -> MPCSolution` performs one sparse SciPy/HiGHS
MILP solve, with no simulation, settlement, timestamp handling or side effects.

Required constructor inputs:

- `load_energy`: nonnegative finite `(T,)` kWh.
- `pv_energy`: nonnegative finite `(K,T)` kWh, or `(T,)` for deterministic mode.
- `price`: nonnegative finite `(T,)` monetary units/kWh.
- `initial_soc`: battery stored kWh.

Constructor checks include positive scenario probabilities summing to one,
compatible vector shapes, finite/nonnegative parameters, battery bounds and
efficiencies, positive time limit, and `0 < cvar_alpha < 1`.

Optional procurement inputs:

- `previous_commitment=None` means baseline cost `sum(price * commitment)`.
- A `(T,)` previous commitment means cost `sum(price * (1.5*up + 0.5*down))`,
  relative to that immediate preceding version, including fixed entries.
- `fixed_commitment` is `(T,)`: finite nonnegative entries are frozen values,
  and NaN entries remain free. Omission leaves the whole vector free. Passing
  a complete finite vector freezes all procurement during ten-minute execution.

Battery defaults: SOC `[1200,10800]` kWh; charge/discharge limits `5000/6` kWh
per interval; efficiencies `.9`; terminal target `6000` kWh; terminal penalty
`.1`; throughput regularizer `1e-6`. These are already interval energies: the
model does not multiply any input or battery limit by the time step.

Risk/solver defaults: `cvar_weight=0`, `cvar_alpha=.9`, `time_limit=30` seconds,
`mip_rel_gap=1e-4`. Terminal and risk parameters are caller-configurable for
the later January calibration task.

Solution arrays are `commitment/revision_up/revision_down[T]`,
`charge/discharge/emergency/spill/mode[K,T]`, `soc[K,T+1]`, and
`scenario_cost[K]`. Cost scalars are `procurement_cost`, `expected_cost`,
`terminal_cost`, `throughput_cost`, `objective`, and optional
`cvar/cvar_eta/cvar_excess[K]`. On failure without an accepted incumbent these
arrays and cost scalars are `None`.

## Variables, constraints and economics

The balance equation is `g + pv + emergency + discharge = load + charge + spill`.
SOC follows `E_next = E + eta_charge*charge - discharge/eta_discharge`.
Binary `z=1` permits charge and forces discharge/emergency to zero; `z=0`
permits discharge and emergency together and forces charge to zero.

The common procurement vector is shared throughout the horizon. Five explicit
equalities per additional scenario share current charge, discharge, emergency,
spill and binary mode. The constructor rejects differing current PV values
instead of averaging them. Simulation must set the current PV observation
identically in every scenario; load is already common. All later operational
variables are scenario recourse. This is a two-stage approximation, not a full
multistage scenario tree.

Emergency bounds are `max(load - pv - guaranteed_fixed_procurement, 0)` per
scenario and interval. Since emergency cannot charge and procurement/discharge
are nonnegative, this bound is sufficient to serve every possible deficit.
Procurement upper bounds use maximum scenario net demand plus charge capacity,
expanded for previous/fixed commitments. Spill permits disposal of surplus
PV or contracted energy; no export revenue is introduced.

Terminal SOC uses a weighted absolute-deviation soft penalty. Optional CVaR
uses one eta and K excess variables, with loss equal to procurement plus
emergency monetary cost. Terminal/throughput regularizers are excluded from
economic CVaR. The objective is expected monetary cost plus expected terminal
and throughput penalties, plus `cvar_weight * CVaR_alpha(loss)`.

For K scenarios and T periods, with R=1 for revision mode and V=1 for CVaR:

- Variables: `T + 6*K*T + 2*K + 2*R*T + V*(K+1)`.
- Binary variables: `K*T`.
- Linear rows: `5*K*T + 2*K + 5*(K-1) + R*T + V*K`.
- Initial SOC and fixed commitments use bounds, not extra rows.
- Construction uses COO triplets converted to CSC; no dense constraint matrix.

A synthetic K=12, T=144, baseline, CVaR-weight=.2 check returned:

```text
status=optimal, has_solution=True
n_variables=10549, n_binary=1728, n_constraints=8731
matrix_nnz=29558, matrix_format=csc
elapsed_seconds=0.2522375, mip_gap=0, mip_node_count=1
max_bound_violation=1.2733e-11
max_constraint_violation=1.2619e-11
max_integrality_violation=0
```

This test used identical zero-PV scenarios and constant load/price. Its purpose
was sparse dimensional validation, not a realistic runtime benchmark.

## Solver status and verification coverage

`has_solution` indicates a finite incumbent that passed row, bound and integer
checks at absolute tolerance `1e-5`, with solver status 0 or 1.
`optimal` additionally requires status 0 (optimal to configured solver gap).
Status 1 may return an executable incumbent, but never claims optimality.
Other statuses and missing/rejected incumbents return no execution arrays.

Diagnostics retain numeric solver status, message, limit/timeout flags, gap,
dual bound, node count, elapsed seconds, sparse dimensions and incumbent
feasibility residuals. No heuristic fallback or silent success is substituted.

Tests cover: no second energy conversion; baseline cost; energy balance;
efficiency-aware SOC; both mode exclusions; permitted discharge plus emergency;
non-anticipativity with a case where separate first actions would be cheaper;
future recourse; weighted expectations; soft terminal deviations on both sides;
frozen/free masks and adjacent-version costs; weighted CVaR tail value and its
effect on procurement; surplus fixed procurement; malformed input rejection;
and time-limited solver results both without and with a real feasible incumbent.
Only the nondeterministic solver deadline/status boundary is substituted in
tests; physical/economic solves use the real HiGHS backend.

## Concerns and handoff

No unresolved correctness concerns within Task 2. Operational handoff:

- `expected_cost` is a horizon planning cost. It must not be accumulated on
  every ten-minute solve; the simulator settles procurement revisions once and
  records only actual first-action emergency energy.
- Future scenario recourse is the approved two-stage approximation; it can be
  optimistic versus a full scenario tree. Only current common actions execute.
- Caller must preserve causality, provide common current observations, truncate
  end-of-data horizons and maintain true SOC. The model cannot infer timestamps.
- Real-data smoke optimization, calibration, independent P1 review and full-year
  runtime remain later tasks; this report makes no claims about their completion.

## Review-fix round 1

Addressed both Important review findings. These changes and the regression
tests are committed together with this report update; use `git log -1 --
Q3/src/model.py` for the immutable fix commit.

### Rounded incumbent feasibility

The decoder now copies the finite solver vector, records its original objective
and integrality residual, rounds binary mode entries, and checks every bound,
linear row and binary value on that complete rounded vector. Continuous actions
are not silently adjusted. Any violation above absolute `1e-5` rejects the
incumbent and returns `has_solution=False` with no execution arrays. A candidate
also must have been within the original integrality tolerance before rounding.

New regression cases use solver candidates that satisfy every original linear
row and are only `5e-6` away from integral modes. One has `0.004` kWh charge with
a mode rounding to zero; the other has `0.004` kWh each of discharge/emergency
with a mode rounding to one. Both must be rejected because the rounded vector
violates the mode constraints by `0.004`, despite the original vector's small
integrality residual. The diagnostic row violation now describes exactly the
rounded candidate that would otherwise be returned.

### Authoritative monetary costs and CVaR

Revision quantities are reconstructed as the positive/negative parts of returned
commitment minus the immediate previous commitment. Procurement cost is now
computed from those quantities, or directly from baseline commitment and price.
Neither result depends on potentially loose up/down auxiliaries.

Scenario monetary losses and expected cost use the reconstructed procurement
cost plus returned emergency quantities. For risk-enabled problems, the decoder
computes the weighted alpha quantile of these losses, then their nonnegative
excess above that quantile, yielding the actual finite-scenario CVaR. Terminal
and throughput penalties continue to use returned SOC/actions. Public
`objective` is rebuilt from these authoritative expected costs, regularizers
and weighted CVaR.

`diagnostics.solver_objective` retains the original modeled incumbent objective;
`mip_gap` and `mip_dual_bound` remain unmodified solver diagnostics associated
with that original optimization report. They are not presented as recomputed
gaps for the canonicalized monetary objective.

The added economic regression supplies a feasible nonoptimal candidate with
unchanged commitment but redundant `up=down=1`. Without CVaR its original
modeled objective is 2; with CVaR weight 2 and a loose eta it is 6. In both cases
the authoritative revision charge, expected cost and returned objective are
zero; in the latter case returned CVaR/eta/excess are also zero.

### Red/green evidence

Before changing production code:

```text
python -m pytest Q3/tests/test_model.py -q
4 failed, 23 passed in 0.91s
```

The two rounding cases failed because invalid candidates were accepted. The
two auxiliary-slack cases failed because procurement cost was 2 instead of 0.

After the fix:

```text
python -m pytest Q3/tests/test_model.py -q
27 passed in 0.60s
python -m pytest Q3/tests -q
48 passed in 1.16s
```

`git diff --check` passed. No unresolved review concerns remain in this round;
the original scope and later-task runtime/causality handoff limitations apply.
