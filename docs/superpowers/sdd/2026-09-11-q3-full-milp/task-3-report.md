# Task 3 — commitment versions and causal rolling execution

Status: DONE. Implementation commit: `79c99bbc98f4b4fed0b625a1e09f708a2947f1dc`
(`feat: implement causal rolling Q3 MILP simulation and settlement`).

Created `Q3/src/simulation.py` and `Q3/tests/test_simulation.py`. Extended
`Q3/src/model.py` and its tests minimally for approved mixed settlement horizons.
Work remained on `codex/q3-full-milp` in `D:\-`; no subagents were spawned.

## TDD and verification

The test module was created before the simulator. Initial focused red:

```text
python -m pytest Q3/tests/test_simulation.py::test_simulation_interface_exists -q
1 failed in 0.46s
AssertionError: Task 3 rolling simulator is missing
```

The full pre-implementation simulator suite returned `1 failed, 12 errors`;
fixture errors were exclusively the absent simulation module. The mixed-price
model regression separately failed before production changes with unexpected
constructor keyword `baseline_mask`: `1 failed, 27 passed in 0.84s`.

The first simulator implementation passed 12 tests; one fixture assertion
incorrectly assumed a revealed midnight historical cell. Correcting this
expectation to the Task 1 timestamp contract yielded all tests passing. The
baseline is 286, not 288, in that one-prior-row synthetic fixture because the
previous row's tail is still unrevealed at issue midnight.

Final verification, after adding a real-optimizer observer test with differing
future scenarios and the model interface documentation:

```text
python -m pytest Q3/tests -q
63 passed in 2.00s
git diff --check
git diff --cached --check
```

Both diff checks had no whitespace errors. Git emitted only its configured
LF-to-CRLF conversion notices. The installed TDD skill's referenced
`writing-good-tests.md` is absent; its available main instructions were followed.

## Public interface for Task 4

`SimulationConfig` accepts `data: InputData | None`, `attachment_dir`,
`horizon_steps` (1..144, default 144), `max_steps` (positive or None),
`n_scenarios` (default 12), `lookback_days` (default 28), `seed`,
`deterministic`, `january_warmup`, and configurable `revision_hours`
(default `(6,12,18)`). All battery, terminal, CVaR and solver settings are
forwarded explicitly to `MPCProblem`. Baseline remains mandatory at midnight.

`simulate(config, date_start, date_end)` takes inclusive operating-row dates.
It loads actual attachments if `config.data` is omitted. Inputs must have
consecutive dates; invalid ranges and nonpositive counts fail explicitly.

`SimulationResult` has:

- `dates`: selected operating dates, in original order.
- `(days,144)` arrays: `timestamps`, `executed`, `baseline`,
  `final_commitment`, `load_energy`, `pv_energy`, `price`, `charge`,
  `discharge`, `emergency`, `spill`, `mode`, `soc_before`, `soc_after`.
- `versions`: chronological immutable `CommitmentVersion` snapshots, including
  `issued_at`, `kind`, `target_times`, `commitment`, `revision_up`,
  `revision_down`, `baseline_cost`, `up_cost`, `down_cost`.
- `costs`: exactly `baseline`, `revision_up`, `revision_down`, `emergency`.
- `total_cost`, `solve_log`, `config`, `elapsed_seconds`.

Unexecuted actuals/actions/SOC/mode are NaN and `executed=False`. An issued
baseline/final commitment contains the complete 144-cell row even in a
three-step smoke run. Unissued later rows remain NaN. Persist this distinction;
do not write planned scenario actions or replace unknown actuals with zeros.
When serializing configuration, omit the optional in-memory `config.data`
object; all decisions and audit arrays have their own result fields.

`solve_log` records every baseline, revision and execution solve, including
timestamp, forecast issue, horizon, number of scenarios, observed-current flag,
initial SOC, virtual-baseline count, MILP diagnostics, incumbent/optimal status,
and execution current-action disagreement. It does not persist all future
scenario decision tensors, which would be large in annual execution.

## Information clock and physical execution

Every ten-minute execution calls the complete SciPy/HiGHS `solve_mpc`.
Already issued procurement is fixed during execution; future unissued rows
are virtual baseline purchases for planning. Only the current scenario-common
charge, discharge, emergency, spill and mode are executed. SOC is advanced
from the real prior SOC and those actions, then checked against the solver's
next SOC. No daily reset, scenario averaging, or dispatch-rule fallback exists.
No feasible incumbent raises `SimulationSolveError` with the failing time and
diagnostics. Accepted time-limited feasible incumbents retain their status.

At every 0:00, an entire new business-row baseline is optimized and issued.
That row spans 00:10 through the following 00:00 left endpoint. At later
midnights, the planning model includes the old row's fixed 00:00 current cell
before all 144 new-row cells, accounting for its SOC effect. The new baseline
never overwrites that old tail. Its current observation may enter planning,
but its actual execution still has a separate complete execution solve.
For the initial selected row, the first baseline starts at future 00:10; its
first PV is the common published point forecast, never the future actual.

At 6/12/18, the current row's remaining commitments are revised against their
immediately preceding version. Executed or earlier-time ledger cells cannot
change; attempted mutations fail atomically. Ledger snapshots are read-only.
Issuance optimization covers at least the affected row's remainder, even when
the smoke execution horizon is shorter. End-of-dataset horizons are truncated.

Forecasts use the most recently published release at or before the decision
clock. Historical load information uses Task 1's strict timestamp cutoff.
Only the current target's load/PV actual is revealed for a current-step solve.
Future actuals never enter decision inputs. A test changes all future actual
load/PV values by thousands and confirms identical actions and cash costs.

Residual cells are populated only after their real left endpoint is strictly
past. Each residual is actual PV minus the latest forecast published at that
historical target. Scenario sampling uses Task 1's chronological paths, prior
operating dates, real timestamp cutoff and 28-day window. January uses K=1
deterministically by default; February onward uses the configured K. Starting
a run in February uses preceding attachment history but takes `initial_soc`
as the supplied initial condition; it does not silently simulate January.

The residual library represents errors of a rolling latest-release predictor,
not separate error libraries calibrated for each lead/release. No final-period
actuals tune it. Calendar warm-up policy is implemented; January parameter
calibration itself remains the later experiment/CLI task.

## Settlement and minimal model extension

`MPCProblem.baseline_mask` is an optional boolean `(T,)` vector, requiring a
previous-commitment vector. True entries use normal 1x purchase price, with
zero returned revision quantities; false entries use 1.5x upward and .5x
downward adjacent-version costs. CVaR includes those same monetary losses.
The default behavior and existing positional parameters remain compatible.

This lets a horizon carry real committed cells and virtual next-row purchases
without charging the latter at the 1.5x adjustment rate. Only the current row's
issuance slice enters its ledger. Virtual procurement, scenario emergency
expectations, terminal penalties, throughput regularizers and risk penalties
are never accumulated as real cash settlement.

Cash costs are full issued baseline plus every actual adjacent-version upward
and downward fee plus `5 * price * executed emergency`. In a partial smoke run,
baseline is still the whole issued day's cost; do not compare it to a complete
year or describe it as the cost of only the three executed intervals.

## Real-attachment runtime evidence

These calls loaded `attachment/附件2.xlsx`, `附件3.xlsx`, `附件4.xlsx` and
the existing result template through `load_inputs`. No output workbook, annual
artifact or formal experiment was generated in Task 3.

Shared settings: `horizon_steps=6`, `n_scenarios=2`, `time_limit=10`; all other
values were explicit `SimulationConfig` defaults (not calibrated values).

| Selected rows | Executed steps | MILP solves | Internal elapsed seconds |
|---|---:|---:|---:|
| 2025-02-01 | 3 | 4 | 1.017129 |
| 2025-02-01 | 37 | 39 | 3.615922 |
| 2025-02-01 through 2025-02-02 | 146 | 151 | 11.532999 |

The smallest run returned four optimal statuses and final SOC 6000 kWh.
Its cash costs were baseline 36112.27435560076; both revision costs and actual
emergency cost were zero. The 37-step run included the 06:00 revision and had
maximum actual balance residual `1.1368683772161603e-13` kWh.

The 146-step run included versions at Feb 1 00/06/12/18 and Feb 2 00.
Cash costs: baseline 82482.87320547359; upward adjustment 9578.818979257354;
downward adjustment and emergency zero. Consecutive real SOC discrepancy was
exactly zero; minimum recorded SOC 4575.777155442056 and final SOC
5529.62705962067 kWh. These are smoke diagnostics, not full-year model claims.

Reproduce the smallest run from repository root:

```powershell
python -c "import sys; sys.path.insert(0,'Q3/src'); from simulation import SimulationConfig,simulate; r=simulate(SimulationConfig(horizon_steps=6,max_steps=3,n_scenarios=2,time_limit=10),'2025-02-01','2025-02-01'); print(r.elapsed_seconds,r.costs,len(r.solve_log))"
```

No known unresolved Task 3 correctness blocker. Independent verification/P1,
persisted CLI smoke artifacts, workbook output, calibrated full-year execution
and experiment figures remain the planned later tasks.
