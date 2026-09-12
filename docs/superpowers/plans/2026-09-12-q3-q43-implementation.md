# Q3 Refactor and Q4-3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a selectable fast/reference Q3 engine, a causal dynamic-price Q4-3 engine, publication figures, and executable 12/18/24-hour representative-day sensitivity results without running a full year.

**Architecture:** Keep the validated Q3 rolling MILP as the reference backend. Add a shared event-policy executor that solves the same MILP only at 0:00/6:00/12:00/18:00 and executes intervening intervals from current observations and the latest discharge reference. Q4-3 owns its dynamic-price forecast, CLI, output, and template writer while reusing Q3 physics and the shared executor.

**Tech Stack:** Python 3, NumPy, SciPy/HiGHS MILP, openpyxl, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-q3-q43-performance-design.md`

## Global Constraints

- Attachment rows remain `00:10 ... next-day 00:00` left endpoints; no circular shift.
- Future actual load, PV, and Q4-3 prices must never enter a decision.
- Emergency energy may coexist with discharge, cannot charge the battery, and costs five times the target interval price.
- Baseline, upward revision, and downward penalty ledgers retain 1.0/1.5/0.5 multipliers.
- SOC is 1200--10800 kWh; only boundary error at or below `1e-5` kWh may be clipped.
- Formal horizon remains 24 hours; sensitivity covers only 12/18/24 hours.
- Do not run annual optimization or overwrite tracked/ignored Q3 outputs.

---

### Task 1: Shared causal execution and exact scenario reduction

**Files:**
- Create: `dispatch_core/__init__.py`
- Create: `dispatch_core/causal_dispatch.py`
- Create: `dispatch_core/scenario.py`
- Create: `tests/test_dispatch_core.py`

**Interfaces:**
- Produces: `dispatch_step(net_demand, commitment, soc, discharge_reference, limits) -> DispatchStep`.
- Produces: `deduplicate_scenarios(values, probabilities=None) -> (unique, probabilities, inverse)`.

- [ ] **Step 1: Write failing tests** for surplus charging, deficit/reference discharge, emergency-with-retained-SOC, `1e-5` clipping, real bound failure, and duplicate probability merging.

```python
step = dispatch_step(net_demand=900, commitment=0, soc=6000,
                     discharge_reference=300, limits=DEFAULT_LIMITS)
assert step.discharge == 300
assert step.emergency == 600
assert step.charge == 0
```

- [ ] **Step 2: Verify RED.**

```powershell
python -m pytest tests/test_dispatch_core.py -q
```

Expected: import failure because `dispatch_core` does not exist.

- [ ] **Step 3: Implement the immutable limits/result dataclasses, energy balance, SOC audit, and stable row deduplication.**

```python
available = commitment - net_demand
if available >= 0:
    charge = min(available, limits.charge_limit, (limits.soc_max - soc) / limits.charge_efficiency)
else:
    discharge = min(-available, discharge_reference, limits.discharge_limit,
                    (soc - limits.soc_min) * limits.discharge_efficiency)
    emergency = -available - discharge
```

- [ ] **Step 4: Verify GREEN and commit.**

```powershell
python -m pytest tests/test_dispatch_core.py -q
git add dispatch_core tests/test_dispatch_core.py
git commit -m "feat: add causal event dispatch core"
```

### Task 2: Selectable Q3 event-policy backend

**Files:**
- Modify: `Q3/src/simulation.py`
- Modify: `Q3/src/run_problem3.py`
- Modify: `Q3/src/make_results.py`
- Modify: `Q3/src/verify_problem3.py`
- Modify: `Q3/tests/test_simulation.py`
- Modify: `Q3/tests/test_result_writer.py`

**Interfaces:**
- `SimulationConfig.backend` is `"rolling-milp"` or `"event-policy"`.
- `SimulationResult` adds `backend`-compatible solve-log fields and saved reference arrays.

- [ ] **Step 1: Add failing tests** proving the event backend calls the optimizer only for baseline/revision events, still issues four commitment versions, never reads a changed future actual before its timestamp, clips `1199.999999`, and remains independently verifiable.

```python
fast = sim.simulate(config(sim, data, backend="event-policy"), "2025-02-01", "2025-02-01")
assert {row["kind"] for row in fast.solve_log} <= {"baseline", "revision"}
assert len(fast.solve_log) == 4
```

- [ ] **Step 2: Verify RED.**

```powershell
python -m pytest Q3/tests/test_simulation.py Q3/tests/test_result_writer.py -q
```

Expected: `SimulationConfig` rejects unknown `backend` and result fields are absent.

- [ ] **Step 3: Implement event reference storage and current-step execution.** Event solves continue to call `solve_mpc`; their probability-weighted discharge trajectories are mapped by literal timestamp. Rolling mode retains its existing per-interval solve path.

```python
if config.backend == "rolling-milp":
    action = solve_at(now, first, config.horizon_steps, "execution")
else:
    action = dispatch_step(actual_net, ledger.commitment[column], soc,
                           reference_by_time.get(now, 0.0), limits)
```

- [ ] **Step 4: Add `--backend`, defaulting smoke/full to `event-policy`, save backend/reference diagnostics, and retain backward-compatible NPZ loading.**

- [ ] **Step 5: Run Q3 unit tests and a six-step smoke test.**

```powershell
python -m pytest Q3/tests tests/test_dispatch_core.py -q
python Q3/src/run_problem3.py --mode smoke --backend event-policy --max-steps 6 --scenarios 1 --horizon-steps 12 --output-dir Q3/test-output
```

- [ ] **Step 6: Commit.**

```powershell
git add Q3/src Q3/tests
git commit -m "refactor: add fast event backend to Q3"
```

### Task 3: Causal Q4-3 dynamic-price inputs and forecasts

**Files:**
- Create: `Q4-3/src/data_io.py`
- Create: `Q4-3/src/price_forecast.py`
- Create: `Q4-3/tests/test_q43_data_forecast.py`
- Create: `Q4-3/requirements.txt`

**Interfaces:**
- Produces `Q43InputData` with Attachment 2 actual load/PV, Attachment 3 PV releases, Attachment 4 actual price, and result4-3 labels.
- Produces `build_price_scenarios(actual_price, target_times, issue_time, n_scenarios, lookback_days, seed)`.

- [ ] **Step 1: Write failing tests** for attachment selection, source order, current-price reveal, future-price perturbation invariance, week-lag/available-history fallback, and paired historical-day indices.

- [ ] **Step 2: Verify RED.**

```powershell
python -m pytest Q4-3/tests/test_q43_data_forecast.py -q
```

- [ ] **Step 3: Implement the exact readers and causal price forecast.**

```python
known = historical_times < issue_time
if target_time == issue_time:
    forecast = actual_price[current_index]
elif target_time - timedelta(days=7) in revealed:
    forecast = revealed[target_time - timedelta(days=7)]
else:
    forecast = mean_of_revealed_same_slot_or_last_known
```

- [ ] **Step 4: Verify GREEN and commit.**

```powershell
python -m pytest Q4-3/tests/test_q43_data_forecast.py -q
git add Q4-3/src Q4-3/tests Q4-3/requirements.txt
git commit -m "feat: add causal Q4-3 price forecasts"
```

### Task 4: Q4-3 MILP, simulation, settlement, and CLI

**Files:**
- Modify: `Q3/src/model.py`
- Modify: `Q3/tests/test_model.py`
- Create: `Q4-3/src/simulation.py`
- Create: `Q4-3/src/run_problem43.py`
- Create: `Q4-3/tests/test_q43_simulation.py`

**Interfaces:**
- `MPCProblem.scenario_price` optionally supplies `(K,T)` causal scenario prices; the existing one-dimensional `price` remains the expected procurement price.
- Q4-3 `simulate(...)` returns the same physical/ledger fields as Q3 plus actual/forecast price and backend diagnostics.

- [ ] **Step 1: Write failing model tests** proving scenario emergency costs use scenario prices while procurement uses their weighted expected price.

- [ ] **Step 2: Verify RED, implement optional scenario prices, and rerun all Q3 model tests.**

```powershell
python -m pytest Q3/tests/test_model.py -q
```

- [ ] **Step 3: Write failing Q4-3 tests** for four event solves, no future-price leakage, actual-price settlement, adjacent revisions, SOC/flow physics, and both selectable backends.

- [ ] **Step 4: Implement Q4-3 simulation and CLI using the shared event executor.**

- [ ] **Step 5: Verify GREEN with unit and six-step smoke tests.**

```powershell
python -m pytest Q3/tests/test_model.py Q4-3/tests/test_q43_simulation.py -q
python Q4-3/src/run_problem43.py --mode smoke --backend event-policy --max-steps 6 --scenarios 1 --horizon-steps 12
```

- [ ] **Step 6: Commit.**

```powershell
git add Q3/src/model.py Q3/tests/test_model.py Q4-3/src Q4-3/tests
git commit -m "feat: implement Q4-3 dynamic-price dispatch"
```

### Task 5: Q4-3 result workbook and independent verification

**Files:**
- Create: `Q4-3/src/make_results.py`
- Create: `Q4-3/src/verify_problem43.py`
- Create: `Q4-3/tests/test_q43_results.py`

**Interfaces:**
- `save_result43(result, output_dir, prefix, template_path)` writes NPZ/JSON/CSV/result4-3.xlsx.
- `verify_solution43(result, workbook_path=None, template_path=None)` recomputes physics and actual-price cash fees without solving.

- [ ] **Step 1: Write failing tests** for literal template ordering, next-day midnight placement, dynamic target-price fees, float clipping, tamper rejection, and no overwriting attachments.

- [ ] **Step 2: Verify RED.**

```powershell
python -m pytest Q4-3/tests/test_q43_results.py -q
```

- [ ] **Step 3: Implement writer and verifier from Q3 semantics with result4-3 paths and dynamic actual prices.**

- [ ] **Step 4: Run Q3/Q4-3 result tests and commit.**

```powershell
python -m pytest Q3/tests/test_result_writer.py Q4-3/tests/test_q43_results.py -q
git add Q4-3/src Q4-3/tests
git commit -m "feat: add verified Q4-3 deliverables"
```

### Task 6: Publication figures and horizon sensitivity

**Files:**
- Create: `dispatch_core/horizon_sensitivity.py`
- Create: `Q3/src/make_figures.py`
- Create: `Q3/src/run_horizon_sensitivity.py`
- Create: `Q4-3/src/make_figures.py`
- Create: `Q4-3/src/run_horizon_sensitivity.py`
- Create: `tests/test_horizon_sensitivity.py`
- Create: `Q3/tests/test_make_figures.py`
- Create: `Q4-3/tests/test_make_figures.py`

**Interfaces:**
- Sensitivity CSV columns are `question,date,horizon_hours,total_cost,emergency_kwh,end_soc_kwh,solve_seconds,max_residual,backend`.
- Figure scripts only read saved NPZ/CSV and never invoke optimization.

- [ ] **Step 1: Write failing tests** for hour-to-step conversion, fixed 12/18/24 set, smoke labelling, required NPZ fields, and PNG/SVG/PDF output.

- [ ] **Step 2: Verify RED.**

```powershell
python -m pytest tests/test_horizon_sensitivity.py Q3/tests/test_make_figures.py Q4-3/tests/test_make_figures.py -q
```

- [ ] **Step 3: Implement seven plot families** using fixed palette `#0072B2/#E69F00/#009E73/#D55E00/#F0E442/#000000`, separate units into panels, and refuse annual labels for smoke data.

- [ ] **Step 4: Implement representative-day sensitivity runners** for four seasonal dates, common SOC/seed/scenarios/tolerances, and event-policy backend.

- [ ] **Step 5: Run the bounded 12/18/24 experiments and generate CSV/JSON/figures.**

```powershell
python Q3/src/run_horizon_sensitivity.py --scenarios 1
python Q4-3/src/run_horizon_sensitivity.py --scenarios 1
```

- [ ] **Step 6: Verify files and commit code plus small sensitivity results, excluding ordinary runtime output.**

```powershell
python -m pytest tests/test_horizon_sensitivity.py Q3/tests/test_make_figures.py Q4-3/tests/test_make_figures.py -q
git add dispatch_core Q3/src Q3/tests Q4-3
git commit -m "feat: add Q3 and Q4-3 analysis figures"
```

### Task 7: Documentation, complete verification, and main push

**Files:**
- Modify: `Q3/README.md`
- Create: `Q4-3/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Documents exact smoke/full/reference/sensitivity/figure commands and warns that representative-day results are not annual totals.

- [ ] **Step 1: Document backend selection, causal information, output schemas, plotting, and no-48/72 decision.**

- [ ] **Step 2: Ignore Q4-3 ordinary full/smoke output while explicitly tracking compact sensitivity CSV/JSON/figure artifacts.**

- [ ] **Step 3: Run syntax and full bounded test suite.**

```powershell
python -m compileall -q dispatch_core Q3/src Q4-3/src
python -m pytest tests Q3/tests Q4-3/tests -q
git diff --check
```

- [ ] **Step 4: Inspect worktree, ensure Q3/output and attachments are untouched, commit documentation, rebase on current origin/main, and push main.**

```powershell
git status --short
git add .gitignore Q3/README.md Q4-3/README.md docs/superpowers/plans/2026-09-12-q3-q43-implementation.md
git commit -m "docs: explain Q3 and Q4-3 execution"
git pull --rebase origin main
git push origin main
```

