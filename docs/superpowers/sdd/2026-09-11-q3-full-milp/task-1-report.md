# Q3 Task 1 — timestamp-safe data and forecasts

## Scope completed

- Created `Q3/src/data_io.py` with an `InputData` contract and exact read-only
  readers for Attachments 2, 3, 4 and the `result3.xlsx` template.
- Kept all 144 operating-day columns in their supplied/template order:
  `0:10` through the following day's `0:00`.  No `roll`, rotation, or
  reindexing occurs in the input layer.
- Converted actual load/PV power from kW to kWh exactly once at the input
  boundary.  The published hourly PV forecast is interpolated in kW and then
  converted exactly once in the forecast layer; optimisation receives kWh.
- Created `Q3/src/forecast.py`: explicit release-plus-hours timestamp mapping,
  strictly pre-issue-day causal load history, and residual PV scenarios whose
  candidate paths finish before the issue day.
- Added `Q3/tests/test_data_forecast.py` for left-endpoint/template handling,
  trailing totals, 18:00 plus 24-hour rollover, no day wrapping, causal
  cutoffs and PV-scenario residual cutoff.

## TDD evidence

Red command:

```powershell
python -m pytest Q3/tests/test_data_forecast.py -q
```

Initial output (before production modules existed):

```text
ERROR collecting Q3/tests/test_data_forecast.py
ModuleNotFoundError: No module named 'data_io'
1 error in 0.35s
```

Additional reader edge cases were introduced test-first.  For example, the
non-padded Attachment 3 date test failed with `ValueError: Invalid isoformat
string: '2025-1-1'`, and the template interval-label test failed before its
left-endpoint parser was implemented.

Green/final command:

```powershell
python -m pytest Q3/tests/test_data_forecast.py -q
```

Final output:

```text
........                                                                 [100%]
8 passed in 0.38s
```

Read-only real-input check also succeeded:

```text
365 (365, 144) (365, 144) (365, 144) 1460
2025-01-01 2025-12-31 588.2882666666667 616.1166666666666
0:10-0:20 0:00-0:10+1
```

## Commit

- `014ef17b70f0647949d692448a4c92c042f0702f` — `feat: add Q3 timestamp-safe forecast inputs`

## Risks and handoff notes

- The causal load forecast intentionally uses only completed operating days;
  Task 3 may add an explicit observed-prefix nowcast only if it is passed with
  an as-of timestamp and cannot expose future target-day values.
- PV interpolation holds the first/last published hourly value outside the
  1-to-24-hour knots rather than wrapping to another day.  If a later MPC
  horizon needs more than one published forecast window, it should select a
  later *available* release instead of reusing any actual data.
- End-of-dataset forecast horizons must remain truncated by the simulator and
  retain terminal-SOC treatment, per the ledger ruling; Task 1 does not invent
  any 2026 data.
