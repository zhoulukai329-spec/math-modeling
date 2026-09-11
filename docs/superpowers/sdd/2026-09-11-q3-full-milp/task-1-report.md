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

## Review-fix round 1

The review correctly identified that a source-row cutoff was not sufficient:
the final `00:00+1` cell of an earlier operating row has the following
calendar day's timestamp.  The implementation now constructs a `(D, 144)`
real-calendar datetime matrix and applies strict `< issue_time` filtering to
each individual load-history and residual-scenario cell.

Added regression coverage for:

- a midnight issue excluding the prior operating row's `00:00+1` load value;
- PV residual paths excluding that same target-midnight residual, with PV
  scenario values clipped at zero;
- unique `np.arange(144)` markers at 18:00, 23:50, next-day 00:00 and 00:10;
- temporary XLSX readers preserving the source order/tail column and rejecting
  price-date mismatch; and
- strict positive-integer checks for scenario count, lookback and horizon.

Red command/output before the implementation fix:

```powershell
python -m pytest Q3/tests/test_data_forecast.py -q
```

```text
5 failed, 9 passed, 2 errors in 1.02s
```

The meaningful failures were the midnight load forecast (`5001.0` observed
instead of `3.0`), target-midnight PV residual leakage (`999.05` observed),
and missing positive-size validation.  The two temporary-workbook fixture
errors were repaired by locating test scratch work below `Q3/tests` rather
than an inaccessible system pytest temporary directory.

Final verification:

```powershell
python -m pytest Q3/tests/test_data_forecast.py -q
```

```text
................                                                         [100%]
16 passed in 0.50s
```

## Review-fix round 2

Added exact temporary-workbook tests for the Attachment 3 reader and complete
`load_inputs()` orchestration.  They verify four releases per day in source
order (00:00/06:00/12:00/18:00), exactly 24 horizon columns per release,
forecast-date consistency with actual-data dates, and preservation of the
source first/tail columns through the public loader.

The PV reader now rejects incomplete/out-of-order release groups, duplicate or
non-increasing dates, and any row that is not exactly 24 hourly values.  The
public loader rejects forecast dates that differ from the actual-data calendar.

`build_pv_scenarios()` now requires `operating_dates` and `issue_time` to be
provided together (or both omitted), preventing a caller from combining a
calendar-aware cutoff with an implicit synthetic row calendar.

Red command/output before these changes:

```powershell
python -m pytest Q3/tests/test_data_forecast.py -q
```

```text
3 failed, 18 passed in 0.93s
```

Final verification:

```powershell
python -m pytest Q3/tests/test_data_forecast.py -q
```

```text
.....................                                                    [100%]
21 passed in 0.70s
```

Real-input read-only smoke result: 365 operating days, 1,460 published PV
releases, three `(365, 144)` data matrices, and unmodified template labels
`0:10-0:20` through `0:00-0:10+1`.
