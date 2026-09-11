# Task 4 — CLI, template-preserving outputs and independent verification

Status: DONE. Commit: the commit containing this report, titled
`feat: add Q3 CLI persisted outputs and independent verification`.
No subagents were spawned; work stayed on `codex/q3-full-milp` in `D:\-`.

Created `Q3/src/run_problem3.py`, `make_results.py`, `verify_problem3.py`,
`Q3/tests/test_result_writer.py`, `Q3/README.md`. Produced and independently
verified a real 37-step smoke NPZ, metrics JSON, trajectory CSV, official
result3 workbook and verification JSON in `Q3/output`.

## TDD and checks

Tests were written before production modules. First red:

```text
python -m pytest Q3/tests/test_result_writer.py::test_output_modules_exist -q
1 failed in 1.04s
AssertionError: Task 4 make_results is missing
```

The first complete focused run had 13 passes and four setup errors because
the existing system `pytest-of-27893` temporary directory denied access.
A fresh directory under the repository's `.pytest_cache` resolved that
environment issue without changing or deleting the existing system directory.
The next focused run found an actual verifier issue: openpyxl changes a missing
default style to an equivalent all-zero style on a populated cell. Comparing
normalized style tuples fixed it; no template appearance was changed.

Two additional regressions were observed red before their fixes:

- Supplying `--calibration` to smoke caused an unbound local after running;
  now rejected by argument parsing before any simulation or output writes.
- Emergency energy with binary charge mode 1 escaped a zero-charge-only
  exclusivity check; the verifier now checks the binary mode itself too.

Final complete suite:

```text
python -m pytest Q3/tests -q --basetemp D:/-/.pytest_cache/task4-final
87 passed in 17.65s
git diff --check
```

Diff check had no whitespace errors. Tests cover all four battery template
anchors; source-time versus calendar-time blocks; baseline versus final
commitments; unknown unexecuted values; emergency actual intervals and fixed
template rows; NPZ roundtrip/CSV; and independent rejection of tampered
physics, SOC, mode, time, costs, versions, frozen past, forecast release,
execution log, non-anticipative current action, and workbook cells.
The TDD skill references an unavailable `writing-good-tests.md`; its complete
available main instructions were used. The Excel tool skill guided read-only
template preservation. No spreadsheet formulas need recalculation here.

## Real CLI smoke evidence

```powershell
python Q3/src/run_problem3.py --mode smoke --date-start 2025-02-01 --date-end 2025-02-01 --horizon-steps 6 --max-steps 37 --scenarios 2 --time-limit 10
python Q3/src/verify_problem3.py Q3/output/smoke_solution.npz --workbook Q3/output/result3.xlsx --report Q3/output/smoke_verification.json
```

Actual persisted result:

| Metric | Value |
|---|---:|
| Executed ten-minute intervals | 37 / 144 |
| MILP solves / optimal solves | 39 / 39 |
| Simulator elapsed seconds | 3.6389112000033492 |
| Baseline cash cost | 36112.27435560076 |
| Upward revision cash cost | 4906.290447738141 |
| Downward revision cash cost | 0 |
| Actual emergency cash cost | 0 |
| Total cash cost | 41018.5648033389 |
| Final actual SOC, kWh | 6001.241185185185 |
| Independent verifier | PASS, no errors |

Versions are Feb 1 00:00 baseline and 06:00 revision. The saved result remains
`complete: false`. Costs include complete issued commitments, not just the
energy consumed in 37 executed intervals; these are not annual results.
The saved verifier's balance, SOC dynamics and SOC continuity residuals are
zero at recorded floating-point precision.

## Output contract

`write_result3(result, output_path, template_path=DEFAULT_TEMPLATE)` loads the
official workbook and saves a distinct destination. Sheet order, dimensions,
date anchors, headers, ellipses, cell styles and unused slots are retained.
The original attachment is not overwritten.

- Plan sheet B:EO is 0:00 baseline; EP total quantity, EQ baseline cash cost.
- Adjusted sheet B:EO is latest cumulative applicable commitment; EP its
  total quantity, EQ the sum of actual adjacent-version up/down fees.
- Battery sheet has only its four official date anchors. Four-hour quantities
  use true calendar left endpoints. A complete 24-interval block is necessary
  to fill its aggregate. Initial-date 0:00 SOC comes from explicit config;
  24:00 SOC is before the action whose left endpoint is midnight, not after it.
- Emergency sheet fills only Feb 1/Feb 2/Dec 31 template anchors. A multiline
  cell lists observed positive ten-minute intervals, and the adjacent numeric
  cell gives their sum. Other reserved rows and the ellipsis are preserved;
  no rows or arbitrary dates are appended. A partial day's positive output is
  explicitly marked partial; a partial day with no positive value stays blank.

The source business row runs 00:10 through next-day 00:00. A standalone Feb 1
run does not execute Feb 1 00:00, so its literal 00:00-04:00 battery block cannot
honestly be populated. This is documented, rather than rotated or zero-filled.
All actual ten-minute trajectories remain available in NPZ/CSV regardless of
the template's limited illustrative date anchors.

`save_result(result, directory, prefix='smoke')` returns Paths keyed by
`solution`, `metrics`, `trajectory`, `workbook`. NPZ has schema-versioned JSON
metadata, separate non-object arrays and every commitment version. It can be
loaded with `load_result(path)` using `allow_pickle=False`; `config.data` is
omitted. Unexecuted physical arrays remain NaN, not zeros. CSV includes every
selected interval with blank unknown fields. Each run overwrites its named
artifacts in that output directory; use separate output directories to retain
separate experiments.

## Verifier contract and limits

`verify_solution(result, workbook_path=None, template_path=None, tolerance=1e-5)`
returns `passed`, `errors`, `residuals`, reconstructed costs, execution count,
completion marker and explicit causality scope. It does not solve a MILP or
call the writer to derive expected workbook cells. It recomputes physical
constraints and settlement and reconstructs template-cell expectations
independently, including unknown cells and preserved styles.

Saved audit fields verify publication clocks, current-observation flags,
unknown future actuals, mandatory update clocks, chronological version freezes,
and one common MILP action per actual execution. These summaries cannot prove
all internal scenario/forecast arrays causal; the existing simulator's
future-actual perturbation tests provide separate behavioral evidence. Logs
claiming optimality are retained, not treated as a standalone optimality proof.

## Task 6 handoff / deferred work

CLI exposes smoke/full/experiments. Full requires a frozen calibration JSON:
`frozen: true`, `training_start: '2025-01-01'`, `training_end: '2025-01-31'`,
`parameters: {terminal_penalty, terminal_soc, cvar_weight, optional cvar_alpha}`.
Those calibrated fields cannot be overridden in full mode. The default full
evaluation is Feb 1–Dec 31 at H=144/K=12, with explicit initial SOC, default
6000 kWh; January is not silently simulated. CLI validates this protocol but
cannot prove a user-created JSON is the product of actual calibration.

Task 6 owns generating that calibration and may extend provenance/initial-SOC
handoff. Experiments currently dispatches `experiments.run_experiments(args)`
if that module exists; otherwise it exits nonzero with an explicit missing
Task 6 error. This is a deliberate unfinished dependency, not fake success.
P1 review, formal experiments, figures and full-year execution were not run.
