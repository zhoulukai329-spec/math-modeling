"""Independent output audit: recompute physics and settlement without solving."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


def verify_solution(result, *, workbook_path=None, template_path=None, tolerance=1e-5):
    errors = []
    residuals = {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    def near(actual, expected, label):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        valid = a.shape == b.shape and np.allclose(a, b, atol=tolerance, rtol=1e-10, equal_nan=True)
        require(valid, label)
        if a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all():
            residuals[label] = float(np.max(np.abs(a - b)))

    shape = (len(result.dates), 144)
    for name in ("timestamps", "executed", "baseline", "final_commitment", "load_energy", "pv_energy",
                 "price", "charge", "discharge", "emergency", "spill", "mode", "soc_before", "soc_after"):
        require(np.shape(getattr(result, name)) == shape, f"{name} shape")
    if errors:
        return dict(passed=False, errors=errors, residuals=residuals)
    require(result.executed.dtype.kind == "b", "executed must be boolean")
    if errors:
        return dict(passed=False, errors=errors, residuals=residuals)
    mask = result.executed
    require(bool(mask.any()), "no executed intervals")
    expected_times = np.asarray([[datetime.combine(day, time()) + timedelta(minutes=10 * (j + 1))
                                  for j in range(144)] for day in result.dates], dtype=object)
    require(np.array_equal(result.timestamps, expected_times), "timestamp source order or midnight boundary")
    require(all(b - a == timedelta(days=1) for a, b in zip(result.dates, result.dates[1:])), "timestamp date chronology")
    require(np.array_equal(mask.ravel(), np.arange(mask.size) < mask.sum()), "execution must be a chronological prefix")
    for name in ("load_energy", "pv_energy", "charge", "discharge", "emergency", "spill", "mode", "soc_before", "soc_after"):
        values = getattr(result, name)
        require(np.isfinite(values[mask]).all(), f"executed {name} nonfinite")
        require(np.isnan(values[~mask]).all(), f"unexecuted {name} must be unknown")
    require(np.isfinite(result.price).all() and np.all(result.price >= 0), "price finite/nonnegative")
    for name in ("load_energy", "pv_energy", "charge", "discharge", "emergency", "spill"):
        require(np.all(getattr(result, name)[mask] >= -tolerance), f"negative {name}")
    c, d, e, spill = [getattr(result, name)[mask] for name in ("charge", "discharge", "emergency", "spill")]
    before, after, mode = result.soc_before[mask], result.soc_after[mask], result.mode[mask]
    cfg = result.config
    near(result.final_commitment[mask] + result.pv_energy[mask] + d + e,
         result.load_energy[mask] + c + spill, "energy balance")
    near(after, before + cfg.charge_efficiency * c - d / cfg.discharge_efficiency, "SOC dynamics")
    if mask.any():
        near(before[:1], [cfg.initial_soc], "SOC initial")
        near(before[1:], after[:-1], "SOC continuity")
    require(np.all(before >= cfg.soc_min - tolerance) and np.all(after >= cfg.soc_min - tolerance)
            and np.all(before <= cfg.soc_max + tolerance) and np.all(after <= cfg.soc_max + tolerance), "SOC bounds")
    require(np.all(c <= cfg.charge_limit + tolerance) and np.all(d <= cfg.discharge_limit + tolerance), "power bounds")
    require(np.all(np.abs(mode - np.rint(mode)) <= tolerance) and np.all((mode >= -tolerance) & (mode <= 1 + tolerance)), "binary mode")
    require(np.all(c <= cfg.charge_limit * mode + tolerance), "charge mode")
    require(np.all(d <= cfg.discharge_limit * (1 - mode) + tolerance), "discharge mode")
    require(not np.any((c > tolerance) & (e > tolerance)), "emergency/charge mode exclusion")
    require(not np.any((mode > .5) & (e > tolerance)), "emergency binary mode")

    date_rows = {day: i for i, day in enumerate(result.dates)}
    ledger = {}
    computed_costs = dict(baseline=0.0, revision_up=0.0, revision_down=0.0, emergency=float(5 * result.price[mask] @ e))
    issuance = []
    last_issue = None
    for version in result.versions:
        target = tuple(version.target_times)
        if not target or target[0].date() not in date_rows:
            require(False, "version unknown target day")
            continue
        i = date_rows[target[0].date()]
        require(target == tuple(expected_times[i]), "version target timestamps")
        if any(np.shape(getattr(version, key)) != (144,) for key in ("commitment", "revision_up", "revision_down")):
            require(False, "version vector shape")
            continue
        current = version.commitment
        require(np.isfinite(current).all() and np.all(current >= -tolerance), "version commitment nonnegative/finite")
        require(last_issue is None or version.issued_at >= last_issue, "version issue chronology")
        last_issue = version.issued_at
        issuance.append((version.issued_at.isoformat(), version.kind))
        midnight = datetime.combine(result.dates[i], time())
        if version.kind == "baseline":
            require(i not in ledger, "version duplicate baseline")
            require(version.issued_at == midnight, "version baseline must be 0:00")
            near(current, result.baseline[i], "version baseline mismatch")
            up = down = np.zeros(144)
            base_fee = float(result.price[i] @ current)
        elif version.kind == "revision" and i in ledger:
            require(version.issued_at.date() == result.dates[i] and version.issued_at.hour in cfg.revision_hours
                    and version.issued_at.minute == 0 and version.issued_at.second == 0, "version revision clock")
            previous = ledger[i]
            frozen = expected_times[i] < version.issued_at
            near(current[frozen], previous[frozen], "version frozen past")
            up, down = np.maximum(current - previous, 0), np.maximum(previous - current, 0)
            base_fee = 0.0
        else:
            require(False, "version missing baseline or invalid kind")
            continue
        near(version.revision_up, up, "version upward delta")
        near(version.revision_down, down, "version downward delta")
        up_fee, down_fee = float(1.5 * result.price[i] @ up), float(.5 * result.price[i] @ down)
        for actual, expected, label in ((version.baseline_cost, base_fee, "baseline"), (version.up_cost, up_fee, "up"), (version.down_cost, down_fee, "down")):
            near(actual, expected, f"version {label} cost")
        computed_costs["baseline"] += base_fee
        computed_costs["revision_up"] += up_fee
        computed_costs["revision_down"] += down_fee
        ledger[i] = current
    for i in range(shape[0]):
        if i in ledger:
            near(result.final_commitment[i], ledger[i], "version final cumulative commitment")
        else:
            require(np.isnan(result.baseline[i]).all() and np.isnan(result.final_commitment[i]).all() and not mask[i].any(), "unissued version row")
    require(set(result.costs) == set(computed_costs), "four cost components")
    for name, expected in computed_costs.items():
        near(result.costs.get(name, np.nan), expected, f"cash cost {name}")

    execution_times, issue_logs = [], []
    last_log_time = None
    for entry in result.solve_log:
        try:
            now, forecast = datetime.fromisoformat(entry["timestamp"]), datetime.fromisoformat(entry["forecast_issue"])
            require(last_log_time is None or now >= last_log_time, "solve timestamp chronology")
            last_log_time = now
            latest = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
            require(forecast <= now and forecast == latest, "forecast publication clock")
            require(entry["has_solution"] is True and entry["n_binary"] > 0, "MILP incumbent/size audit")
            expected_k = 1 if cfg.deterministic or (cfg.january_warmup and now.month == 1) else cfg.n_scenarios
            require(entry["scenario_count"] == expected_k, "scenario count")
            require(entry["kind"] in ("baseline", "revision", "execution"), "unknown solve kind")
            if entry["kind"] == "execution":
                execution_times.append(now)
                require(entry["observed_current"] is True, "execution current observation")
                require(entry["current_action_disagreement"] <= tolerance, "scenario common current action")
                index = len(execution_times) - 1
                if index < len(before):
                    near(entry["initial_soc"], before[index], "execution initial SOC")
            else:
                issue_logs.append((entry["timestamp"], entry["kind"]))
                if now == datetime.combine(result.dates[0], time()) and entry["kind"] == "baseline":
                    require(entry["observed_current"] is False, "future first baseline observation")
        except (KeyError, ValueError, TypeError) as exc:
            require(False, f"solve audit fields: {exc}")
    require(execution_times == list(expected_times[mask]), "execution log must cover every executed timestamp")
    require(issue_logs == issuance, "version issuance log mismatch")
    if mask.any():
        mandatory = [(datetime.combine(result.dates[0], time()).isoformat(), "baseline")]
        for now in expected_times[mask]:
            if now.time() == time() and now.date() in date_rows:
                mandatory.append((now.isoformat(), "baseline"))
            elif now.minute == 0 and now.hour in cfg.revision_hours:
                mandatory.append((now.isoformat(), "revision"))
        require(issuance == mandatory, "version mandatory update clocks")

    if workbook_path is not None:
        _verify_workbook(result, workbook_path, template_path, require, tolerance)
    return dict(passed=not errors, errors=errors, residuals=residuals, recomputed_costs=computed_costs,
                executed_steps=int(mask.sum()), complete=bool(mask.all()),
                causality_scope="Publication clocks, observation flags, one common action per execution, unknown future actuals. "
                "Saved summaries cannot prove all forecast/scenario inputs; future-actual perturbation is tested separately.")


def _verify_workbook(result, path, template_path, require, tolerance):
    """Reconstruct expected cells independently of the writer, including blanks."""
    if template_path is None:
        template_path = Path(__file__).resolve().parents[2] / "attachment/附件5/result3.xlsx"
    original, workbook = load_workbook(template_path), load_workbook(path)
    require(workbook.sheetnames == original.sheetnames, "workbook sheets")
    if workbook.sheetnames != original.sheetnames:
        return
    expected = {}
    def put(name, r, c, value):
        expected[name, r, c] = value
    by_day = {day: i for i, day in enumerate(result.dates)}
    for name, array in (("计划购电量", result.baseline), ("调整购电量", result.final_commitment)):
        for row in range(2, original[name].max_row + 1):
            day = original[name].cell(row, 1).value.date()
            if day not in by_day:
                continue
            i = by_day[day]
            for j in range(144):
                put(name, row, j + 2, float(array[i, j]) if np.isfinite(array[i, j]) else None)
            if np.isfinite(array[i]).all():
                put(name, row, 146, float(array[i].sum()))
                fees = float(result.price[i] @ result.baseline[i]) if name == "计划购电量" else sum(
                    v.up_cost + v.down_cost for v in result.versions if v.target_times[0].date() == day)
                put(name, row, 147, fees)
    stamps = list(result.timestamps[result.executed])
    data = {name: getattr(result, name)[result.executed] for name in ("charge", "discharge", "soc_before", "soc_after")}
    stamp_index = {stamp: i for i, stamp in enumerate(stamps)}
    sheet = original["充放电量"]
    for r in range(2, sheet.max_row + 1):
        day = sheet.cell(r, 1).value
        if not isinstance(day, datetime):
            continue
        for block in range(6):
            lo, hi = day + timedelta(hours=block * 4), day + timedelta(hours=(block + 1) * 4)
            positions = [i for i, stamp in enumerate(stamps) if lo <= stamp < hi]
            if len(positions) == 24:
                put(sheet.title, r + block, 3, float(data["charge"][positions].sum()))
                put(sheet.title, r + block, 4, float(data["discharge"][positions].sum()))
        for offset in (0, 1):
            boundary = day + timedelta(days=offset)
            if boundary in stamp_index:
                put(sheet.title, r + offset, 6, float(data["soc_before"][stamp_index[boundary]]))
            elif boundary - timedelta(minutes=10) in stamp_index:
                put(sheet.title, r + offset, 6, float(data["soc_after"][stamp_index[boundary - timedelta(minutes=10)]]))
            elif offset == 0 and day.date() == result.dates[0]:
                put(sheet.title, r, 6, result.config.initial_soc)
    sheet = original["紧急购电量"]
    for r in range(2, sheet.max_row + 1):
        day = sheet.cell(r, 1).value
        if not isinstance(day, datetime) or day.date() not in by_day:
            continue
        i = by_day[day.date()]
        positive = np.flatnonzero(result.executed[i] & (result.emergency[i] > 1e-7))
        if len(positive) or result.executed[i].all():
            labels = [original["计划购电量"].cell(1, int(j) + 2).value for j in positive]
            text = "\n".join(labels) if labels else "无"
            if not result.executed[i].all():
                text = "已执行部分：\n" + text
            put(sheet.title, r, 2, text)
            put(sheet.title, r, 3, float(result.emergency[i, result.executed[i]].sum()))
    mismatches = []
    for source in original:
        target = workbook[source.title]
        require((source.max_row, source.max_column) == (target.max_row, target.max_column), f"workbook dimensions {source.title}")
        require(list(source.merged_cells.ranges) == list(target.merged_cells.ranges), f"workbook merges {source.title}")
        for row in source:
            for cell in row:
                value = expected.get((source.title, cell.row, cell.column), cell.value)
                observed = target.cell(cell.row, cell.column)
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                valid = (isinstance(observed.value, (int, float)) and abs(value - observed.value) <= tolerance) if numeric else observed.value == value
                source_style = tuple(cell._style) if cell._style is not None else (0,) * 9
                target_style = tuple(observed._style) if observed._style is not None else (0,) * 9
                if not valid or source_style != target_style:
                    mismatches.append(f"{source.title}!{cell.coordinate}")
    require(not mismatches, "workbook cell mismatch: " + ", ".join(mismatches[:12]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    from make_results import load_result
    report = verify_solution(load_result(args.solution), workbook_path=args.workbook, template_path=args.template)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
