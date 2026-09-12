"""Preserve the official template; persist actual trajectories and full ledgers."""
from __future__ import annotations
import io
import csv
from dataclasses import fields, replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from data_io import ATTACHMENT_DIR, coerce_date
from simulation import CommitmentVersion, SimulationConfig, SimulationResult

ARRAY_FIELDS = ("executed", "baseline", "final_commitment", "load_energy", "pv_energy",
                "pv_forecast", "price", "charge", "discharge", "emergency", "spill", "mode",
                "charge_reference", "discharge_reference", "soc_before", "soc_after")
DEFAULT_TEMPLATE = ATTACHMENT_DIR / "附件5" / "result3.xlsx"


def trim_result(result: SimulationResult, date_start) -> SimulationResult:
    """Remove warm-up rows and recompute the exact reported cash scope."""
    start = coerce_date(date_start)
    keep = np.asarray([day >= start for day in result.dates], dtype=bool)
    if not keep.any():
        raise ValueError("trim start is after the result period")
    dates = tuple(day for day, selected in zip(result.dates, keep) if selected)
    arrays = {name: getattr(result, name)[keep].copy() for name in ARRAY_FIELDS}
    timestamps = result.timestamps[keep].copy()
    versions = tuple(v for v in result.versions if v.target_times[0].date() in set(dates))
    mask = arrays["executed"]
    if not mask.any():
        raise ValueError("trimmed result has no executed intervals")
    midnight = datetime.combine(dates[0], time())
    boundary = result.calendar_boundary
    if boundary and boundary["timestamp"] != midnight.isoformat():
        boundary = None
    positions = np.argwhere((result.timestamps == midnight) & result.executed)
    if len(positions):
        i, j = positions[0]
        boundary = dict(timestamp=midnight.isoformat(), **{
            name: float(getattr(result, name)[i, j])
            for name in ("charge", "discharge", "soc_before", "soc_after")})
    costs = {
        "baseline": float(sum(v.baseline_cost for v in versions)),
        "revision_up": float(sum(v.up_cost for v in versions)),
        "revision_down": float(sum(v.down_cost for v in versions)),
        "emergency": float(np.sum(5 * arrays["price"][mask] * arrays["emergency"][mask])),
    }
    actual_times = set(timestamps[mask])
    issue_keys = {(v.issued_at, v.kind) for v in versions}
    solve_log = []
    for entry in result.solve_log:
        now = datetime.fromisoformat(entry["timestamp"])
        if ((entry["kind"] == "execution" and now in actual_times)
                or (entry["kind"] != "execution" and (now, entry["kind"]) in issue_keys)):
            solve_log.append(dict(entry))
    for entry in solve_log:
        if (entry["kind"] == "baseline"
                and datetime.fromisoformat(entry["timestamp"]) == datetime.combine(dates[0], time())):
            entry["trimmed_warmup_boundary"] = True
    initial_soc = float(arrays["soc_before"][mask][0])
    return replace(result, dates=dates, timestamps=timestamps, versions=versions,
                   costs=costs, solve_log=solve_log, calendar_boundary=boundary,
                   config=replace(result.config, initial_soc=initial_soc), **arrays)


def _json_value(value):
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _configuration(config):
    return {f.name: getattr(config, f.name) for f in fields(config) if f.name != "data"}


def write_result3(result: SimulationResult, output_path: str | Path,
                  template_path: str | Path = DEFAULT_TEMPLATE) -> Path:
    from dispatch_core.result_workbook import write_workbook
    return write_workbook(result, output_path, template_path)


def save_result(result: SimulationResult, output_dir: str | Path, *, prefix: str = "smoke",
                template_path: str | Path = DEFAULT_TEMPLATE) -> dict[str, Path]:
    """NPZ contains no pickle objects; JSON metadata is embedded for reconstruction."""
    from dispatch_core.workbook_contract import preflight_template
    from dispatch_core.calendar_boundary import boundary_errors
    preflight_template(template_path, Path(output_dir) / "result3.xlsx")
    errors = boundary_errors(result)
    if errors:
        raise ValueError("; ".join(errors))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(solution=output_dir / f"{prefix}_solution.npz", metrics=output_dir / f"{prefix}_metrics.json",
                 trajectory=output_dir / f"{prefix}_trajectory.csv", workbook=output_dir / "result3.xlsx")
    metadata = dict(schema_version=1, dates=[str(d) for d in result.dates], costs=result.costs,
                    config=_configuration(result.config), solve_log=result.solve_log,
                    elapsed_seconds=result.elapsed_seconds, calendar_boundary=result.calendar_boundary,
                    versions=[dict(issued_at=v.issued_at.isoformat(), kind=v.kind,
                                   target_times=[t.isoformat() for t in v.target_times],
                                   baseline_cost=v.baseline_cost, up_cost=v.up_cost, down_cost=v.down_cost)
                              for v in result.versions])
    arrays = {name: getattr(result, name) for name in ARRAY_FIELDS}
    arrays["timestamps"] = np.asarray([[t.isoformat() for t in row] for row in result.timestamps])
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, default=_json_value, allow_nan=False))
    for key in ("commitment", "revision_up", "revision_down"):
        arrays[f"version_{key}"] = np.asarray([getattr(v, key) for v in result.versions], dtype=float).reshape(-1, 144)
    np.savez_compressed(paths["solution"], **arrays)
    metrics = dict(schema_version=1, dates=metadata["dates"], executed_steps=int(result.executed.sum()),
                   selected_steps=int(result.executed.size), complete=bool(result.executed.all()),
                   costs=result.costs, total_cost=result.total_cost, elapsed_seconds=result.elapsed_seconds,
                   solve_count=len(result.solve_log), optimal_solve_count=sum(bool(v["optimal"]) for v in result.solve_log),
                   config=metadata["config"], final_soc=float(result.soc_after[result.executed][-1]),
                   cost_scope="Full issued commitments and adjacent-version fees; emergency only for executed steps.",
                   calibration="Explicit configuration; no calibration inferred from this result.")
    paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_value, allow_nan=False) + "\n", encoding="utf-8")
    with paths["trajectory"].open("w", newline="", encoding="utf-8-sig") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["operating_date", "timestamp", *ARRAY_FIELDS])
        for i, day in enumerate(result.dates):
            for j in range(144):
                values = [getattr(result, name)[i, j] for name in ARRAY_FIELDS]
                csv_writer.writerow([str(day), result.timestamps[i, j].isoformat(),
                                     *["" if isinstance(v, (float, np.floating)) and np.isnan(v) else v for v in values]])
    write_result3(result, paths["workbook"], template_path)
    return paths


def load_result(path: str | Path) -> SimulationResult:
    path = Path(path)
    # Some file-transfer tools prepend a UTF-8 BOM to binary files.
    # Strip only that known wrapper; pickle loading remains disabled.
    raw = path.read_bytes()
    source = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    with np.load(io.BytesIO(source), allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        if metadata["schema_version"] != 1:
            raise ValueError("unsupported solution schema")
        versions = []
        for i, record in enumerate(metadata["versions"]):
            record["issued_at"] = datetime.fromisoformat(record["issued_at"])
            record["target_times"] = tuple(datetime.fromisoformat(t) for t in record["target_times"])
            versions.append(CommitmentVersion(**record, **{name: saved[f"version_{name}"][i].copy()
                for name in ("commitment", "revision_up", "revision_down")}))
        config = metadata["config"]
        config["revision_hours"] = tuple(config["revision_hours"])
        restored = {name: saved[name].copy() for name in ARRAY_FIELDS if name in saved.files}
        if "discharge_reference" not in restored:
            restored["discharge_reference"] = restored["discharge"].copy()
        if "charge_reference" not in restored:
            restored["charge_reference"] = restored["charge"].copy()
        if "pv_forecast" not in restored:
            restored["pv_forecast"] = restored["pv_energy"].copy()
        return SimulationResult(dates=tuple(date.fromisoformat(d) for d in metadata["dates"]),
            timestamps=np.asarray([[datetime.fromisoformat(t) for t in row] for row in saved["timestamps"]], dtype=object),
            versions=tuple(versions), config=SimulationConfig(**config), costs=metadata["costs"],
            solve_log=metadata["solve_log"], elapsed_seconds=metadata["elapsed_seconds"],
            calendar_boundary=metadata.get("calendar_boundary"),
            **restored)
