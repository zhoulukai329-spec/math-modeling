# -*- coding: utf-8 -*-
"""Q4-2 data paths, physical constants and dynamic-price input."""
import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
REPO_ROOT = Q_DIR.parent
Q2_SRC = REPO_ROOT / "Q2" / "src"
ATTACH_DIR = REPO_ROOT / "attachment"
OUTPUT_DIR = Q_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(Q2_SRC) not in sys.path:
    sys.path.append(str(Q2_SRC))

import xlsx_reader as xr
import importlib.util


def _load_q2_data_io():
    spec = importlib.util.spec_from_file_location("_q2_data_io", Q2_SRC / "data_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_q2 = _load_q2_data_io()

DT = _q2.DT
N = _q2.N
ETA_C = _q2.ETA_C
ETA_D = _q2.ETA_D
E_MIN = _q2.E_MIN
E_MAX = _q2.E_MAX
P_MAX = _q2.P_MAX
C_MAX = _q2.C_MAX
E0_START = _q2.E0_START
EMERGENCY_MULT = _q2.EMERGENCY_MULT
THROUGHPUT_PENALTY = _q2.THROUGHPUT_PENALTY
SOC_TOL = 1e-5

ATTACH2 = ATTACH_DIR / "附件2.xlsx"
ATTACH4 = ATTACH_DIR / "附件4.xlsx"
TEMPLATE42 = ATTACH_DIR / "附件5" / "result4-2.xlsx"
RESULT42 = OUTPUT_DIR / "result4-2.xlsx"
SOLUTION_NPZ = OUTPUT_DIR / "prob4-2_solution.npz"
SMOKE_NPZ = OUTPUT_DIR / "smoke_solution.npz"

# Compatibility aliases used by the proven Q2 workbook writer.
TEMPLATE2 = TEMPLATE42
RESULT2 = RESULT42

excel_serial_to_date = _q2.excel_serial_to_date
read_actual_data = _q2.read_actual_data
terminal_value = _q2.terminal_value
time_label = _q2.time_label
plan_header = _q2.plan_header
block_labels = _q2.block_labels
_parse_time_fraction = _q2._parse_time_fraction


def align_price_cross_day(raw, first_price):
    """Convert rows 00:10..24:00 to calendar intervals 00:00..24:00.

    The last cell of row d is the midnight interval belonging to day d+1.
    It must therefore feed the next row, never be rolled within the same row.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != N:
        raise ValueError(f"电价矩阵期望形状 (D,{N})，实际 {raw.shape}")
    out = np.empty_like(raw)
    out[:, 1:] = raw[:, :-1]
    out[0, 0] = float(first_price)
    out[1:, 0] = raw[:-1, -1]
    return out


def read_dynamic_prices(first_price=None):
    """Read Attachment 4 as (dates, actual_price[D,144]) in calendar order."""
    rows = xr.read_sheet_rows(str(ATTACH4))["Sheet1"]
    if len(rows) != 366:
        raise ValueError(f"附件4期望表头加365天，实际 {len(rows)} 行")
    header = rows[0]
    if len(header) != 145:
        raise ValueError(f"附件4表头期望145列，实际 {len(header)}")
    times = np.array([_parse_time_fraction(v) for v in header[1:]], dtype=float)
    if times[0] != 10 or times[-1] != 1440 or not np.all(np.diff(times) == 10):
        raise ValueError("附件4时间列不是连续的00:10至次日00:00")

    data = rows[1:]
    dates = np.array([float(r[0]) for r in data])
    raw = np.array([[float(v) for v in r[1:145]] for r in data], dtype=float)
    if raw.shape != (365, N) or not np.isfinite(raw).all():
        raise ValueError("附件4电价数据形状错误或含非有限值")
    if np.any(raw < 0):
        raise ValueError("附件4出现负电价，当前题目费用模型未定义该情形")

    if first_price is None:
        # Attachment 4 has no 2025-01-01 00:00 quote. Use its nearest
        # available quote only for this single boundary cell; no Attachment 1
        # series is introduced into Q4-2.
        first_price = raw[0, 0]
    return dates, align_price_cross_day(raw, first_price)
