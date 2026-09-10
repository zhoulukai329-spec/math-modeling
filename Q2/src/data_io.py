# -*- coding: utf-8 -*-
"""问题 2 的数据读取、预处理与通用参数。

只使用标准库 + numpy，并复用 Q2/src/xlsx_reader.py 解析 xlsx。
所有路径在本文件中统一路由，业务代码不直接写死路径。
"""
import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
REPO_ROOT = Q_DIR.parent
ATTACH_DIR = REPO_ROOT / "attachment"
OUTPUT_DIR = Q_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import xlsx_reader as xr


ATTACH1 = ATTACH_DIR / "附件1.xlsx"
ATTACH2 = ATTACH_DIR / "附件2.xlsx"
TEMPLATE2 = ATTACH_DIR / "附件5" / "result2.xlsx"
RESULT2 = OUTPUT_DIR / "result2.xlsx"
SOLUTION_NPZ = OUTPUT_DIR / "prob2_solution.npz"


DT = 1.0 / 6.0
N = 144
ETA_C = 0.9
ETA_D = 0.9
E_MIN = 1200.0
E_MAX = 10800.0
P_MAX = 5000.0
C_MAX = P_MAX * DT
E0_START = 6000.0
EMERGENCY_MULT = 5.0
THROUGHPUT_PENALTY = 1e-6


def excel_serial_to_date(serial):
    """Excel 1900 日期序列号 -> (year, month, day)。仅用于展示/校验。"""
    import datetime
    base = datetime.date(1899, 12, 30)
    d = base + datetime.timedelta(days=int(round(float(serial))))
    return d.year, d.month, d.day


def _parse_time_fraction(t):
    """附件 1/2 时间列转分钟。数值为日分数，'0:00+1' 表示次日 0:00。"""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return round(float(t) * 24 * 60)
    s = str(t).strip()
    offset = 0
    if "+" in s:
        s, off = s.split("+")
        offset = int(off) * 1440
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm) + offset


def read_price_typical():
    """读附件 1 的固定日内电价，并转成自然时钟顺序。

    返回 price: shape (144,)，price[k] 对应 [10k,10(k+1)] 分钟区间。
    """
    rows = xr.read_sheet_rows(str(ATTACH1))["Sheet1"]
    data = rows[1:]
    assert len(data) == 144, f"附件1 期望 144 行，实际 {len(data)}"
    t_min = np.array([_parse_time_fraction(r[0]) for r in data], dtype=float)
    price_raw = np.array([float(r[1]) for r in data], dtype=float)
    load_raw = np.array([float(r[2]) for r in data], dtype=float)
    pv_raw = np.array([float(r[3]) for r in data], dtype=float)
    assert t_min[0] == 10 and t_min[-1] == 1440
    assert np.all(np.diff(t_min) == 10)
    # 原始行首 0:10 对应区间 [0:10,0:20]，末行 0:00+1 对应当日 [0:00,0:10]。
    # 循环右移一格，使下标 k 直接对应时钟区间 [10k,10(k+1)]。
    price = np.roll(price_raw, 1)
    load_kw = np.roll(load_raw, 1)
    pv_kw = np.roll(pv_raw, 1)
    return price, load_kw, pv_kw


def read_actual_data():
    """读附件 2 全年负载/光伏实际功率，转成自然时钟顺序和能量。

    返回:
      dates: (365,)  Excel 日期序列号
      net:   (365,144) 净负荷 L-G 的能量 (kWh)
      load:  (365,144) 负载能量 (kWh)
      pv:    (365,144) 光伏能量 (kWh)
    """
    sheets = xr.read_sheet_rows(str(ATTACH2))
    # 附件2 第一张表=小区负载，第二张表=光伏发电实际功率
    names = list(sheets)
    load_sheet = sheets[names[0]]
    pv_sheet = sheets[names[1]]

    header = load_sheet[0]
    assert len(header) == 145, f"附件2 表头期望 145 列，实际 {len(header)}"
    time_head = header[1:]
    t_min = np.array([_parse_time_fraction(v) for v in time_head], dtype=float)
    assert t_min[0] == 10 and t_min[-1] == 1440
    assert np.all(np.diff(t_min) == 10)

    def load_matrix(sheet):
        data = sheet[1:]
        assert len(data) == 365, f"附件2 期望 365 天，实际 {len(data)}"
        dates = np.array([float(r[0]) for r in data])
        mat = np.array([[float(v) for v in r[1:]] for r in data], dtype=float)
        assert mat.shape == (365, 144)
        return dates, mat

    dates, load_kw = load_matrix(load_sheet)
    _, pv_kw = load_matrix(pv_sheet)

    # 循环右移一格，使下标 k = [10k,10(k+1)]。
    load_kw = np.roll(load_kw, 1, axis=1)
    pv_kw = np.roll(pv_kw, 1, axis=1)

    load = load_kw * DT
    pv = pv_kw * DT
    net = load - pv
    return dates, net, load, pv


def terminal_value(price):
    """储能量 E 在当日 24:00 的线性终值系数。

    用固定日内电价均值近似未来电能的边际价值，并乘以放电效率：
    1 kWh 存储能量次日可放出 eta_d kWh，按平均电价节省购电费。
    """
    return ETA_D * float(np.mean(price))


def time_label(k, end=None):
    """返回第 k 个自然时钟区间的 'HH:MM-HH:MM' 文本。

    k=0..143；end 为 True 时右侧为 24:00，否则为 (k+1)*10 分钟。
    """
    def fmt(minutes):
        minutes = minutes % 1440
        h = minutes // 60
        m = minutes % 60
        if minutes == 1440 or (h == 0 and m == 0 and minutes == 0):
            return "0:00"
        if minutes % 60 == 0 and minutes > 0:
            return f"{h}:00"
        return f"{h}:{m:02d}"

    a = k * 10
    b = (k + 1) * 10
    if b == 1440:
        right = "0:00+1"
    else:
        right = fmt(b)
    return f"{fmt(a)}-{right}"


def plan_header():
    """问题 2 计划购电量工作表的 147 列表头。"""
    labels = []
    # 与模板一致：先 0:10-0:20 ... 23:50-0:00+1，最后补 0:00-0:10+1。
    for k in range(1, 144):
        labels.append(time_label(k))
    labels.append("23:50-0:00+1")  # 已有；为避免奇偶，显式按模板顺序
    # 上面循环 k=1..143 已包含 23:50-0:00+1，去掉末尾重复并补 0:00-0:10+1。
    labels = labels[:-1] + ["0:00-0:10+1"]
    labels = ["日期\\时间"] + labels + ["全天购电量", "全天购电费"]
    return labels


def block_labels():
    return ["0:00-4:00", "4:00-8:00", "8:00-12:00",
            "12:00-16:00", "16:00-20:00", "20:00-24:00"]

