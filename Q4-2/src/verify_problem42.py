# -*- coding: utf-8 -*-
"""Independent numerical and workbook verification for Q4-2 outputs."""
import argparse
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import data_io as dio
import forecast as fc
import xlsx_reader as xr


TOL = 1e-5


def check_solution(z):
    g, c, d, w, e, E, net = (z[k] for k in ["g","c","d","w","e","E","net"])
    price = z["price"]
    if price.shape != g.shape:
        raise AssertionError(f"动态电价应与购电矩阵同形，实际{price.shape}和{g.shape}")
    balance = g + e + d - c - w - net
    dynamics = np.diff(E, axis=1) - (dio.ETA_C*c - d/dio.ETA_D)
    cross = np.max(np.abs(E[:-1,-1] - E[1:,0])) if len(E) > 1 else 0.0
    planned = np.sum(price*g, axis=1)
    emergency = np.sum(dio.EMERGENCY_MULT*price*e, axis=1)
    checks = {
        "能量平衡": np.max(np.abs(balance)),
        "SOC递推": np.max(np.abs(dynamics)),
        "SOC下界": max(0.0, dio.E_MIN-float(E.min())),
        "SOC上界": max(0.0, float(E.max())-dio.E_MAX),
        "跨日SOC": cross,
        "计划费用": np.max(np.abs(planned-z["planned_cost"])),
        "应急费用": np.max(np.abs(emergency-z["emergency_cost"])),
        "总费用": np.max(np.abs(planned+emergency-z["total_cost"])),
    }
    for name, value in checks.items():
        print(f"  {name}: {value:.3e}")
        if value > TOL:
            raise AssertionError(f"{name}超过容差{TOL}: {value}")
    if np.any(g < -TOL) or np.any(c < -TOL) or np.any(d < -TOL) or np.any(e < -TOL):
        raise AssertionError("出现负的购电/充电/放电/应急量")
    if np.any((e > TOL) & (c > TOL)) or np.any((e > TOL) & (w > TOL)):
        raise AssertionError("紧急购电与充电或弃电同时发生")


def check_causality():
    typical, _, _ = dio.read_price_typical()
    _, actual = dio.read_dynamic_prices(first_price=typical[0])
    predicted, _ = fc.build_causal_price_forecasts(actual, typical)
    changed = actual.copy()
    changed[100:] += 1000.0
    predicted_changed, _ = fc.build_causal_price_forecasts(changed, typical)
    if not np.array_equal(predicted[:100], predicted_changed[:100]):
        raise AssertionError("改变未来真实电价影响了此前预测")
    if not np.array_equal(predicted[100,1:], predicted_changed[100,1:]):
        raise AssertionError("当天00:10后的未来真实电价泄漏进0:00预测")
    if predicted_changed[100,0] != changed[100,0]:
        raise AssertionError("0:00已经发布的实时电价未进入当前决策")
    print("  电价预测非前视扰动检验: PASS")


def check_workbook(z, workbook):
    sheets = xr.read_sheet_rows(str(workbook))
    if list(sheets) != ["计划购电量", "充放电量", "紧急购电量"]:
        raise AssertionError(f"工作表名称错误: {list(sheets)}")
    dates, g = z["dates"], z["g"]
    plan = sheets["计划购电量"]
    if len(plan) != len(dates)+1:
        raise AssertionError("计划购电量工作表行数错误")
    for i in range(len(dates)):
        ordered = np.concatenate([g[i,1:], g[i,:1]])
        actual = np.asarray(plan[i+1][1:145], dtype=float)
        if np.max(np.abs(actual-ordered)) > 1e-4:
            raise AssertionError(f"计划购电量第{i+2}行与NPZ不一致")
        if abs(float(plan[i+1][146])-float(z["planned_cost"][i])) > 1e-4:
            raise AssertionError(f"计划购电费用第{i+2}行不一致")
    print("  result4-2.xlsx计划表与NPZ: PASS")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", nargs="?", default=str(dio.SOLUTION_NPZ))
    parser.add_argument("--workbook", default=str(dio.RESULT42))
    args = parser.parse_args(argv)
    with np.load(args.solution, allow_pickle=False) as z:
        print("[1] 数值约束与动态费用")
        check_solution(z)
        print("[2] 价格预测非前视")
        check_causality()
        if Path(args.workbook).exists():
            print("[3] 正式工作簿")
            check_workbook(z, Path(args.workbook))
    print("Q4-2校验 PASS")


if __name__ == "__main__":
    main()
