# -*- coding: utf-8 -*-
"""问题 2 的独立校验脚本。

不直接调用 run_problem2.py，而是读取其保存的 npz 和生成的 result2.xlsx，
从以下层次复核：
  1) 逐时段能量平衡与储能动态残差；
  2) 储能电量/充放电/购电/紧急购电非负性及上下限；
  3) 跨日 SOC 连续性与费用重算；
  4) result2.xlsx 的行数、表头、数值与 npz 对齐；
  5) 预测因果性抽查：0:00 只使用目标日之前的历史。
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import data_io as dio
import forecast as fc
import xlsx_reader as xr


TOL = 1e-6


def check_primal(z):
    g, c, d, w, e, E, net = z["g"], z["c"], z["d"], z["w"], z["e"], z["E"], z["net"]
    price = z["price"]
    eta_c, eta_d = dio.ETA_C, dio.ETA_D

    bal = g + e + d - c - w - net
    dyn = np.diff(E, axis=1) - (eta_c * c - d / eta_d)
    checks = {
        "能量平衡残差": float(np.max(np.abs(bal))),
        "储能动态残差": float(np.max(np.abs(dyn))),
        "储能越下限": float(max(0.0, dio.E_MIN - E.min())),
        "储能越上限": float(max(0.0, E.max() - dio.E_MAX)),
        "充电越上限": float(max(0.0, c.max() - dio.C_MAX)),
        "放电越上限": float(max(0.0, d.max() - dio.C_MAX)),
        "购电为负": float(max(0.0, -g.min())),
        "紧急购电为负": float(max(0.0, -e.min())),
        "弃光为负": float(max(0.0, -w.min())),
        "同时充放电时段数": int(((c > 1e-8) & (d > 1e-8)).sum()),
    }
    ok = True
    print("\n[1] 原始可行性")
    for name, val in checks.items():
        passed = val <= TOL if "时段数" not in name else val == 0
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val:.3e}")
    return ok


def check_continuity_and_cost(z):
    E = z["E"]
    price = z["price"]
    g, e = z["g"], z["e"]
    planned = z["planned_cost"]
    emergency = z["emergency_cost"]
    total = z["total_cost"]

    # 结果数组本身已按 2/1~12/31 连续；需重算跨日连续。
    cross = np.max(np.abs(E[:-1, -1] - E[1:, 0]))
    rec_planned = np.sum(price[None, :] * g, axis=1)
    rec_emergency = np.sum(dio.EMERGENCY_MULT * price[None, :] * e, axis=1)
    rec_total = rec_planned + rec_emergency
    print("\n[2] 跨日连续性与费用重算")
    print(f"  跨日 SOC 最大断点: {cross:.3e}")
    print(f"  计划费用最大误差: {np.max(np.abs(rec_planned - planned)):.3e}")
    print(f"  紧急费用最大误差: {np.max(np.abs(rec_emergency - emergency)):.3e}")
    print(f"  总费用最大误差:   {np.max(np.abs(rec_total - total)):.3e}")
    ok = (
        cross <= TOL
        and np.max(np.abs(rec_planned - planned)) <= TOL
        and np.max(np.abs(rec_emergency - emergency)) <= TOL
        and np.max(np.abs(rec_total - total)) <= TOL
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] 跨日连续性与费用重算")
    return ok


def check_causality():
    price, load_kw, pv_kw = dio.read_price_typical()
    dates, net, load, pv = dio.read_actual_data()
    typical_load = load_kw * dio.DT
    typical_pv = pv_kw * dio.DT
    f, r, load_hat, pv_hat, load_resid, pv_resid = fc.build_causal_forecasts(
        load, pv, typical_load, typical_pv
    )
    ok = True
    print("\n[3] 预测因果性抽查")
    for d in [31, 100, 200, 300, 364]:
        # 场景生成只能使用 <d 的历史；这里检查负载/光伏点预测不引用未来行。
        if d == 0:
            ok &= np.allclose(load_hat[d], typical_load)
            ok &= np.allclose(pv_hat[d], typical_pv)
        elif d < 7:
            ok &= np.allclose(load_hat[d], load[:d].mean(axis=0))
            ok &= np.allclose(pv_hat[d], pv[:d].mean(axis=0))
        else:
            ok &= np.allclose(load_hat[d], load[d - 7])
            ok &= np.allclose(pv_hat[d], pv[d - 7])
        sc = fc.scenarios_for_day(d, f, load_resid, pv_resid,
                                  n_scenarios=4, lookback=14, seed=7)
        ok &= np.isfinite(sc).all()
    print(f"  [{'PASS' if ok else 'FAIL'}] 负载/光伏点预测仅用历史，场景生成无 NaN/Inf")
    return ok


def check_emergency_semantics(z):
    """语义不变量：紧急购电只弥补当期缺口，禁止给储能充电。"""
    g, c, e, net = z["g"], z["c"], z["e"], z["net"]
    tol = 1e-6
    # 1) 计划购电量 >= 当期净负荷时不应有紧急购电
    surplus_emergency = int(((g >= net - tol) & (e > tol)).sum())
    # 2) 紧急购电与充电不应同时发生
    e_and_c = int(((e > tol) & (c > tol)).sum())
    # 3) 紧急购电不超过当期缺口 max(0, net - g)
    deficit = np.maximum(0.0, net - g)
    over = float(np.max(np.maximum(0.0, e - deficit)))
    print("\n[5] 紧急购电语义不变量")
    print(f"  计划>=净负荷仍紧急购电的时段数: {surplus_emergency}")
    print(f"  紧急购电与充电同时发生的时段数: {e_and_c}")
    print(f"  紧急购电超过当期缺口的最大量: {over:.3e} kWh")
    ok = surplus_emergency == 0 and e_and_c == 0 and over <= 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] 紧急购电语义不变量")
    return ok


def check_xlsx(z):
    sheets = xr.read_sheet_rows(str(dio.RESULT2))
    names = list(sheets)
    g, c, d, E, e = z["g"], z["c"], z["d"], z["E"], z["e"]
    dates = z["dates"]
    planned = z["planned_cost"]
    ok = True
    print("\n[4] result2.xlsx 复核")
    print(f"  工作表名: {names}")
    ok &= names == ["计划购电量", "充放电量", "紧急购电量"]

    plan = sheets["计划购电量"]
    ok &= len(plan) == 1 + len(dates)
    print(f"  计划购电量行数: {len(plan)} (期望 {1 + len(dates)})")

    # 检查计划购电量数值：模板顺序为 g[:,1:] 后跟 g[:,0]
    if len(plan) == 1 + len(dates):
        max_diff = 0.0
        for i in range(len(dates)):
            row = plan[i + 1]
            ordered = np.concatenate([g[i, 1:], g[i, :1]])
            vals = np.array(row[1 : 1 + dio.N], dtype=float)
            max_diff = max(max_diff, float(np.max(np.abs(vals - ordered))))
            ok &= abs(row[0] - dates[i]) <= 1e-8
            ok &= abs(row[1 + dio.N] - g[i].sum()) <= 1e-4
            ok &= abs(row[2 + dio.N] - planned[i]) <= 1e-4
        print(f"  计划购电量最大数值误差: {max_diff:.3e}")

    charge = sheets["充放电量"]
    ok &= len(charge) == 1 + 6 * len(dates)
    print(f"  充放电量行数: {len(charge)} (期望 {1 + 6 * len(dates)})")
    if len(charge) == 1 + 6 * len(dates):
        for i in range(len(dates)):
            rows = charge[1 + 6 * i : 1 + 6 * (i + 1)]
            for j, (a, b) in enumerate(
                zip([0, 24, 48, 72, 96, 120], [24, 48, 72, 96, 120, 144])
            ):
                ok &= abs(rows[j][2] - c[i, a:b].sum()) <= 1e-4
                ok &= abs(rows[j][3] - d[i, a:b].sum()) <= 1e-4
            ok &= abs(rows[0][5] - E[i, 0]) <= 1e-4
            ok &= abs(rows[1][5] - E[i, -1]) <= 1e-4

    # 紧急购电量按日期分组重算
    emergency = sheets["紧急购电量"]
    # 将每行归到其最近的非空日期
    current_idx = None
    rec_sum = np.zeros(len(dates))
    for row in emergency[1:]:
        if row[0] is not None:
            current_idx = int(np.where(np.isclose(dates, float(row[0])))[0][0])
        if current_idx is not None and row[2] is not None:
            rec_sum[current_idx] += float(row[2])
    e_sum = e.sum(axis=1)
    diff = np.max(np.abs(rec_sum - e_sum))
    print(f"  紧急购电量逐日汇总最大误差: {diff:.3e}")
    ok &= diff <= 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] result2.xlsx 数值对齐")
    return ok


def main():
    print("=" * 76)
    print("问题 2 模型检验")
    print("=" * 76)
    z = np.load(str(dio.SOLUTION_NPZ))
    ok1 = check_primal(z)
    ok2 = check_continuity_and_cost(z)
    ok3 = check_causality()
    ok4 = check_xlsx(z)
    ok5 = check_emergency_semantics(z)
    print("\n" + "=" * 76)
    if ok1 and ok2 and ok3 and ok4 and ok5:
        print("结论: 问题 2 模型检验 PASS。")
    else:
        print("结论: 存在 FAIL 项，请检查模型、数值精度或结果文件。")
    print("=" * 76)


if __name__ == "__main__":
    main()

