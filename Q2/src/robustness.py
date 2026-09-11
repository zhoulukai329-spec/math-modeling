# -*- coding: utf-8 -*-
"""问题 2 的稳健性检验与同一信息集基准对照。

1) 场景数 × 随机种子稳健性：对 11 个代表日，用场景数 {3,6,12}、
   随机种子 {7,2025,99173} 重算日前计划并做因果回测，比较平均日成本波动。
2) 同一信息集基准对照：统一用“因果点预测 + 因果逐时调度”，比较
   - 无储能因果策略；
   - 确定性点预测策略（点预测计划 + 储能因果调度）；
   - 当前随机策略（两阶段随机规划计划 + 储能因果调度）。

运行方式（在仓库根目录）:
    python Q2/src/robustness.py
"""
import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import data_io as dio
import forecast as fc
import optimization as opt
from run_problem2 import OUTPUT_START_DAY, N_SCENARIOS, LOOKBACK


ROBUSTNESS_CSV = dio.OUTPUT_DIR / "robustness_scenario_seed.csv"
BASELINES_CSV = dio.OUTPUT_DIR / "baselines.csv"


def output_stat(planned, emergency, e_all, full_year_mask):
    """汇总正式输出期指标。

    兼容两种输入：全年 365 天数组，或主程序已裁剪的
    2/1–12/31 共 334 天数组。
    """
    planned = np.asarray(planned)
    emergency = np.asarray(emergency)
    e_all = np.asarray(e_all)
    expected_output_days = int(np.count_nonzero(full_year_mask))
    if len(planned) == len(full_year_mask):
        selected_planned = planned[full_year_mask]
        selected_emergency = emergency[full_year_mask]
        selected_e = e_all[full_year_mask]
    elif len(planned) == expected_output_days:
        selected_planned = planned
        selected_emergency = emergency
        selected_e = e_all
    else:
        raise ValueError(
            f"费用数组长度 {len(planned)} 既不是全年 {len(full_year_mask)} "
            f"也不是输出期 {expected_output_days}"
        )
    return {
        "planned": float(selected_planned.sum()),
        "emergency": float(selected_emergency.sum()),
        "total": float((selected_planned + selected_emergency).sum()),
        "emergency_kwh": float(selected_e.sum()),
        "emergency_days": int((selected_e.sum(axis=1) > 1e-8).sum()),
    }


def _load_context():
    price, load_kw_typ, pv_kw_typ = dio.read_price_typical()
    dates, net, load, pv = dio.read_actual_data()
    typical_load = load_kw_typ * dio.DT
    typical_pv = pv_kw_typ * dio.DT
    f, r, load_hat, pv_hat, load_resid, pv_resid = fc.build_causal_forecasts(
        load, pv, typical_load, typical_pv
    )
    v = dio.terminal_value(price)
    return price, dates, net, load, pv, f, load_resid, pv_resid, v


def run_scenario_seed_grid(price, net, f, load_resid, pv_resid, v):
    day_idx = np.linspace(OUTPUT_START_DAY, len(net) - 1, 11, dtype=int)
    print("=" * 82)
    print("场景数 × 随机种子稳健性（11 个代表日，因果回测）")
    print("=" * 82)
    print(f"{'场景数':<6}{'种子':<8}{'平均日成本(元)':>16}")
    print("-" * 82)
    rows = []
    for n_sc in [3, 6, 12]:
        for seed in [7, 2025, 99173]:
            costs = []
            for d in day_idx:
                scenarios = fc.scenarios_for_day(
                    d, f, load_resid, pv_resid,
                    n_scenarios=n_sc, lookback=LOOKBACK, seed=seed,
                )
                g, first = opt.build_first_stage(price, scenarios, dio.E0_START, v)
                _c, _d, _w, e, _E = opt.causal_dispatch(
                    net[d], g, dio.E0_START,
                    discharge_reference=first["stats"]["d_mean"],
                )
                _p, _em, total = opt.dispatch_cost(price, g, e)
                costs.append(total)
            avg = float(np.mean(costs))
            rows.append((n_sc, seed, avg))
            print(f"{n_sc:<6}{seed:<8}{avg:>16.2f}")

    print("\n各场景数下跨种子波动范围")
    print("-" * 82)
    summary = []
    for n_sc in [3, 6, 12]:
        vals = [r[2] for r in rows if r[0] == n_sc]
        lo, hi = min(vals), max(vals)
        summary.append((n_sc, lo, hi, hi - lo))
        print(f"  {n_sc} 场景: 平均日成本 {lo:,.2f} ~ {hi:,.2f} 元 "
              f"(跨度 {hi - lo:,.2f} 元)")

    with open(ROBUSTNESS_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["场景数", "随机种子", "平均日成本_元"])
        for r in rows:
            writer.writerow(r)
        writer.writerow([])
        writer.writerow(["场景数", "最低平均日成本", "最高平均日成本", "跨度"])
        for s in summary:
            writer.writerow(s)
    print(f"\n稳健性结果已保存: {ROBUSTNESS_CSV}")
    return rows, summary


def run_baselines(price, net, f):
    D = net.shape[0]
    mask = np.arange(D) >= OUTPUT_START_DAY

    # 1) 无储能因果策略：计划 = 点预测(截断非负)，无储能，缺口用紧急电。
    g_det = np.maximum(0.0, f)
    planned0 = np.zeros(D)
    emergency0 = np.zeros(D)
    e0_all = np.zeros_like(net)
    for d in range(D):
        e = np.maximum(0.0, net[d] - g_det[d])
        e0_all[d] = e
        planned0[d] = float(np.sum(price * g_det[d]))
        emergency0[d] = float(np.sum(dio.EMERGENCY_MULT * price * e))

    # 2) 确定性点预测策略：计划 = 点预测(截断非负)，储能因果调度。
    planned1 = np.zeros(D)
    emergency1 = np.zeros(D)
    e1_all = np.zeros_like(net)
    E_cur = dio.E0_START
    for d in range(D):
        _c, _d, _w, e, E = opt.causal_dispatch(net[d], g_det[d], E_cur)
        e1_all[d] = e
        planned1[d] = float(np.sum(price * g_det[d]))
        emergency1[d] = float(np.sum(dio.EMERGENCY_MULT * price * e))
        E_cur = float(E[-1])

    # 3) 当前随机策略：读取主程序保存的解。
    z = np.load(str(dio.SOLUTION_NPZ))
    planned2 = z["planned_cost"]
    emergency2 = z["emergency_cost"]
    total2 = z["total_cost"]
    e2_all = z["e"]

    strategies = [
        ("无储能因果策略", output_stat(planned0, emergency0, e0_all, mask)),
        ("确定性点预测策略", output_stat(planned1, emergency1, e1_all, mask)),
        ("当前随机策略", output_stat(planned2, emergency2, e2_all, mask)),
    ]

    print("\n" + "=" * 82)
    print("同一信息集基准对照（输出期 2025-02-01 ~ 12-31）")
    print("=" * 82)
    header = (f"{'策略':<18}{'计划费(元)':>16}{'紧急费(元)':>16}{'总成本(元)':>16}"
              f"{'应急电量(kWh)':>16}{'应急天数':>10}")
    print(header)
    print("-" * 82)
    for name, s in strategies:
        print(f"{name:<18}{s['planned']:>16.2f}{s['emergency']:>16.2f}"
              f"{s['total']:>16.2f}{s['emergency_kwh']:>16.2f}{s['emergency_days']:>10}")

    base = strategies[0][1]["total"]
    print("\n相对无储能因果策略的总成本变化")
    print("-" * 60)
    for name, s in strategies:
        print(f"  {name:<18} 总成本 {s['total']:,.2f} 元 "
              f"({(s['total'] - base) / base * 100:+.2f}%)")

    with open(BASELINES_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["策略", "计划费_元", "紧急费_元", "总成本_元",
                         "应急电量_kWh", "应急天数"])
        for name, s in strategies:
            writer.writerow([name, f"{s['planned']:.2f}", f"{s['emergency']:.2f}",
                             f"{s['total']:.2f}", f"{s['emergency_kwh']:.2f}",
                             s["emergency_days"]])
    print(f"\n基准对照结果已保存: {BASELINES_CSV}")
    return strategies


def main():
    price, dates, net, load, pv, f, load_resid, pv_resid, v = _load_context()
    run_scenario_seed_grid(price, net, f, load_resid, pv_resid, v)
    run_baselines(price, net, f)
    print("\n稳健性检验与基准对照全部完成。")


if __name__ == "__main__":
    main()
