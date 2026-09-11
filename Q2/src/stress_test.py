# -*- coding: utf-8 -*-
"""问题 2 的补充压力测试与蒙特卡洛检验。

与 verify_problem2.py 的确定性可行性检查互补，本脚本关注：
  1. lookback 窗口长度的稳健性；
  2. 运行起点初始 SOC 的敏感性；
  3. 连续阴雨（光伏为零）下的紧急购电压力；
  4. 极端高负荷下的紧急购电压力；
  5. 历史残差块蒙特卡洛：对代表日重采样历史残差，评估固定日前计划的日成本分布。
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
from run_problem2 import simulate_days


OUTPUT_START_DAY = 31


def _load_context():
    z = np.load(str(dio.SOLUTION_NPZ))
    price, load_kw_typ, pv_kw_typ = dio.read_price_typical()
    dates, net, load, pv = dio.read_actual_data()
    typical_net = (load_kw_typ - pv_kw_typ) * dio.DT
    f, r = fc.build_causal_forecasts(net, typical_net)
    return z, price, net, load, pv, f, r


def _summary(sim, mask):
    return {
        "planned": float(sim["planned_cost"][mask].sum()),
        "emergency": float(sim["emergency_cost"][mask].sum()),
        "total": float(sim["total_cost"][mask].sum()),
        "emergency_kwh": float(sim["e"][mask].sum()),
        "emergency_days": int((sim["e"][mask].sum(axis=1) > 1e-8).sum()),
    }


def run_lookback_sensitivity(z, price, net, f, r):
    q = float(z["q_selected"][0])
    v = float(z["v_terminal"][0])
    D = len(net)
    mask = np.arange(D) >= OUTPUT_START_DAY
    print("\n[1] lookback 窗口敏感性")
    rows = []
    for lb in [7, 14, 28, 56]:
        sim = simulate_days(net, price, f, r, q, v, 0, D - 1, dio.E0_START,
                            lookback=lb)
        s = _summary(sim, mask)
        rows.append((lb, s["planned"], s["emergency"], s["total"],
                     s["emergency_kwh"], s["emergency_days"]))
        print(f"  lookback={lb:2d}  总成本={s['total']:,.2f}  紧急费={s['emergency']:,.2f}")
    return rows


def run_initial_soc_sensitivity(z, price, net, f, r):
    q = float(z["q_selected"][0])
    v = float(z["v_terminal"][0])
    D = len(net)
    mask = np.arange(D) >= OUTPUT_START_DAY
    print("\n[2] 初始 SOC 敏感性")
    rows = []
    for e0 in [3000.0, 6000.0, 9000.0]:
        sim = simulate_days(net, price, f, r, q, v, 0, D - 1, e0)
        s = _summary(sim, mask)
        rows.append((e0, s["planned"], s["emergency"], s["total"],
                     s["emergency_kwh"], s["emergency_days"]))
        print(f"  E0={e0:7.1f}  总成本={s['total']:,.2f}  紧急费={s['emergency']:,.2f}")
    return rows


def _run_fixed_plan_stress(price, stress_net, g_base, E0):
    planned_total = 0.0
    emergency_total = 0.0
    emergency_kwh = 0.0
    emergency_days = 0
    E_cur = float(E0)
    for i in range(len(g_base)):
        c, d, w, e, E = opt.causal_dispatch(stress_net[i], g_base[i], E_cur)
        planned, emergency, total = opt.dispatch_cost(price, g_base[i], e, c, d)
        planned_total += planned
        emergency_total += emergency
        emergency_kwh += float(e.sum())
        if float(e.sum()) > 1e-8:
            emergency_days += 1
        E_cur = float(E[-1])
    return planned_total, emergency_total, emergency_kwh, emergency_days


def run_extreme_stress(z, price, net, load, pv):
    mask = np.arange(len(net)) >= OUTPUT_START_DAY
    load_out = load[mask]
    pv_out = pv[mask]
    g_base = z["g"]
    E0 = float(z["E"][0, 0])
    print("\n[3] 极端天气/负荷压力测试（固定现有日前计划）")
    rows = []

    overcast_net = load_out - 0.0 * pv_out
    p, e, kwh, days = _run_fixed_plan_stress(price, overcast_net, g_base, E0)
    print(f"  连续阴雨(PV=0): 总成本={p+e:,.2f}  紧急费={e:,.2f}  "
          f"紧急电量={kwh:,.2f} kWh  紧急天数={days}")
    rows.append(("连续阴雨PV=0", p, e, p + e, kwh, days))

    high_load_net = load_out * 1.2 - pv_out
    p, e, kwh, days = _run_fixed_plan_stress(price, high_load_net, g_base, E0)
    print(f"  高负荷(load×1.2): 总成本={p+e:,.2f}  紧急费={e:,.2f}  "
          f"紧急电量={kwh:,.2f} kWh  紧急天数={days}")
    rows.append(("高负荷load×1.2", p, e, p + e, kwh, days))
    return rows


def run_residual_monte_carlo(z, price, net, f, r, n_days=11, n_samples=100):
    print(f"\n[4] 历史残差块蒙特卡洛（{n_days} 个代表日 × {n_samples} 次 = {n_days * n_samples} 次）")
    day_idx = np.linspace(OUTPUT_START_DAY, len(net) - 1, n_days, dtype=int)
    costs = []
    emergency_kwh = []
    for d in day_idx:
        i = int(d - OUTPUT_START_DAY)
        g = z["g"][i]
        E0 = float(z["E"][i, 0])
        scenarios = fc.scenarios_for_day(
            d, net, f, r, n_scenarios=n_samples, lookback=28, seed=2025
        )
        for s in range(n_samples):
            c, dd, w, e, E = opt.causal_dispatch(scenarios[s], g, E0)
            _p, _em, total = opt.dispatch_cost(price, g, e, c, dd)
            costs.append(total)
            emergency_kwh.append(float(e.sum()))
    costs = np.array(costs)
    emergency_kwh = np.array(emergency_kwh)
    q_cost = np.quantile(costs, [0.05, 0.5, 0.95])
    q_em = np.quantile(emergency_kwh, [0.05, 0.5, 0.95])
    print(f"  日成本均值={costs.mean():,.2f}  5%/50%/95%分位="
          f"{q_cost[0]:,.2f}/{q_cost[1]:,.2f}/{q_cost[2]:,.2f} 元")
    print(f"  紧急购电量均值={emergency_kwh.mean():,.2f} kWh  "
          f"95%分位={q_em[2]:,.2f} kWh")
    return costs, emergency_kwh


def main():
    z, price, net, load, pv, f, r = _load_context()
    lookback_rows = run_lookback_sensitivity(z, price, net, f, r)
    soc_rows = run_initial_soc_sensitivity(z, price, net, f, r)
    stress_rows = run_extreme_stress(z, price, net, load, pv)
    costs, emergency_kwh = run_residual_monte_carlo(z, price, net, f, r)

    out = dio.OUTPUT_DIR / "stress_test_results.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["实验", "参数", "计划费", "紧急费", "总费", "紧急电量", "紧急天数"])
        for lb, p, e, tot, kwh, days in lookback_rows:
            writer.writerow(["lookback", lb, p, e, tot, kwh, days])
        for e0, p, e, tot, kwh, days in soc_rows:
            writer.writerow(["初始SOC", e0, p, e, tot, kwh, days])
        for name, p, e, tot, kwh, days in stress_rows:
            writer.writerow([name, "", p, e, tot, kwh, days])
        writer.writerow(["残差块MC", "日成本均值", "", "", float(costs.mean()), "", ""])
        writer.writerow(["残差块MC", "日成本5%", "", "", float(np.quantile(costs, 0.05)), "", ""])
        writer.writerow(["残差块MC", "日成本50%", "", "", float(np.quantile(costs, 0.5)), "", ""])
        writer.writerow(["残差块MC", "日成本95%", "", "", float(np.quantile(costs, 0.95)), "", ""])
        writer.writerow(["残差块MC", "紧急电量均值", "", "", "", float(emergency_kwh.mean()), ""])
        writer.writerow(["残差块MC", "紧急电量95%", "", "", "", float(np.quantile(emergency_kwh, 0.95)), ""])
    print(f"\n压力测试结果已保存: {out}")


if __name__ == "__main__":
    main()
