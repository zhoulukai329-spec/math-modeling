# -*- coding: utf-8 -*-
"""问题 2 主流程：逐日制定计划并回测，保存完整结果。

运行方式（在仓库根目录）:
    python Q2/src/run_problem2.py
"""
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import data_io as dio
import forecast as fc
import optimization as opt


OUTPUT_START_DAY = 31  # 2025-02-01 的索引（0 基；0 = 2025-01-01）
N_SCENARIOS = 6
LOOKBACK = 28


def main():
    t0 = time.time()

    price, load_typical_kw, pv_typical_kw = dio.read_price_typical()
    dates, net, load, pv = dio.read_actual_data()
    typical_net = (load_typical_kw - pv_typical_kw) * dio.DT

    f, r = fc.build_causal_forecasts(net, typical_net)
    v_terminal = dio.terminal_value(price)

    D = net.shape[0]
    N = dio.N

    g_all = np.zeros((D, N))
    c_all = np.zeros((D, N))
    d_all = np.zeros((D, N))
    w_all = np.zeros((D, N))
    e_all = np.zeros((D, N))
    E_all = np.zeros((D, N + 1))

    planned_cost = np.zeros(D)
    emergency_cost = np.zeros(D)
    total_cost = np.zeros(D)
    first_obj = np.zeros(D)
    first_expected_emergency = np.zeros(D)

    E_cur = dio.E0_START
    solve_status = []

    for day in range(D):
        if day % 30 == 0:
            print(f"[{time.time() - t0:6.1f}s] 处理日期索引 {day}/{D-1}", flush=True)

        scenarios = fc.scenarios_for_day(
            day, net, f, r, n_scenarios=N_SCENARIOS, lookback=LOOKBACK, seed=2025
        )
        g, first = opt.build_first_stage(price, scenarios, E_cur, v_terminal)
        c, d_act, w, e, E, second = opt.build_second_stage(
            price, net[day], g, E_cur, v_terminal
        )

        g_all[day] = g
        c_all[day] = c
        d_all[day] = d_act
        w_all[day] = w
        e_all[day] = e
        E_all[day] = E

        planned_cost[day] = float(np.sum(price * g))
        emergency_cost[day] = float(np.sum(dio.EMERGENCY_MULT * price * e))
        total_cost[day] = planned_cost[day] + emergency_cost[day]
        first_obj[day] = first["objective"]
        first_expected_emergency[day] = first["stats"].get("expected_emergency", 0.0)
        solve_status.append(second.status)

        E_cur = float(E[-1])

    print(f"\n全年回测完成，用时 {time.time() - t0:.1f} s")

    mask = np.arange(D) >= OUTPUT_START_DAY
    dates_out = dates[mask]
    g_out = g_all[mask]
    c_out = c_all[mask]
    d_out = d_all[mask]
    w_out = w_all[mask]
    e_out = e_all[mask]
    E_out = E_all[mask]
    net_out = net[mask]
    planned_out = planned_cost[mask]
    emergency_out = emergency_cost[mask]
    total_out = total_cost[mask]

    np.savez_compressed(
        str(dio.SOLUTION_NPZ),
        dates=dates_out,
        price=price,
        net=net_out,
        g=g_out,
        c=c_out,
        d=d_out,
        w=w_out,
        e=e_out,
        E=E_out,
        planned_cost=planned_out,
        emergency_cost=emergency_out,
        total_cost=total_out,
        first_obj=first_obj[mask],
        first_expected_emergency=first_expected_emergency[mask],
        v_terminal=np.array([v_terminal]),
        n_scenarios=np.array([N_SCENARIOS]),
        eta=np.array([dio.ETA_C, dio.ETA_D]),
    )
    print(f"结果已保存: {dio.SOLUTION_NPZ}")

    print("\n===== 输出期（2025-02-01 ~ 2025-12-31）汇总 =====")
    print(f"计划购电总费用: {planned_out.sum():.2f} 元")
    print(f"紧急购电总费用: {emergency_out.sum():.2f} 元")
    print(f"总购电费用:     {total_out.sum():.2f} 元")
    print(f"紧急购电总量:   {e_out.sum():.2f} kWh")
    print(f"存在紧急购电的天数: {(e_out.sum(axis=1) > 1e-8).sum()} / {len(e_out)}")
    print(f"计划购电总量:   {g_out.sum():.2f} kWh")


if __name__ == "__main__":
    main()
