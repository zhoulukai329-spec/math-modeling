# -*- coding: utf-8 -*-
"""问题 2 主流程：逐日制定计划并回测，保存完整结果。

运行方式（在仓库根目录）:
    python Q2/src/run_problem2.py

本文件同时导出 simulate_days()，供敏感性检验与蒙特卡洛脚本复用，
保证这些脚本与主程序使用完全相同的两阶段随机线性规划逻辑。
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
N_SCENARIOS = 12
LOOKBACK = 28


def simulate_days(net, price, f, load_resid, pv_resid,
                  n_scenarios=N_SCENARIOS, lookback=LOOKBACK,
                  seed=2025, E_start=dio.E0_START, v_terminal=None,
                  start_day=0, end_day=None):
    """逐日运行两阶段随机线性规划并回测。

    参数:
      net:        (D,144) 全年实际净负荷能量 (kWh)
      price:      (144,) 日内电价 (元/kWh)
      f:           净负荷因果点预测 (D,144)
      load_resid:  负载因果残差 (D,144)
      pv_resid:    光伏因果残差 (D,144)
      n_scenarios: 场景数
      lookback:    残差场景池回看窗口
      seed:        确定性抽样种子
      E_start:     第 start_day 天 0:00 的储电量 (kWh)
      v_terminal:  24:00 储能量终值系数；None 时用 data_io.terminal_value
      start_day:   起算日期索引（含）
      end_day:     结束日期索引（含）；None 表示到最后一天

    返回 dict，键为 g/c/d/w/e/E/planned_cost/emergency_cost/total_cost/
    first_obj/first_expected_emergency/solve_status。
    """
    if v_terminal is None:
        v_terminal = dio.terminal_value(price)

    D = net.shape[0]
    N = dio.N
    if end_day is None:
        end_day = D - 1

    g_all = np.zeros((D, N))
    c_all = np.zeros((D, N))
    d_all = np.zeros((D, N))
    w_all = np.zeros((D, N))
    e_all = np.zeros((D, N))
    E_all = np.zeros((D, N + 1))
    d_reference_all = np.zeros((D, N))

    planned_cost = np.zeros(D)
    emergency_cost = np.zeros(D)
    total_cost = np.zeros(D)
    first_obj = np.zeros(D)
    first_expected_emergency = np.zeros(D)

    E_cur = float(E_start)
    solve_status = []

    for day in range(start_day, end_day + 1):
        if day % 30 == 0:
            print(f"    处理日期索引 {day}/{D-1}", flush=True)

        scenarios = fc.scenarios_for_day(
            day, f, load_resid, pv_resid,
            n_scenarios=n_scenarios, lookback=lookback, seed=seed,
        )
        g, first = opt.build_first_stage(price, scenarios, E_cur, v_terminal)
        d_reference = first["stats"]["d_mean"]
        c, d_act, w, e, E = opt.causal_dispatch(
            net[day], g, E_cur, discharge_reference=d_reference
        )

        g_all[day] = g
        c_all[day] = c
        d_all[day] = d_act
        w_all[day] = w
        e_all[day] = e
        E_all[day] = E
        d_reference_all[day] = d_reference

        planned_cost[day] = float(np.sum(price * g))
        emergency_cost[day] = float(np.sum(dio.EMERGENCY_MULT * price * e))
        total_cost[day] = planned_cost[day] + emergency_cost[day]
        first_obj[day] = first["objective"]
        first_expected_emergency[day] = first["stats"].get("expected_emergency", 0.0)
        solve_status.append(first["status"])

        E_cur = float(E[-1])

    return {
        "g": g_all,
        "c": c_all,
        "d": d_all,
        "w": w_all,
        "e": e_all,
        "E": E_all,
        "d_reference": d_reference_all,
        "planned_cost": planned_cost,
        "emergency_cost": emergency_cost,
        "total_cost": total_cost,
        "first_obj": first_obj,
        "first_expected_emergency": first_expected_emergency,
        "solve_status": solve_status,
    }


def main():
    t0 = time.time()

    price = dio.read_price()
    dates, net, load, pv = dio.read_actual_data()

    f, r, load_hat, pv_hat, load_resid, pv_resid = fc.build_causal_forecasts(
        load, pv
    )
    v_terminal = dio.terminal_value(price)

    D = net.shape[0]

    print("=" * 70)
    print("问题 2 逐日两阶段随机规划回测")
    print("=" * 70)
    sim = simulate_days(
        net, price, f, load_resid, pv_resid,
        n_scenarios=N_SCENARIOS, lookback=LOOKBACK,
        seed=2025, E_start=dio.E0_START, v_terminal=v_terminal,
    )
    print(f"\n全年回测完成，用时 {time.time() - t0:.1f} s")

    mask = np.arange(D) >= OUTPUT_START_DAY
    dates_out = dates[mask]
    g_out = sim["g"][mask]
    c_out = sim["c"][mask]
    d_out = sim["d"][mask]
    w_out = sim["w"][mask]
    e_out = sim["e"][mask]
    E_out = sim["E"][mask]
    net_out = net[mask]
    planned_out = sim["planned_cost"][mask]
    emergency_out = sim["emergency_cost"][mask]
    total_out = sim["total_cost"][mask]

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
        d_reference=sim["d_reference"][mask],
        planned_cost=planned_out,
        emergency_cost=emergency_out,
        total_cost=total_out,
        first_obj=sim["first_obj"][mask],
        first_expected_emergency=sim["first_expected_emergency"][mask],
        v_terminal=np.array([v_terminal]),
        n_scenarios=np.array([N_SCENARIOS]),
        forecast_initialization=np.array(["zero_no_attachment1"]),
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
