# -*- coding: utf-8 -*-
"""Q4-2 场景数与随机种子敏感性检验。"""
import csv
import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_io as dio
import forecast as fc
import optimization as opt
from run_problem42 import OUTPUT_START_DAY, N_SCENARIOS, LOOKBACK, _inputs


SCENARIO_COUNTS = (3, 6, 12)
SEEDS = (7, 2025, 99173)
OUTPUT_CSV = dio.OUTPUT_DIR / "sensitivity_scenario_seed.csv"


def formal_start_soc(energy, calendar_days, output_start_day=OUTPUT_START_DAY):
    """Return each representative day's 00:00 SOC from the saved formal run."""
    energy = np.asarray(energy, dtype=float)
    calendar_days = np.asarray(calendar_days, dtype=int)
    rows = calendar_days - int(output_start_day)
    if energy.ndim != 2 or energy.shape[1] < 1:
        raise ValueError("正式SOC数组必须为二维且至少包含00:00列")
    if np.any(rows < 0) or np.any(rows >= len(energy)):
        raise ValueError("代表日不在正式结果保存期内")
    return energy[rows, 0]


def run_grid():
    dates, net, price_actual, net_f, price_f, net_r, price_r = _inputs()
    day_idx = np.linspace(OUTPUT_START_DAY, len(net) - 1, 11, dtype=int)
    with np.load(dio.SOLUTION_NPZ, allow_pickle=False) as formal:
        if not bool(formal["complete"][0]):
            raise ValueError("Q4-2敏感性检验需要完整全年正式结果")
        formal_E = formal["E"].copy()
        formal_g = formal["g"].copy()
    start_soc = formal_start_soc(formal_E, day_idx)

    rows = []
    for n_sc in SCENARIO_COUNTS:
        for seed in SEEDS:
            totals = []
            planned = []
            emergency = []
            emergency_kwh = []
            baseline_max_g_error = 0.0
            for day, E0 in zip(day_idx, start_soc):
                net_s, price_s, _ = fc.joint_scenarios_for_day(
                    day, net_f, price_f, net_r, price_r,
                    n_scenarios=n_sc, lookback=LOOKBACK, seed=seed,
                )
                v_terminal = dio.terminal_value(price_s.mean(axis=0))
                g, first = opt.build_first_stage(price_s, net_s, E0, v_terminal)
                _c, _d, _w, e, _E = opt.causal_dispatch(
                    net[day], g, E0,
                    discharge_reference=first["stats"]["d_mean"],
                )
                p_cost, e_cost, total = opt.dispatch_cost(price_actual[day], g, e)
                planned.append(p_cost)
                emergency.append(e_cost)
                totals.append(total)
                emergency_kwh.append(float(e.sum()))
                if n_sc == N_SCENARIOS and seed == 2025:
                    row = day - OUTPUT_START_DAY
                    baseline_max_g_error = max(
                        baseline_max_g_error,
                        float(np.max(np.abs(g - formal_g[row]))),
                    )
            if n_sc == N_SCENARIOS and seed == 2025 and baseline_max_g_error > 1e-5:
                raise AssertionError(
                    "正式参数未能复现Q4-2主结果："
                    f"计划购电最大误差 {baseline_max_g_error:.3e} kWh"
                )
            row = (
                n_sc, seed, float(np.mean(planned)), float(np.mean(emergency)),
                float(np.mean(totals)), float(np.mean(emergency_kwh)),
                baseline_max_g_error if n_sc == N_SCENARIOS and seed == 2025 else "",
            )
            rows.append(row)
            print(
                f"场景={n_sc:2d} 种子={seed:5d} 平均日总成本={row[4]:,.2f}元 "
                f"平均应急电量={row[5]:,.2f}kWh"
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "场景数", "随机种子", "平均计划费_元", "平均紧急费_元",
            "平均日总成本_元", "平均日应急电量_kWh", "正式参数复现最大购电误差_kWh",
        ])
        writer.writerows(rows)
    print(f"敏感性结果已保存: {OUTPUT_CSV}")
    return rows


if __name__ == "__main__":
    run_grid()
