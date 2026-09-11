# -*- coding: utf-8 -*-
"""问题 2 的 1 月 1 日 0:00 插补敏感性检验 + 历史残差块蒙特卡洛。

只改变 1 月 1 日首个 10 分钟（0:00-0:10）的负荷插补值，其余日期全部采用
跨日拼接，模型（两阶段随机线性规划 + 因果预测）保持不变。比较三种插补对
正式输出期（2 月 1 日起）的影响，并对主方案做历史残差块蒙特卡洛。

运行方式（在仓库根目录）:
    python Q2/src/imputation_sensitivity.py
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
from run_problem2 import simulate_days, OUTPUT_START_DAY, N_SCENARIOS, LOOKBACK


MONTE_CARLO_N_DAYS = 11
MONTE_CARLO_N_SAMPLES = 40
SENSITIVITY_CSV = dio.OUTPUT_DIR / "imputation_sensitivity.csv"
MONTE_CARLO_CSV = dio.OUTPUT_DIR / "monte_carlo_results.csv"


def _load_context():
    price, load_typ_kw, pv_typ_kw = dio.read_price_typical()
    dates, load_raw, pv_raw = dio._read_attachment2_matrices()
    typical_net = (load_typ_kw - pv_typ_kw) * dio.DT
    v_terminal = dio.terminal_value(price)
    return price, dates, load_raw, pv_raw, load_typ_kw, typical_net, v_terminal


def _build_net(load_raw, pv_raw, first_load_kw, first_pv_kw=0.0):
    """按给定 1 月 1 日 0:00 负荷插补值，用跨日拼接构造全年净负荷。"""
    load_kw = dio._align_cross_day(load_raw, first_load_kw)
    pv_kw = dio._align_cross_day(pv_raw, first_pv_kw)
    return (load_kw - pv_kw) * dio.DT


def _run_case(price, net, typical_net, v_terminal, label):
    f, r = fc.build_causal_forecasts(net, typical_net)
    print(f"\n—— 运行插补方案：{label} ——")
    sim = simulate_days(
        net, price, f, r, n_scenarios=N_SCENARIOS, lookback=LOOKBACK,
        seed=2025, E_start=dio.E0_START, v_terminal=v_terminal,
    )
    mask = np.arange(net.shape[0]) >= OUTPUT_START_DAY
    return {
        "label": label,
        "sim": sim,
        "net": net,
        "total_cost": float(sim["total_cost"][mask].sum()),
        "planned_cost": float(sim["planned_cost"][mask].sum()),
        "emergency_cost": float(sim["emergency_cost"][mask].sum()),
        "emergency_kwh": float(sim["e"][mask].sum()),
        "emergency_days": int((sim["e"][mask].sum(axis=1) > 1e-8).sum()),
        "E_feb1_0": float(sim["E"][OUTPUT_START_DAY, 0]),
        "E_final_24": float(sim["E"][-1, -1]),
    }


def run_imputation_sensitivity(price, load_raw, pv_raw, load_typ_kw,
                               typical_net, v_terminal):
    extrap = 2.0 * load_raw[0, 0] - load_raw[0, 1]  # 线性外推 3449.9305
    typical0 = load_typ_kw[0]                        # 附件1 典型 3444.7259
    hold0 = load_raw[0, 0]                           # 向后持有 3529.7296

    cases = [
        ("线性外推(主方案)", extrap),
        ("附件1典型值", typical0),
        ("向后持有0:10", hold0),
    ]

    print("=" * 82)
    print("问题 2 · 1 月 1 日 0:00 插补敏感性检验")
    print(f"三种 1 月 1 日 0:00 负荷取值: 线性外推={extrap:.4f}, "
          f"典型={typical0:.4f}, 向后持有={hold0:.4f} kW（光伏均取 0）")
    print("=" * 82)

    results = []
    main_sim = None
    for label, first_load in cases:
        net = _build_net(load_raw, pv_raw, first_load)
        r = _run_case(price, net, typical_net, v_terminal, label)
        results.append(r)
        if main_sim is None:
            main_sim = r

    main = results[0]
    print("\n" + "=" * 82)
    print("三种插补的正式输出期（2025-02-01 ~ 2025-12-31）结果对比")
    print("=" * 82)
    header = (
        f"{'插补方案':<18}{'总成本(元)':>16}{'计划费(元)':>16}{'紧急费(元)':>16}"
        f"{'应急电量(kWh)':>16}{'2/1初始SOC':>14}{'年末SOC':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['label']:<18}{r['total_cost']:>16.2f}{r['planned_cost']:>16.2f}"
              f"{r['emergency_cost']:>16.2f}{r['emergency_kwh']:>16.2f}"
              f"{r['E_feb1_0']:>14.2f}{r['E_final_24']:>14.2f}")

    print("\n相对主方案（线性外推）的差异")
    print("-" * 60)
    for r in results[1:]:
        print(f"{r['label']:<18} 总成本差={r['total_cost'] - main['total_cost']:+,.2f} 元 "
              f"({(r['total_cost'] - main['total_cost']) / main['total_cost'] * 100:+.4f}%)")

    print("\nSOC 轨迹最大差异（相对主方案，覆盖全年 D×145 个点）")
    print("-" * 60)
    E_main = main["sim"]["E"]
    for r in results[1:]:
        E_alt = r["sim"]["E"]
        max_diff = float(np.max(np.abs(E_alt - E_main)))
        print(f"{r['label']:<18} max|ΔSOC|={max_diff:.4f} kWh")

    # 应急购电量差异（主方案为基准）
    print("\n应急购电量差异（相对主方案）")
    print("-" * 60)
    for r in results[1:]:
        print(f"{r['label']:<18} Δ应急电量={r['emergency_kwh'] - main['emergency_kwh']:+,.4f} kWh")

    with open(SENSITIVITY_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["插补方案", "首段负荷_kW", "总成本_元", "计划费_元",
                         "紧急费_元", "应急电量_kWh", "应急天数",
                         "2月1日初始SOC", "年末24时SOC"])
        for (label, first_load), r in zip(cases, results):
            writer.writerow([label, f"{first_load:.4f}", f"{r['total_cost']:.2f}",
                             f"{r['planned_cost']:.2f}", f"{r['emergency_cost']:.2f}",
                             f"{r['emergency_kwh']:.2f}", r["emergency_days"],
                             f"{r['E_feb1_0']:.4f}", f"{r['E_final_24']:.4f}"])
    print(f"\n敏感性结果已保存: {SENSITIVITY_CSV}")
    return main_sim, results


def run_residual_monte_carlo(price, net, typical_net, v_terminal, main_sim):
    f, r = fc.build_causal_forecasts(net, typical_net)
    day_idx = np.linspace(OUTPUT_START_DAY, net.shape[0] - 1,
                          MONTE_CARLO_N_DAYS, dtype=int)
    print("\n" + "=" * 82)
    print(f"历史残差块蒙特卡洛（主方案）: {MONTE_CARLO_N_DAYS} 个代表日 × "
          f"{MONTE_CARLO_N_SAMPLES} 次 = {MONTE_CARLO_N_DAYS * MONTE_CARLO_N_SAMPLES} 次")
    print("=" * 82)

    all_cost = []
    all_emergency_kwh = []
    for d in day_idx:
        i = int(d - OUTPUT_START_DAY)
        g = main_sim["sim"]["g"][d]
        E0 = float(main_sim["sim"]["E"][d, 0])
        scenarios = fc.scenarios_for_day(
            d, net, f, r, n_scenarios=MONTE_CARLO_N_SAMPLES, lookback=LOOKBACK, seed=2025
        )
        for s in range(MONTE_CARLO_N_SAMPLES):
            _c, _d, _w, e, _E, _res = opt.build_second_stage(
                price, scenarios[s], g, E0, v_terminal
            )
            cost = float(np.sum(price * g) + np.sum(dio.EMERGENCY_MULT * price * e))
            all_cost.append(cost)
            all_emergency_kwh.append(float(e.sum()))

    cost = np.array(all_cost)
    em = np.array(all_emergency_kwh)
    q_cost = np.quantile(cost, [0.05, 0.5, 0.95])
    q_em = np.quantile(em, [0.05, 0.5, 0.95])
    print(f"  日成本均值={cost.mean():,.2f} 元   5%/50%/95%分位="
          f"{q_cost[0]:,.2f}/{q_cost[1]:,.2f}/{q_cost[2]:,.2f} 元")
    print(f"  应急购电量均值={em.mean():,.2f} kWh   5%/50%/95%分位="
          f"{q_em[0]:,.2f}/{q_em[1]:,.2f}/{q_em[2]:,.2f} kWh")

    with open(MONTE_CARLO_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["指标", "均值", "5%分位", "50%分位", "95%分位"])
        writer.writerow(["日成本_元", f"{cost.mean():.2f}", f"{q_cost[0]:.2f}",
                         f"{q_cost[1]:.2f}", f"{q_cost[2]:.2f}"])
        writer.writerow(["应急电量_kWh", f"{em.mean():.2f}", f"{q_em[0]:.2f}",
                         f"{q_em[1]:.2f}", f"{q_em[2]:.2f}"])
    print(f"蒙特卡洛结果已保存: {MONTE_CARLO_CSV}")
    return cost, em


def main():
    price, dates, load_raw, pv_raw, load_typ_kw, typical_net, v_terminal = _load_context()
    main_sim, results = run_imputation_sensitivity(
        price, load_raw, pv_raw, load_typ_kw, typical_net, v_terminal
    )
    run_residual_monte_carlo(price, main_sim["net"], typical_net, v_terminal, main_sim)
    print("\n插补敏感性检验与蒙特卡洛全部完成。")


if __name__ == "__main__":
    main()
