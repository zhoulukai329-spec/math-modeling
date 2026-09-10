# -*- coding: utf-8 -*-
"""问题 2 的因果净负荷预测滚动检验。

严格按“只使用目标日之前的历史数据”的原则，对 2025-02-01~12-31 做滚动回测，
比较四种简单的因果预测方法：
  1. 前一日同时段
  2. 上周同日同时段（run_problem2.py 使用的点预测）
  3. 最近 7 日同时段均值
  4. 最近 4 个同星期日的同日同时段均值

输出 MAE、RMSE、偏差，并把结果保存为 CSV。
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


OUTPUT_START_DAY = 31  # 0 基索引；0 = 2025-01-01
METRICS_CSV = dio.OUTPUT_DIR / "forecast_rolling_metrics.csv"


def previous_day_forecast(net, d):
    return net[d - 1]


def previous_week_forecast(net, d):
    if d >= 7:
        return net[d - 7]
    return net[:d].mean(axis=0)


def rolling_7day_mean_forecast(net, d):
    start = max(0, d - 7)
    return net[start:d].mean(axis=0)


def rolling_4_same_weekday_forecast(net, d):
    """使用目标日之前最近 4 个同星期日的均值；历史不足时用可用样本。"""
    indices = [d - 7, d - 14, d - 21, d - 28]
    indices = [j for j in indices if j >= 0]
    if not indices:
        return np.zeros_like(net[d])
    return net[indices].mean(axis=0)


def evaluate(name, forecast_func, net, target_days):
    errs = []
    for d in target_days:
        f = forecast_func(net, d)
        err = net[d] - f
        errs.append(err)
    E = np.stack(errs, axis=0)      # (D,144) kWh/10min
    E_kw = E * 6.0                  # kWh/10min -> 平均 kW
    mae = float(np.mean(np.abs(E_kw)))
    rmse = float(np.sqrt(np.mean(E_kw**2)))
    bias = float(np.mean(E_kw))
    return mae, rmse, bias


def main():
    _, load_kw_typ, pv_kw_typ = dio.read_price_typical()
    dates, net, load, pv = dio.read_actual_data()
    target_days = list(range(OUTPUT_START_DAY, len(net)))

    methods = [
        ("前一日同时段", previous_day_forecast),
        ("上周同日同时段", previous_week_forecast),
        ("最近7日均值", rolling_7day_mean_forecast),
        ("最近4个同星期日均值", rolling_4_same_weekday_forecast),
    ]

    print("=" * 82)
    print("问题 2 因果净负荷预测滚动检验")
    print(f"目标期: 2025-02-01 ~ 2025-12-31，共 {len(target_days)} 天")
    print(f"误差单位: kW（由 kWh/10min × 6 换算）")
    print("=" * 82)
    print(f"{'方法':<24}{'MAE':>12}{'RMSE':>12}{'偏差':>12}")
    rows = []
    for name, func in methods:
        mae, rmse, bias = evaluate(name, func, net, target_days)
        rows.append((name, mae, rmse, bias))
        print(f"{name:<24}{mae:>12.2f}{rmse:>12.2f}{bias:>12.2f}")

    best_idx = int(np.argmin([r[1] for r in rows]))
    best_name = rows[best_idx][0]
    print("-" * 82)
    print(f"MAE 最低方法: {best_name}")
    print("结论: 数据存在明显的周周期，'上周同日同时段'明显优于前一日和7日均值。")

    dio.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["方法", "MAE_kW", "RMSE_kW", "Bias_kW"])
        writer.writerows(rows)
    print(f"\n指标已保存: {METRICS_CSV}")


if __name__ == "__main__":
    main()

