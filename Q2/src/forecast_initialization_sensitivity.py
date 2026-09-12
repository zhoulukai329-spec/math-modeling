# -*- coding: utf-8 -*-
"""Compare two admissible first-day forecast initializations for Q2."""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_io as dio
import forecast as fc
from run_problem2 import simulate_days, OUTPUT_START_DAY, LOOKBACK, N_SCENARIOS


def build_priors(load):
    """Return zero and flat-current-load day-one forecast profiles (kWh)."""
    load = np.asarray(load, dtype=float)
    if load.ndim != 2 or load.shape[1] != dio.N:
        raise ValueError("负载矩阵必须为(D,144)")
    zero = np.zeros(dio.N)
    flat = np.full(dio.N, float(load[0, 0]))
    return {
        "zero_prior": (zero.copy(), zero.copy()),
        "flat_current_load_prior": (flat, zero.copy()),
    }


def _run_case(name, priors, net, load, pv, price, args, end_day):
    load_prior, pv_prior = priors
    f, _r, _lh, _ph, load_resid, pv_resid = fc.build_causal_forecasts(
        load, pv, first_day_load=load_prior, first_day_pv=pv_prior
    )
    sim = simulate_days(
        net, price, f, load_resid, pv_resid,
        n_scenarios=args.scenarios, lookback=args.lookback,
        seed=args.seed, E_start=dio.E0_START, start_day=0, end_day=end_day,
    )
    return {"name": name, "sim": sim, "forecast": f,
            "load_prior": load_prior, "pv_prior": pv_prior}


def _date_text(serial):
    y, m, d = dio.excel_serial_to_date(serial)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _metrics(case, end_day, args):
    s = case["sim"]
    jan_end = min(end_day, OUTPUT_START_DAY - 1)
    formal_exists = end_day >= OUTPUT_START_DAY
    formal = slice(OUTPUT_START_DAY, end_day + 1)
    return {
        "方案": case["name"],
        "场景数": args.scenarios,
        "回看天数": args.lookback,
        "随机种子": args.seed,
        "运行终止日索引": end_day,
        "首日预测负载_kWh每10分钟": float(case["load_prior"][0]),
        "1月成本_元": float(s["total_cost"][:jan_end+1].sum()),
        "2月1日初始SOC_kWh": float(s["E"][OUTPUT_START_DAY, 0]) if formal_exists else np.nan,
        "正式期计划费_元": float(s["planned_cost"][formal].sum()) if formal_exists else np.nan,
        "正式期紧急费_元": float(s["emergency_cost"][formal].sum()) if formal_exists else np.nan,
        "正式期总费用_元": float(s["total_cost"][formal].sum()) if formal_exists else np.nan,
        "正式期紧急电量_kWh": float(s["e"][formal].sum()) if formal_exists else np.nan,
        "正式期紧急购电天数": int((s["e"][formal].sum(axis=1) > 1e-8).sum()) if formal_exists else 0,
        "末日24点SOC_kWh": float(s["E"][end_day, -1]),
    }


def _save_case(case, dates, net, price, end_day, output_dir, args):
    s = case["sim"]
    n = end_day + 1
    np.savez_compressed(
        output_dir / f"{case['name']}.npz",
        dates=dates[:n], net=net[:n], price=price,
        forecast=case["forecast"][:n], load_prior=case["load_prior"],
        pv_prior=case["pv_prior"], g=s["g"][:n], c=s["c"][:n],
        d=s["d"][:n], w=s["w"][:n], e=s["e"][:n], E=s["E"][:n],
        planned_cost=s["planned_cost"][:n], emergency_cost=s["emergency_cost"][:n],
        total_cost=s["total_cost"][:n], seed=np.array([args.seed]),
        n_scenarios=np.array([args.scenarios]), lookback=np.array([args.lookback]),
        end_day=np.array([end_day]),
    )


def _write_comparison(cases, dates, end_day, output_dir, args):
    metrics = [_metrics(case, end_day, args) for case in cases]
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    a, b = cases[0]["sim"], cases[1]["sim"]
    daily_path = output_dir / "daily_comparison.csv"
    fields = [
        "日期", "零先验_日初SOC", "当前值持续_日初SOC", "SOC初值差",
        "零先验_日末SOC", "当前值持续_日末SOC", "SOC末值差",
        "零先验_当日费用", "当前值持续_当日费用", "当日费用差",
        "累计费用差", "计划购电量绝对差_kWh", "紧急购电量差_kWh",
    ]
    cumulative = 0.0
    with daily_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for d in range(end_day + 1):
            cost_diff = float(b["total_cost"][d] - a["total_cost"][d])
            cumulative += cost_diff
            writer.writerow({
                "日期": _date_text(dates[d]),
                "零先验_日初SOC": float(a["E"][d,0]),
                "当前值持续_日初SOC": float(b["E"][d,0]),
                "SOC初值差": float(b["E"][d,0]-a["E"][d,0]),
                "零先验_日末SOC": float(a["E"][d,-1]),
                "当前值持续_日末SOC": float(b["E"][d,-1]),
                "SOC末值差": float(b["E"][d,-1]-a["E"][d,-1]),
                "零先验_当日费用": float(a["total_cost"][d]),
                "当前值持续_当日费用": float(b["total_cost"][d]),
                "当日费用差": cost_diff,
                "累计费用差": cumulative,
                "计划购电量绝对差_kWh": float(np.abs(b["g"][d]-a["g"][d]).sum()),
                "紧急购电量差_kWh": float(b["e"][d].sum()-a["e"][d].sum()),
            })

    report_path = output_dir / "如何看结果.txt"
    m0, m1 = metrics
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("Q2首日预测初始化敏感性分析\n\n")
        fh.write(f"运行设置：场景数={args.scenarios}，回看天数={args.lookback}，随机种子={args.seed}，终止日期={_date_text(dates[end_day])}。\n\n")
        fh.write("summary.csv：比较两行的2月1日初始SOC和正式期总费用。\n")
        fh.write("daily_comparison.csv：查看SOC差何时接近0，以及累计费用差是否停止变化。\n\n")
        if end_day >= OUTPUT_START_DAY:
            fh.write(f"2月1日初始SOC差：{m1['2月1日初始SOC_kWh']-m0['2月1日初始SOC_kWh']:.6f} kWh\n")
            fh.write(f"本次正式期总费用差：{m1['正式期总费用_元']-m0['正式期总费用_元']:.6f} 元\n")
        else:
            fh.write("本次smoke尚未运行到2月1日，不能判断正式输出期影响。\n")
    return summary_path, daily_path, report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Q2首日预测初始化敏感性分析")
    parser.add_argument("--mode", choices=["smoke", "feb-boundary", "full"], default="smoke")
    parser.add_argument("--scenarios", type=int, default=N_SCENARIOS)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args(argv)
    end_day = {"smoke": 1, "feb-boundary": 31, "full": 364}[args.mode]
    output_dir = dio.OUTPUT_DIR / f"forecast_initialization_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    price = dio.read_price()
    dates, net, load, pv = dio.read_actual_data()
    priors = build_priors(load)
    t0 = time.time()
    cases = []
    for name, prior in priors.items():
        print(f"\n运行方案: {name}", flush=True)
        case = _run_case(name, prior, net, load, pv, price, args, end_day)
        _save_case(case, dates, net, price, end_day, output_dir, args)
        cases.append(case)
    paths = _write_comparison(cases, dates, end_day, output_dir, args)
    print(f"\n敏感性分析完成，用时 {time.time()-t0:.1f}s")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
