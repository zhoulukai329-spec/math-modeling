"""Fill Q2 report tables from the current solution and evaluation CSV files."""
import csv
import re

import numpy as np

import data_io as dio
from emergency_table import DATE_NAMES, emergency_groups, target_indices


def fmt(value, decimals=4):
    value = float(value)
    if abs(value) < 0.5 * 10**(-decimals):
        value = 0.0
    return f"{value:.{decimals}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def replace_table(text, anchor, replacement):
    start = text.index(anchor)
    match = re.search(r"(?m)^\|.*\|\n(?:\|.*\|\n)*", text[start:])
    if match is None:
        raise ValueError(f"No table after {anchor}")
    a, b = start + match.start(), start + match.end()
    return text[:a] + replacement + "\n" + text[b:]


def csv_rows(name):
    with (dio.OUTPUT_DIR / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    z = np.load(dio.SOLUTION_NPZ)
    indices = target_indices(z["dates"])
    groups = emergency_groups(z)
    forecasts = csv_rows("forecast_rolling_metrics.csv")
    baselines = csv_rows("baselines.csv")
    sensitivity = csv_rows("imputation_sensitivity.csv")
    monte_carlo = csv_rows("monte_carlo_results.csv")
    with (dio.OUTPUT_DIR / "robustness_scenario_seed.csv").open(encoding="utf-8-sig", newline="") as stream:
        grid = list(csv.reader(stream))[1:10]

    planned_rows = []
    charge_rows = []
    emergency_rows = []
    blocks = dio.block_labels()
    for name, i, group in zip(DATE_NAMES, indices, groups):
        planned_rows.append([name] + [fmt(z["g"][i, k]) for k in [60, 72, 84, 96, 108, 120]]
                            + [fmt(z["g"][i].sum()), fmt(z["planned_cost"][i])])
        charge_rows.append([name] + [f'{fmt(z["c"][i, a:a+24].sum())} / {fmt(z["d"][i, a:a+24].sum())}'
                                    for a in range(0, 144, 24)]
                           + [fmt(z["E"][i, 0]), fmt(z["E"][i, -1])])
        emergency_rows.extend([[name, label, fmt(amount)] for label, amount in group])
        emergency_rows.append([name, "合计", fmt(z["e"][i].sum())])
    plan_table = table(["日期"] + [f"{h}:00-{h}:10" for h in [10, 12, 14, 16, 18, 20]]
                       + ["全天计划购电量 (kWh)", "全天计划购电费 (元)"], planned_rows)
    charge_table = table(["日期"] + [b + " 充电/放电" for b in blocks]
                         + ["0:00 储电量", "24:00 储电量"], charge_rows)
    emergency_table = table(["日期", "紧急购电时间段", "紧急购电量 (kWh)"], emergency_rows)
    overall_table = table(["指标", "数值"], [
        ["计划购电总费用", fmt(z["planned_cost"].sum(), 2) + " 元"],
        ["紧急购电总费用", fmt(z["emergency_cost"].sum(), 2) + " 元"],
        ["总购电费用", fmt(z["total_cost"].sum(), 2) + " 元"],
        ["紧急购电总量", fmt(z["e"].sum(), 2) + " kWh"],
        ["有紧急购电的天数", f'{int((z["e"].sum(axis=1)>1e-8).sum())} / {len(z["dates"])} 天']])
    base = float(baselines[0]["总成本_元"])
    saving = (1-float(baselines[2]["总成本_元"])/base)*100
    baseline_rows = [[r["策略"]] + [r[k] for k in ["计划费_元", "紧急费_元", "总成本_元"]]
                     for r in baselines]
    baseline_paper = [row + [f'{(float(r["总成本_元"])/base-1)*100:+.2f}%']
                      for row, r in zip(baseline_rows, baselines)]
    forecast_table = table(["方法", "MAE (kW)", "RMSE (kW)", "偏差 (kW)"],
                           [[r["方法"]] + [fmt(r[k], 2) for k in ["MAE_kW", "RMSE_kW", "Bias_kW"]]
                            for r in forecasts])
    lo, hi = min(float(r[2]) for r in grid), max(float(r[2]) for r in grid)
    sens_base = float(sensitivity[0]["总成本_元"])
    delta = max(abs(float(r["总成本_元"])-sens_base) for r in sensitivity)
    robust_table = table(["检验项", "结果"], [
        ["场景数×种子 平均日成本波动范围", f"{lo:.2f}–{hi:.2f} 元/日（9 组配置，跨度 {hi-lo:.2f} 元/日）"],
        ["插补敏感性（相对主方案最大差异）", f"输出期总成本差 {delta:.2f} 元（{delta/sens_base*100:.4f}%）；基于 CSV 两位小数精度"],
        ["蒙特卡洛 日成本 5%/50%/95% 分位", " / ".join(monte_carlo[0][k] for k in ["5%分位", "50%分位", "95%分位"]) + " 元"],
        ["蒙特卡洛 应急电量 5%/50%/95% 分位", " / ".join(monte_carlo[1][k] for k in ["5%分位", "50%分位", "95%分位"]) + " kWh"]])

    source_note = "> 数据来自本次运行的 prob2_solution.npz 与检验 CSV；能量单位为 kWh，费用单位为元，合计按未舍入值计算。"
    for filename, paper in [("问题2_求解报告.md", False), ("问题2_求解报告_论文版.md", True)]:
        path = dio.OUTPUT_DIR / filename
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^> .*?(?:待填|待重跑).*?$", source_note, text)
        text = replace_table(text, "### 7.1" if paper else "### 9.1", plan_table)
        text = replace_table(text, "### 7.3" if paper else "### 9.3", emergency_table)
        text = replace_table(text, "### 7.5" if paper else "### 9.4", overall_table)
        if paper:
            text = replace_table(text, "### 7.2", charge_table)
            text = replace_table(text, "### 7.4", forecast_table)
            text = replace_table(text, "### 7.6", table(["策略", "计划费(元)", "紧急费(元)", "总成本(元)", "相对无储能因果策略"], baseline_paper))
            text = replace_table(text, "## 8 ", robust_table)
            text = text.replace("结果（待填入数值）表明", f"输出期总成本为 {fmt(z['total_cost'].sum(), 2)} 元，相对无储能因果策略降低 {saving:.2f}%。结果表明")
        else:
            text = replace_table(text, "四种简单预测方法的误差如下", table(["预测方法", "平均误差 MAE (kW)", "RMSE (kW)"],
                                 [[r["方法"], fmt(r["MAE_kW"], 2), fmt(r["RMSE_kW"], 2)] for r in forecasts]))
            for name, i in zip(DATE_NAMES, indices):
                text = replace_table(text, "#### " + name, table(["时间段", "充电量 (kWh)", "放电量 (kWh)"],
                    [[label, fmt(z["c"][i, a:a+24].sum()), fmt(z["d"][i, a:a+24].sum())]
                     for label, a in zip(blocks, range(0, 144, 24))]))
                start = text.index("#### " + name)
                end = text.find("\n###", start+1)
                if end < 0:
                    end = len(text)
                section = re.sub(r"0 点储电量：.*?kWh。", f'0 点储电量：{fmt(z["E"][i, 0])} kWh；24 点储电量：{fmt(z["E"][i, -1])} kWh。', text[start:end])
                text = text[:start]+section+text[end:]
            text = replace_table(text, '#### 怎么公平', table(["策略", "计划费(元)", "紧急费(元)", "总成本(元)"], baseline_rows))
            text = text.replace("可以看到大多数天紧急购电费是 0，少数预测不准的日子会突然变高", "输出期 334 天均有紧急购电，费用随预测误差和调度情况变化")
            text = text.replace("这组对比能说明电池和滚动计划确实帮微网省了不少钱。", f"当前随机策略相对无储能因果策略降低总成本 {saving:.2f}%。")
        if "待填" in text or "待重跑" in text:
            raise ValueError(f"Unfilled report: {filename}")
        validate_markdown_tables(text)
        path.write_text(text, encoding="utf-8")
        print(f"Filled {filename}")


def validate_markdown_tables(text):
    expected = None
    for line in text.splitlines():
        if line.startswith("|"):
            count = len(line.split("|"))
            if expected is not None and count != expected:
                raise ValueError(f"Table column count mismatch: {line}")
            expected = count
        else:
            expected = None


if __name__ == "__main__":
    main()
