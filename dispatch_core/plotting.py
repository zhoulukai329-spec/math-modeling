"""Publication-oriented Q3/Q4-3 plots that never invoke an optimizer."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
TEAL = "#009E73"
VERMILION = "#D55E00"
YELLOW = "#F0E442"
BLACK = "#000000"
PALE_BLUE = "#D1E5F0"
PALE_RED = "#FDDBC7"
WHITE = "#FFFFFF"


def configure_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 360,
    })


def _save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=360, bbox_inches="tight", facecolor=WHITE)
        paths.append(path)
    plt.close(fig)
    return paths


def _time_axis(n):
    return np.arange(1, n + 1) / 6.0


def _style_time(ax, xmax=24.0):
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.linspace(0, xmax, 5) if xmax < 4 else np.arange(0, xmax + .01, 4))
    ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)


def plot_horizon_sensitivity(records, output_dir, title):
    """Show the cost/energy/speed trade-off for the continuous horizon length."""
    configure_style()
    records = list(records)
    if not records:
        raise ValueError("sensitivity records are empty")
    dates = list(dict.fromkeys(str(row["date"]) for row in records))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), layout="constrained")
    metrics = (("total_cost", "当日费用（元）"),
               ("emergency_kwh", "紧急购电量（kWh）"),
               ("solve_seconds", "MILP求解时间（s）"))
    colors = (BLUE, ORANGE, TEAL, VERMILION)
    markers = ("o", "s", "^", "D")
    for ax, (key, label) in zip(axes, metrics):
        for i, day in enumerate(dates):
            rows = sorted((row for row in records if str(row["date"]) == day),
                          key=lambda row: row["horizon_hours"])
            ax.plot([row["horizon_hours"] for row in rows], [row[key] for row in rows],
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    lw=1.25, ms=4, label=day[5:])
        ax.set_xlabel("预测前瞻长度（h）")
        ax.set_ylabel(label)
        ax.set_xticks((12, 18, 24))
        ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    axes[-1].legend(title="代表日", frameon=False, fontsize=7, title_fontsize=8)
    fig.suptitle(title)
    return _save(fig, Path(output_dir), "fig6_horizon_sensitivity")


def load_sensitivity_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = ("horizon_hours", "total_cost", "emergency_kwh", "end_soc_kwh",
               "first_commitment_kwh", "first_discharge_kwh", "solve_seconds", "max_residual")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def make_solution_figures(solution_path, output_dir, *, question, dynamic_price=False,
                          sensitivity_csv=None):
    """Render five solution figures and, when supplied, one sensitivity figure."""
    configure_style()
    output_dir = Path(output_dir)
    with np.load(solution_path, allow_pickle=False) as z:
        required = {"executed", "baseline", "final_commitment", "load_energy",
                    "pv_energy", "pv_forecast", "price", "charge", "discharge",
                    "emergency", "spill", "soc_before", "soc_after", "metadata_json"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"solution is missing fields: {sorted(missing)}")
        values = {key: z[key].copy() for key in z.files if key != "metadata_json"}
        metadata = json.loads(str(z["metadata_json"].item()))
    dates = metadata["dates"]
    executed = values["executed"].astype(bool)
    daily_emergency = np.nansum(np.where(executed, values["emergency"], 0), axis=1)
    index = int(np.argmax(daily_emergency)) if len(dates) else 0
    smoke = not bool(executed.all()) or len(dates) < 28
    qualifier = "（冒烟预览）" if smoke else ""
    t = _time_axis(values["price"].shape[1])
    executed_columns = np.flatnonzero(executed[index])
    display_end = max(1.0, float(t[executed_columns[-1]])) if smoke and executed_columns.size else 24.0
    paths = []

    forecast_rows = 2 if dynamic_price else 1
    fig, axes = plt.subplots(forecast_rows, 1, figsize=(7.2, 2.8 if forecast_rows == 1 else 4.7),
                             sharex=True, layout="constrained", squeeze=False)
    ax = axes[0, 0]
    ax.plot(t, values["pv_energy"][index] * 6, color=TEAL, lw=1.55, label="实际光伏")
    ax.plot(t, values["pv_forecast"][index] * 6, color=BLUE, lw=1.3, ls="--", label="当时可用预测")
    ax.set_ylabel("光伏功率（kW）")
    ax.legend(frameon=False, ncol=2)
    _style_time(ax, display_end)
    if dynamic_price:
        ax = axes[1, 0]
        ax.step(t, values["price"][index], where="mid", color=ORANGE, lw=1.45, label="实际电价")
        ax.step(t, values["price_forecast"][index], where="mid", color=BLUE, lw=1.25,
                ls="--", label="当时可用预测")
        ax.set_ylabel("电价（元/kWh）")
        ax.legend(frameon=False, ncol=2)
        _style_time(ax, display_end)
    axes[-1, 0].set_xlabel("左端点时刻（h）")
    fig.suptitle(f"{question} 预测与实际信息对照：{dates[index]}{qualifier}")
    paths += _save(fig, output_dir, "fig1_forecast_actual")

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.5), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [0.8, 1.5, 1.1, 1.0]})
    axes[0].step(t, values["price"][index], where="mid", color=ORANGE, lw=1.4)
    axes[0].set_ylabel("电价\n（元/kWh）")
    net = values["load_energy"][index] - values["pv_energy"][index]
    axes[1].step(t, values["baseline"][index] * 6, where="mid", color=PALE_BLUE, lw=1.1, label="0点基准")
    axes[1].step(t, values["final_commitment"][index] * 6, where="mid", color=BLUE, lw=1.5, label="调整后购电")
    axes[1].plot(t, net * 6, color=BLACK, lw=1.0, label="实际净负荷")
    axes[1].bar(t, values["emergency"][index] * 6, width=1/6, color=VERMILION, alpha=.8, label="紧急购电")
    axes[1].set_ylabel("功率（kW）")
    axes[1].legend(frameon=False, ncol=4, fontsize=7)
    axes[2].fill_between(t, 0, values["charge"][index] * 6, step="mid", color=ORANGE, alpha=.55, label="充电")
    axes[2].fill_between(t, 0, -values["discharge"][index] * 6, step="mid", color=TEAL, alpha=.55, label="放电")
    axes[2].set_ylabel("储能功率（kW）")
    axes[2].legend(frameon=False, ncol=2)
    axes[3].plot(t, values["soc_after"][index] / 1000, color=BLUE, lw=1.6)
    axes[3].axhline(1.2, color=BLACK, ls="--", lw=.7)
    axes[3].axhline(10.8, color=BLACK, ls="--", lw=.7)
    axes[3].set_ylabel("SOC（MWh）")
    axes[3].set_xlabel("左端点时刻（h）")
    for ax in axes:
        _style_time(ax, display_end)
    fig.suptitle(f"{question} 购电承诺与储能执行：{dates[index]}{qualifier}")
    paths += _save(fig, output_dir, "fig2_commitment_dispatch")

    daily_base = np.nansum(values["price"] * values["baseline"], axis=1)
    daily_revision = np.zeros(len(dates))
    date_index = {day: i for i, day in enumerate(dates)}
    for version in metadata.get("versions", []):
        day = str(version.get("target_times", [""])[0])[:10]
        if day in date_index:
            daily_revision[date_index[day]] += float(version.get("up_cost", 0)) + float(version.get("down_cost", 0))
    daily_emergency_cost = np.nansum(5 * values["price"] * values["emergency"], axis=1)
    calendar = [datetime.fromisoformat(day) for day in dates]
    if len(dates) >= 28:
        labels = list(dict.fromkeys(day.strftime("%Y-%m") for day in calendar))
        groups = np.array([labels.index(day.strftime("%Y-%m")) for day in calendar])
        base_plot = np.array([daily_base[groups == i].sum() for i in range(len(labels))])
        revision_plot = np.array([daily_revision[groups == i].sum() for i in range(len(labels))])
        emergency_plot = np.array([daily_emergency_cost[groups == i].sum() for i in range(len(labels))])
        x = np.arange(len(labels)); xlabels = [label[-2:] + "月" for label in labels]
        cost_unit, scale = "月费用（万元）", 1e4
    else:
        base_plot, revision_plot, emergency_plot = daily_base, daily_revision, daily_emergency_cost
        x = np.arange(len(dates)); xlabels = [day[5:] for day in dates]
        cost_unit, scale = "日费用（万元）", 1e4
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), layout="constrained")
    axes[0].bar(x, base_plot / scale, color=BLUE, label="基准购电费")
    axes[0].bar(x, revision_plot / scale, bottom=base_plot / scale, color=ORANGE, label="调整费用")
    axes[0].bar(x, emergency_plot / scale, bottom=(base_plot + revision_plot) / scale,
                color=VERMILION, label="紧急购电费")
    axes[0].set_xticks(x, xlabels)
    axes[0].set_ylabel(cost_unit)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].scatter(np.arange(len(dates)), daily_emergency / 1000, s=13, color=VERMILION, alpha=.75)
    axes[1].set_ylabel("紧急购电（MWh/日）")
    axes[1].set_xlabel("结果内日期序号")
    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", lw=.55, alpha=.7)
    fig.suptitle(f"{question} 日费用与紧急购电风险{qualifier}")
    paths += _save(fig, output_dir, "fig3_cost_emergency")

    cmap = LinearSegmentedColormap.from_list("emergency", [WHITE, PALE_RED, ORANGE, VERMILION, "#B2182B"])
    matrix = np.nan_to_num(values["emergency"], nan=0.0)
    vmax = max(float(matrix.max()), 1e-12)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    mesh = ax.pcolormesh(np.arange(145) / 6, np.arange(len(dates) + 1), matrix,
                         cmap=cmap, norm=PowerNorm(gamma=.55, vmin=0, vmax=vmax), shading="flat")
    ax.set_xlabel("左端点时刻（h）")
    ax.set_ylabel("结果内日期序号")
    ax.set_title(f"{question} 紧急购电日期—时段热力图{qualifier}")
    colorbar = fig.colorbar(mesh, ax=ax, pad=.02)
    colorbar.set_label("紧急购电量（kWh/10 min）")
    paths += _save(fig, output_dir, "fig4_emergency_heatmap")

    logs = metadata.get("solve_log", [])
    elapsed = np.asarray([row.get("elapsed_seconds", np.nan) for row in logs], dtype=float)
    variables = np.asarray([row.get("n_variables", np.nan) for row in logs], dtype=float)
    valid = np.isfinite(elapsed) & np.isfinite(variables)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    axes[0].hist(elapsed[np.isfinite(elapsed)], bins=min(20, max(1, len(elapsed))), color=BLUE, alpha=.8)
    axes[0].set_xlabel("单次MILP求解时间（s）")
    axes[0].set_ylabel("次数")
    axes[1].scatter(variables[valid], elapsed[valid], color=TEAL, s=18, alpha=.75)
    axes[1].set_xlabel("变量数")
    axes[1].set_ylabel("求解时间（s）")
    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", lw=.55, alpha=.7)
    fig.suptitle(f"{question} 计算性能诊断{qualifier}")
    paths += _save(fig, output_dir, "fig5_solver_performance")

    if sensitivity_csv is not None:
        paths += plot_horizon_sensitivity(load_sensitivity_csv(sensitivity_csv), output_dir,
                                          f"{question} 12/18/24小时前瞻敏感性")
    return paths
