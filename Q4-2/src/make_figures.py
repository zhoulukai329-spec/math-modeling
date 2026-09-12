# -*- coding: utf-8 -*-
"""Generate publication-ready Q4-2 figures from a saved solution.

This module only reads an NPZ result; it never invokes the optimizer.  It works
with both the full result and the one-day smoke archive, although the full
archive is required for meaningful monthly and seasonal conclusions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np


SRC = Path(__file__).resolve().parent
OUTPUT = SRC.parent / "output"
FIGURES = OUTPUT / "figures"

# User-selected, color-blind-friendly palette.  Meanings remain fixed in all figures.
BLUE = "#0072B2"       # planned / forecast
ORANGE = "#E69F00"     # price / charging
TEAL = "#009E73"       # actual / discharging
VERMILION = "#D55E00" # emergency / warning
YELLOW = "#F0E442"     # highlight only
BLACK = "#000000"
PALE_BLUE = "#D1E5F0"
PALE_RED = "#FDDBC7"
WHITE = "#FFFFFF"


def configure_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 360,
    })


def excel_dates(values: np.ndarray) -> np.ndarray:
    epoch = datetime(1899, 12, 30)
    return np.array([epoch + timedelta(days=float(v)) for v in values])


def time_axis(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / 6.0


def style_time_axis(ax) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)


def save(fig, basename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"{basename}.{suffix}", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def representative_index(z) -> int:
    """Choose a data-driven day: the largest daily price range."""
    price = np.asarray(z["price"])
    return int(np.argmax(np.ptp(price, axis=1)))


def render_forecast(z) -> plt.Figure:
    """Show that both forecasts are causal approximations, not future truth."""
    dates = excel_dates(z["dates"])
    idx = representative_index(z)
    t = time_axis(z["price"].shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True, layout="constrained")
    fig.suptitle(f"代表日预测与实际值对照（{dates[idx]:%Y-%m-%d}）")

    axes[0].plot(t, z["price"][idx], color=TEAL, lw=1.7, label="实际电价")
    axes[0].plot(t, z["price_forecast"][idx], color=BLUE, lw=1.4, ls="--", label="0:00可用预测")
    axes[0].set_ylabel("电价（元/kWh）")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(t, z["net"][idx] * 6, color=TEAL, lw=1.7, label="实际净负荷")
    axes[1].plot(t, z["net_forecast"][idx] * 6, color=BLUE, lw=1.4, ls="--", label="日前预测")
    axes[1].axhline(0, color=BLACK, lw=0.7)
    axes[1].set_ylabel("净负荷功率（kW）")
    axes[1].set_xlabel("时刻（h）")
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        style_time_axis(ax)
    return fig


def render_dispatch(z) -> plt.Figure:
    """Connect price, actions, and the storage state without a misleading twin axis."""
    dates = excel_dates(z["dates"])
    idx = representative_index(z)
    t = time_axis(z["g"].shape[1])
    te = np.arange(z["E"].shape[1]) / 6.0
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.1), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.0, 1.8, 1.2, 1.0]})
    fig.suptitle(f"动态电价下的购电—储能协同响应（{dates[idx]:%Y-%m-%d}）")

    axes[0].step(t, z["price"][idx], where="mid", color=ORANGE, lw=1.5)
    axes[0].set_ylabel("电价\n（元/kWh）")

    axes[1].step(t, z["g"][idx] * 6, where="mid", color=BLUE, lw=1.5, label="计划购电")
    axes[1].step(t, z["net"][idx] * 6, where="mid", color=BLACK, lw=1.0, label="实际净负荷")
    axes[1].bar(t, z["e"][idx] * 6, width=1/6, color=VERMILION, alpha=0.82, label="紧急购电")
    axes[1].axhline(0, color=BLACK, lw=0.6)
    axes[1].set_ylabel("功率（kW）")
    axes[1].legend(frameon=False, ncol=3)

    axes[2].fill_between(t, 0, z["c"][idx] * 6, step="mid", color=ORANGE, alpha=0.55, label="充电")
    axes[2].fill_between(t, 0, -z["d"][idx] * 6, step="mid", color=TEAL, alpha=0.55, label="放电")
    axes[2].axhline(0, color=BLACK, lw=0.6)
    axes[2].set_ylabel("充放电功率\n（kW）")
    axes[2].legend(frameon=False, ncol=2)

    axes[3].plot(te, z["E"][idx] / 1000, color=BLUE, lw=1.7, label="储能电量")
    axes[3].axhline(1.2, color=BLACK, lw=0.7, ls="--", label="允许范围")
    axes[3].axhline(10.8, color=BLACK, lw=0.7, ls="--")
    axes[3].set_ylim(0, 12)
    axes[3].set_ylabel("储能电量\n（MWh）")
    axes[3].set_xlabel("时刻（h）")
    axes[3].legend(frameon=False, ncol=2)
    for ax in axes:
        style_time_axis(ax)
    return fig


def render_annual(z) -> plt.Figure:
    """Summarize additive costs and daily emergency risk in separate panels."""
    dates = excel_dates(z["dates"])
    month_keys = np.array([(d.year, d.month) for d in dates], dtype=object)
    labels, inverse = np.unique([f"{y}-{m:02d}" for y, m in month_keys], return_inverse=True)
    planned = np.array([z["planned_cost"][inverse == i].sum() for i in range(len(labels))]) / 1e4
    emergency_cost = np.array([z["emergency_cost"][inverse == i].sum() for i in range(len(labels))]) / 1e4
    daily_e = np.asarray(z["e"]).sum(axis=1) / 1000
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), layout="constrained")
    fig.suptitle("月度费用构成与逐日应急风险")

    axes[0].bar(x, planned, color=BLUE, width=0.68, label="计划购电费")
    axes[0].bar(x, emergency_cost, bottom=planned, color=VERMILION, width=0.68, label="紧急购电费")
    axes[0].set_xticks(x, [s[-2:] + "月" for s in labels])
    axes[0].set_ylabel("月累计费用（万元）")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)

    axes[1].scatter(dates, daily_e, s=11, color=VERMILION, alpha=0.75, label="每日紧急购电量")
    axes[1].axhline(np.median(daily_e), color=BLACK, lw=1, ls="--", label="日中位数")
    axes[1].set_ylabel("紧急购电量（MWh/日）")
    axes[1].set_xlabel("日期")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    return fig


def render_emergency_heatmap(z) -> plt.Figure:
    """Locate emergency energy jointly by date and time of day."""
    dates = excel_dates(z["dates"])
    e = np.asarray(z["e"])
    day_edges = mdates.date2num(list(dates) + [dates[-1] + timedelta(days=1)])
    hour_edges = np.arange(e.shape[1] + 1) / 6
    cmap = LinearSegmentedColormap.from_list(
        "q42_emergency", [WHITE, PALE_RED, ORANGE, VERMILION, "#B2182B"]
    )
    vmax = max(float(e.max()), 1e-12)
    fig, ax = plt.subplots(figsize=(7.2, 5.1), layout="constrained")
    mesh = ax.pcolormesh(hour_edges, day_edges, e, shading="flat", cmap=cmap,
                         norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax), rasterized=True)
    ax.set_title("紧急购电发生的日期—时段分布")
    ax.set_xlabel("时刻（h）")
    ax.set_ylabel("日期")
    ax.set_xticks(np.arange(0, 25, 4))
    ax.yaxis_date()
    ax.yaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%m月" if len(dates) > 2 else "%m-%d"))
    ax.grid(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("紧急购电量（kWh/10 min）")
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="从Q4-2结果生成论文图，不运行优化模型")
    parser.add_argument("solution", nargs="?", default=str(OUTPUT / "prob4-2_solution.npz"))
    args = parser.parse_args(argv)
    path = Path(args.solution)
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}；可传入Q4-2/output/smoke_solution.npz试画")
    configure_style()
    with np.load(path, allow_pickle=False) as z:
        required = {"dates", "price", "price_forecast", "net", "net_forecast",
                    "g", "c", "d", "e", "E", "planned_cost", "emergency_cost"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"结果文件缺少字段：{sorted(missing)}")
        renderers = [
            (render_forecast, "fig1_forecast_validation"),
            (render_dispatch, "fig2_dynamic_price_dispatch"),
            (render_annual, "fig3_annual_cost_emergency"),
            (render_emergency_heatmap, "fig4_emergency_heatmap"),
        ]
        for renderer, name in renderers:
            save(renderer(z), name)
    print(f"已生成4组论文图（PNG/SVG/PDF）：{FIGURES}")


if __name__ == "__main__":
    main()
