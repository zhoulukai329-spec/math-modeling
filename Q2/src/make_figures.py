# -*- coding: utf-8 -*-
"""问题 2 结果可视化。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_io as dio


for _font in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def _load():
    return np.load(str(dio.SOLUTION_NPZ))


def _serial_to_month_axis(dates):
    # 将 Excel 序列号转为 2025 年的月分数（1.0=1月1日，12.0=12月初）
    base = dates[0]
    day_of_year = dates - 45658.0  # 2025-01-01 = 45658
    return 1.0 + day_of_year / 30.44


def fig_annual_cost_emergency(z):
    dates = z["dates"]
    planned = z["planned_cost"]
    emergency_cost = z["emergency_cost"]
    e_daily = z["e"].sum(axis=1)
    x = _serial_to_month_axis(dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(x, planned / 1e4, color="#16a085", lw=1.2, label="计划购电费")
    ax1.plot(x, emergency_cost / 1e4, color="#c0392b", lw=1.0, alpha=0.85, label="紧急购电费")
    ax1.plot(x, (planned + emergency_cost) / 1e4, color="#2c3e50", lw=1.0, alpha=0.5, label="总购电费")
    ax1.set_ylabel("费用 (万元)")
    ax1.set_title("图1  2025-02-01~12-31 逐日购电费用")
    ax1.legend(loc="upper left", ncol=3)
    ax1.grid(alpha=0.3)

    ax2.bar(x, e_daily / 1e3, width=0.55, color="#e67e22", label="紧急购电量")
    ax2.set_ylabel("紧急购电量 (MWh)")
    ax2.set_xlabel("月份 (2025 年)")
    ax2.set_title("逐日紧急购电量")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(dio.OUTPUT_DIR / "fig1_annual_cost_emergency.png"), dpi=130)
    plt.close(fig)


def fig_specified_days(z):
    target_serials = [45736, 45829, 45923, 46012]
    target_names = ["2025.3.20", "2025.6.21", "2025.9.23", "2025.12.21"]
    dates, price = z["dates"], z["price"]
    net, g, e, E = z["net"], z["g"], z["e"], z["E"]
    t_start = np.arange(dio.N) / 6.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, name, serial in zip(axes.ravel(), target_names, target_serials):
        i = int(np.where(np.isclose(dates, serial))[0][0])
        ax.plot(t_start, net[i] * 6, color="#7f8c8d", lw=1.1, label="实际净负荷 L−G")
        ax.plot(t_start, g[i] * 6, color="#16a085", lw=1.5, label="计划购电功率")
        em = e[i] * 6
        if em.sum() > 1e-8:
            ax.fill_between(
                t_start, 0, em, color="#c0392b", alpha=0.55, step="post",
                label="紧急购电功率",
            )
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(name)
        ax.set_ylabel("功率 (kW)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-3500, 9000)
        if ax is axes.ravel()[0]:
            ax.legend(loc="upper left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("时刻 (h)")
    fig.suptitle("图2  指定日期的实际净负荷、计划购电与紧急购电")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(dio.OUTPUT_DIR / "fig2_specified_days_dispatch.png"), dpi=130)
    plt.close(fig)


def fig_emergency_heatmap(z):
    e = z["e"]  # (334,144) kWh
    hours = np.arange(dio.N + 1) / 6.0
    day_axis = _serial_to_month_axis(z["dates"])
    day_edges = np.linspace(day_axis[0], day_axis[-1] + 1 / 30.44, len(day_axis) + 1)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.pcolormesh(
        hours, day_edges, e, shading="auto", cmap="YlOrRd", vmin=0, vmax=800
    )
    ax.set_xlim(0, 24)
    ax.set_ylim(day_edges[0], day_edges[-1])
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("月份 (2025 年)")
    ax.set_title("图3  逐日逐时段紧急购电量热力图 (kWh/10min)")
    cb = fig.colorbar(im, ax=ax, label="紧急购电量 (kWh)")
    fig.tight_layout()
    fig.savefig(str(dio.OUTPUT_DIR / "fig3_emergency_heatmap.png"), dpi=130)
    plt.close(fig)


def fig_storage_soc(z):
    E = z["E"]  # (334,145)
    x = _serial_to_month_axis(z["dates"])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(x, E.min(axis=1), E.max(axis=1), color="#9b59b6", alpha=0.2, label="日内电量范围")
    ax.plot(x, E[:, 0], color="#2980b9", lw=1.0, label="0:00 储电量")
    ax.plot(x, E[:, -1], color="#8e44ad", lw=1.0, label="24:00 储电量")
    ax.axhline(dio.E_MAX, color="#c0392b", ls="--", lw=1, label="上限 10800 kWh")
    ax.axhline(dio.E_MIN, color="#c0392b", ls="--", lw=1, label="下限 1200 kWh")
    ax.set_xlabel("月份 (2025 年)")
    ax.set_ylabel("储电量 (kWh)")
    ax.set_title("图4  储能设备电量范围与 0:00/24:00 储电量")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(dio.OUTPUT_DIR / "fig4_storage_soc.png"), dpi=130)
    plt.close(fig)


def main():
    z = _load()
    fig_annual_cost_emergency(z)
    fig_specified_days(z)
    fig_emergency_heatmap(z)
    fig_storage_soc(z)
    print("图片已生成: Q2/output/fig1~fig4.png")


if __name__ == "__main__":
    main()
