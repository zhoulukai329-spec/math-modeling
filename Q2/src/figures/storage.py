"""Across-day quantiles at each time plus unconnected midnight states."""
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from plot_support import SKY, BLUE, GREEN, BLACK, date_values, calendar_axis, style, legend, note


def render(z):
    E = z["E"]
    dates = date_values(z["dates"])
    h = np.arange(E.shape[1]) / 6
    q10, q25, median, q75, q90 = np.quantile(E, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0) / 1000
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.8), layout="constrained")
    fig.suptitle("Q2 · 图4  日内储能分布与跨日衔接")
    a.vlines(h, q10, q90, color=SKY, lw=3.8, alpha=0.55, label="同一时刻 P10—P90")
    a.vlines(h, q25, q75, color=BLUE, lw=2.5, label="同一时刻 P25—P75")
    a.scatter(h, median, s=9, color=BLACK, zorder=4, label="中位数")
    for val, label in [(1.2, "下限 1.2 MWh"), (10.8, "上限 10.8 MWh")]:
        a.axhline(val, color=BLACK, ls="--", lw=0.8)
        a.text(23.8, val + 0.18, label, ha="right", fontsize=9)
    a.set_ylim(0, 12)
    a.set_ylabel("储电量 (MWh)")
    style(a, time=True)
    legend(a, ncol=3)
    # Day-end d and day-start d+1 coincide by construction, so plot them once.
    b.scatter(dates, E[:, 0] / 1000, s=14, color=GREEN, label="每日 0:00（等于前一日 24:00）")
    b.scatter([dates[-1] + timedelta(days=1)], [E[-1, -1] / 1000],
              s=22, marker="s", color=BLACK, clip_on=False, label="12月31日 24:00")
    b.axhline(10.8, color=BLACK, ls="--", lw=0.8)
    i = int(E[:, 0].argmin())
    note(b, f"最低日界电量 {E[i, 0] / 1000:.2f} MWh\n{dates[i]:%m-%d} 00:00",
         (dates[i], E[i, 0] / 1000), (18, -14), GREEN)
    b.set_ylim(6, 11.5)
    b.set_ylabel("日界储电量 (MWh)")
    calendar_axis(b, dates)
    style(b)
    legend(b, ncol=2)
    return fig
