"""Exact daily interval heatmap with an explicit nonlinear scale and hourly totals."""
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from plot_support import ORANGE, RED, BLACK, date_values, calendar_axis, style, note, stamp


def render(z):
    e = z["e"]
    dates = date_values(z["dates"])
    day_edges = mdates.date2num(list(dates) + [dates[-1] + timedelta(days=1)])
    hours = np.arange(e.shape[1] + 1) / 6
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8.5), layout="constrained",
                               gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Q2 · 图3  紧急购电的日期与时段分布")
    cmap = LinearSegmentedColormap.from_list("emergency", ["#FFFFFF", "#FBE9C3", ORANGE, RED, "#682600"])
    image = a.pcolormesh(hours, day_edges, e, shading="flat", cmap=cmap,
                        norm=PowerNorm(gamma=0.5, vmin=0, vmax=float(e.max())), rasterized=True)
    a.grid(False)
    a.set_xlim(0, 24)
    a.set_xticks(np.arange(0, 25, 4))
    a.set_xlabel("时刻 (h)")
    calendar_axis(a, dates, axis="y")
    a.set_title("每格 = 1 天 × 10 分钟；色阶采用平方根映射", loc="left")
    cb = fig.colorbar(image, ax=a, pad=0.02, fraction=0.035)
    cb.set_label("紧急购电量 (kWh/10 min)")
    cb.set_ticks([0, 25, 100, 225, 400, 625, float(e.max())])
    cb.set_ticklabels(["0", "25", "100", "225", "400", "625", f"{e.max():.0f}"])
    i, j = np.unravel_index(e.argmax(), e.shape)
    a.scatter([(j + 0.5) / 6], [day_edges[i] + 0.5], s=36, facecolor="none", edgecolor=BLACK, lw=1)
    note(a, f"最大 {e[i, j]:.1f} kWh\n{dates[i]:%m-%d} {stamp(j)}",
         ((j + 0.5) / 6, day_edges[i] + 0.5), (-18, 24))
    hourly = e.reshape(e.shape[0], 24, 6).sum(axis=(0, 2)) / 1000
    b.bar(np.arange(24) + 0.5, hourly, width=0.82, color=RED)
    peak = int(hourly.argmax())
    note(b, f"累计最多：{peak:02d}:00—{peak+1:02d}:00\n{hourly[peak]:.1f} MWh",
         (peak + 0.5, hourly[peak]), (12, 10), RED)
    b.set_ylim(0, max(1, hourly.max() * 1.55))
    b.set_ylabel("2—12月累计\n紧急购电量 (MWh)")
    style(b, time=True)
    return fig
