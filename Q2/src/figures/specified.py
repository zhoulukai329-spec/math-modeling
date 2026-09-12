"""Daily dispatch bars and net-demand observations; emergency has its own scale."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from plot_support import GREEN, RED, BLACK, date_values, style


def render(z):
    dates = date_values(z["dates"])
    targets = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    lookup = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    indices = [lookup[d] for d in targets]
    t = np.arange(z["net"].shape[1]) / 6 + 1 / 12
    buy, net, emergency = z["g"] * 6, z["net"] * 6, z["e"] * 6
    low = min(0, net[indices].min()) - 1000
    high = max(buy[indices].max(), net[indices].max()) + 1800
    emax = max(1, emergency[indices].max() * 1.35)
    fig = plt.figure(figsize=(14, 9.5), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[2.5, 1, 2.5, 1], hspace=0.16)
    fig.suptitle("四个指定日的计划购电与实时缺口")
    for n, (target, i) in enumerate(zip(targets, indices)):
        row, col = (n // 2) * 2, n % 2
        a = fig.add_subplot(grid[row, col])
        b = fig.add_subplot(grid[row + 1, col])
        a.bar(t, buy[i], width=0.14, color=GREEN, alpha=0.85)
        a.scatter(t, net[i], s=9, facecolor="white", edgecolor=BLACK, lw=0.6, zorder=4)
        a.axhline(0, color=BLACK, lw=0.7)
        a.set_ylim(low, high)
        a.set_title(target, loc="left")
        a.set_ylabel("功率 (kW)")
        a.text(0.98, 0.97, f"计划购电 {z['g'][i].sum() / 1000:.2f} MWh",
               transform=a.transAxes, ha="right", va="top", color=GREEN, fontsize=9)
        b.bar(t, emergency[i], width=0.14, color=RED)
        b.set_ylim(0, emax)
        b.set_ylabel("紧急购电\n功率 (kW)")
        b.text(0.98, 0.94, f"紧急购电 {z['e'][i].sum() / 1000:.2f} MWh",
               transform=b.transAxes, ha="right", va="top", color=RED, fontsize=9)
        for ax in (a, b):
            style(ax, time=True)
        a.set_xlabel("")
        a.tick_params(labelbottom=False)
        if n == 0:
            a.legend(handles=[Patch(facecolor=GREEN, label="计划购电（10 分钟柱）"),
                              Line2D([], [], color=BLACK, marker="o", linestyle="none",
                                     markerfacecolor="white", markersize=4, label="实际净负荷")],
                     frameon=False, ncol=2, loc="lower left")
    return fig
