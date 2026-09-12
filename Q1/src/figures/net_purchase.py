"""Compare discrete purchases with the signed net demand."""
import numpy as np
import matplotlib.pyplot as plt
from plot_support import GREEN, ORANGE, BLACK, style, legend, note, stamp


def render(z):
    t = z["t_min"] / 60 + 1 / 12
    net = z["load_kw"] - z["pv_kw"]
    buy = z["x"] * 6
    fig, ax = plt.subplots(figsize=(12, 5.6), layout="constrained")
    fig.suptitle("储能调节前后的供电需求")
    ax.bar(t, buy, width=0.14, color=GREEN, alpha=0.8, label="计划购电（每柱 10 分钟）")
    ax.scatter(t, net, s=16, facecolor="white", edgecolor=BLACK,
               linewidth=0.8, label="净负荷（负荷 − 光伏）", zorder=4)
    ax.scatter(t[net < 0], net[net < 0], s=16, color=ORANGE,
               label="光伏富余", zorder=5)
    ax.axhline(0, color=BLACK, lw=0.8)
    i = int(net.argmin())
    note(ax, f"最大光伏富余 {-net[i]:,.0f} kW · {stamp(i)}",
         (t[i], net[i]), (18, -14), ORANGE)
    i = int(buy.argmax())
    note(ax, f"购电峰值 {buy[i]:,.0f} kW · {stamp(i)}",
         (t[i], buy[i]), (20, 15), GREEN)
    ax.set_ylim(net.min() - 1800, buy.max() * 1.22)
    ax.set_ylabel("功率 (kW)")
    legend(ax, ncol=3)
    style(ax, time=True)
    return fig
