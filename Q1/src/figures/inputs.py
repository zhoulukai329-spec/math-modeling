"""Tariff intervals and load/PV observations."""
import numpy as np
import matplotlib.pyplot as plt
from plot_support import BLUE, ORANGE, RED, style, legend, note, stamp


def render(z):
    edges = np.r_[z["t_min"] / 60, 24.0]
    t = edges[:-1] + 1 / 12
    price, load, pv = z["price"], z["load_kw"], z["pv_kw"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.2), layout="constrained",
                               gridspec_kw={"height_ratios": [1, 1.5]})
    fig.suptitle("Q1 · 图1  电价与供需的日内结构")
    a.stairs(price, edges, color=RED, lw=1.5, baseline=None)
    a.set_ylabel("电价 (元/kWh)")
    a.set_ylim(0, price.max() * 1.34)
    i, j = int(price.argmax()), int(price.argmin())
    note(a, f"最高 {price[i]:.4f} 元/kWh · {stamp(i)}", (t[i], price[i]), (-12, 22), RED)
    note(a, f"最低 {price[j]:.4f} 元/kWh · {stamp(j)}", (t[j], price[j]), (10, 18), RED)
    b.scatter(t, load, s=14, color=BLUE, label="小区负荷")
    b.scatter(t, pv, s=15, marker="^", color=ORANGE, label="光伏预测")
    b.set_ylabel("功率 (kW)")
    b.set_ylim(-350, max(load.max(), pv.max()) * 1.25)
    i = int(pv.argmax())
    note(b, f"光伏峰值 {pv[i]:,.0f} kW · {stamp(i)}", (t[i], pv[i]), (12, 18), ORANGE)
    legend(b, ncol=2)
    for ax in (a, b):
        style(ax, time=True)
    return fig
