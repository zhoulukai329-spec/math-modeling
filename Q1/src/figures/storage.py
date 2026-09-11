"""Energy is an accumulated state: connect boundary samples, without smoothing."""
import numpy as np
import matplotlib.pyplot as plt
from plot_support import PURPLE, BLACK, style, note, stamp


def render(z):
    E = z["E"]
    t = np.arange(len(E)) / 6
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    fig.suptitle("Q1 · 图4  储能电量与容量约束")
    ax.plot(t, E, color=PURPLE, lw=1.5, marker="o", markersize=2.3)
    for value, label in [(10800, "上限 10,800 kWh"), (1200, "下限 1,200 kWh")]:
        ax.axhline(value, color=BLACK, lw=0.9, ls="--")
        ax.text(23.8, value + 160, label, ha="right", fontsize=9)
    ax.axhline(E[0], color=BLACK, lw=0.7, ls=":")
    ax.scatter([0, 24], E[[0, -1]], color=BLACK, s=30, zorder=4, clip_on=False)
    note(ax, f"0:00 = 24:00\n{E[0]:,.0f} kWh", (0, E[0]), (16, -18))
    i = int(E.argmin())
    note(ax, f"首次触下限 · {stamp(i)}", (t[i], E[i]), (-22, 26), PURPLE)
    ax.set_ylim(0, 12400)
    ax.set_ylabel("储电量 (kWh)")
    style(ax, time=True)
    return fig
