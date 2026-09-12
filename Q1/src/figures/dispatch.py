"""Separate purchase and signed battery actions to avoid crowded grouped bars."""
import numpy as np
import matplotlib.pyplot as plt
from plot_support import GREEN, BLUE, RED, BLACK, style, legend, note


def render(z):
    t = z["t_min"] / 60 + 1 / 12
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.2), layout="constrained")
    fig.suptitle("逐 10 分钟购电与储能动作")
    a.bar(t, z["x"] * 6, width=0.14, color=GREEN, label="计划购电")
    a.set_ylabel("购电功率 (kW)")
    a.set_ylim(0, z["x"].max() * 6 * 1.23)
    a.text(0.5, 0.92, f"全天购电 {z['x'].sum() / 1000:.2f} MWh",
           transform=a.transAxes, ha="center", va="top", color=GREEN)
    b.bar(t, z["c"] * 6, width=0.14, color=BLUE, label="充电（正）")
    b.bar(t, -z["d"] * 6, width=0.14, color=RED, label="放电（负）")
    b.axhline(0, color=BLACK, lw=0.8)
    b.set_ylim(-6500, 6500)
    b.set_ylabel("储能功率 (kW)")
    b.text(0.48, 0.97, f"充电 {z['c'].sum() / 1000:.2f} MWh",
           transform=b.transAxes, ha="center", va="top", color=BLUE)
    b.text(0.72, 0.04, f"全天放电 {z['d'].sum() / 1000:.2f} MWh",
           transform=b.transAxes, ha="center", va="bottom", color=RED)
    for ax in (a, b):
        style(ax, time=True)
        legend(ax, ncol=2)
    return fig
