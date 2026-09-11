"""Monthly additive costs and daily emergency extremes."""
import numpy as np
import matplotlib.pyplot as plt
from plot_support import GREEN, RED, BLACK, date_values, style, legend, note, calendar_axis


def render(z):
    dates = date_values(z["dates"])
    months = np.array([d.month for d in dates])
    groups = np.unique(months)
    planned = np.array([z["planned_cost"][months == m].sum() for m in groups]) / 1e4
    emergency = np.array([z["emergency_cost"][months == m].sum() for m in groups]) / 1e4
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.8), layout="constrained")
    fig.suptitle("Q2 · 图1  月度费用构成与逐日应急风险")
    a.bar(groups, planned, width=0.65, color=GREEN, label="计划购电费")
    a.bar(groups, emergency, bottom=planned, width=0.65, color=RED, label="紧急购电费")
    a.set_xticks(groups, [f"{m}月" for m in groups])
    a.set_ylabel("月累计费用 (万元)")
    a.set_ylim(0, (planned + emergency).max() * 1.32)
    total = z["planned_cost"].sum() + z["emergency_cost"].sum()
    share = z["emergency_cost"].sum() / total
    a.text(0.5, 0.97, f"2—12月总费用 {total / 1e4:,.2f} 万元  |  紧急费用占比 {share:.1%}",
           transform=a.transAxes, ha="center", va="top", fontsize=10)
    i = int(np.argmax(planned + emergency))
    note(a, f"最高 {planned[i] + emergency[i]:.1f} 万元",
         (groups[i], planned[i] + emergency[i]), (-12, 10))
    legend(a, ncol=2)
    daily = z["e"].sum(axis=1) / 1000
    b.scatter(dates, daily, color=RED, s=13, alpha=0.8, label="每日紧急购电量")
    median = float(np.median(daily))
    b.axhline(median, color=BLACK, lw=1, ls="--", label=f"日中位数 {median:.2f} MWh")
    i = int(daily.argmax())
    note(b, f"{dates[i]:%m-%d} · {daily[i]:.2f} MWh\n当日总费用 {z['total_cost'][i] / 1e4:.2f} 万元",
         (dates[i], daily[i]), (20, -12), RED)
    b.set_ylim(0, daily.max() * 1.24)
    b.set_ylabel("日紧急购电量 (MWh)")
    calendar_axis(b, dates)
    legend(b, ncol=2)
    style(a)
    style(b)
    return fig
