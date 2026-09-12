"""Presentation helpers shared by Q1 and Q2; no model dependencies."""
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
BLACK = "#000000"
GRAY = "#777777"


def configure():
    for font in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            break
        except ValueError:
            continue
    plt.rcParams.update({
        "axes.unicode_minus": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titlepad": 12, "axes.labelsize": 10,
        "figure.titlesize": 15, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 180, "axes.axisbelow": True,
    })


def style(ax, time=False):
    ax.grid(axis="y", color="#E2E2E2", lw=0.7)
    if time:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xlabel("时刻 (h)")


def legend(ax, **kwargs):
    return ax.legend(frameon=False, loc="upper left",
                     bbox_to_anchor=(0, 1.16), borderaxespad=0, **kwargs)


def stamp(index):
    minute = int(index) * 10
    return f"{minute // 60:02d}:{minute % 60:02d}"


def note(ax, text, xy, offset=(12, 18), color=BLACK):
    ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points",
                ha="left" if offset[0] >= 0 else "right",
                va="bottom" if offset[1] >= 0 else "top", fontsize=9,
                color=color, zorder=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 2},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8})


def date_values(serials):
    return np.array([datetime(1899, 12, 30) + timedelta(days=float(s))
                     for s in serials])


def calendar_axis(ax, dates, axis="x"):
    target = ax.xaxis if axis == "x" else ax.yaxis
    months = sorted({d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                     for d in dates})
    target.set_major_locator(FixedLocator(mdates.date2num(months)))
    target.set_major_formatter(mdates.DateFormatter("%m月"))
    bounds = (dates[0], dates[-1] + timedelta(days=1))
    if axis == "x":
        ax.set_xlim(*bounds)
        ax.set_xlabel("日期 (2025 年)")
    else:
        ax.set_ylim(*bounds)
        ax.set_ylabel("日期 (2025 年)")


def finish(fig, output, filename):
    fig.savefig(output / filename, facecolor="white")
    plt.close(fig)