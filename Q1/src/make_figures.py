"""问题 1 结果可视化"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
REPO_ROOT = Q_DIR.parent
ATTACH_DIR = REPO_ROOT / 'attachment'
OUTPUT_DIR = Q_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for f in ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

d = np.load(str(OUTPUT_DIR / 'prob1_solution.npz'))
t_min, price = d['t_min'], d['price']
load_kw, pv_kw = d['load_kw'], d['pv_kw']
L, G = d['L'], d['G']
x, c, dd, w, E = d['x'], d['c'], d['d'], d['w'], d['E']
cost = float(d['cost'])

tt = t_min / 60.0          # 小时
# 以区间起点画阶梯图，其中区间 k 对应 [k/6, (k+1)/6) 小时
t_start = np.arange(144) / 6.0

def step(ax, y, **kw):
    ax.step(np.append(t_start, 24.0), np.append(y, y[-1]), where='post', **kw)

# 图1： 电价 / 负载 / 光伏
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
a1b = a1.twinx()
a1b.plot(tt, price, color='#c0392b', lw=1.6, label='电价(元/kWh)')
a1b.set_ylabel('电价 (元/kWh)', color='#c0392b')
a1b.tick_params(axis='y', labelcolor='#c0392b')
a1b.set_ylim(0, 1.6)
a1.set_zorder(a1b.get_zorder() + 1); a1.patch.set_visible(False)
a1.set_title('图1  电价 / 小区负载 / 光伏预测功率 (24h)')
a1b.legend(loc='upper right'); a1b.grid(alpha=.3)

a2.plot(tt, load_kw, color='#2980b9', lw=1.4, label='小区负载 (kW)')
a2.plot(tt, pv_kw, color='#e67e22', lw=1.4, label='光伏预测 (kW)')
a2.set_xlabel('时刻 (h)'); a2.set_ylabel('功率 (kW)')
a2.legend(loc='upper left'); a2.grid(alpha=.3); a2.set_xlim(0, 24)
fig.tight_layout(); fig.savefig(str(OUTPUT_DIR / 'fig1_price_load_pv.png'), dpi=130)
plt.close(fig)

# 图2: 净负载 与 购电量
fig, ax = plt.subplots(figsize=(11, 4.6))
net = load_kw - pv_kw
ax.plot(tt, net, color='#7f8c8d', lw=1.3, label='净负载 L−G (kW)')
ax.plot(tt, x*6, color='#16a085', lw=1.6, label='计划购电功率 x·6 (kW)')
ax.axhline(0, color='k', lw=.7)
ax.set_title('图2  净负载与计划购电功率')
ax.set_xlabel('时刻 (h)'); ax.set_ylabel('功率 (kW)')
ax.legend(loc='upper left'); ax.grid(alpha=.3); ax.set_xlim(0, 24)
fig.tight_layout(); fig.savefig(str(OUTPUT_DIR / 'fig2_netload_purchase.png'), dpi=130)
plt.close(fig)

# 图3: 购电 / 充电 / 放电
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.step(np.append(t_start, 24), np.append(x*6, x[-1]*6), where='post',
        color='#16a085', lw=1.5, label='购电功率 x·6 (kW)')
ax.step(np.append(t_start, 24), np.append(c*6, c[-1]*6), where='post',
        color='#2980b9', lw=1.4, label='充电功率 c·6 (kW)')
ax.step(np.append(t_start, 24), np.append(-dd*6, -dd[-1]*6), where='post',
        color='#e67e22', lw=1.4, label='放电功率 −d·6 (kW)')
ax.axhline(0, color='k', lw=.7)
ax.set_title('图3  计划购电 / 储能充电 / 储能放电 功率曲线')
ax.set_xlabel('时刻 (h)'); ax.set_ylabel('功率 (kW)')
ax.legend(loc='upper left', ncol=3); ax.grid(alpha=.3); ax.set_xlim(0, 24)
fig.tight_layout(); fig.savefig(str(OUTPUT_DIR / 'fig3_buy_charge_discharge.png'), dpi=130)
plt.close(fig)

# 图4: 储能电量
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(np.arange(145)/6.0, E, color='#8e44ad', lw=1.8, label='储能电量 E (kWh)')
ax.axhline(10800, color='#c0392b', ls='--', lw=1, label='上限 10800')
ax.axhline(1200, color='#c0392b', ls='--', lw=1, label='下限 1200')
ax.axhline(6000, color='#7f8c8d', ls=':', lw=1, label='初值/终值 6000')
ax.set_title('图4  储能设备电量轨迹 (0:00 = 24:00 = 6000 kWh)')
ax.set_xlabel('时刻 (h)'); ax.set_ylabel('储电量 (kWh)')
ax.legend(loc='upper right'); ax.grid(alpha=.3); ax.set_xlim(0, 24)
fig.tight_layout(); fig.savefig(str(OUTPUT_DIR / 'fig4_storage_energy.png'), dpi=130)
plt.close(fig)

print(f'图片已生成: {OUTPUT_DIR}/fig1~fig4.png')
print(f'全天购电费: {cost:.4f} 元')
