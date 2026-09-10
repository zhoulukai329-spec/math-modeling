# -*- coding: utf-8 -*-
"""
问题 1 求解：微网计划购电策略（确定性线性规划 LP）

场景：单日，已知（逐 10 分钟）电价 p_t、小区负载 L_t（kW）、光伏预测功率 G_t（kW）。
目标：最小化当日购电费  min Σ p_t · x_t
约束：功率平衡、储能动态、储能上下限、充放电功率上限、0:00 与 24:00 储电量相等。

储能参数（附录 1）：
  E_cap = 12000 kWh   最大容量
  E_min = 1200, E_max = 10800 kWh   电量允许区间
  P_max = 5000 kW     最大充/放电功率
  E0    = 6000 kWh    0:00 初值
  eta   = 0.9         充放电效率（η_c = η_d = 0.9，往返效率 0.81）
"""
import numpy as np
from scipy.optimize import linprog
import xlsx_reader as xr

# ---------------- 数据读取 ----------------
rows = xr.read_sheet_rows('attachment/附件1.xlsx')['Sheet1']
# 表头: 时间, 电价, 小区负载, 光伏发电预测功率
header = rows[0]
data = rows[1:]          # 144 行, 对应 144 个 10 分钟区间
assert len(data) == 144, f"期望 144 个区间, 实际 {len(data)}"

def parse_time(t):
    """把附件 1 的时间列(数值型天分数 或 'HH:MM' 文本)转成 距 0:00 的分钟数。"""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return round(float(t) * 24 * 60)      # 天分数 -> 分钟
    s = str(t).strip()
    offset = 0
    if '+' in s:                               # '0:00+1' 表示次日 0:00 = 当日 24:00
        s, off = s.split('+')
        offset = int(off) * 1440
    hh, mm = s.split(':')
    return int(hh) * 60 + int(mm) + offset

t_min = np.array([parse_time(r[0]) for r in data], dtype=float)   # 各区间结束时刻(分钟)
price = np.array([float(r[1]) for r in data])                     # 元/kWh
load_kw  = np.array([float(r[2]) for r in data])                  # kW
pv_kw    = np.array([float(r[3]) for r in data])                  # kW

# 确认时间列从 10 分钟递增到 1440 分钟
assert t_min[0] == 10 and t_min[-1] == 1440, (t_min[0], t_min[-1])
assert np.all(np.diff(t_min) == 10), "时间步长应为 10 分钟"

DT = 1 / 6.0                       # 区间长度 = 10 min = 1/6 h
L = load_kw * DT                   # 负载能量 kWh/区间
G = pv_kw * DT                     # 光伏能量 kWh/区间

# ---------------- 参数 ----------------
N = 144                            # 区间数
eta_c = 0.9                        # 充电效率
eta_d = 0.9                        # 放电效率
E_cap, E_min, E_max = 12000.0, 1200.0, 10800.0
P_max = 5000.0                     # kW
C_max = P_max * DT                 # 每区间最大充/放电能量 (kWh) = 833.333
E0 = 6000.0                        # 0:00 储电量

# ---------------- 决策变量 ----------------
# 每区间 t: x[t](购电), c[t](充电量), d[t](放电量), w[t](弃光量)   (均为 kWh)
# 储能状态 E[t], t=0..N (E[0] 为 0:00, E[N] 为 24:00), 共 N+1 个
nv_x = N; nv_c = N; nv_d = N; nv_w = N; nv_E = N + 1
def ix(t):  return t
def ic(t):  return nv_x + t
def id_(t): return nv_x + nv_c + t
def iw(t):  return nv_x + nv_c + nv_d + t
def iE(t):  return nv_x + nv_c + nv_d + nv_w + t
n_vars = nv_x + nv_c + nv_d + nv_w + nv_E

# 目标: min Σ p_t · x_t  (c 系数)
c_obj = np.zeros(n_vars)
for t in range(N):
    c_obj[ix(t)] = price[t]

# ---------------- 约束: A_ub x <= b_ub , A_eq x == b_eq ----------------
A_ub, b_ub = [], []
A_eq, b_eq = [], []

def add_eq(row, rhs):
    A_eq.append(row); b_eq.append(rhs)
def add_ub(row, rhs):
    A_ub.append(row); b_ub.append(rhs)

# (1) 母线功率平衡: x_t + G_t + d_t - c_t - w_t = L_t   (购电+光伏+放电 = 负载+充电+弃光)
for t in range(N):
    row = np.zeros(n_vars)
    row[ix(t)] = 1.0
    row[id_(t)] = 1.0
    row[ic(t)] = -1.0
    row[iw(t)] = -1.0
    add_eq(row, L[t] - G[t])

# (2) 储能动态: E_{t+1} = E_t + η_c·c_t − d_t/η_d
for t in range(N):
    row = np.zeros(n_vars)
    row[iE(t+1)] = 1.0
    row[iE(t)]   = -1.0
    row[ic(t)]   = -eta_c
    row[id_(t)]  = 1.0 / eta_d
    add_eq(row, 0.0)

# (3) 储能电量上下限: E_min <= E_t <= E_max  (t=0..N)
for t in range(N + 1):
    row = np.zeros(n_vars); row[iE(t)] = 1.0
    add_ub(row, E_max)
    row = np.zeros(n_vars); row[iE(t)] = -1.0
    add_ub(row, -E_min)

# (4) 边界条件: E[0] = E0, E[N] = E0  (0:00 与 24:00 储电量相同)
row = np.zeros(n_vars); row[iE(0)] = 1.0; add_eq(row, E0)
row = np.zeros(n_vars); row[iE(N)] = 1.0; add_eq(row, E0)

# (5) 变量上下界
lb = np.zeros(n_vars)
ub = np.full(n_vars, np.inf)
for t in range(N):
    ub[ic(t)] = C_max      # 充电量 <= 833.333 kWh/区间
    ub[id_(t)] = C_max     # 放电量 <= 833.333 kWh/区间
for t in range(N + 1):
    lb[iE(t)] = -np.inf    # E 由约束(3)限制
    ub[iE(t)] = np.inf
# x, w 下界 0, 上界 inf (已由 lb=0 保证)

# ---------------- 求解 ----------------
res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
              A_eq=np.array(A_eq), b_eq=np.array(b_eq),
              bounds=list(zip(lb, ub)), method='highs')

print('求解状态:', res.message)
print('目标值(全天购电费, 元):', res.fun)
if not res.success:
    raise RuntimeError('LP 求解失败: ' + res.message)

x_sol = res.x[ix(0):ix(N-1)+1]
c_sol = res.x[ic(0):ic(N-1)+1]
d_sol = res.x[id_(0):id_(N-1)+1]
w_sol = res.x[iw(0):iw(N-1)+1]
E_sol = res.x[iE(0):iE(N)+1]

# ---------------- 校验 ----------------
def report():
    bal = x_sol + G + d_sol - c_sol - w_sol - L
    print('\n=== 校验 ===')
    print('功率平衡最大残差 (kWh):', np.max(np.abs(bal)))
    print('弃光量总和 (kWh):', w_sol.sum(), '  弃光区间数:', (w_sol > 1e-6).sum())
    print('同时充放电的区间数:', ((c_sol > 1e-6) & (d_sol > 1e-6)).sum())
    print('储能初值/终值:', E_sol[0], E_sol[-1])
    print('储能 min/max:', E_sol.min(), E_sol.max())
    print('充电量总和:', c_sol.sum(), ' 放电量总和:', d_sol.sum())
    print('全天购电量(kWh):', x_sol.sum(), ' 全天购电费(元):', res.fun)
    print('光伏总发电(kWh):', G.sum(), ' 负载总耗电(kWh):', L.sum())
report()

# 保存供后续步骤使用
np.savez('output/prob1_solution.npz', t_min=t_min, price=price, load_kw=load_kw,
         pv_kw=pv_kw, L=L, G=G, x=x_sol, c=c_sol, d=d_sol, w=w_sol, E=E_sol,
         cost=res.fun)
print('\n已保存 output/prob1_solution.npz')
