# -*- coding: utf-8 -*-
"""问题 2 的两阶段线性规划求解器。

第一阶段（每天 0:00）：
  在净负荷场景下决定计划购电量 g，目标为
      sum_t p_t g_t + (1/S) sum_s [ 5 p_t e_{s,t} + eps(c+d) - v_T E_{s,N} ]
  其中 g 为跨场景共享的 here-and-now 决策，储能运行与紧急购电为 wait-and-see 决策。

第二阶段（实际回测）：
  固定计划 g，逐时段因果调度（只观察当期实际净负荷与当前储电量），
  紧急购电只弥补当期缺口、禁止给储能充电，得到实际充放电量、储电量和紧急购电量。
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from data_io import (
    N, DT, ETA_C, ETA_D, E_MIN, E_MAX, C_MAX,
    EMERGENCY_MULT, THROUGHPUT_PENALTY,
)


def _solve_lp(c, A_eq_coo, b_eq, bounds, label="", A_ub_coo=None, b_ub=None,
              integrality=None):
    """统一调用 HiGHS，带基本失败检查；支持不等式与整数变量。"""

    def _to_csr(coo):
        if coo is None or (hasattr(coo, "nnz") and coo.nnz == 0):
            return None
        return coo_matrix(coo).tocsr()

    res = linprog(
        c,
        A_ub=_to_csr(A_ub_coo),
        b_ub=b_ub,
        A_eq=_to_csr(A_eq_coo),
        b_eq=b_eq,
        bounds=bounds,
        integrality=integrality,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"{label} 求解失败: {res.message}")
    return res


def build_first_stage(price, scenarios, E_start, v_terminal):
    """构建并求解 0:00 计划购电 LP（两阶段随机线性规划第一阶段）。

    price: (144,) 元/kWh
    scenarios: (S,144) 净负荷场景 kWh
    E_start: 当日 0:00 储电量 kWh
    v_terminal: 24:00 储能量的线性终值系数

    语义约束：对场景净负荷先拆成正净需求和光伏富余。
    计划电与光伏富余只能分配给当期负荷、充电或弃电；
    紧急电只出现在负荷供给方程中，因此不可能进入充电链路。

    返回:
      g: (144,) 计划购电量
      result: 字典，含目标值与各场景变量的期望统计
    """
    S = scenarios.shape[0]
    per = 6 * N + 1  # c(N)+d(N)+w(N)+e(N)+u(N)+E(N+1)
    n_z = S * N
    n_vars = N + S * per + n_z

    g_start = 0

    def s_start(s):
        return N + s * per

    def c_idx(s, t):
        return s_start(s) + t

    def d_idx(s, t):
        return s_start(s) + N + t

    def w_idx(s, t):
        return s_start(s) + 2 * N + t

    def e_idx(s, t):
        return s_start(s) + 3 * N + t

    def u_idx(s, t):
        return s_start(s) + 4 * N + t

    def E_idx(s, t):
        return s_start(s) + 5 * N + t

    def z_idx(s, t):
        return N + S * per + s * N + t

    M_e = float(max(np.max(scenarios), 0.0)) + 1.0

    c_obj = np.zeros(n_vars)
    for t in range(N):
        c_obj[g_start + t] = price[t]
    for s in range(S):
        for t in range(N):
            c_obj[e_idx(s, t)] = EMERGENCY_MULT * price[t] / S
            c_obj[c_idx(s, t)] = THROUGHPUT_PENALTY / S
            c_obj[d_idx(s, t)] = THROUGHPUT_PENALTY / S
        c_obj[E_idx(s, N)] = -v_terminal / S

    bounds = [(0.0, None)] * N
    for _s in range(S):
        for _t in range(N):
            bounds.append((0.0, C_MAX))       # c
        for _t in range(N):
            bounds.append((0.0, C_MAX))       # d
        for _t in range(N):
            bounds.append((0.0, None))        # w
        for _t in range(N):
            bounds.append((0.0, None))        # e
        for _t in range(N):
            bounds.append((0.0, None))        # u：计划电/光伏直接供负荷
        for _t in range(N + 1):
            bounds.append((E_MIN, E_MAX))     # E
    bounds += [(0.0, 1.0)] * n_z

    integrality = np.zeros(n_vars, dtype=int)
    integrality[N + S * per:] = 1

    rows = []
    cols = []
    vals = []
    rhs = []
    row_idx = 0

    def add_row(col_indices, coefs, rhs_val):
        nonlocal row_idx
        for col, val in zip(col_indices, coefs):
            rows.append(row_idx)
            cols.append(col)
            vals.append(val)
        rhs.append(rhs_val)
        row_idx += 1

    for s in range(S):
        for t in range(N):
            demand = max(float(scenarios[s, t]), 0.0)
            renewable_surplus = max(-float(scenarios[s, t]), 0.0)
            # 计划电+净光伏富余 = 直供负荷+充电+弃电
            add_row(
                [g_start + t, u_idx(s, t), c_idx(s, t), w_idx(s, t)],
                [1.0, -1.0, -1.0, -1.0],
                -renewable_surplus,
            )
            # 净负荷需求 = 计划电/光伏直供+放电+紧急电
            add_row(
                [u_idx(s, t), d_idx(s, t), e_idx(s, t)],
                [1.0, 1.0, 1.0],
                demand,
            )
        for t in range(N):
            add_row(
                [E_idx(s, t + 1), E_idx(s, t), c_idx(s, t), d_idx(s, t)],
                [1.0, -1.0, -ETA_C, 1.0 / ETA_D],
                0.0,
            )
        add_row([E_idx(s, 0)], [1.0], E_start)

    n_rows = row_idx
    A_eq = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_vars))

    # e>0 时 z=1，并强制 c=0；从而禁止紧急电直接或间接充电。
    ub_rows, ub_cols, ub_vals, ub_rhs = [], [], [], []

    def add_ub(col_indices, coefs, rhs_val):
        row = len(ub_rhs)
        for col, val in zip(col_indices, coefs):
            ub_rows.append(row)
            ub_cols.append(col)
            ub_vals.append(val)
        ub_rhs.append(rhs_val)

    for s in range(S):
        for t in range(N):
            add_ub([e_idx(s, t), z_idx(s, t)], [1.0, -M_e], 0.0)
            add_ub([c_idx(s, t), z_idx(s, t)], [1.0, C_MAX], C_MAX)

    A_ub = coo_matrix(
        (ub_vals, (ub_rows, ub_cols)), shape=(len(ub_rhs), n_vars)
    )
    res = _solve_lp(
        c_obj, A_eq, rhs, bounds, "问题2 第一阶段",
        A_ub_coo=A_ub, b_ub=ub_rhs, integrality=integrality,
    )

    g = res.x[g_start : g_start + N]
    stats = {}
    if S > 0:
        e_list, c_list, d_list, E_list = [], [], [], []
        for s in range(S):
            e_list.append(res.x[e_idx(s, 0) : e_idx(s, N - 1) + 1])
            c_list.append(res.x[c_idx(s, 0) : c_idx(s, N - 1) + 1])
            d_list.append(res.x[d_idx(s, 0) : d_idx(s, N - 1) + 1])
            E_list.append(res.x[E_idx(s, 0) : E_idx(s, N) + 1])
        stats = {
            "e_mean": np.mean(e_list, axis=0),
            "c_mean": np.mean(c_list, axis=0),
            "d_mean": np.mean(d_list, axis=0),
            "E_mean": np.mean(E_list, axis=0),
            "expected_emergency": float(np.mean([np.sum(EMERGENCY_MULT * price * e) for e in e_list])),
            "emergency_charge_overlap": int(np.sum(
                (np.asarray(e_list) > 1e-8) & (np.asarray(c_list) > 1e-8)
            )),
        }

    result = {
        "objective": float(res.fun),
        "planned_cost": float(np.sum(price * g)),
        "stats": stats,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "model_type": "two_stage_stochastic_milp",
    }
    return g, result


def causal_dispatch(net_actual, g, E_start, discharge_reference=None):
    """因果逐时调度（第二阶段真实回测）。

    固定计划购电量 g，逐时段只用当期实际净负荷 net_actual[t]、当前储电量
    和 0:00 已确定的参考放电量 discharge_reference[t]，不读取未来实际值。
    紧急购电只弥补当期缺口，禁止进入充电链路：

      r = net_actual[t] - g[t]
      r > 0（缺口）：放电不超过参考量、缺口、功率与当前可用电量；紧急 e = r - d；
                     充电 c = 0，弃光 w = 0。
      r <= 0（富余）：充电 c = min(-r, C_MAX, (E_MAX-E)/ETA_C)；弃光 w = -r - c；
                     放电 d = 0，紧急 e = 0。

    返回 c, d, w, e: (144,)，E: (145,)。构造上保证逐时段能量平衡
    g + e + d - c - w = net_actual、SOC 动态一致、上下限与非负性成立。
    """
    n = len(net_actual)
    if discharge_reference is None:
        discharge_reference = np.full(n, C_MAX)
    discharge_reference = np.asarray(discharge_reference, dtype=float)
    if discharge_reference.shape != (n,):
        raise ValueError(f"discharge_reference 期望形状 {(n,)}，实际 {discharge_reference.shape}")
    c = np.zeros(n)
    d = np.zeros(n)
    w = np.zeros(n)
    e = np.zeros(n)
    E = np.zeros(n + 1)
    E[0] = float(E_start)
    for t in range(n):
        r = float(net_actual[t]) - float(g[t])
        if r > 0.0:
            d[t] = min(
                r,
                max(0.0, discharge_reference[t]),
                C_MAX,
                (E[t] - E_MIN) * ETA_D,
            )
            e[t] = r - d[t]
        else:
            s = -r
            c[t] = min(s, C_MAX, (E_MAX - E[t]) / ETA_C)
            w[t] = s - c[t]
        E[t + 1] = E[t] + ETA_C * c[t] - d[t] / ETA_D
    return c, d, w, e, E


def dispatch_cost(price, g, e):
    """给定固定计划 g 与紧急购电 e，返回 (计划费, 紧急费, 总费)。"""
    planned = float(np.sum(price * g))
    emergency = float(np.sum(EMERGENCY_MULT * price * e))
    return planned, emergency, planned + emergency
