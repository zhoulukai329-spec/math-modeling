# -*- coding: utf-8 -*-
"""问题 2 的两阶段线性规划求解器。

第一阶段（每天 0:00）：
  在净负荷场景下决定计划购电量 g，目标为
      sum_t p_t g_t + (1/S) sum_s [ 5 p_t e_{s,t} + eps(c+d) - v_T E_{s,N} ]
  其中 g 为跨场景共享的 here-and-now 决策，储能运行与紧急购电为 wait-and-see 决策。

第二阶段（实际回测）：
  固定计划 g，用当日实际净负荷重解储能/紧急购电问题，得到实际充放电量、储电量和紧急购电量。
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from data_io import (
    N, DT, ETA_C, ETA_D, E_MIN, E_MAX, C_MAX,
    EMERGENCY_MULT, THROUGHPUT_PENALTY,
)


def _solve_lp(c, A_eq_coo, b_eq, bounds, label=""):
    """统一调用 HiGHS，带基本失败检查。"""
    n_vars = len(c)
    if A_eq_coo is None or (hasattr(A_eq_coo, "nnz") and A_eq_coo.nnz == 0):
        A_eq = None
    else:
        A_eq = coo_matrix(A_eq_coo).tocsr()
    res = linprog(
        c,
        A_ub=None,
        b_ub=None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"{label} 求解失败: {res.message}")
    return res


def build_first_stage(price, scenarios, E_start, v_terminal):
    """构建并求解 0:00 计划购电 LP。

    price: (144,) 元/kWh
    scenarios: (S,144) 净负荷场景 kWh
    E_start: 当日 0:00 储电量 kWh
    v_terminal: 24:00 储能量的线性终值系数

    返回:
      g: (144,) 计划购电量
      result: 字典，含目标值与各场景变量的期望统计
    """
    S = scenarios.shape[0]
    per = 5 * N + 1  # c(N)+d(N)+w(N)+e(N)+E(N+1)
    n_vars = N + S * per

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

    def E_idx(s, t):
        return s_start(s) + 4 * N + t

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
        for _t in range(N + 1):
            bounds.append((E_MIN, E_MAX))     # E

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
        base = s_start(s)
        for t in range(N):
            add_row(
                [g_start + t, e_idx(s, t), d_idx(s, t), c_idx(s, t), w_idx(s, t)],
                [1.0, 1.0, 1.0, -1.0, -1.0],
                scenarios[s, t],
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

    res = _solve_lp(c_obj, A_eq, rhs, bounds, "问题2 第一阶段")

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
            "expected_emergency": float(np.mean([np.sum(price * e) for e in e_list])),
        }

    result = {
        "objective": float(res.fun),
        "planned_cost": float(np.sum(price * g)),
        "stats": stats,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
    }
    return g, result


def build_second_stage(price, net_actual, g, E_start, v_terminal):
    """固定计划购电量 g，按当日实际净负荷回测实际运行。

    返回:
      c,d,w,e: (144,)
      E: (145,)
      res: linprog 结果对象
    """
    per = 5 * N + 1
    n_vars = per

    def c_idx(t):
        return t

    def d_idx(t):
        return N + t

    def w_idx(t):
        return 2 * N + t

    def e_idx(t):
        return 3 * N + t

    def E_idx(t):
        return 4 * N + t

    c_obj = np.zeros(n_vars)
    for t in range(N):
        c_obj[e_idx(t)] = EMERGENCY_MULT * price[t]
        c_obj[c_idx(t)] = THROUGHPUT_PENALTY
        c_obj[d_idx(t)] = THROUGHPUT_PENALTY
    c_obj[E_idx(N)] = -v_terminal

    bounds = []
    for _t in range(N):
        bounds.append((0.0, C_MAX))    # c
    for _t in range(N):
        bounds.append((0.0, C_MAX))    # d
    for _t in range(N):
        bounds.append((0.0, None))     # w
    for _t in range(N):
        bounds.append((0.0, None))     # e
    for _t in range(N + 1):
        bounds.append((E_MIN, E_MAX))  # E

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

    for t in range(N):
        add_row(
            [e_idx(t), d_idx(t), c_idx(t), w_idx(t)],
            [1.0, 1.0, -1.0, -1.0],
            net_actual[t] - g[t],
        )
    for t in range(N):
        add_row(
            [E_idx(t + 1), E_idx(t), c_idx(t), d_idx(t)],
            [1.0, -1.0, -ETA_C, 1.0 / ETA_D],
            0.0,
        )
    add_row([E_idx(0)], [1.0], E_start)

    A_eq = coo_matrix((vals, (rows, cols)), shape=(row_idx, n_vars))
    res = _solve_lp(c_obj, A_eq, rhs, bounds, "问题2 第二阶段")

    c = res.x[c_idx(0) : c_idx(N - 1) + 1]
    d = res.x[d_idx(0) : d_idx(N - 1) + 1]
    w = res.x[w_idx(0) : w_idx(N - 1) + 1]
    e = res.x[e_idx(0) : e_idx(N - 1) + 1]
    E = res.x[E_idx(0) : E_idx(N) + 1]
    return c, d, w, e, E, res
