# -*- coding: utf-8 -*-
"""Joint dynamic-price/net-load two-stage MILP and causal dispatch."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import data_io as dio


def _csr(matrix):
    if matrix is None or matrix.nnz == 0:
        return None
    return matrix.tocsr()


def build_first_stage(price_scenarios, net_scenarios, E_start, v_terminal):
    """Choose one day-ahead plan shared by all joint future scenarios."""
    prices = np.asarray(price_scenarios, dtype=float)
    net = np.asarray(net_scenarios, dtype=float)
    if prices.ndim == 1:
        prices = np.repeat(prices[None, :], len(net), axis=0)
    if prices.shape != net.shape or net.ndim != 2 or net.shape[1] != dio.N:
        raise ValueError("价格场景和净负荷场景必须同为(S,144)")
    if len(net) == 0 or not np.isfinite(prices).all() or not np.isfinite(net).all():
        raise ValueError("场景不能为空且不得包含NaN/Inf")
    E_start = _clip_soc(E_start)

    S, N = net.shape
    per = 6 * N + 1
    n_binary = S * N
    n_vars = N + S * per + n_binary

    def base(s): return N + s * per
    def c_i(s, t): return base(s) + t
    def d_i(s, t): return base(s) + N + t
    def w_i(s, t): return base(s) + 2 * N + t
    def e_i(s, t): return base(s) + 3 * N + t
    def u_i(s, t): return base(s) + 4 * N + t
    def E_i(s, t): return base(s) + 5 * N + t
    def z_i(s, t): return N + S * per + s * N + t

    objective = np.zeros(n_vars)
    objective[:N] = prices.mean(axis=0)
    for s in range(S):
        for t in range(N):
            objective[e_i(s, t)] = dio.EMERGENCY_MULT * prices[s, t] / S
            objective[c_i(s, t)] = dio.THROUGHPUT_PENALTY / S
            objective[d_i(s, t)] = dio.THROUGHPUT_PENALTY / S
        objective[E_i(s, N)] = -float(v_terminal) / S

    bounds = [(0.0, None)] * N
    for _ in range(S):
        bounds += [(0.0, dio.C_MAX)] * N
        bounds += [(0.0, dio.C_MAX)] * N
        bounds += [(0.0, None)] * N
        bounds += [(0.0, None)] * N
        bounds += [(0.0, None)] * N
        bounds += [(dio.E_MIN, dio.E_MAX)] * (N + 1)
    bounds += [(0.0, 1.0)] * n_binary
    integrality = np.zeros(n_vars, dtype=int)
    integrality[N + S * per:] = 1

    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    row = 0

    def add_eq(cols, vals, rhs):
        nonlocal row
        eq_r.extend([row] * len(cols)); eq_c.extend(cols); eq_v.extend(vals)
        eq_b.append(rhs); row += 1

    for s in range(S):
        for t in range(N):
            demand = max(float(net[s, t]), 0.0)
            surplus = max(-float(net[s, t]), 0.0)
            add_eq([t, u_i(s,t), c_i(s,t), w_i(s,t)], [1,-1,-1,-1], -surplus)
            add_eq([u_i(s,t), d_i(s,t), e_i(s,t)], [1,1,1], demand)
        for t in range(N):
            add_eq(
                [E_i(s,t+1), E_i(s,t), c_i(s,t), d_i(s,t)],
                [1,-1,-dio.ETA_C,1/dio.ETA_D], 0,
            )
        add_eq([E_i(s,0)], [1], E_start)
    A_eq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(row, n_vars))

    max_demand = float(np.maximum(net, 0.0).max())
    big_m = max_demand + 1.0
    ub_r, ub_c, ub_v, ub_b = [], [], [], []

    def add_ub(cols, vals, rhs):
        r = len(ub_b)
        ub_r.extend([r] * len(cols)); ub_c.extend(cols); ub_v.extend(vals); ub_b.append(rhs)

    for s in range(S):
        for t in range(N):
            add_ub([e_i(s,t), z_i(s,t)], [1,-big_m], 0)
            add_ub([c_i(s,t), z_i(s,t)], [1,dio.C_MAX], dio.C_MAX)
    A_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(len(ub_b), n_vars))

    result = linprog(
        objective, A_ub=_csr(A_ub), b_ub=np.asarray(ub_b),
        A_eq=_csr(A_eq), b_eq=np.asarray(eq_b), bounds=bounds,
        integrality=integrality, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Q4-2第一阶段求解失败: {result.message}")

    g = result.x[:N]
    e_list, c_list, d_list, E_list = [], [], [], []
    for s in range(S):
        e_list.append(result.x[e_i(s,0):e_i(s,N-1)+1])
        c_list.append(result.x[c_i(s,0):c_i(s,N-1)+1])
        d_list.append(result.x[d_i(s,0):d_i(s,N-1)+1])
        E_list.append(result.x[E_i(s,0):E_i(s,N)+1])
    stats = {
        "e_mean": np.mean(e_list, axis=0),
        "c_mean": np.mean(c_list, axis=0),
        "d_mean": np.mean(d_list, axis=0),
        "E_mean": np.mean(E_list, axis=0),
        "mean_plan_price": prices.mean(axis=0),
        "expected_emergency": float(np.mean([
            np.sum(dio.EMERGENCY_MULT * prices[s] * e_list[s]) for s in range(S)
        ])),
        "emergency_charge_overlap": int(np.sum(
            (np.asarray(e_list) > 1e-8) & (np.asarray(c_list) > 1e-8)
        )),
    }
    return g, {
        "objective": float(result.fun), "planned_cost": float(np.sum(prices.mean(0)*g)),
        "stats": stats, "success": True, "status": int(result.status),
        "message": str(result.message),
        "model_type": "joint_price_netload_stochastic_milp",
    }


def _clip_soc(value):
    value = float(value)
    if value < dio.E_MIN - dio.SOC_TOL or value > dio.E_MAX + dio.SOC_TOL:
        raise ValueError(f"SOC真实越界: {value:.12f} kWh")
    if value < dio.E_MIN:
        return dio.E_MIN
    if value > dio.E_MAX:
        return dio.E_MAX
    return value


def causal_dispatch(net_actual, g, E_start, discharge_reference=None):
    """Execute causally; clip only solver-scale SOC boundary roundoff."""
    net_actual = np.asarray(net_actual, dtype=float)
    g = np.asarray(g, dtype=float)
    n = len(net_actual)
    if g.shape != (n,):
        raise ValueError("计划购电量形状错误")
    if discharge_reference is None:
        discharge_reference = np.full(n, dio.C_MAX)
    discharge_reference = np.asarray(discharge_reference, dtype=float)
    if discharge_reference.shape != (n,):
        raise ValueError("参考放电量形状错误")

    c = np.zeros(n); d = np.zeros(n); w = np.zeros(n); e = np.zeros(n)
    E = np.zeros(n + 1); E[0] = _clip_soc(E_start)
    for t in range(n):
        E[t] = _clip_soc(E[t])
        gap = float(net_actual[t] - g[t])
        if gap > 0:
            available = max(0.0, (E[t] - dio.E_MIN) * dio.ETA_D)
            d[t] = min(gap, max(0.0, discharge_reference[t]), dio.C_MAX, available)
            e[t] = max(0.0, gap - d[t])
        else:
            surplus = -gap
            room = max(0.0, (dio.E_MAX - E[t]) / dio.ETA_C)
            c[t] = min(surplus, dio.C_MAX, room)
            w[t] = max(0.0, surplus - c[t])
        E[t + 1] = _clip_soc(E[t] + dio.ETA_C*c[t] - d[t]/dio.ETA_D)
    return c, d, w, e, E


def dispatch_cost(price_actual, g, e):
    planned = float(np.sum(np.asarray(price_actual) * np.asarray(g)))
    emergency = float(np.sum(dio.EMERGENCY_MULT * np.asarray(price_actual) * np.asarray(e)))
    return planned, emergency, planned + emergency

