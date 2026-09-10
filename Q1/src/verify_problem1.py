# -*- coding: utf-8 -*-
"""
问题 1 模型检验脚本（独立于 solve_problem1.py 重算，并做 KKT 认证）

检验内容：
1. 用 HiGHS 的 dual simplex / interior point 两种算法交叉重算，目标值应一致；
2. 检查原问题的所有等式、不等式和变量上下界是否被严格满足；
3. 提取 HiGHS 返回的对偶变量，验证 KKT 条件：
   - 拉格朗日函数对每个变量的梯度为 0（stationarity）；
   - 互补松弛条件成立（complementary slackness）；
   - 对偶变量符号正确（dual feasibility）；
4. 由对偶变量计算对偶目标值，验证 primal-dual gap 约等于 0；
5. 复核报告表 1/表 2 的数值，并校验 result1.xlsx 中的填写位置。
"""

import numpy as np
from scipy.optimize import linprog

import xlsx_reader as xr


TOL = 1e-6


def read_data():
    rows = xr.read_sheet_rows("attachment/附件1.xlsx")["Sheet1"]
    data = rows[1:]
    assert len(data) == 144, f"期望 144 个区间, 实际 {len(data)}"

    def parse_time(t):
        if t is None:
            return None
        if isinstance(t, (int, float)):
            return round(float(t) * 24 * 60)
        s = str(t).strip()
        offset = 0
        if "+" in s:
            s, off = s.split("+")
            offset = int(off) * 1440
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm) + offset

    t_min = np.array([parse_time(r[0]) for r in data], dtype=float)
    price = np.array([float(r[1]) for r in data])
    load_kw = np.array([float(r[2]) for r in data])
    pv_kw = np.array([float(r[3]) for r in data])

    assert t_min[0] == 10 and t_min[-1] == 1440
    assert np.all(np.diff(t_min) == 10)

    dt = 1.0 / 6.0
    L = load_kw * dt
    G = pv_kw * dt
    return t_min, price, load_kw, pv_kw, L, G


def build_lp(price, L, G):
    N = 144
    eta_c = 0.9
    eta_d = 0.9
    E_min, E_max = 1200.0, 10800.0
    P_max = 5000.0
    C_max = P_max / 6.0
    E0 = 6000.0

    nv_x = nv_c = nv_d = nv_w = N
    nv_E = N + 1

    def ix(t):
        return t

    def ic(t):
        return nv_x + t

    def id_(t):
        return nv_x + nv_c + t

    def iw(t):
        return nv_x + nv_c + nv_d + t

    def iE(t):
        return nv_x + nv_c + nv_d + nv_w + t

    n_vars = nv_x + nv_c + nv_d + nv_w + nv_E

    c_obj = np.zeros(n_vars)
    for t in range(N):
        c_obj[ix(t)] = price[t]

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    def add_eq(row, rhs):
        A_eq.append(row)
        b_eq.append(rhs)

    def add_ub(row, rhs):
        A_ub.append(row)
        b_ub.append(rhs)

    for t in range(N):
        row = np.zeros(n_vars)
        row[ix(t)] = 1.0
        row[id_(t)] = 1.0
        row[ic(t)] = -1.0
        row[iw(t)] = -1.0
        add_eq(row, L[t] - G[t])

    for t in range(N):
        row = np.zeros(n_vars)
        row[iE(t + 1)] = 1.0
        row[iE(t)] = -1.0
        row[ic(t)] = -eta_c
        row[id_(t)] = 1.0 / eta_d
        add_eq(row, 0.0)

    for t in range(N + 1):
        row = np.zeros(n_vars)
        row[iE(t)] = 1.0
        add_ub(row, E_max)
        row = np.zeros(n_vars)
        row[iE(t)] = -1.0
        add_ub(row, -E_min)

    row = np.zeros(n_vars)
    row[iE(0)] = 1.0
    add_eq(row, E0)
    row = np.zeros(n_vars)
    row[iE(N)] = 1.0
    add_eq(row, E0)

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    for t in range(N):
        ub[ic(t)] = C_max
        ub[id_(t)] = C_max
    for t in range(N + 1):
        lb[iE(t)] = -np.inf
        ub[iE(t)] = np.inf

    meta = {
        "N": N,
        "eta_c": eta_c,
        "eta_d": eta_d,
        "E_min": E_min,
        "E_max": E_max,
        "C_max": C_max,
        "E0": E0,
        "ix": ix,
        "ic": ic,
        "id_": id_,
        "iw": iw,
        "iE": iE,
    }
    return (
        np.array(c_obj),
        np.array(A_ub),
        np.array(b_ub),
        np.array(A_eq),
        np.array(b_eq),
        list(zip(lb, ub)),
        meta,
    )


def slice_solution(res, meta):
    N = meta["N"]
    ix, ic, id_, iw, iE = meta["ix"], meta["ic"], meta["id_"], meta["iw"], meta["iE"]
    x = res.x[ix(0) : ix(N - 1) + 1]
    c = res.x[ic(0) : ic(N - 1) + 1]
    d = res.x[id_(0) : id_(N - 1) + 1]
    w = res.x[iw(0) : iw(N - 1) + 1]
    E = res.x[iE(0) : iE(N) + 1]
    return x, c, d, w, E


def check_primal(res, meta, L, G):
    N = meta["N"]
    eta_c, eta_d = meta["eta_c"], meta["eta_d"]
    E_min, E_max, C_max, E0 = meta["E_min"], meta["E_max"], meta["C_max"], meta["E0"]
    x, c, d, w, E = slice_solution(res, meta)

    bal = x + G + d - c - w - L
    dyn = np.diff(E) - (eta_c * c - d / eta_d)
    checks = {
        "功率平衡残差": np.max(np.abs(bal)),
        "储能动态残差": np.max(np.abs(dyn)),
        "储能初值误差": abs(E[0] - E0),
        "储能终值误差": abs(E[-1] - E0),
        "储能越下限": max(0.0, E_min - E.min()),
        "储能越上限": max(0.0, E.max() - E_max),
        "充电越上限": max(0.0, c.max() - C_max),
        "放电越上限": max(0.0, d.max() - C_max),
        "购电为负": max(0.0, -x.min()),
        "弃光为负": max(0.0, -w.min()),
    }
    ok = True
    for name, val in checks.items():
        passed = val <= TOL
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val:.3e}")
    return ok, x, c, d, w, E


def check_kkt(res, meta, price, L, G, A_ub, b_ub, A_eq, b_eq, bounds):
    N = meta["N"]
    eta_c, eta_d = meta["eta_c"], meta["eta_d"]
    C_max = meta["C_max"]
    ix, ic, id_, iw, iE = meta["ix"], meta["ic"], meta["id_"], meta["iw"], meta["iE"]

    # SciPy 返回的等式对偶变量是标准拉格朗日乘子的相反数，
    # 即 res.eqlin.marginals = -lambda_eq。下面公式按该约定书写。
    y_bal = res.eqlin.marginals[0:N]
    y_dyn = res.eqlin.marginals[N : 2 * N]
    y_b0 = res.eqlin.marginals[2 * N]
    y_bN = res.eqlin.marginals[2 * N + 1]
    # A_ub 按每个时刻交替添加：偶数行是 E_t <= E_max，奇数行是 -E_t <= -E_min。
    lam_hi = res.ineqlin.marginals[0 : 2 * (N + 1) : 2]
    lam_lo = res.ineqlin.marginals[1 : 2 * (N + 1) : 2]
    lower_m = res.lower.marginals
    upper_m = res.upper.marginals

    x, c, d, w, E = slice_solution(res, meta)
    n_vars = len(res.x)
    stat = np.zeros(n_vars)

    for t in range(N):
        stat[ix(t)] = price[t] - y_bal[t] - lower_m[ix(t)]
        stat[iw(t)] = y_bal[t] - lower_m[iw(t)]
        stat[ic(t)] = (
            y_bal[t] + eta_c * y_dyn[t] - upper_m[ic(t)] - lower_m[ic(t)]
        )
        stat[id_(t)] = (
            -y_bal[t] - y_dyn[t] / eta_d - upper_m[id_(t)] - lower_m[id_(t)]
        )

    for t in range(N + 1):
        val = -lam_hi[t] + lam_lo[t]
        if t == 0:
            val += y_dyn[0] - y_b0
        elif t == N:
            val += -y_dyn[N - 1] - y_bN
        else:
            val += y_dyn[t] - y_dyn[t - 1]
        stat[iE(t)] = val

    comp = []
    for t in range(N + 1):
        comp.append(abs(lam_hi[t] * (meta["E_max"] - E[t])))
        comp.append(abs(lam_lo[t] * (E[t] - meta["E_min"])))
    for t in range(N):
        comp.append(abs(lower_m[ix(t)] * x[t]))
        comp.append(abs(lower_m[iw(t)] * w[t]))
        comp.append(abs(lower_m[ic(t)] * c[t]))
        comp.append(abs(lower_m[id_(t)] * d[t]))
        comp.append(abs(upper_m[ic(t)] * (C_max - c[t])))
        comp.append(abs(upper_m[id_(t)] * (C_max - d[t])))

    stationarity = np.max(np.abs(stat))
    complementarity = max(comp)
    dual_sign_ok = (
        np.all(lam_hi <= TOL)
        and np.all(lam_lo <= TOL)
        and np.all(lower_m >= -TOL)
        and np.all(upper_m <= TOL)
    )

    finite_upper = np.array([C_max if np.isfinite(bounds[i][1]) else 0.0 for i in range(n_vars)])
    dual_obj = float(np.dot(res.eqlin.marginals, b_eq))
    dual_obj += float(np.dot(res.ineqlin.marginals, b_ub))
    dual_obj += float(np.dot(upper_m, finite_upper))
    gap = abs(res.fun - dual_obj)

    print(f"  [{'PASS' if stationarity <= TOL else 'FAIL'}] 拉格朗日梯度最大残差: {stationarity:.3e}")
    print(f"  [{'PASS' if complementarity <= TOL else 'FAIL'}] 互补松弛最大残差: {complementarity:.3e}")
    print(f"  [{'PASS' if dual_sign_ok else 'FAIL'}] 对偶变量符号: {dual_sign_ok}")
    print(f"  [{'PASS' if gap <= TOL else 'FAIL'}] primal-dual gap: {gap:.3e}")

    return stationarity <= TOL and complementarity <= TOL and dual_sign_ok and gap <= TOL


def check_tables(x, c, d, E, cost):
    spec = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
            ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
    blocks = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
              ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144)]

    expected_spec = {name: x[k] for name, k in spec}
    report_spec = {
        "10:00-10:10": 0.00,
        "12:00-12:10": 480.41,
        "14:00-14:10": 0.00,
        "16:00-16:10": 445.43,
        "18:00-18:10": 531.89,
        "20:00-20:10": 0.00,
    }
    max_diff = max(abs(expected_spec[k] - v) for k, v in report_spec.items())
    print(f"  [{'PASS' if max_diff < 0.01 else 'FAIL'}] 表 1 数值与重算最大误差: {max_diff:.4f}")
    print(f"  [{'PASS' if abs(x.sum() - 59482.70) < 0.01 else 'FAIL'}] 全天购电量: {x.sum():.4f}")
    print(f"  [{'PASS' if abs(cost - 35126.95) < 0.01 else 'FAIL'}] 全天购电费: {cost:.4f}")

    expected_blocks = []
    for name, a, b in blocks:
        expected_blocks.append((c[a:b].sum(), d[a:b].sum()))
    report_blocks = [
        (4500.00, 0.00),
        (833.33, 6365.84),
        (4787.96, 1703.00),
        (5286.04, 91.10),
        (0.00, 5780.13),
        (5333.33, 2859.87),
    ]
    max_cd = max(
        max(abs(e[0] - r[0]), abs(e[1] - r[1]))
        for e, r in zip(expected_blocks, report_blocks)
    )
    print(f"  [{'PASS' if max_cd < 0.01 else 'FAIL'}] 表 2 数值与重算最大误差: {max_cd:.4f}")
    print(f"  [{'PASS' if abs(E[0] - 6000) < 0.01 and abs(E[-1] - 6000) < 0.01 else 'FAIL'}] "
          f"0:00/24:00 储电量: {E[0]:.4f} / {E[-1]:.4f}")


def check_result_file(x, c, d, E):
    try:
        sheets = xr.read_sheet_rows("result1.xlsx")
        names = list(sheets)
        plan = sheets[names[0]]
        charge = sheets[names[1]]
        numeric = []
        for row in plan[1:145]:
            numeric.append(row[1] if len(row) > 1 else None)
        if len(numeric) == len(x) and all(v is not None for v in numeric):
            diff = np.max(np.abs(np.array(numeric, dtype=float) - x))
            print(f"  [{'PASS' if diff <= 1e-4 else 'FAIL'}] result1.xlsx 购电量与重算最大误差: {diff:.3e}")
        else:
            print("  [FAIL] result1.xlsx 购电量行数或空值异常")

        block_rows = charge[1:7]
        got_blocks = [(r[1], r[2]) for r in block_rows]
        expected_blocks = [
            (c[0:24].sum(), d[0:24].sum()),
            (c[24:48].sum(), d[24:48].sum()),
            (c[48:72].sum(), d[48:72].sum()),
            (c[72:96].sum(), d[72:96].sum()),
            (c[96:120].sum(), d[96:120].sum()),
            (c[120:144].sum(), d[120:144].sum()),
        ]
        cd_diff = max(
            max(abs(g[0] - e[0]), abs(g[1] - e[1]))
            for g, e in zip(got_blocks, expected_blocks)
        )
        print(f"  [{'PASS' if cd_diff < 0.01 else 'FAIL'}] result1.xlsx 充放电量块与重算最大误差: {cd_diff:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] 读取 result1.xlsx 失败: {exc}")


def main():
    print("=" * 76)
    print("问题 1 模型检验")
    print("=" * 76)

    t_min, price, load_kw, pv_kw, L, G = read_data()
    c_obj, A_ub, b_ub, A_eq, b_eq, bounds, meta = build_lp(price, L, G)

    print("\n[1] 不同算法交叉重算")
    ref_obj = None
    for method in ("highs", "highs-ds", "highs-ipm"):
        r = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method=method)
        print(f"  {method:10s} success={r.success} objective={r.fun:.10f}")
        if not r.success:
            print(f"  [FAIL] {method} 未得到最优解: {r.message}")
            continue
        if ref_obj is None:
            ref_obj = r.fun
        elif abs(r.fun - ref_obj) > 1e-5:
            print(f"  [FAIL] {method} 目标值与 highs 不一致")

    print("\n[2] 原始可行性")
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    ok_primal, x, c, d, w, E = check_primal(res, meta, L, G)

    print("\n[3] KKT 条件")
    ok_kkt = check_kkt(res, meta, price, L, G, A_ub, b_ub, A_eq, b_eq, bounds)

    print("\n[4] 报告表 1 / 表 2 复核")
    check_tables(x, c, d, E, res.fun)

    print("\n[5] result1.xlsx 复核")
    check_result_file(x, c, d, E)

    print("\n" + "=" * 76)
    if ok_primal and ok_kkt:
        print("结论: 模型检验 PASS。求解结果满足全部约束，且 KKT 条件成立。")
    else:
        print("结论: 模型检验存在 FAIL 项，请检查建模或数值精度。")
    print("=" * 76)


if __name__ == "__main__":
    main()
