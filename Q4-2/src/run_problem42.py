# -*- coding: utf-8 -*-
"""Q4-2 full runner: Q2 under causal forecasts of real-time prices."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
Q2_SRC = SRC_DIR.parents[1] / "Q2" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.append(str(Q2_SRC))

import data_io as dio
import forecast as fc
import optimization as opt


OUTPUT_START_DAY = 31
N_SCENARIOS = 12
LOOKBACK = 28


def simulate_days(net, price_actual, net_forecast, price_forecast,
                  net_residual, price_residual, n_scenarios=N_SCENARIOS,
                  lookback=LOOKBACK, seed=2025, E_start=dio.E0_START,
                  start_day=0, end_day=None):
    """Plan with forecast scenarios, then settle against actual daily prices."""
    net = np.asarray(net, dtype=float)
    price_actual = np.asarray(price_actual, dtype=float)
    D, N = net.shape
    expected_shape = (D, dio.N)
    for name, value in [
        ("price_actual", price_actual), ("net_forecast", net_forecast),
        ("price_forecast", price_forecast), ("net_residual", net_residual),
        ("price_residual", price_residual),
    ]:
        if np.asarray(value).shape != expected_shape:
            raise ValueError(f"{name}期望形状{expected_shape}，实际{np.asarray(value).shape}")
    if end_day is None:
        end_day = D - 1
    if not (0 <= start_day <= end_day < D):
        raise ValueError("起止日期索引不合法")

    arrays = {name: np.zeros((D, N)) for name in ["g","c","d","w","e","d_reference"]}
    E_all = np.zeros((D, N + 1))
    costs = {name: np.zeros(D) for name in [
        "planned_cost","emergency_cost","total_cost","first_obj","first_expected_emergency"
    ]}
    statuses = []
    source_days = np.full((D, n_scenarios), -2, dtype=int)
    E_cur = opt._clip_soc(E_start)
    for day in range(start_day, end_day + 1):
        if day % 30 == 0:
            print(f"    处理日期索引 {day}/{D-1}", flush=True)
        net_s, price_s, sources = fc.joint_scenarios_for_day(
            day, net_forecast, price_forecast, net_residual, price_residual,
            n_scenarios=n_scenarios, lookback=lookback, seed=seed,
        )
        v_terminal = dio.terminal_value(price_s.mean(axis=0))
        g, first = opt.build_first_stage(price_s, net_s, E_cur, v_terminal)
        c, d, w, e, E = opt.causal_dispatch(
            net[day], g, E_cur, discharge_reference=first["stats"]["d_mean"]
        )
        arrays["g"][day] = g; arrays["c"][day] = c; arrays["d"][day] = d
        arrays["w"][day] = w; arrays["e"][day] = e
        arrays["d_reference"][day] = first["stats"]["d_mean"]
        E_all[day] = E
        p_cost, e_cost, total = opt.dispatch_cost(price_actual[day], g, e)
        costs["planned_cost"][day] = p_cost
        costs["emergency_cost"][day] = e_cost
        costs["total_cost"][day] = total
        costs["first_obj"][day] = first["objective"]
        costs["first_expected_emergency"][day] = first["stats"]["expected_emergency"]
        statuses.append(first["status"]); source_days[day] = sources
        E_cur = opt._clip_soc(E[-1])
    return {**arrays, "E": E_all, **costs, "solve_status": statuses,
            "scenario_source_days": source_days}


def _inputs():
    dates, net, load, pv = dio.read_actual_data()
    price_dates, price_actual = dio.read_dynamic_prices()
    if not np.array_equal(dates, price_dates):
        raise ValueError("附件2与附件4日期不一致")
    # Q4 names only Attachments 2 and 4 for the Q2 recalculation. With no
    # history before 2025-01-01, use a neutral zero prior for that warm-up day;
    # every later forecast is built solely from already observed Attachment 2.
    zero_prior = np.zeros(dio.N)
    net_f, _r, _lh, _ph, load_r, pv_r = fc.build_causal_forecasts(
        load, pv, zero_prior, zero_prior
    )
    net_r = load_r - pv_r
    price_f, price_r = fc.build_causal_price_forecasts(price_actual)
    return dates, net, price_actual, net_f, price_f, net_r, price_r


def _save(path, dates, price_actual, price_forecast, net, net_forecast,
          sim, mask, args):
    np.savez_compressed(
        path, dates=dates[mask], price=price_actual[mask],
        price_forecast=price_forecast[mask], net=net[mask],
        net_forecast=net_forecast[mask],
        g=sim["g"][mask], c=sim["c"][mask], d=sim["d"][mask],
        w=sim["w"][mask], e=sim["e"][mask], E=sim["E"][mask],
        d_reference=sim["d_reference"][mask],
        planned_cost=sim["planned_cost"][mask],
        emergency_cost=sim["emergency_cost"][mask], total_cost=sim["total_cost"][mask],
        first_obj=sim["first_obj"][mask],
        first_expected_emergency=sim["first_expected_emergency"][mask],
        scenario_source_days=sim["scenario_source_days"][mask],
        n_scenarios=np.array([args.scenarios]), lookback=np.array([args.lookback]),
        seed=np.array([args.seed]), eta=np.array([dio.ETA_C, dio.ETA_D]),
        complete=np.array([args.mode == "full"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Q4-2波动电价购电策略")
    parser.add_argument("--mode", choices=["smoke","full"], default="smoke")
    parser.add_argument("--scenarios", type=int, default=N_SCENARIOS)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--smoke-day", type=int, default=31,
                        help="smoke日期索引，31表示2025-02-01")
    args = parser.parse_args(argv)
    t0 = time.time()
    dates, net, price_actual, net_f, price_f, net_r, price_r = _inputs()
    if args.mode == "full":
        start, end, E0, out = 0, len(dates)-1, dio.E0_START, dio.SOLUTION_NPZ
    else:
        start = end = args.smoke_day
        E0, out = dio.E0_START, dio.SMOKE_NPZ
    sim = simulate_days(
        net, price_actual, net_f, price_f, net_r, price_r,
        n_scenarios=args.scenarios, lookback=args.lookback, seed=args.seed,
        E_start=E0, start_day=start, end_day=end,
    )
    mask = np.zeros(len(dates), dtype=bool)
    if args.mode == "full": mask[OUTPUT_START_DAY:] = True
    else: mask[start:end+1] = True
    _save(out, dates, price_actual, price_f, net, net_f, sim, mask, args)
    print(f"Q4-2 {args.mode}完成: {out}，用时{time.time()-t0:.1f}s")
    print(f"所选期间总费用: {sim['total_cost'][mask].sum():.2f} 元")


if __name__ == "__main__":
    main()
