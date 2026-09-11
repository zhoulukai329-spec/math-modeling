# -*- coding: utf-8 -*-
"""问题 2 的因果负载/光伏预测与场景生成。

预测规则只使用"当天 0:00 之前"的历史数据，且分别对负载与光伏建模：
  - d == 0: 用附件 1 的典型日负载/光伏作为初始先验；
  - 1 <= d < 7: 用此前所有历史日的平均负载/光伏；
  - d >= 7: 用上周同一天的负载/光伏（周周期明显，MAE 最低）。

净负荷点预测 = 负载点预测 - 光伏点预测；净负荷残差 = 负载残差 - 光伏残差。
场景采用"净负荷点预测 + 同一历史日的负载残差块 - 光伏残差块"生成，
保留负载与光伏误差之间的时段间相关性。
"""
import numpy as np


def build_causal_forecasts(load, pv, typical_load, typical_pv):
    """分别生成 0..D-1 每日的因果负载/光伏点预测与残差，再合成净负荷。

    参数:
      load:         (D,144) 全年实际负载能量 (kWh)
      pv:           (D,144) 全年实际光伏能量 (kWh)
      typical_load: (144,) 附件 1 典型日负载能量 (kWh)
      typical_pv:   (144,) 附件 1 典型日光伏能量 (kWh)

    返回:
      f:          (D,144) 净负荷点预测 = load_hat - pv_hat
      r:          (D,144) 净负荷残差   = load_resid - pv_resid
      load_hat:   (D,144) 负载点预测
      pv_hat:     (D,144) 光伏点预测
      load_resid: (D,144) 负载残差 = 实际负载 - 负载点预测
      pv_resid:   (D,144) 光伏残差 = 实际光伏 - 光伏点预测
    """
    D = load.shape[0]
    load_hat = np.zeros_like(load)
    pv_hat = np.zeros_like(pv)
    for d in range(D):
        if d == 0:
            load_hat[d] = typical_load
            pv_hat[d] = typical_pv
        elif d < 7:
            load_hat[d] = load[:d].mean(axis=0)
            pv_hat[d] = pv[:d].mean(axis=0)
        else:
            load_hat[d] = load[d - 7]
            pv_hat[d] = pv[d - 7]
    load_resid = load - load_hat
    pv_resid = pv - pv_hat
    f = load_hat - pv_hat
    r = load_resid - pv_resid
    return f, r, load_hat, pv_hat, load_resid, pv_resid


def scenarios_for_day(d, f, load_resid, pv_resid, n_scenarios=12, lookback=28,
                      seed=2025):
    """生成第 d 天 0:00 可用的一组净负荷场景。

    随机抽取一个往期历史日 j 的负载残差块 load_resid[j] 与光伏残差块
    pv_resid[j]（取自同一历史日，保留负载与光伏误差之间的相关性），
    净负荷场景 = f[d] + load_resid[j] - pv_resid[j]。

    参数:
      d: 目标日期索引
      f: 净负荷点预测，f[d] 为第 d 天预测
      load_resid, pv_resid: 因果残差，仅使用 j<d 的行
      n_scenarios: 场景数
      lookback: 使用最近多少天残差作为场景池
      seed: 确定性抽样种子

    返回 (S,144) 场景矩阵。若历史不足，则自动重复可用残差块。
    """
    rng = np.random.default_rng(seed + d)
    available = list(range(max(0, d - lookback), d))
    if len(available) == 0:
        pool = [np.zeros_like(f[d])]
    else:
        pool = [load_resid[j] - pv_resid[j] for j in available]

    if len(pool) >= n_scenarios:
        idx = rng.choice(len(pool), size=n_scenarios, replace=False)
    else:
        # 历史不足时用重复抽样补齐；仅发生在年初热身阶段，不影响 2 月后输出。
        idx = rng.choice(len(pool), size=n_scenarios, replace=True)

    base = f[d]
    out = np.stack([base + pool[i] for i in idx], axis=0)
    return out


def actual_scenario(d, net):
    """返回第 d 天实际净负荷，供第二阶段的真实回测使用。"""
    return net[d]
