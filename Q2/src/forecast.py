# -*- coding: utf-8 -*-
"""问题 2 的因果净负荷预测与场景生成。

预测规则只使用“当天 0:00 之前”的历史数据：
  - d == 0: 用附件 1 的典型日净负荷作为初始先验；
  - 1 <= d < 7: 用此前所有历史日的平均净负荷；
  - d >= 7: 用上周同一天的净负荷（周周期明显，MAE 最低）。

场景采用“预测值 + 历史整日残差块”的方式生成，保留时段间的相关性。
"""
import numpy as np


def build_causal_forecasts(net, typical_net):
    """生成 0..D-1 每日的因果点预测和残差。

    返回:
      f: (D,144) 点预测
      r: (D,144) 当日实际 - 点预测
    """
    D = net.shape[0]
    f = np.zeros_like(net)
    r = np.zeros_like(net)
    for d in range(D):
        if d == 0:
            f[d] = typical_net
        elif d < 7:
            f[d] = net[:d].mean(axis=0)
        else:
            f[d] = net[d - 7]
        r[d] = net[d] - f[d]
    return f, r


def scenarios_for_day(d, net, f, r, n_scenarios=8, lookback=28, seed=2025):
    """生成第 d 天 0:00 可用的一组净负荷场景。

    参数:
      d: 目标日期索引
      net: 全年实际净负荷 (仅允许用 <d 的行)
      f: 因果点预测，f[d] 为第 d 天预测
      r: 因果残差，仅使用 r[j], j<d
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
        pool = [r[j] for j in available]

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

