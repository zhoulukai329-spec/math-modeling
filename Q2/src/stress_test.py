# -*- coding: utf-8 -*-
"""[已废弃] 本脚本已失效，请删除。

本脚本使用旧版 API，与新模型不兼容：
  - z["q_selected"]（旧字段，已移除）
  - simulate_days(net, price, f, r, q, v, 0, D-1, e0, lookback=...)（旧位置参数签名）
  - dispatch_cost(price, g, e, c, d)（旧 5 参签名，现为 3 参 dispatch_cost(price, g, e)）

其功能已被拆分到：
  - robustness.py            ：场景数 × 随机种子稳健性、同一信息集基准对照
  - imputation_sensitivity.py：历史残差块蒙特卡洛、插补敏感性

彻底删除本文件：git rm Q2/src/stress_test.py
"""


def main():
    raise SystemExit(
        "stress_test.py 已废弃，功能见 robustness.py / imputation_sensitivity.py。"
    )


if __name__ == "__main__":
    main()
