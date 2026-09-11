# 问题 3：十分钟滚动场景 MPC-MILP

从仓库根目录运行。依赖 Python、NumPy、SciPy（支持 `optimize.milp`）、openpyxl；测试使用 pytest。
附件保持只读。每个实际十分钟步都解 MILP，只执行当前场景共同动作，SOC 连续传递。
0:00 发布全天基线，6/12/18 点修订剩余承诺；没有启发式调度替代或全年预计算结果。

## 真实 smoke 与校验

```powershell
python Q3/src/run_problem3.py --mode smoke --date-start 2025-02-01 --date-end 2025-02-01 --horizon-steps 6 --max-steps 37 --scenarios 2 --time-limit 10
python Q3/src/verify_problem3.py Q3/output/smoke_solution.npz --workbook Q3/output/result3.xlsx --report Q3/output/smoke_verification.json
python -m pytest Q3/tests -q
```

默认 smoke 即 2025-02-01、H=6、K=2、最多 37 步，含 06:00 修订。发承诺时仍覆盖整行，
短预测窗仅缩短普通执行的前瞻范围。`--date-end 2025-02-02 --max-steps 146` 可检查跨日。
January warm-up 默认确定性；2 月起使用场景，历史窗口 28 天，默认种子 2025。
`--deterministic`、`--cvar-weight`、`--terminal-penalty`、`--terminal-soc` 等可显式设定 smoke 参数。
时间上限下有效整数可行解允许执行，是否最优和 MILP 诊断均保存；没有可行解即失败。

Windows 如系统旧 `pytest-of-*` 临时目录权限异常，可指定一个尚不存在的临时子目录：
`python -m pytest Q3/tests -q --basetemp D:/-/.pytest_cache/q3-check-new`。

## 文件与数值语义

`Q3/output/smoke_solution.npz` 保存 144 列原始顺序数组、执行掩码、每个承诺版本、求解日志与配置。
加载用 `make_results.load_result(path)`，禁用 pickle；`config.data` 不序列化。
`smoke_trajectory.csv` 是全部所选业务行的逐十分钟轨迹，未执行物理值为空。
`smoke_metrics.json` 含费用、求解数、运行时间、完整性标记与配置。
`smoke_verification.json` 是独立校验报告。输出前后均校验，失败返回非零状态。

`result3.xlsx` 从 `attachment/附件5/result3.xlsx` 载入后另存，保留工作表、日期、表头、样式、
省略号和固定行列数；禁止输出到原模板路径。模板四张表的填写含义：

| 表 | 写入内容 |
|---|---|
| 计划购电量 | B:EO 为当日 0:00 基线；EP 为总量，EQ 为基线金额 |
| 调整购电量 | B:EO 为最后已发布版本的累计有效承诺；EP 为该累计总量，EQ 为各相邻版本上调费与下调费之和 |
| 充放电量 | 只填模板 2 月 1 日、2 月 2 日、3 月 20 日、12 月 31 日锚点；C/D 为实际日历四小时块总量，F 为 0:00/24:00 的状态 |
| 紧急购电量 | 只填模板 2 月 1 日、2 月 2 日、12 月 31 日锚点；B 列多行列出实际正应急十分钟区间，C 列为这些执行值总量；其余预留槽位保留 |

购电量表的列左端点是 **00:10 到次日 00:00**，不得循环移位。
充放电表的 `0:00-4:00` 等区间按日历左闭右开汇总，因此 04:00 属于第二块，
日历午夜动作来自前一业务行的末列。缺少任一十分钟执行时，该四小时汇总留空。
例如从 2 月 1 日 00:10 开始的独立运行，不能生成完整的该日日历 00:00-04:00 汇总。
开始日 0:00 SOC 用显式初始状态；24:00 SOC 是该时刻动作执行前的状态，不能误用 00:10 状态。
只执行部分日且没有正应急值时，不能写“全天无应急”，故该模板单元格留空。
若部分日已有正应急，标注“已执行部分”。完整逐步数据始终在 NPZ/CSV 中。

现金费用严格为：基线 `price * baseline`，相邻版本上调 `1.5 * price * up`、
下调 `0.5 * price * down`，实际应急 `5 * price * emergency`。
调整表的累计承诺不能再按基线差值一次收费，也不能与基线量直接相加。
终端惩罚、吞吐正则项、CVaR 项和未执行场景应急费用均不进入现金费用。
部分 smoke 的费用包含整日已经发布的基线和修订，不能称为 37 个区间的净消费成本或全年结果。

## 独立验证范围

校验器不调用优化器，独立重算时间顺序、功率/能量平衡、SOC 递推/连续/边界、
二元模式、充放电互斥、充电与应急互斥、承诺冻结与逐版本增减、强制修订时钟、四部分费用，
并逐单元格核对输出与模板位置及未知值。允许放电与应急共存。
每个执行时间必须有真实 MILP 求解日志且当前动作分歧在容差内。
保存字段可检查预测发布不晚于决策、当前观察标记、未来实际值为空；
这些摘要不能单独证明内部所有预测输入无泄漏，因此另有改变未来实际值的决策不变性测试。
独立校验不将日志中的求解最优状态当成数学最优性证明。

## 全年与实验接口

```powershell
python Q3/src/run_problem3.py --mode full --calibration Q3/results/frozen_calibration.json --date-start 2025-02-01 --date-end 2025-12-31
python Q3/src/run_problem3.py --mode experiments
```

`full` 默认 H=144、K=12、全年所选区间无步数限制，运行量可能很大。
必须先经 P1 和 Task 6 的一月校准；Task 4 没有运行全年或自动生成校准结论。
冻结配置 JSON 协议是 `frozen: true`、`training_start: "2025-01-01"`、
`training_end: "2025-01-31"`、`parameters` 对象，后者必须含 `terminal_penalty`、
`terminal_soc`、`cvar_weight`，可含 `cvar_alpha`。
full 禁止命令行覆盖这些冻结值。JSON 是实验流程的输入凭据，CLI 只验证协议，
不能凭一份手填文件证明实际完成了校准。初始 SOC 由 `--initial-soc` 显式提供，默认 6000 kWh；
从 2 月开始不隐式执行 1 月，也不自动继承一月终态。
`--max-steps` 可限制 full 调试，但输出仍记录 `complete: false`，不得称作全年完成。
`experiments` 路由到后续 `experiments.run_experiments(args)`；模块未实现时明确失败。
每次运行覆盖同一输出目录内对应模式文件及 `result3.xlsx`；保留不同试验请使用不同 `--output-dir`。
