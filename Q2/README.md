# 问题 2：含紧急购电的日前计划与滚动回测

> 当前代码已修正为附件1只读取电价，负载和光伏仅来自附件2。仓库中既有的
> `output/prob2_solution.npz` 与 `output/result2.xlsx` 是修正前的历史结果；
> 必须重新运行主程序后再生成和校验正式结果，脚本会拒绝把旧NPZ当作修正版。

本目录与 `Q1/` 同级，所有输入路径在 `src/data_io.py` 中统一路由，不修改原始附件。

## 运行顺序

```powershell
python Q2/src/validate_forecast.py
python Q2/src/run_problem2.py
python Q2/src/make_results.py
python Q2/src/make_figures.py
python Q2/src/verify_problem2.py
python Q2/src/robustness.py
python Q2/src/imputation_sensitivity.py
python Q2/src/fill_reports.py
python Q2/src/emergency_table.py
python -m unittest discover -s Q2/tests -v
```

首日预测初始化敏感性分析单独运行：

```powershell
# 最小检查：两个方案各跑前两天
python Q2/src/forecast_initialization_sensitivity.py --mode smoke --scenarios 1

# 跑到2月1日：重点看第一月是否消除了首日差异
python Q2/src/forecast_initialization_sensitivity.py --mode feb-boundary

# 两个方案各跑全年：比较正式输出期全部费用和应急电量
python Q2/src/forecast_initialization_sensitivity.py --mode full
```

结果位于 `Q2/output/forecast_initialization_<mode>/`。先看 `summary.csv` 中的
2月1日初始SOC和正式期总费用，再看 `daily_comparison.csv` 中SOC差异何时收敛。

`fill_reports.py` 按章节回填两份报告的数值表；须在以上求解与分析完成后运行。
`emergency_table.py` 生成四个指定日期并排的紧急购电表，依赖 `numpy`、
`matplotlib` 和 `openpyxl`。连续紧急购电时段合并，购电量采用 kWh。

计划购电工作表按官方模板的跨日口径填写：每行覆盖当天00:10至次日00:10，
最后一列取下一自然日00:00的计划值，行合计与费用也按这144格重新计算。
12月31日的最后一格由模型在2026年1月1日00:00仅用届时可用历史信息额外规划，
不计入2025年正式评价期。旧版NPZ不含该边界字段，修改后须重新运行主程序。

> 注：`src/stress_test.py` 已废弃（旧 API），功能由 `robustness.py` 与
> `imputation_sensitivity.py` 覆盖，请用 `git rm Q2/src/stress_test.py` 删除。

## 输出

- `output/result2.xlsx`：最终提交结果文件
- `output/微网在指定日期的紧急购电量.md/.xlsx/.png`：四个指定日期各两列的紧急购电表
- `output/prob2_solution.npz`：完整逐日逐时段解
- `output/forecast_rolling_metrics.csv`：因果预测滚动检验误差对比
- `output/imputation_sensitivity.csv`：1 月 1 日 0:00 插补敏感性检验
- `output/forecast_initialization_<mode>/`：零先验与首时刻负载持续法的预测初始化对比
- `output/monte_carlo_results.csv`：历史残差块蒙特卡洛结果
- `output/robustness_scenario_seed.csv`：场景数 × 随机种子稳健性
- `output/baselines.csv`：同一信息集基准对照（无储能因果 / 确定性点预测 / 随机策略）
- `output/fig1~fig4.png`：结果图
- `output/问题2_求解报告.md`：求解报告
- `Q2问题解决方案步骤.md`：当前模型、调度语义与完整实现链路说明
