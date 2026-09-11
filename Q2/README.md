# 问题 2：含紧急购电的日前计划与滚动回测

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

`fill_reports.py` 按章节回填两份报告的数值表；须在以上求解与分析完成后运行。
`emergency_table.py` 生成四个指定日期并排的紧急购电表，依赖 `numpy`、
`matplotlib` 和 `openpyxl`。连续紧急购电时段合并，购电量采用 kWh。

> 注：`src/stress_test.py` 已废弃（旧 API），功能由 `robustness.py` 与
> `imputation_sensitivity.py` 覆盖，请用 `git rm Q2/src/stress_test.py` 删除。

## 输出

- `output/result2.xlsx`：最终提交结果文件
- `output/微网在指定日期的紧急购电量.md/.xlsx/.png`：四个指定日期各两列的紧急购电表
- `output/prob2_solution.npz`：完整逐日逐时段解
- `output/forecast_rolling_metrics.csv`：因果预测滚动检验误差对比
- `output/imputation_sensitivity.csv`：1 月 1 日 0:00 插补敏感性检验
- `output/monte_carlo_results.csv`：历史残差块蒙特卡洛结果
- `output/robustness_scenario_seed.csv`：场景数 × 随机种子稳健性
- `output/baselines.csv`：同一信息集基准对照（无储能因果 / 确定性点预测 / 随机策略）
- `output/fig1~fig4.png`：结果图
- `output/问题2_求解报告.md`：求解报告
- `Q2问题解决方案步骤.md`：当前模型、调度语义与完整实现链路说明
