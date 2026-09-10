# 问题 2：含紧急购电的日前计划与滚动回测

本目录与 `Q1/` 同级，所有输入路径在 `src/data_io.py` 中统一路由，不修改原始附件。

## 运行顺序

```powershell
python Q2/src/validate_forecast.py
python Q2/src/run_problem2.py
python Q2/src/make_results.py
python Q2/src/make_figures.py
python Q2/src/verify_problem2.py
```

## 输出

- `output/result2.xlsx`：最终提交结果文件
- `output/prob2_solution.npz`：完整逐日逐时段解
- `output/forecast_rolling_metrics.csv`：因果预测滚动检验误差对比
- `output/fig1~fig4.png`：结果图
- `output/问题2_求解报告.md`：求解报告
