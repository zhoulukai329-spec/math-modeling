# 问题4-3：动态电价下的四时点购电调整

Q4-3读取附件2的实际负荷/光伏、附件3每天0/6/12/18点发布的未来24小时光伏预测、
附件4动态电价和附件5的`result4-3.xlsx`模板；不读取附件1。

## 模型口径

- 数组和工作簿保持附件原始左端点顺序：00:10至次日00:00，不循环移动。
- 当前十分钟开始时才可读取该格实际负荷、光伏和电价；未来实际值只用于事后结算。
- 未来电价依次使用上周同一时段、已发生日期同一时段均值和首日边界先验预测。
- 光伏与价格误差场景使用相同历史来源日；完全重复的联合场景自动合并概率。
- 0:00发布基准，6/12/18点只调整未执行部分；1/1.5/0.5/5倍费用逐版本结算。
- 紧急电只能供当前负荷，不能充电；允许保留电池并同时购买紧急电。
- SOC限制为1200--10800 kWh。最多`1e-5` kWh的边界毛刺被钳位，更大的越界直接失败。

默认`event-policy`在0/6/12/18点运行完整场景MILP，并把场景加权的充放电参考用于区间内
快速执行。`rolling-milp`在每10分钟重解MILP，仅建议用于少数日期精度对照。

## 命令

```powershell
python Q4-3/src/run_problem43.py --mode smoke --backend event-policy
python Q4-3/src/run_problem43.py --mode smoke --backend rolling-milp --max-steps 12
python Q4-3/src/run_problem43.py --mode full --backend event-policy
python Q4-3/src/verify_problem43.py Q4-3/output/full_solution.npz --workbook Q4-3/output/result4-3.xlsx
```

全年模式从1月1日暖机以连续传递SOC并建立历史库，保存前裁去2月1日前的暖机行和费用；
正式工作簿只填模板的2月1日至12月31日。

## 精度对照、前瞻测试和绘图

```powershell
python Q4-3/src/compare_backends.py
python Q4-3/src/run_horizon_sensitivity.py --scenarios 3
python Q4-3/src/make_figures.py Q4-3/output/full_solution.npz
```

前两个命令只运行代表日。敏感性固定为12/18/24小时和四个季节代表日，不生成48/72小时
外推；CSV、JSON和敏感性图保存在`Q4-3/analysis`。绘图程序只读取已有结果并输出六组
360 DPI PNG、SVG和PDF；冒烟图明确标注“冒烟预览”，不得作为全年结论。
