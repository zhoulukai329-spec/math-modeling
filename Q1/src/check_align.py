# -*- coding: utf-8 -*-
"""核对 result1.xlsx 与附件5模板的对齐情况（结构 + 数值语义）。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
REPO_ROOT = Q_DIR.parent
ATTACH_DIR = REPO_ROOT / 'attachment'
OUTPUT_DIR = Q_DIR / 'output'
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import xlsx_reader as xr

TEMPLATE = str(ATTACH_DIR / '附件5' / 'result1.xlsx')
OUT = str(OUTPUT_DIR / 'result1.xlsx')

tpl = xr.read_sheet_rows(TEMPLATE)
out = xr.read_sheet_rows(OUT)

print('1. 工作表名')
print('模板:', list(tpl.keys()))
print('输出:', list(out.keys()))
print('工作表名一致:', list(tpl.keys()) == list(out.keys()))

print('\n2. 计划购电量 表头与行标签')
tp1, ou1 = tpl[list(tpl.keys())[0]], out[list(out.keys())[0]]
print('表头一致:', tp1[0] == ou1[0], '->', ou1[0])
tlabels = [r[0] for r in tp1[1:145]]
olabels = [r[0] for r in ou1[1:145]]
print('行标签数:', len(tlabels), 'vs', len(olabels))
print('行标签完全一致:', tlabels == olabels)
if tlabels != olabels:
    for i, (a, b) in enumerate(zip(tlabels, olabels)):
        if a != b:
            print(f'  首个不一致 第{i+2}行: 模板[{a}] vs 输出[{b}]')
            break
print('首行标签:', olabels[0], '| 末行标签:', olabels[-1])

print('\n3. 计划购电量 数值语义对齐')
d = np.load(str(OUTPUT_DIR / 'prob1_solution.npz'))
x = d['x']
vals = [r[1] for r in ou1[1:145]]
# 模板 R2 "0:10-0:20" 应对应 时钟区间1 = x[1]；R145 "0:00+1-0:10+1" 对应 时钟区间0 = x[0]
expect = [x[(k + 1) % 144] for k in range(144)]
diff = np.max(np.abs(np.array(vals, dtype=float) - np.array(expect)))
print(f'B2..B145 与 x[(k+1)%144] 最大误差: {diff:.3e}  (应≈0)')
print(f'  B2(标签"0:10-0:20")  = {vals[0]:.4f}  == x[1] = {x[1]:.4f}')
print(f'  B145(标签"0:00+1-0:10+1") = {vals[-1]:.4f}  == x[0] = {x[0]:.4f}')

print('\n4. 充放电量 表头与标签')
tp2, ou2 = tpl[list(tpl.keys())[1]], out[list(out.keys())[1]]
print('表头一致:', tp2[0] == ou2[0], '->', ou2[0])
tl2 = [(r[0], r[3] if len(r) > 3 else None) for r in tp2[1:7]]
ol2 = [(r[0], r[3] if len(r) > 3 else None) for r in ou2[1:7]]
print('块标签一致:', [a for a, _ in tl2] == [a for a, _ in ol2])
print('0:00/24:00 标签:', [b for _, b in ol2 if b is not None])

print('\n5. 充放电量 数值语义对齐')
c, dd, E = d['c'], d['d'], d['E']
blocks = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
ok = True
for i, (a, b) in enumerate(blocks):
    row = ou2[i + 1]
    ch, di = row[1], row[2]
    exp_ch, exp_di = c[a:b].sum(), dd[a:b].sum()
    good = abs(ch - exp_ch) < 0.01 and abs(di - exp_di) < 0.01
    ok &= good
    print(f'  块{i+1} {row[0]:12s} 充{ch:9.2f}(应{exp_ch:9.2f}) 放{di:9.2f}(应{exp_di:9.2f}) {"OK" if good else "MISMATCH"}')
print(f'  E2(0:00储电) = {ou2[1][4]:.1f} (应 {E[0]:.1f})   E3(24:00储电) = {ou2[2][4]:.1f} (应 {E[-1]:.1f})')

print('\n结论')
struct_ok = (list(tpl.keys()) == list(out.keys()) and tp1[0] == ou1[0]
             and tlabels == olabels and tp2[0] == ou2[0])
print('结构与标签对齐:', 'PASS' if struct_ok else 'FAIL')
print('数值语义对齐:', 'PASS' if (diff < 1e-3 and ok) else 'FAIL')
