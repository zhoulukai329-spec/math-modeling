# -*- coding: utf-8 -*-
"""
问题 1 结果生成：
  (a) 填写 result1.xlsx 模板（计划购电量 / 充放电量 两个工作表）
  (b) 输出论文中表 1、表 2 的数值
  (c) 基线对比（无储能）
"""
import sys
from pathlib import Path

#路径
SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
REPO_ROOT = Q_DIR.parent
ATTACH_DIR = REPO_ROOT / 'attachment'
OUTPUT_DIR = Q_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np, zipfile, shutil, os
from xml.etree import ElementTree as ET

MAIN_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
ET.register_namespace('', MAIN_NS)

d = np.load(str(OUTPUT_DIR / 'prob1_solution.npz'))
t_min, price = d['t_min'], d['price']
L, G = d['L'], d['G']
x, c, dd, w, E = d['x'], d['c'], d['d'], d['w'], d['E']
cost = float(d['cost'])
N = len(x)

# 表1: 指定时间段购电量
spec = [('10:00-10:10', 60), ('12:00-12:10', 72), ('14:00-14:10', 84),
        ('16:00-16:10', 96), ('18:00-18:10', 108), ('20:00-20:10', 120)]
print('=' * 70)
print('表 1  微网在指定时间段的购电量及全天的购电量和购电费')
print('=' * 70)
for name, k in spec:
    print(f'  {name:12s}  购电量 = {x[k]:10.4f} kWh')
print(f'  {"全天购电量":12s}  = {x.sum():10.4f} kWh')
print(f'  {"全天购电费":12s}  = {cost:10.4f} 元')

# 表2: 指定时间段充放电量
blocks = [('0:00-4:00', 0, 24), ('4:00-8:00', 24, 48), ('8:00-12:00', 48, 72),
          ('12:00-16:00', 72, 96), ('16:00-20:00', 96, 120), ('20:00-24:00', 120, 144)]
print('\n' + '=' * 70)
print('表 2  储能设备在指定时间段的充放电量及 0:00 / 24:00 储电量')
print('=' * 70)
chg_blocks, dis_blocks = [], []
for name, a, b in blocks:
    ch = c[a:b].sum(); di = dd[a:b].sum()
    chg_blocks.append(ch); dis_blocks.append(di)
    print(f'  {name:12s}  充电量 = {ch:10.4f} kWh   放电量 = {di:10.4f} kWh')
print(f'  {"0:00 储电量":12s} = {E[0]:10.4f} kWh')
print(f'  {"24:00 储电量":12s} = {E[-1]:10.4f} kWh')

# 基线：无储能
x_base = np.maximum(0.0, L - G)      # 无储能时每区间购电量 = max(0, 净负载)
cost_base = float((price * x_base).sum())
print('\n' + '=' * 70)
print('基线对比（无储能）')
print('=' * 70)
print(f'  无储能全天购电量 = {x_base.sum():.4f} kWh,  购电费 = {cost_base:.4f} 元')
print(f'  有储能全天购电量 = {x.sum():.4f} kWh,  购电费 = {cost:.4f} 元')
print(f'  储能节省购电费   = {cost_base - cost:.4f} 元 ({(cost_base-cost)/cost_base*100:.2f}%)')

# 填写 result1.xlsx
def fill_cell(row_el, col_letter, value):
    """在指定行元素中, 设置某列单元格为数值."""
    ref = col_letter + row_el.get('r')
    # 查找已存在的 cell
    c = None
    for child in row_el.findall(f'{{{MAIN_NS}}}c'):
        if child.get('r') == ref:
            c = child; break
    if c is None:
        c = ET.SubElement(row_el, f'{{{MAIN_NS}}}c')
        c.set('r', ref)
    # 清除已有 v
    for v in c.findall(f'{{{MAIN_NS}}}v'):
        c.remove(v)
    c.attrib.pop('t', None)          # 数值类型 (去掉共享字符串标记)
    v = ET.SubElement(c, f'{{{MAIN_NS}}}v')
    v.text = f'{value:.4f}'

TEMPLATE = str(ATTACH_DIR / '附件5' / 'result1.xlsx')
OUT = str(OUTPUT_DIR / 'result1.xlsx')
zin = zipfile.ZipFile(TEMPLATE, 'r')

def write_sheet(zout, name, root):
    data = ET.tostring(root, encoding='UTF-8', xml_declaration=True)
    zout.writestr(f'xl/worksheets/{name}.xml', data)

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename in ('xl/worksheets/sheet1.xml', 'xl/worksheets/sheet2.xml'):
            continue
        zout.writestr(item, zin.read(item.filename))

    # sheet1: 计划购电量 (B2..B145)
    # 内部数组是自然日顺序：x[0]=0:00-0:10 ... x[143]=23:50-24:00；
    # 官方模板行顺序是：0:10-0:20 ... 23:50-24:00, 次日0:00-0:10。
    # 因此模板第 k+2 行应写 x[(k+1) % N]。
    root1 = ET.fromstring(zin.read('xl/worksheets/sheet1.xml'))
    rows1 = {r.get('r'): r for r in root1.findall(f'{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row')}
    for k in range(N):
        fill_cell(rows1[str(k + 2)], 'B', x[(k + 1) % N])
    write_sheet(zout, 'sheet1', root1)

    # 充放电量
    root2 = ET.fromstring(zin.read('xl/worksheets/sheet2.xml'))
    rows2 = {r.get('r'): r for r in root2.findall(f'{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row')}
    for i in range(6):
        fill_cell(rows2[str(i + 2)], 'B', chg_blocks[i])
        fill_cell(rows2[str(i + 2)], 'C', dis_blocks[i])
    fill_cell(rows2['2'], 'E', E[0])    # 0:00 储电量
    fill_cell(rows2['3'], 'E', E[-1])   # 24:00 储电量
    write_sheet(zout, 'sheet2', root2)

zin.close()
print(f'\n已生成结果文件: {OUT}')
print('工作表“计划购电量”: 144 个 10 分钟区间购电量 (B2:B145)')
print('工作表“充放电量”:   6 个 4 小时块充/放电量 (B2:C7) + 0:00/24:00 储电量 (E2:E3)')
