# -*- coding: utf-8 -*-
"""问题 2 结果文件生成：论文表 1/2/3 与 result2.xlsx。

从 run_problem2.py 保存的 prob2_solution.npz 读取结果，复制附件 5 的
result2.xlsx 模板结构，生成完整 334 天结果。
"""
import re
import sys
import zipfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
Q_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import data_io as dio


def _escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _col_name(idx):
    """0-based 列号 -> Excel 列名。"""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


# 每列数据单元格的样式索引（复用附件 5 模板 cellXfs）：
# sheet1 计划购电量: 日期(3, 日期格式) + 144 时间(4) + 全天购电量/费(20,20)
# sheet2 充放电量: 日期(6) 时间段(7) 充电量(7) 放电量(8) 时刻(9) 储电量(7)
# sheet3 紧急购电量: 日期(6) 时间段(7) 购电量(7)
SHEET_COL_STYLES = [
    [3] + [4] * 144 + [20, 20],
    [6, 7, 7, 8, 9, 7],
    [6, 7, 7],
]


def _cell_xml(col, row_num, value, style=None):
    ref = f"{_col_name(col)}{row_num}"
    s_attr = f' s="{style}"' if style is not None else ""
    if value is None:
        return f'<c r="{ref}"{s_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}"{s_attr} t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and np.isnan(value):
            return f'<c r="{ref}"{s_attr}/>'
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return f'<c r="{ref}"{s_attr}><v>{text}</v></c>'
    # 字符串统一用 inlineStr，避免 sharedStrings 索引复杂化
    text = _escape(value)
    return f'<c r="{ref}"{s_attr} t="inlineStr"><is><t>{text}</t></is></c>'


def _data_row_xml(row_num, values, col_styles):
    cells = "".join(
        _cell_xml(col, row_num, value,
                  col_styles[col] if col < len(col_styles) else None)
        for col, value in enumerate(values)
    )
    return f'<row r="{row_num}">{cells}</row>'


def _fmt_minutes(minutes):
    minutes = minutes % 1440
    h = minutes // 60
    m = minutes % 60
    if minutes == 1440:
        return "24:00"
    if m == 0:
        return f"{h}:00"
    return f"{h}:{m:02d}"


def _find_date_idx(dates, target_serial):
    idx = np.where(np.isclose(dates, target_serial))[0]
    if len(idx) == 0:
        raise ValueError(f"日期序列号 {target_serial} 不在结果中")
    return int(idx[0])


def build_plan_rows(dates, g, planned_cost):
    header = dio.plan_header()
    rows = [header]
    N = dio.N
    for i, serial in enumerate(dates):
        # 模板顺序：x[1]..x[143], x[0]
        ordered = np.concatenate([g[i, 1:], g[i, :1]])
        row = [float(serial)] + list(ordered) + [
            float(g[i].sum()),
            float(planned_cost[i]),
        ]
        rows.append(row)
    return rows


def build_charge_rows(dates, c, d, E):
    header = ["日期", "时间段", "充电量", "放电量", "时刻", "储电量"]
    rows = [header]
    blocks = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
    labels = dio.block_labels()
    for i, serial in enumerate(dates):
        for j, (a, b) in enumerate(blocks):
            row = [None, labels[j], float(c[i, a:b].sum()), float(d[i, a:b].sum()), None, None]
            if j == 0:
                row[0] = float(serial)
                row[4] = "0:00"
                row[5] = float(E[i, 0])
            elif j == 1:
                row[4] = "24:00"
                row[5] = float(E[i, -1])
            rows.append(row)
    return rows


def build_emergency_rows(dates, e, threshold=1e-7):
    header = ["日期", "购电时间段", "购电量"]
    rows = [header]
    for i, serial in enumerate(dates):
        runs = []
        in_run = False
        start = None
        for t in range(dio.N):
            if e[i, t] > threshold:
                if not in_run:
                    start = t
                    in_run = True
            else:
                if in_run:
                    runs.append((start, t - 1))
                    in_run = False
        if in_run:
            runs.append((start, dio.N - 1))

        if not runs:
            continue
        for j, (a, b) in enumerate(runs):
            label = f"{_fmt_minutes(a * 10)}-{_fmt_minutes((b + 1) * 10)}"
            amount = float(e[i, a : b + 1].sum())
            date_val = float(serial) if j == 0 else None
            rows.append([date_val, label, amount])
    return rows


def _build_formatted_sheet(template_xml, rows, col_styles):
    """保留模板的列宽/样式/冻结窗格与表头行，只替换数据区。"""
    xml = template_xml.decode("utf-8")
    start = xml.index("<sheetData>")
    end = xml.index("</sheetData>") + len("</sheetData>")
    sheet_data_str = xml[start:end]
    row_start = sheet_data_str.index("<row ")
    row_end = sheet_data_str.index("</row>") + len("</row>")
    header_row = sheet_data_str[row_start:row_end]
    data_rows_xml = "".join(
        _data_row_xml(i + 2, values, col_styles)
        for i, values in enumerate(rows[1:])
    )
    new_xml = xml[:start] + f"<sheetData>{header_row}{data_rows_xml}</sheetData>" + xml[end:]
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 1
    new_ref = f"A1:{_col_name(n_cols - 1)}{n_rows}"
    new_xml = re.sub(r'<dimension ref="[^"]*"/>', f'<dimension ref="{new_ref}"/>', new_xml, count=1)
    return new_xml


def write_result_xlsx(sheets):
    """将三个工作表写入 result2.xlsx，保留附件 5 模板格式（列宽/样式/表头）。"""
    template = str(dio.TEMPLATE2)
    out = str(dio.RESULT2)
    z_in = zipfile.ZipFile(template, "r")
    z_out = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    worksheet_names = ["xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml", "xl/worksheets/sheet3.xml"]
    for item in z_in.infolist():
        if item.filename.startswith("xl/worksheets/sheet") and item.filename.endswith(".xml"):
            continue
        z_out.writestr(item, z_in.read(item.filename))
    for filename, rows, col_styles in zip(worksheet_names, sheets, SHEET_COL_STYLES):
        new_xml = _build_formatted_sheet(z_in.read(filename), rows, col_styles)
        z_out.writestr(filename, new_xml)
    z_in.close()
    z_out.close()
    print(f"结果文件已生成: {out}")


def main():
    z = np.load(str(dio.SOLUTION_NPZ))
    dates, price = z["dates"], z["price"]
    g, c, d, e, E = z["g"], z["c"], z["d"], z["e"], z["E"]
    planned_cost = z["planned_cost"]
    emergency_cost = z["emergency_cost"]
    total_cost = z["total_cost"]

    target_serials = [45736, 45829, 45923, 46012]
    target_names = ["2025.3.20", "2025.6.21", "2025.9.23", "2025.12.21"]
    spec_idx = [60, 72, 84, 96, 108, 120]
    spec_names = [
        "10:00-10:10", "12:00-12:10", "14:00-14:10",
        "16:00-16:10", "18:00-18:10", "20:00-20:10",
    ]
    block_names = dio.block_labels()

    print("=" * 90)
    print("表 1  指定日期的计划购电量（kWh）")
    print("=" * 90)
    for name, serial in zip(target_names, target_serials):
        i = _find_date_idx(dates, serial)
        print(f"\n{name}")
        for sname, k in zip(spec_names, spec_idx):
            print(f"  {sname:12s}  {g[i, k]:12.4f}")
        print(f"  全天购电量   {g[i].sum():12.4f} kWh")
        print(f"  全天购电费   {planned_cost[i]:12.4f} 元")

    print("\n" + "=" * 90)
    print("表 2  指定日期的储能充放电量（kWh）")
    print("=" * 90)
    for name, serial in zip(target_names, target_serials):
        i = _find_date_idx(dates, serial)
        print(f"\n{name}")
        for j, (bname, a, b) in enumerate(
            zip(block_names, [0, 24, 48, 72, 96, 120], [24, 48, 72, 96, 120, 144])
        ):
            print(f"  {bname:12s}  充电 {c[i, a:b].sum():12.4f}   放电 {d[i, a:b].sum():12.4f}")
        print(f"  0:00 储电量  {E[i, 0]:12.4f} kWh")
        print(f"  24:00 储电量 {E[i, -1]:12.4f} kWh")

    print("\n" + "=" * 90)
    print("表 3  指定日期的紧急购电量（kWh）")
    print("=" * 90)
    for name, serial in zip(target_names, target_serials):
        i = _find_date_idx(dates, serial)
        print(f"\n{name}")
        runs = []
        in_run = False
        start = None
        for t in range(dio.N):
            if e[i, t] > 1e-7:
                if not in_run:
                    start = t
                    in_run = True
            else:
                if in_run:
                    runs.append((start, t - 1))
                    in_run = False
        if in_run:
            runs.append((start, dio.N - 1))
        if not runs:
            print("  无紧急购电")
        for a, b in runs:
            label = f"{_fmt_minutes(a * 10)}-{_fmt_minutes((b + 1) * 10)}"
            print(f"  {label:14s}  {e[i, a:b + 1].sum():12.4f}")

    print("\n" + "=" * 90)
    print("输出期总费用")
    print("=" * 90)
    print(f"  计划购电总费用: {planned_cost.sum():.2f} 元")
    print(f"  紧急购电总费用: {emergency_cost.sum():.2f} 元")
    print(f"  总购电费用:     {total_cost.sum():.2f} 元")

    plan_rows = build_plan_rows(dates, g, planned_cost)
    charge_rows = build_charge_rows(dates, c, d, E)
    emergency_rows = build_emergency_rows(dates, e)
    write_result_xlsx([plan_rows, charge_rows, emergency_rows])


if __name__ == "__main__":
    main()

