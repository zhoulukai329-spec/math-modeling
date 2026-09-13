# -*- coding: utf-8 -*-
"""Create result4-2.xlsx from the full Q4-2 solution archive."""
import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np

import data_io as dio


Q2_MODULE = dio.REPO_ROOT / "Q2" / "src" / "make_results.py"
spec = importlib.util.spec_from_file_location("_q2_make_results", Q2_MODULE)
_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)

build_plan_rows = _base.build_plan_rows
build_charge_rows = _base.build_charge_rows


def build_emergency_rows(dates, e, threshold=1e-7):
    """Return one row for every 10-minute interval with emergency purchase."""
    rows = [["日期", "购电时间段", "购电量"]]
    for i, serial in enumerate(dates):
        indices = np.flatnonzero(e[i] > threshold)
        for j, t in enumerate(indices):
            start = int(t) * 10
            end = start + 10
            label = f"{_base._fmt_minutes(start)}-{_base._fmt_minutes(end)}"
            date_val = float(serial) if j == 0 else None
            rows.append([date_val, label, float(e[i, t])])
    return rows


def write_result_xlsx(sheets, output_path=None):
    """Write three sheets using the official result4-2 template, never in place."""
    template = Path(dio.TEMPLATE42)
    output = Path(output_path) if output_path is not None else Path(dio.RESULT42)
    if template.resolve() == output.resolve():
        raise ValueError("结果文件不能覆盖附件模板")
    output.parent.mkdir(parents=True, exist_ok=True)
    worksheet_names = [
        "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml",
        "xl/worksheets/sheet3.xml",
    ]
    with zipfile.ZipFile(template, "r") as z_in, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED
    ) as z_out:
        for item in z_in.infolist():
            if item.filename.startswith("xl/worksheets/sheet") and item.filename.endswith(".xml"):
                continue
            z_out.writestr(item, z_in.read(item.filename))
        for filename, rows, styles in zip(
            worksheet_names, sheets, _base.SHEET_COL_STYLES
        ):
            xml = _base._build_formatted_sheet(z_in.read(filename), rows, styles)
            z_out.writestr(filename, xml)
    print(f"结果文件已生成: {output}")


def main():
    if not dio.SOLUTION_NPZ.exists():
        raise FileNotFoundError(
            f"缺少全年结果 {dio.SOLUTION_NPZ}，请先运行 --mode full"
        )
    with np.load(dio.SOLUTION_NPZ, allow_pickle=False) as z:
        complete = bool(z["complete"][0]) if "complete" in z.files else False
        if not complete:
            raise ValueError("结果档案不是完整全年运行，拒绝生成正式result4-2.xlsx")
        dates = z["dates"]
        if len(dates) != 334:
            raise ValueError(f"正式输出期应为334天，实际{len(dates)}天")
        for key in ("boundary_g0", "boundary_price0"):
            if key not in z.files or not np.isfinite(float(z[key][0])):
                raise ValueError(f"正式结果缺少有效跨年边界字段{key}，请重新运行full")
        sheets = [
            build_plan_rows(
                dates, z["g"], z["price"],
                float(z["boundary_g0"][0]), float(z["boundary_price0"][0]),
            ),
            build_charge_rows(dates, z["c"], z["d"], z["E"]),
            build_emergency_rows(dates, z["e"]),
        ]
    write_result_xlsx(sheets)


if __name__ == "__main__":
    main()
