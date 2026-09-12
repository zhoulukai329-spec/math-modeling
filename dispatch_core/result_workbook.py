"""Preserve the official submission template and independently audit all cells."""
from datetime import datetime, timedelta
from pathlib import Path
import os
import tempfile
import numpy as np
from openpyxl import load_workbook
from .calendar_boundary import boundary_errors
from .workbook_contract import preflight_template


def _physical(result):
    rows = {stamp: (float(result.charge[i,j]), float(result.discharge[i,j]),
                    float(result.soc_before[i,j]), float(result.soc_after[i,j]))
            for i,row in enumerate(result.timestamps) for j,stamp in enumerate(row)
            if result.executed[i,j]}
    boundary = getattr(result, "calendar_boundary", None)
    if boundary:
        rows[datetime.fromisoformat(boundary["timestamp"])] = tuple(
            float(boundary[k]) for k in ("charge","discharge","soc_before","soc_after"))
    return rows


def write_workbook(result, output_path, template_path):
    output = Path(output_path)
    preflight_template(template_path, output)
    errors = boundary_errors(result)
    if errors:
        raise ValueError("; ".join(errors))
    book = load_workbook(template_path)
    temporary = None
    try:
        by_day = {day:i for i,day in enumerate(result.dates)}
        for name, matrix in (("计划购电量",result.baseline),("调整购电量",result.final_commitment)):
            sheet = book[name]
            for row in range(2,sheet.max_row+1):
                day = sheet.cell(row,1).value.date()
                if day not in by_day:
                    continue
                i = by_day[day]
                for column,value in enumerate(matrix[i],2):
                    sheet.cell(row,column).value = float(value) if np.isfinite(value) else None
                if np.isfinite(matrix[i]).all():
                    sheet.cell(row,146).value = float(matrix[i].sum())
                    sheet.cell(row,147).value = (float(result.price[i] @ matrix[i]) if name == "计划购电量"
                        else float(sum(v.up_cost+v.down_cost for v in result.versions if v.target_times[0].date() == day)))
        physical = _physical(result)
        sheet = book["充放电量"]
        for row in range(2,sheet.max_row+1):
            anchor = sheet.cell(row,1).value
            if not isinstance(anchor,datetime):
                continue
            for block in range(6):
                stamps = [anchor+timedelta(hours=4*block,minutes=10*k) for k in range(24)]
                for column,index in ((3,0),(4,1)):
                    sheet.cell(row+block,column).value = (
                        float(sum(physical[t][index] for t in stamps)) if all(t in physical for t in stamps) else None)
            for offset in (0,1):
                t = anchor+timedelta(days=offset)
                value = physical[t][2] if t in physical else (
                    physical[t-timedelta(minutes=10)][3] if t-timedelta(minutes=10) in physical else (
                    result.config.initial_soc if offset == 0 and anchor.date() == result.dates[0] else None))
                sheet.cell(row+offset,6).value = value
        sheet = book["紧急购电量"]
        for row in range(2,sheet.max_row+1):
            anchor = sheet.cell(row,1).value
            if not isinstance(anchor,datetime) or anchor.date() not in by_day:
                continue
            i = by_day[anchor.date()]
            active = result.executed[i]
            indices = np.flatnonzero(active & (result.emergency[i] > 1e-7))
            if len(indices) or active.all():
                text = "\n".join(book["计划购电量"].cell(1,int(j)+2).value for j in indices) if len(indices) else "无"
                sheet.cell(row,2).value = ("" if active.all() else "已执行部分：\n")+text
                sheet.cell(row,3).value = float(result.emergency[i,active].sum())
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent,suffix=".xlsx",delete=False) as handle:
            temporary = Path(handle.name)
        book.save(temporary)
        os.replace(temporary,output)
        temporary = None
        return output
    finally:
        book.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_workbook(result, path, template_path, require, tolerance):
    """Reconstruct allowed cells separately from the writer; retain all template constants."""
    errors = boundary_errors(result,tolerance)
    for error in errors:
        require(False,error)
    if errors:
        return
    preflight_template(template_path)
    book, source = load_workbook(path), load_workbook(template_path)
    try:
        require(book.sheetnames == source.sheetnames,"workbook sheets")
        if book.sheetnames != source.sheetnames:
            return
        expected = {}
        by_day = {day:i for i,day in enumerate(result.dates)}
        for name,matrix in (("计划购电量",result.baseline),("调整购电量",result.final_commitment)):
            for row in range(2,source[name].max_row+1):
                day = source[name].cell(row,1).value.date()
                if day not in by_day:
                    continue
                i = by_day[day]
                for j in range(144):
                    expected[name,row,j+2] = float(matrix[i,j]) if np.isfinite(matrix[i,j]) else None
                if np.isfinite(matrix[i]).all():
                    expected[name,row,146] = float(np.sum(matrix[i]))
                    expected[name,row,147] = (float(np.sum(result.price[i]*matrix[i])) if name == "计划购电量"
                        else float(sum(v.up_cost+v.down_cost for v in result.versions if v.target_times[0].date() == day)))
        # Independent index: do not call the writer's physical lookup.
        timestamps = list(result.timestamps[result.executed])
        values = [list(getattr(result,k)[result.executed]) for k in ("charge","discharge","soc_before","soc_after")]
        boundary = getattr(result,"calendar_boundary",None)
        if boundary:
            timestamps.append(datetime.fromisoformat(boundary["timestamp"]))
            for vector,key in zip(values,("charge","discharge","soc_before","soc_after")):
                vector.append(boundary[key])
        index = {stamp:i for i,stamp in enumerate(timestamps)}
        name = "充放电量"
        for row in range(2,source[name].max_row+1):
            anchor = source[name].cell(row,1).value
            if not isinstance(anchor,datetime):
                continue
            for block in range(6):
                keys = [anchor+timedelta(minutes=240*block+10*j) for j in range(24)]
                for column,vector in ((3,values[0]),(4,values[1])):
                    expected[name,row+block,column] = float(np.sum([vector[index[t]] for t in keys])) if all(t in index for t in keys) else None
            for offset in (0,1):
                t = anchor+timedelta(days=offset)
                expected[name,row+offset,6] = (float(values[2][index[t]]) if t in index else
                    float(values[3][index[t-timedelta(minutes=10)]]) if t-timedelta(minutes=10) in index else
                    result.config.initial_soc if offset == 0 and anchor.date() == result.dates[0] else None)
        name = "紧急购电量"
        for row in range(2,source[name].max_row+1):
            anchor = source[name].cell(row,1).value
            if not isinstance(anchor,datetime) or anchor.date() not in by_day:
                continue
            i = by_day[anchor.date()]
            selected = [j for j in range(144) if result.executed[i,j] and result.emergency[i,j] > 1e-7]
            if selected or result.executed[i].all():
                text = "\n".join(source["计划购电量"].cell(1,j+2).value for j in selected) if selected else "无"
                expected[name,row,2] = ("" if result.executed[i].all() else "已执行部分：\n")+text
                expected[name,row,3] = float(sum(result.emergency[i,j] for j in range(144) if result.executed[i,j]))
        for original in source:
            sheet = book[original.title]
            require((sheet.max_row,sheet.max_column) == (original.max_row,original.max_column),f"workbook {sheet.title} dimensions")
            require(str(sheet.merged_cells) == str(original.merged_cells),f"workbook {sheet.title} merges")
            require(sheet.freeze_panes == original.freeze_panes and sheet.print_area == original.print_area,f"workbook {sheet.title} view")
            column_layout = lambda ws: {k:(v.width,v.hidden,v.min,v.max) for k,v in ws.column_dimensions.items()}
            row_layout = lambda ws: {k:(v.height,v.hidden,v.outlineLevel) for k,v in ws.row_dimensions.items()}
            require(column_layout(sheet) == column_layout(original),f"workbook {sheet.title} column layout")
            require(row_layout(sheet) == row_layout(original),f"workbook {sheet.title} row layout")
            for row in original:
                for cell in row:
                    actual = sheet.cell(cell.row,cell.column)
                    wanted = expected.get((sheet.title,cell.row,cell.column),cell.value)
                    valid = (isinstance(actual.value,(int,float)) and np.isfinite(actual.value) and abs(actual.value-wanted) <= tolerance
                             if isinstance(wanted,(int,float,np.number)) else actual.value == wanted)
                    label = f"workbook {sheet.title}!{cell.coordinate}"
                    require(valid,label)
                    require(tuple(actual._style or (0,)*9) == tuple(cell._style or (0,)*9),label+" style")
    finally:
        book.close()
        source.close()
