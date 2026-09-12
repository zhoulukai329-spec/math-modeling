"""Complete calendar battery tables and business-row emergency records."""
from copy import copy
from datetime import datetime, time, timedelta

import numpy as np
from openpyxl import load_workbook


def _physical(result):
    values = {
        stamp: {key: float(getattr(result, key)[i, j])
                for key in ('charge', 'discharge', 'soc_before', 'soc_after')}
        for i, row in enumerate(result.timestamps)
        for j, stamp in enumerate(row) if result.executed[i, j]
    }
    boundary = getattr(result, 'calendar_boundary', None)
    if boundary:
        values[datetime.fromisoformat(boundary['timestamp'])] = boundary
    return values


def _reset_sheet(sheet):
    styles = [copy(cell._style) for cell in sheet[2]]
    for merged in list(sheet.merged_cells.ranges):
        sheet.unmerge_cells(str(merged))
    sheet.delete_rows(2, sheet.max_row)
    return styles


def _append(sheet, values, styles):
    sheet.append(values)
    for cell, style in zip(sheet[sheet.max_row], styles):
        cell._style = copy(style)


def _clock(minutes):
    days, minute = divmod(minutes, 1440)
    return f'{minute // 60}:{minute % 60:02d}' + (f'+{days}' if days else '')


def write_workbook(result, output_path, template_path):
    from pathlib import Path
    output, template = Path(output_path), Path(template_path)
    if output.resolve() == template.resolve():
        raise ValueError('output must not overwrite the attachment template')
    book = load_workbook(template)
    lookup = {day: i for i, day in enumerate(result.dates)}
    for name, matrix in [('计划购电量', result.baseline), ('调整购电量', result.final_commitment)]:
        sheet = book[name]
        for r in range(2, sheet.max_row + 1):
            day = sheet.cell(r, 1).value.date()
            if day not in lookup:
                continue
            i = lookup[day]
            for j, value in enumerate(matrix[i], 2):
                sheet.cell(r, j, float(value) if np.isfinite(value) else None)
            if np.isfinite(matrix[i]).all():
                sheet.cell(r, 146, float(matrix[i].sum()))
                fee = (float(result.price[i] @ result.baseline[i]) if name == '计划购电量' else
                       sum(v.up_cost + v.down_cost for v in result.versions
                           if v.target_times[0].date() == day))
                sheet.cell(r, 147, fee)
    physical = _physical(result)
    sheet = book['充放电量']
    styles = _reset_sheet(sheet)
    for day in result.dates:
        midnight = datetime.combine(day, time())
        for block in range(6):
            stamps = [midnight + timedelta(hours=block * 4, minutes=10 * j) for j in range(24)]
            sums = ([sum(physical[t][key] for t in stamps) for key in ('charge', 'discharge')]
                    if all(t in physical for t in stamps) else [None, None])
            state = None
            if block < 2:
                target = midnight + timedelta(days=block)
                if target in physical:
                    state = physical[target]['soc_before']
                elif target - timedelta(minutes=10) in physical:
                    state = physical[target - timedelta(minutes=10)]['soc_after']
            _append(sheet, [midnight if block == 0 else None,
                           f'{block * 4}:00-{(block + 1) * 4}:00', *sums,
                           ('0:00' if block == 0 else '24:00') if block < 2 else None,
                           state], styles)
    sheet = book['紧急购电量']
    styles = _reset_sheet(sheet)
    for i, day in enumerate(result.dates):
        positive = np.flatnonzero(result.executed[i] & (result.emergency[i] > 1e-7))
        groups = np.split(positive, np.flatnonzero(np.diff(positive) != 1) + 1) if len(positive) else []
        midnight = datetime.combine(day, time())
        if not groups:
            _append(sheet, [midnight, '无' if result.executed[i].all() else '未完整执行',
                            0.0 if result.executed[i].all() else None], styles)
        for k, group in enumerate(groups):
            label = f'{_clock((int(group[0]) + 1) * 10)}-{_clock((int(group[-1]) + 2) * 10)}'
            if not result.executed[i].all():
                label = '已执行部分：' + label
            _append(sheet, [midnight if k == 0 else None, label,
                            float(result.emergency[i, group].sum())], styles)
    output.parent.mkdir(parents=True, exist_ok=True)
    book.save(output)
    book.close()
    return output


def verify_workbook(result, path, template_path, require, tolerance):
    """Check full coverage and recompute cells directly from physical arrays."""
    book, template = load_workbook(path), load_workbook(template_path)
    try:
        require(book.sheetnames == template.sheetnames, 'workbook sheets')
        if book.sheetnames != template.sheetnames:
            return
        def equal(actual, expected, label):
            if expected is None:
                require(actual is None, label)
            elif isinstance(expected, (int, float, np.number)):
                require(isinstance(actual, (int, float)) and np.isfinite(actual)
                        and abs(actual - expected) <= tolerance, label)
            else:
                require(actual == expected, label)
        by_day = {day: i for i, day in enumerate(result.dates)}
        for name, matrix in [('计划购电量', result.baseline), ('调整购电量', result.final_commitment)]:
            sheet, source = book[name], template[name]
            require((sheet.max_row, sheet.max_column) == (source.max_row, source.max_column), name + ' dimensions')
            for r in range(1, source.max_row + 1):
                day = source.cell(r, 1).value.date() if r > 1 else None
                for c in range(1, 148):
                    expected = source.cell(r, c).value
                    if day in by_day and c > 1:
                        i = by_day[day]
                        if c <= 145:
                            expected = float(matrix[i, c-2]) if np.isfinite(matrix[i, c-2]) else None
                        elif np.isfinite(matrix[i]).all():
                            expected = (float(matrix[i].sum()) if c == 146 else
                                        float(result.price[i] @ result.baseline[i]) if name == '计划购电量' else
                                        sum(v.up_cost + v.down_cost for v in result.versions if v.target_times[0].date() == day))
                    equal(sheet.cell(r, c).value, expected, f'{name}!{r},{c}')
        # Include the one preserved calendar-boundary action, without adding it to cash costs.
        physical = _physical(result)
        sheet = book['充放电量']
        require(sheet.max_row == 1 + 6 * len(result.dates), 'battery all-date coverage')
        for i, day in enumerate(result.dates):
            midnight = datetime.combine(day, time())
            for block in range(6):
                r = 2 + i * 6 + block
                equal(sheet.cell(r, 1).value, midnight if block == 0 else None, 'battery date')
                equal(sheet.cell(r, 2).value, f'{block*4}:00-{(block+1)*4}:00', 'battery block')
                start = midnight + timedelta(hours=4*block)
                stamps = [start + timedelta(minutes=10*j) for j in range(24)]
                for c, key in [(3, 'charge'), (4, 'discharge')]:
                    expected = sum(physical[t][key] for t in stamps) if all(t in physical for t in stamps) else None
                    equal(sheet.cell(r, c).value, expected, f'battery {day} block {block} {key}')
                expected = None
                if block < 2:
                    target = midnight + timedelta(days=block)
                    if target in physical:
                        expected = physical[target]['soc_before']
                    elif target - timedelta(minutes=10) in physical:
                        expected = physical[target-timedelta(minutes=10)]['soc_after']
                equal(sheet.cell(r, 6).value, expected, 'battery boundary SOC')
        sheet = book['紧急购电量']
        r = 2
        for i, day in enumerate(result.dates):
            intervals = []
            j = 0
            while j < 144:
                if not result.executed[i, j] or result.emergency[i, j] <= 1e-7:
                    j += 1
                    continue
                first = j
                while j < 144 and result.executed[i, j] and result.emergency[i, j] > 1e-7:
                    j += 1
                def stamp(k):
                    minutes = (k + 1) * 10
                    return f'{minutes % 1440 // 60}:{minutes % 60:02d}' + ('+1' if minutes >= 1440 else '')
                label = stamp(first) + '-' + stamp(j)
                if not result.executed[i].all():
                    label = '已执行部分：' + label
                intervals.append((label, float(result.emergency[i, first:j].sum())))
            if not intervals:
                intervals = [('无', 0.0)] if result.executed[i].all() else [('未完整执行', None)]
            for k, (label, amount) in enumerate(intervals):
                equal(sheet.cell(r, 1).value, datetime.combine(day, time()) if k == 0 else None, 'emergency date')
                equal(sheet.cell(r, 2).value, label, 'emergency interval')
                equal(sheet.cell(r, 3).value, amount, 'emergency amount')
                r += 1
        require(sheet.max_row == r-1, 'emergency all-date coverage')
    finally:
        book.close()
        template.close()
