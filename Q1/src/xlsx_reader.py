"""Minimal xlsx reader using only stdlib (zipfile + xml), no openpyxl.

Reads a .xlsx file and returns per-sheet data as a dict:
  { sheet_name: [ [cell_value, ...], ... ] }   (rows; None for empty)
Plus a helper to dump raw cell coordinate -> value for debugging.
"""
import zipfile, re, sys
from xml.etree import ElementTree as ET

NS = {
    'm': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}

def _col_to_idx(col_letters):
    n = 0
    for c in col_letters:
        n = n * 26 + (ord(c) - ord('A') + 1)
    return n - 1  # 0-based

def _cell_ref(coord):
    m = re.match(r'([A-Z]+)(\d+)', coord)
    return m.group(1), int(m.group(2))

def read_sheet_rows(path, sheet=None):
    """Return dict {sheet_name: list_of_rows(list_of_values)} for the file."""
    out = {}
    with zipfile.ZipFile(path) as z:
        # shared strings
        sst = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('m:si', NS):
                # concatenate all t nodes
                text = ''.join(t.text or '' for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'))
                sst.append(text)
        # workbook -> sheet name -> rId
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        sheet_names = []
        sheet_rids = []
        for sh in wb.findall('.//m:sheet', NS):
            sheet_names.append(sh.get('name'))
            sheet_rids.append(sh.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'))
        # rels: rId -> target
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rel_map = {}
        for rel in rels:
            rel_map[rel.get('Id')] = rel.get('Target')
        # read each worksheet
        for name, rid in zip(sheet_names, sheet_rids):
            target = rel_map[rid]
            if not target.startswith('xl/'):
                target = 'xl/' + target.lstrip('/')
            if target not in z.namelist():
                # try resolving via ../
                target = target.replace('xl/worksheets/', 'xl/worksheets/')
            data = z.read(target)
            root = ET.fromstring(data)
            rows = []
            for row in root.findall('.//m:sheetData/m:row', NS):
                cells = {}
                maxc = 0
                for c in row.findall('m:c', NS):
                    ref = c.get('r')
                    col, rn = _cell_ref(ref)
                    ci = _col_to_idx(col)
                    t = c.get('t')
                    v = None
                    vnode = c.find('m:v', NS)
                    isnode = c.find('m:is', NS)
                    if t == 's':
                        idx = int(vnode.text) if vnode is not None and vnode.text else 0
                        v = sst[idx] if idx < len(sst) else ''
                    elif t == 'inlineStr':
                        v = ''.join(tt.text or '' for tt in isnode.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'))
                    elif t == 'b':
                        v = vnode.text if vnode is not None else None
                    elif t == 'str':
                        v = vnode.text if vnode is not None else None
                    else:
                        v = vnode.text if vnode is not None else None
                        if v is not None:
                            try:
                                v = float(v)          # 数值单元格 -> float
                            except ValueError:
                                pass
                    cells[ci] = v
                    maxc = max(maxc, ci)
                rowvals = [cells.get(i) for i in range(maxc + 1)]
                rows.append(rowvals)
            out[name] = rows
    return out

def dump_sheet(path, sheet=None, max_rows=None, max_cols=None):
    data = read_sheet_rows(path)
    for name, rows in data.items():
        if sheet is not None and name != sheet:
            continue
        print(f'===== SHEET: {name}  ({len(rows)} rows) =====')
        for i, r in enumerate(rows):
            if max_rows is not None and i >= max_rows:
                print('   ... (truncated)')
                break
            vals = r if max_cols is None else r[:max_cols]
            print(f'  R{i+1}: {vals}')

if __name__ == '__main__':
    import glob
    for f in sys.argv[1:]:
        print('#' * 70)
        print('FILE:', f)
        dump_sheet(f)
