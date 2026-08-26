"""修复raw_data撕裂行(流式版): 每次只处理1只股票的3个复权文件, 内存安全.

派生字段(振幅9/涨跌幅10/涨跌额11/换手率12)跨复权同值, 从同日期完好行复制
(来源含同文件重复行+兄弟复权文件); 无来源时置空(''->NaN不崩溃, '-'才会崩溃).
撕裂来源: 2026-08-21损坏事件的交错写入, OHLC/量额完好, 仅尾部派生字段残缺.
"""
import csv
import os
import re
import shutil
from pathlib import Path

RAW = Path('/mnt/d/quant/data/stock_data/raw_data')
BACKUP = Path('/mnt/d/quant/data/backup_torn_rows_20260821')
TYPES = ['none', 'qfq', 'hfq']
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DERIVED_IDX = [9, 10, 11, 12]
OHLCV_IDX = [3, 4, 5, 6, 7, 8]


def is_num(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def read_file(path):
    """返回(eol, lines, rows)或None. rows[i]为None表示空行."""
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        content = f.read()
    eol = '\r\n' if '\r\n' in content else '\n'
    lines = content.split(eol)
    if lines and lines[-1] == '':
        lines = lines[:-1]
    rows = []
    for ln in lines[1:]:
        if not ln:
            rows.append(None)
            continue
        rows.append(next(csv.reader([ln])))
    return eol, lines, rows


def main():
    dirs = sorted(p for p in RAW.iterdir() if p.is_dir() and re.fullmatch(r'\d{6}', p.name))
    n_stocks_bad = n_files_changed = n_bad_rows = n_fixed_fields = 0
    no_sibling, ohlcv_bad, bad_dates = [], [], []

    for d in dirs:
        code = d.name
        loaded = {}
        for t in TYPES:
            p = d / f'{t}.csv'
            if not p.exists():
                continue
            try:
                loaded[t] = read_file(p)
            except Exception as e:
                print(f'  读取异常 {code}/{t}: {e}')
                continue
        if not loaded:
            continue

        # 兄弟值表: date -> {idx: 值}, 来源=所有文件中该日期的完好行
        sib = {}
        for eol, lines, rows in loaded.values():
            for row in rows:
                if row is None or len(row) != 13 or not DATE_RE.match(row[2] or ''):
                    continue
                if all(is_num(row[i]) for i in DERIVED_IDX):
                    sib.setdefault(row[2], {}).setdefault(0, [])
                    vals = {i: row[i] for i in DERIVED_IDX if i not in sib[row[2]]}
                    sib[row[2]].update(vals)

        stock_changed = False
        for t, (eol, lines, rows) in loaded.items():
            changed = False
            for i, row in enumerate(rows):
                if row is None or len(row) < 3 or not DATE_RE.match(row[2] or ''):
                    continue
                bad_fields = []
                for idx in DERIVED_IDX:
                    if idx >= len(row):
                        bad_fields.append(idx)
                    elif row[idx] == '' or not is_num(row[idx]):
                        bad_fields.append(idx)
                if not bad_fields:
                    for idx in OHLCV_IDX:
                        if idx >= len(row) or row[idx] == '' or not is_num(row[idx]):
                            ohlcv_bad.append((code, t, row[2], idx))
                            break
                    continue
                n_bad_rows += 1
                bad_dates.append(row[2])
                date_sib = sib.get(row[2], {})
                row = list(row) + [''] * (13 - len(row))
                for idx in bad_fields:
                    if idx in date_sib:
                        row[idx] = date_sib[idx]
                        n_fixed_fields += 1
                    else:
                        row[idx] = ''
                        no_sibling.append((code, t, row[2], idx))
                lines[i + 1] = ','.join(row)
                changed = True
            if changed:
                assert len(lines) == len(rows) + 1, f'{code}/{t} 行数对账失败'
                p = d / f'{t}.csv'
                bak = BACKUP / code
                bak.mkdir(parents=True, exist_ok=True)
                if not (bak / f'{t}.csv').exists():
                    shutil.copy2(p, bak / f'{t}.csv')
                tmp = p.with_suffix('.tmp')
                with open(tmp, 'w', encoding='utf-8', newline='') as f:
                    f.write(eol.join(lines) + eol)
                os.replace(tmp, p)
                n_files_changed += 1
                stock_changed = True
        if stock_changed:
            n_stocks_bad += 1
            if n_stocks_bad % 50 == 0:
                print(f'  已处理 {n_stocks_bad} 只有撕裂行的股票...')

    print(f'\n撕裂行: {n_bad_rows} 行, {n_files_changed} 个文件, {n_stocks_bad} 只股票')
    print(f'修复字段: {n_fixed_fields}, 无来源置空: {len(no_sibling)}')
    print(f'OHLC/量额撕裂(无法本地修复): {len(ohlcv_bad)}')
    for x in ohlcv_bad[:10]:
        print(f'  {x[0]}/{x[1]} {x[2]} idx={x[3]}')
    if bad_dates:
        recent = sum(1 for d in bad_dates if d >= '2026-08-01')
        print(f'撕裂日期分布: 2026-08以来 {recent} 行, 更早历史 {len(bad_dates)-recent} 行')
        print(f'样例日期: {sorted(set(bad_dates))[:5]} ... {sorted(set(bad_dates))[-5:]}')
    print(f'备份: {BACKUP}')


if __name__ == '__main__':
    main()
