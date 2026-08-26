"""截掉raw_data中今天(2026-08-21)的行, 供用户重跑数据更新干净追加.

被删行全部写入journal日志(可完整回滚). 同时报告单文件删除>1行的重复情况.
"""
import csv
import os
import re
import shutil
from pathlib import Path

RAW = Path('/mnt/d/quant/data/stock_data/raw_data')
JOURNAL = Path('/mnt/d/quant/data/backup_torn_rows_20260821/removed_rows_20260821.log')
TYPES = ['none', 'qfq', 'hfq']
TODAY = '2026-08-21'
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def read_file(path):
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        content = f.read()
    eol = '\r\n' if '\r\n' in content else '\n'
    lines = content.split(eol)
    if lines and lines[-1] == '':
        lines = lines[:-1]
    return eol, lines


def row_date(ln):
    """返回该行的日期字段(现格式row[2], 旧格式row[0]), 无法解析返回None."""
    try:
        row = next(csv.reader([ln]))
    except (StopIteration, csv.Error):
        return None
    if len(row) >= 3 and DATE_RE.match(row[2] or ''):
        return row[2]
    if row and DATE_RE.match(row[0] or ''):
        return row[0]
    return None


def main():
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in RAW.iterdir() if p.is_dir() and re.fullmatch(r'\d{6}', p.name))
    n_files_changed = n_removed = 0
    dup_files = []
    jf = open(JOURNAL, 'w', encoding='utf-8')

    for d in dirs:
        for t in TYPES:
            p = d / f'{t}.csv'
            if not p.exists():
                continue
            try:
                eol, lines = read_file(p)
            except Exception as e:
                print(f'  读取异常 {d.name}/{t}: {e}')
                continue
            if len(lines) < 2:
                continue
            keep, removed = lines[:1], []
            for ln in lines[1:]:
                if ln and row_date(ln) == TODAY:
                    removed.append(ln)
                else:
                    keep.append(ln)
            if not removed:
                continue
            for ln in removed:
                jf.write(f'{d.name}/{t}\t{ln}\n')
            tmp = p.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8', newline='') as f:
                f.write(eol.join(keep) + eol)
            os.replace(tmp, p)
            n_files_changed += 1
            n_removed += len(removed)
            if len(removed) > 1:
                dup_files.append((d.name, t, len(removed)))
    jf.close()

    print(f'截掉 {TODAY} 行: {n_removed} 行, {n_files_changed} 个文件')
    print(f'单文件多行(重复迹象): {len(dup_files)}')
    for x in dup_files[:20]:
        print(f'  {x[0]}/{x[1]} 删了{x[2]}行')
    print(f'回滚日志: {JOURNAL}')


if __name__ == '__main__':
    main()
