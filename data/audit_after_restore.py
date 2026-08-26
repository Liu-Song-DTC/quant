"""恢复后全面审计: 检查三复权文件日期一致性+legacy格式, 补算缺失的fq_factors.

目的: 数据恢复完成后, 确定哪些股票真正需要xtquant全量重下,
并把因文件损坏导致"计算fq factors失败"的股票就地补算(数据完整的).
"""
import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path('/mnt/d/quant/data/stock_data/raw_data')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
ADJ_TYPES = ['none', 'qfq', 'hfq']


def read_dates(path):
    """返回(date_set, legacy_flag, read_error)."""
    try:
        with open(path, encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or '股票名称' not in header[0]:
                return set(), True, False
            dates = set()
            for row in reader:
                if len(row) == 13 and DATE_RE.match(row[2] or ''):
                    dates.add(row[2])
            return dates, False, False
    except Exception:
        return set(), False, True


def recalc_fq(code):
    """与xtquant_downloader.calc_fq_factors同逻辑: 增量追加缺失日期."""
    d = RAW / code
    paths = {t: d / f'{t}.csv' for t in ADJ_TYPES}
    if not all(p.exists() for p in paths.values()):
        return 'skip'
    try:
        dfs = {}
        for t, p in paths.items():
            df = pd.read_csv(p, parse_dates=['日期'], encoding='utf-8')
            df = df[['日期', '收盘']].copy()
            df.rename(columns={'收盘': f'收盘_{t}'}, inplace=True)
            dfs[t] = df
        merged = dfs['none'].merge(dfs['qfq'], on='日期', how='inner') \
                            .merge(dfs['hfq'], on='日期', how='inner')
        if merged.empty:
            return 'empty_merge'
        factors = pd.DataFrame({
            '日期': merged['日期'].dt.date,
            'symbol': code,
            'qfq_factor': np.round(merged['收盘_qfq'] / merged['收盘_none'].replace(0, np.nan), 6),
            'hfq_factor': np.round(merged['收盘_hfq'] / merged['收盘_none'].replace(0, np.nan), 6),
        }).dropna()
        fq_path = d / 'fq_factors.csv'
        if fq_path.exists():
            old = pd.read_csv(fq_path, parse_dates=['日期'])
            old['日期'] = old['日期'].dt.date
            new_rows = factors[~factors['日期'].isin(set(old['日期']))]
            if new_rows.empty:
                return 'ok'
            factors = pd.concat([old, new_rows], ignore_index=True).sort_values('日期')
        factors.to_csv(fq_path, index=False, encoding='utf-8')
        return 'ok'
    except Exception as e:
        return f'fail: {str(e)[:60]}'


def main():
    t0 = datetime.now()
    dirs = sorted(d for d in RAW.iterdir()
                  if d.is_dir() and re.fullmatch(r'\d{6}', d.name))
    print(f'审计 {len(dirs)} 只股票...')
    need_redownload, fq_recalc, fq_fails = [], [], []

    for i, d in enumerate(dirs):
        code = d.name
        date_sets, legacy, read_err = {}, False, False
        for t in ADJ_TYPES:
            p = d / f'{t}.csv'
            if not p.exists():
                continue
            dates, lg, re_ = read_dates(p)
            date_sets[t] = dates
            legacy = legacy or lg
            read_err = read_err or re_
        if read_err:
            need_redownload.append(f'{code}\tread_error\t文件读取异常')
            continue
        if legacy:
            need_redownload.append(f'{code}\tlegacy\t旧格式无股票名称列')
            continue
        if len(date_sets) == 3:
            union = set.union(*date_sets.values())
            missing = {t: len(union - s) for t, s in date_sets.items()}
            if any(v > 0 for v in missing.values()):
                need_redownload.append(
                    f'{code}\tmissing_dates\t{missing} (并集{len(union)})')
        # fq_factors补算: 无论是否有缺失, 就地补齐可用日期
        st = recalc_fq(code)
        if st.startswith('fail'):
            fq_fails.append((code, st))
        elif st == 'ok':
            pass  # 无法区分是否追加了, 统一视为成功
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(dirs)} ...')

    # fq_factors日期覆盖检查(补算后再看还缺多少)
    stale_fq = []
    for line in need_redownload:
        code = line.split('\t')[0]
        if 'missing' not in line and 'read' not in line:
            continue
        d = RAW / code
        p = d / 'fq_factors.csv'
        if not p.exists():
            stale_fq.append((code, 'no_fq'))
            continue
        try:
            fq_dates = pd.read_csv(p, usecols=['日期'])['日期']
            fq_last = pd.to_datetime(fq_dates).max()
            raw_dates = read_dates(d / 'none.csv')[0]
            raw_last = max(raw_dates) if raw_dates else None
            if raw_last and fq_last < pd.Timestamp(raw_last):
                stale_fq.append((code, f'fq止于{fq_last.date()} raw止于{raw_last}'))
        except Exception as e:
            stale_fq.append((code, f'检查失败:{str(e)[:40]}'))

    out = Path('/mnt/d/quant/data/backup_raw_corrupt_20260821/redownload_list.txt')
    with open(out, 'w') as f:
        f.write('# 恢复后审计 (2026-08-21): 需xtquant全量重下的股票\n')
        for line in need_redownload:
            f.write(line + '\n')
    print(f'\n===== 审计完成 ({datetime.now()-t0}) =====')
    print(f'需全量重下: {len(need_redownload)} 只')
    for line in need_redownload[:15]:
        print(f'  {line}')
    if len(need_redownload) > 15:
        print(f'  ... 共{len(need_redownload)}只, 详见 {out}')
    print(f'fq_factors补算失败: {len(fq_fails)}')
    for c, m in fq_fails[:10]:
        print(f'  {c}: {m}')
    print(f'fq仍落后于raw: {len(stale_fq)}')
    for c, m in stale_fq[:10]:
        print(f'  {c}: {m}')


if __name__ == '__main__':
    main()
