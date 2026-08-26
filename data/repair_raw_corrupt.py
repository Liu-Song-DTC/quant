"""修复 raw_data 损坏文件: 去垃圾行/去重/排序/原子写 + 重算fq_factors.

损坏原因(2026-08-21): 更新运行被中断后重跑, save_to_raw的except-fallback
用"仅新数据"覆盖旧文件 + 交错写入产生碎片行/乱序/重复.
"""
import csv
import os
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RAW = Path('/mnt/d/quant/data/stock_data/raw_data')
BACKUP = Path('/mnt/d/quant/data/backup_raw_corrupt_20260821')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
ADJ_TYPES = ['none', 'qfq', 'hfq']


def read_clean_rows(path, code):
    """容错读取: 只保留13字段且日期合法的行, 返回(rows, n_garbage, n_legacy)."""
    rows, garbage = [], 0
    try:
        with open(path, encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return [], 0, 0
            # 旧格式(无股票名称列, 日期在首位) -> 不属于本修复范围
            legacy = ('股票名称' not in header[0])
            for row in reader:
                if not row:
                    continue
                if len(row) == 13 and DATE_RE.match(row[2] or '') and (row[1] or '') == code:
                    rows.append(row)
                else:
                    garbage += 1
            return rows, garbage, (1 if legacy else 0)
    except Exception:
        return [], -1, 0


def repair_file(path, code):
    """清洗并原子写回. 返回(状态, 变化描述)."""
    rows, garbage, legacy = read_clean_rows(path, code)
    if legacy:
        return 'legacy', '旧格式文件, 需xtquant全量重下'
    if garbage < 0:
        return 'read_error', '读取异常'
    if garbage == 0:
        # 无垃圾行也可能有重复/乱序
        dates = [r[2] for r in rows]
        if len(dates) == len(set(dates)) and dates == sorted(dates):
            return 'clean', ''

    # 备份
    bak_dir = BACKUP / code
    bak_dir.mkdir(parents=True, exist_ok=True)
    bak_path = bak_dir / path.name
    if not bak_path.exists():
        shutil.copy2(path, bak_path)

    # 去重(保留最后出现的) + 排序
    dedup = {}
    for r in rows:
        dedup[r[2]] = r
    cleaned = [dedup[d] for d in sorted(dedup)]

    # 原子写: 临时文件+rename
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(STOCK_HEADER)
        w.writerows(cleaned)
    os.replace(tmp, path)
    n_dup = len(rows) - len(cleaned)
    return 'repaired', f'垃圾{garbage}行 去重{n_dup}行 {len(rows)}->{len(cleaned)}行'


STOCK_HEADER = ['股票名称', '股票代码', '日期', '开盘', '收盘', '最高', '最低',
                '成交量', '成交额', '换手率', '涨跌幅', '涨跌额', '振幅']


def recalc_fq(code):
    """重算fq_factors(与xtquant_downloader.calc_fq_factors同逻辑)."""
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
    stats = {'clean': 0, 'repaired': 0, 'legacy': 0, 'read_error': 0}
    repaired_codes, redownload_codes, fq_fails = [], [], []

    dirs = sorted(d for d in RAW.iterdir()
                  if d.is_dir() and re.fullmatch(r'\d{6}', d.name))
    print(f'扫描 {len(dirs)} 只股票...')
    for i, d in enumerate(dirs):
        code = d.name
        code_changed = False
        for t in ADJ_TYPES:
            p = d / f'{t}.csv'
            if not p.exists():
                continue
            status, msg = repair_file(p, code)
            stats[status] = stats.get(status, 0) + 1
            if status == 'repaired':
                code_changed = True
            elif status in ('legacy', 'read_error'):
                redownload_codes.append((code, t, msg))
        if code_changed:
            repaired_codes.append(code)
            fq_st = recalc_fq(code)
            if fq_st not in ('ok', 'skip'):
                fq_fails.append((code, fq_st))
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(dirs)} ...')

    # 数据丢失检查: 三个复权文件日期集一致性
    lost = []
    for code in repaired_codes:
        d = RAW / code
        date_sets = {}
        for t in ADJ_TYPES:
            p = d / f'{t}.csv'
            if p.exists():
                rows, _, legacy = read_clean_rows(p, code)
                if not legacy:
                    date_sets[t] = {r[2] for r in rows}
        if len(date_sets) == 3:
            union = set.union(*date_sets.values())
            missing = {t: len(union - s) for t, s in date_sets.items()}
            if any(v > 0 for v in missing.values()):
                lost.append((code, missing))

    print(f'\n===== 修复完成 ({datetime.now()-t0}) =====')
    print(f"干净文件: {stats['clean']}, 修复: {stats['repaired']}, "
          f"旧格式: {stats['legacy']}, 读取异常: {stats['read_error']}")
    print(f'修复股票数: {len(repaired_codes)}')
    if fq_fails:
        print(f'fq_factors重算失败: {len(fq_fails)}')
        for c, m in fq_fails[:10]:
            print(f'  {c}: {m}')
    if lost:
        print(f'\n存在日期缺失(需xtquant全量重下): {len(lost)} 只')
        for c, m in lost[:20]:
            print(f'  {c}: {m}')
    if redownload_codes:
        print(f'\n旧格式/无法读取(需全量重下): {len(redownload_codes)} 个文件')
        for c, t, m in redownload_codes[:10]:
            print(f'  {c}/{t}: {m}')
    # 保存重下清单
    with open(BACKUP / 'redownload_list.txt', 'w') as f:
        for c, t, m in redownload_codes:
            f.write(f'{c}\t{t}\t{m}\n')
        for c, m in lost:
            f.write(f'{c}\tmissing_dates\t{m}\n')
    print(f'\n备份目录: {BACKUP}')
    print(f'重下清单: {BACKUP / "redownload_list.txt"}')


if __name__ == '__main__':
    main()
