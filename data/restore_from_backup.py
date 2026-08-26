"""从备份恢复raw_data: 修正版清洗(代码列zfill匹配), 干净文件字节级还原.

修复repair_raw_corrupt.py的bug: 旧行股票代码未补零('1'而非'000001'),
原脚本按精确匹配丢弃了全部历史行. 本脚本从备份重放:
- 无损坏(无垃圾/无重复/有序) -> copy2字节级还原
- 有损坏 -> 清洗(去垃圾/去重保末次/排序/原子写)
"""
import csv
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

RAW = Path('/mnt/d/quant/data/stock_data/raw_data')
BACKUP = Path('/mnt/d/quant/data/backup_raw_corrupt_20260821')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
ADJ_TYPES = ['none', 'qfq', 'hfq']
STOCK_HEADER = ['股票名称', '股票代码', '日期', '开盘', '收盘', '最高', '最低',
                '成交量', '成交额', '换手率', '涨跌幅', '涨跌额', '振幅']


def read_rows(path, code):
    """容错读取. 返回(valid_rows, n_garbage, legacy_flag)."""
    rows, garbage = [], 0
    try:
        with open(path, encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return [], 0, 0
            if '股票名称' not in header[0]:
                return [], 0, 1
            for row in reader:
                if not row:
                    continue
                if (len(row) == 13 and DATE_RE.match(row[2] or '')
                        and (row[1] or '').zfill(6) == code):
                    rows.append(row)
                else:
                    garbage += 1
            return rows, garbage, 0
    except Exception:
        return [], -1, 0


def main():
    t0 = datetime.now()
    restored, cleaned, errors = 0, 0, []
    dirs = sorted(d for d in BACKUP.iterdir() if d.is_dir())
    print(f'从备份恢复 {len(dirs)} 只股票...')
    for i, bdir in enumerate(dirs):
        code = bdir.name
        for f in sorted(bdir.glob('*.csv')):
            dst = RAW / code / f.name
            rows, garbage, legacy = read_rows(f, code)
            if legacy:
                continue  # 旧格式, 原样未动过, 无备份需求
            if garbage < 0:
                errors.append(f'{code}/{f.name}: 备份读取异常')
                continue
            dates = [r[2] for r in rows]
            needs_clean = garbage > 0 or len(dates) != len(set(dates)) or dates != sorted(dates)
            if not needs_clean:
                shutil.copy2(f, dst)  # 字节级还原
                restored += 1
            else:
                dedup = {r[2]: r for r in rows}
                cleaned_rows = [dedup[d] for d in sorted(dedup)]
                tmp = dst.with_suffix('.tmp')
                with open(tmp, 'w', encoding='utf-8', newline='') as fh:
                    w = csv.writer(fh)
                    w.writerow(STOCK_HEADER)
                    w.writerows(cleaned_rows)
                os.replace(tmp, dst)
                cleaned += 1
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(dirs)} ...')
    print(f'\n===== 恢复完成 ({datetime.now()-t0}) =====')
    print(f'字节级还原: {restored} 个文件, 清洗重写: {cleaned} 个文件')
    for e in errors[:10]:
        print(f'  {e}')


if __name__ == '__main__':
    main()
