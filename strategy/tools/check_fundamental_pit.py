#!/usr/bin/env python
"""守卫: 基本面 `数据可用日期` ≡ 法定披露截止日 (PIT)。

**为什么必须守**: 原约定 `数据可用日期 = 报告期` 让**回测**提前 1~4 个月用未公开财报,
而**实盘**库里物理上没有未披露的行 ⇒ 两边信息集不同 = 违反铁律。
修法是把这个列改成法定截止日(见 data/fundamental_pit.py)。

这个守卫防的是**回退**: `data_manager.py` / `refresh_fundamental_recent.py` 里
只要有一处写回 `= 报告期`, 新入库的行就会静默恢复泄漏口径 —— 而泄漏**不会报错**,
只会让回测悄悄变好看。形态同 live-regime-blind-bug(查表失败返回默认=静默温床)。

退出码: 0=全部合规  1=有行不合规  2=目录/列缺失
用法: check_fundamental_pit.py [--quiet] [--dir DIR] [--max-report N]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'data'))

import pandas as pd  # noqa: E402

from fundamental_pit import pit_available_series  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUIET = '--quiet' in sys.argv
DIR = ROOT + '/data/stock_data/fundamental_data'
if '--dir' in sys.argv:
    DIR = sys.argv[sys.argv.index('--dir') + 1]
MAXREP = 0
if '--max-report' in sys.argv:
    MAXREP = int(sys.argv[sys.argv.index('--max-report') + 1])


def fail(msg, code):
    print(f'[fund-pit] ✗ {msg}')
    sys.exit(code)


if not os.path.isdir(DIR):
    fail(f'目录缺失: {DIR}', 2)

files = sorted(f for f in os.listdir(DIR) if f.endswith('.csv'))
if not files:
    fail(f'目录为空: {DIR}', 2)

bad_files, bad_rows, tot_rows, checked = [], 0, 0, 0
examples = []
for f in files:
    try:
        d = pd.read_csv(os.path.join(DIR, f), dtype=str, usecols=['报告期', '数据可用日期'])
    except Exception:
        continue                      # 缺列/读不动的不算违规(另有新鲜度门禁管)
    exp = pit_available_series(d['报告期'])
    ok = exp != ''
    if not ok.any():
        continue
    checked += 1
    got = d['数据可用日期'].astype(str)
    bad = ok.values & (got.values != exp.values)   # numpy bool 数组(.nonzero 需要)
    n = int(bad.sum())
    tot_rows += int(ok.sum())
    if n:
        bad_files.append(f[:-4])
        bad_rows += n
        if len(examples) < 5:
            j = int(bad.nonzero()[0][0])
            examples.append(f'    {f[:-4]} 报告期 {d["报告期"].iloc[j]}: '
                            f'{got.iloc[j]} (应为 {exp.iloc[j]})')

if bad_rows == 0:
    if not QUIET:
        print(f'[fund-pit] ✓ 数据可用日期 ≡ 法定截止日  '
              f'({checked} 只 / {tot_rows} 行)')
    sys.exit(0)

print(f'[fund-pit] ✗ {bad_rows} 行不合规, 涉及 {len(bad_files)} 只 '
      f'(共查 {checked} 只 / {tot_rows} 行)')
for e in examples:
    print(e)
if len(bad_files) > 5:
    print(f'    ... 另 {len(bad_files)-5} 只')
print('    ⇒ 这些行会让回测提前 1~4 个月看到未公开财报 (实盘看不到) = 实盘≠回测。')
print('    ⇒ 修: /mnt/d/quant/.venv/bin/python data/apply_fundamental_pit_dates.py --apply')
sys.exit(1)
