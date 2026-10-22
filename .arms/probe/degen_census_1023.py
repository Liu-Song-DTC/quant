#!/usr/bin/env python3
"""#356 退化价格普查 (2026-10-23, 只读): 生产 backtrader_data 负价/零价/NaN/高低倒挂存量.

目的: 校准数据层守卫阈值 — 守卫接进 refresh_all.sh 步骤3 (rebuild 产物换名前),
若当前库本身有存量退化格, 直接上"零容忍"会第一天就锁死每日流.
检查项 (逐文件 usecols 只读 OHLC+volume+amount):
  - OHLC 任一 NaN / <=0
  - high < low (倒挂)
  - volume<0 / amount<0 (信息项)
用法: python degen_census_1023.py [目录]
"""
import os
import sys
import numpy as np
import pandas as pd

BT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/quant/data/stock_data/backtrader_data'
files = sorted(f for f in os.listdir(BT) if f.endswith('_qfq.csv') and not f.startswith('._'))
print(f'扫描 {len(files)} 个文件: {BT}')

tot = dict(nan=0, nonpos=0, hilo=0, volneg=0, amtneg=0, dupdt=0)
bad_files = {}
worst = []
for fn in files:
    try:
        d = pd.read_csv(os.path.join(BT, fn),
                        usecols=['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    except Exception as e:
        bad_files.setdefault(fn, []).append(f'READ_ERR:{e}')
        continue
    o = pd.to_numeric(d['open'], errors='coerce').values
    h = pd.to_numeric(d['high'], errors='coerce').values
    l = pd.to_numeric(d['low'], errors='coerce').values
    c = pd.to_numeric(d['close'], errors='coerce').values
    v = pd.to_numeric(d['volume'], errors='coerce').values
    a = pd.to_numeric(d['amount'], errors='coerce').values
    ohlc = np.vstack([o, h, l, c])
    m_nan = np.isnan(ohlc).any(axis=0)
    m_nonpos = (ohlc <= 0).any(axis=0) & ~m_nan
    m_hilo = (h < l) & ~np.isnan(h) & ~np.isnan(l)
    m_v = v < 0
    m_a = a < 0
    dup = int(d['datetime'].duplicated().sum())
    n = dict(nan=int(m_nan.sum()), nonpos=int(m_nonpos.sum()), hilo=int(m_hilo.sum()),
             volneg=int(m_v.sum()), amtneg=int(m_a.sum()), dupdt=dup)
    for k in tot:
        tot[k] += n[k]
    if any(n.values()):
        bad_files[fn] = n
        worst.append((fn, sum(n.values())))

print('\n== 总量 ==')
for k, v in tot.items():
    print(f'  {k:8s}: {v}')
print(f'\n有问题文件: {len(bad_files)} / {len(files)}')
if bad_files:
    print('\n== 明细 (前 30) ==')
    for fn, n in sorted(bad_files.items(), key=lambda x: -sum(x[1].values()))[:30]:
        print(f'  {fn:18s} {n}')
    # 抽 3 个非pos例看原始行
    shown = 0
    print('\n== 样例行 ==')
    for fn, n in sorted(bad_files.items(), key=lambda x: -sum(x[1].values())):
        if shown >= 3:
            break
        d = pd.read_csv(os.path.join(BT, fn), usecols=['datetime', 'open', 'high', 'low', 'close'])
        ohlc = d[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
        m = (ohlc.isna().any(axis=1) | (ohlc <= 0).any(axis=1) | (ohlc['high'] < ohlc['low']))
        print(f'-- {fn} ({int(m.sum())} 行)')
        print(d[m].head(3).to_string(index=False))
        shown += 1
else:
    print('✅ 当前生产库零退化格 ⇒ 守卫可上零容忍')
