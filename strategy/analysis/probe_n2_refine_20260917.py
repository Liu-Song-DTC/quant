#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N2细化 (2026-09-17): 卖出侧隔夜gap的分位数/年度稳定性/涨跌停可卖性/
单日vs多日gap拆分/两种替代成交价(信号日收盘 vs 出场日收盘)对账。"""
import os, pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
KLINE_DIR = os.path.join(BASE, '..', 'data', 'stock_data', 'backtrader_data')

tr = pd.read_csv(os.path.join(RVD, 'trade_realized.csv'), dtype={'code': str})
tr['code'] = tr['code'].str.zfill(6)
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['exit_date'] = pd.to_datetime(tr['exit_date'])

rows = []
for code, g in tr.groupby('code'):
    fp = os.path.join(KLINE_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        continue
    try:
        k = pd.read_csv(fp, usecols=['datetime', 'open', 'close', 'change_percent'])
        k['datetime'] = pd.to_datetime(k['datetime'])
        k = k.sort_values('datetime').set_index('datetime')
    except Exception:
        continue
    o, c, cp = k['open'], k['close'], k['change_percent']
    dates = o.index
    for _, r in g.iterrows():
        ed = r['exit_date']
        # 信号日 = 出场日前最后一个交易日
        prev = dates[dates < ed]
        if len(prev) == 0:
            continue
        sd = prev[-1]
        # 信号日后一天 = 出场日? (确认无停牌间隙)
        nxt = dates[dates > sd]
        if len(nxt) == 0:
            continue
        gap_close_sig_open_exit = c[sd] / o[ed] - 1 if ed in o.index else np.nan
        gap_close_sig_close_exitday = c[sd] / c[ed] - 1 if ed in c.index else np.nan
        daygap = (ed - sd).days  # 1=正常次日, >1=中间停牌/假期
        limit_down_sig = (cp[sd] <= -9.5) if not np.isnan(cp[sd]) else False
        limit_down_exit = (cp[ed] <= -9.5) if not np.isnan(cp[ed]) else False
        rows.append(dict(code=code, sd=sd, ed=ed, year=ed.year, hold=r['hold_days'],
                         gap_open=gap_close_sig_open_exit, gap_close=gap_close_sig_close_exitday,
                         daygap=daygap, ld_sig=limit_down_sig, ld_exit=limit_down_exit))
df = pd.DataFrame(rows).dropna(subset=['gap_open'])
print(f"n={len(df)} (trade_realized {len(tr)})", flush=True)

def pct(x):
    return x * 100

print("\n=== 卖出gap分布 (close(sig)/open(exit)-1) ===", flush=True)
print(df['gap_open'].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).apply(pct).round(3).to_string(), flush=True)
print(f"\nlog sum = {np.log1p(df['gap_open']).sum()*100:.1f}% | 算术和 = {df['gap_open'].sum()*100:.1f}%", flush=True)

print("\n=== 年度稳定性 (mean% / n / logsum%) ===", flush=True)
for y, g in df.groupby('year'):
    print(f"  {y}: mean={g['gap_open'].mean()*100:+.2f}%  n={len(g)}  logsum={np.log1p(g['gap_open']).sum()*100:+.1f}%  pos={(g['gap_open']>0).mean()*100:.0f}%", flush=True)

print("\n=== 单日gap vs 多日gap(停牌/假期) ===", flush=True)
g1 = df[df['daygap'] == 1]
gm = df[df['daygap'] > 1]
print(f"  单日: n={len(g1)} mean={g1['gap_open'].mean()*100:+.2f}% logsum={np.log1p(g1['gap_open']).sum()*100:+.1f}%", flush=True)
print(f"  多日: n={len(gm)} mean={gm['gap_open'].mean()*100:+.2f}% logsum={np.log1p(gm['gap_open']).sum()*100:+.1f}%", flush=True)

print("\n=== 信号日/出场日跌停(不可成交)计数 ===", flush=True)
print(f"  信号日跌停(收盘卖不可行): {df['ld_sig'].sum()} ({df['ld_sig'].mean()*100:.1f}%)", flush=True)
print(f"  出场日跌停(开盘卖不可行): {df['ld_exit'].sum()} ({df['ld_exit'].mean()*100:.1f}%)", flush=True)

print("\n=== 替代1: 信号日收盘卖 (close(sig)/open(exit) vs 1) ===", flush=True)
print(f"  total logsum vs open-fill: {-np.log1p(df['gap_open']).sum()*100:+.1f}% (即开盘卖相比收盘卖多亏这么多)", flush=True)

print("\n=== 替代2: 出场日收盘卖 (close(sig)/close(exit)-1) ===", flush=True)
print(f"  mean={df['gap_close'].mean()*100:+.2f}%  logsum={np.log1p(df['gap_close']).sum()*100:+.1f}%", flush=True)
for y, g in df.groupby('year'):
    print(f"    {y}: mean={g['gap_close'].mean()*100:+.2f}% n={len(g)}", flush=True)

print("\n=== 极端尾部审计: gap<-5%的案例 ===", flush=True)
tail = df[df['gap_open'] < -0.05]
print(f"  n={len(tail)}, 占logsum的份额: {np.log1p(tail['gap_open']).sum()/np.log1p(df['gap_open']).sum()*100:.0f}%", flush=True)
print(tail[['code', 'sd', 'ed', 'hold', 'gap_open']].head(12).to_string(index=False), flush=True)
print("\n=== 剔除gap<-5%尾部后的年度logsum ===", flush=True)
d2 = df[df['gap_open'] >= -0.05]
for y, g in d2.groupby('year'):
    print(f"    {y}: logsum={np.log1p(g['gap_open']).sum()*100:+.1f}% n={len(g)}", flush=True)
