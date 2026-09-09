#!/usr/bin/env python
"""减持排除窗验证+窗后行为探针 (2026-09-09): 现有减持排除在信号日的功效

现有机制(portfolio层硬约束, get_reduction_codes): 满足其一即排除 —
  1) 减持公告(eitime)披露于date前30天内; 2) date落在[变动开始, 变动截止]窗口内。
本探针在信号日层面: 对每个buy信号计算"减持覆盖期"covered_until=
max(公告日, 窗口截止日)的代码级前缀最大值(PIT: 仅用公告时刻≤信号日的记录),
  在窗内(covered≥信号日) → 排除人群, 验证其fwd5是否真低于基线;
  窗后lag=信号日-covered → 1-30/31-60/61-90天桶, 检验排除窗是否该延伸。
输入: rolling_validation_results/backtest_signals.csv (E-E2消融run)
      data/alternative_data/reduction_records.pkl
      data/stock_data/backtrader_data/*_qfq.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'


def main():
    rd = pd.read_pickle('/mnt/d/quant/data/alternative_data/reduction_records.pkl')
    rd['code'] = rd['code'].astype(str).str.zfill(6)
    rd['eit'] = pd.to_datetime(rd['eitime']).dt.normalize()
    rd['end'] = pd.to_datetime(rd['end_date'])
    rd = rd.dropna(subset=['eit', 'end']).copy()
    rd['cov'] = np.maximum(rd['eit'].values.astype('datetime64[ns]'),
                           rd['end'].values.astype('datetime64[ns]'))
    rd = rd.sort_values('eit').reset_index(drop=True)
    print(f"减持记录: {len(rd):,} | 股票 {rd['code'].nunique():,}")

    # 代码级: eit排序 + covered_until前缀max (公告时刻≤t的记录才可见, 同provider)
    idx = {}
    for code, grp in rd.groupby('code', sort=False):
        eit = grp['eit'].values.astype('datetime64[ns]')
        cov = grp['cov'].values.astype('datetime64[ns]')
        pmax = np.maximum.accumulate(cov)
        idx[code] = (eit, pmax)
    print(f"前缀索引: {len(idx)} 只股票")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'],
                      low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    b = sig[sig['buy']].copy().reset_index(drop=True)
    b['bp'] = b['chan_buy_point'].astype(int)
    print(f"buy信号: {len(b):,}")

    t64 = b['d'].values.astype('datetime64[ns]')
    state = np.full(len(b), -999.0)  # lag天数: <0=窗内(排除人群), NaN=无记录
    state[:] = np.nan
    for code, grp in b.groupby('code', sort=False):
        if code not in idx:
            continue
        eit, pmax = idx[code]
        i = np.searchsorted(eit, t64[grp.index.values], side='right') - 1
        ok = i >= 0
        pos = grp.index.values[ok]
        pm = pmax[i[ok]]
        lag = (t64[pos] - pm).astype('timedelta64[D]').astype(float)
        state[pos] = lag  # lag<=0 → 窗内
    b['lag'] = state
    print(f"有减持记录的信号: {b['lag'].notna().sum():,}")

    # fwd5
    closes = {}
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))
    f5s = np.full(len(b), np.nan)
    for code, grp in b.groupby('code', sort=False):
        if code not in closes:
            continue
        dts, cl = closes[code]
        d64 = grp['d'].values
        i = np.searchsorted(dts, d64)
        ok = (i >= 0) & (i < len(cl) - 5)
        f5s[grp.index.values[ok]] = cl[i[ok] + 5] / cl[i[ok]] - 1
    b['fwd5'] = f5s
    bv = b[b['fwd5'].notna()].copy()

    def bucket(l):
        if pd.isna(l):
            return '无减持记录'
        if l <= 0:
            return '窗内(现排除)'
        if l <= 30:
            return '窗后1-30天'
        if l <= 60:
            return '窗后31-60天'
        if l <= 90:
            return '窗后61-90天'
        return '窗后90天+'
    bv['bkt'] = bv['lag'].apply(bucket)

    print("\n=== ① 减持窗口状态 × fwd5 (全bp) ===")
    g = bv.groupby('bkt', observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g['mean5'] = (g['mean5'] * 100).round(2)
    print(g.to_string())

    print("\n=== ② 窗内 vs 窗后1-30天 × bp类 ===")
    for bpv in (0, 1, 2, 7):
        sub = bv[bv['bp'] == bpv]
        inw = sub[sub['bkt'] == '窗内(现排除)']
        p30 = sub[sub['bkt'] == '窗后1-30天']
        none_ = sub[sub['bkt'] == '无减持记录']
        if len(inw) >= 20:
            print(f"bp{bpv}: 窗内 n={len(inw):,} mean5 {inw['fwd5'].mean()*100:+.2f}% | "
                  f"窗后1-30天 n={len(p30):,} {p30['fwd5'].mean()*100:+.2f}% | "
                  f"无记录 n={len(none_):,} {none_['fwd5'].mean()*100:+.2f}%")

    print("\n=== ③ 窗内排除人群 逐年 (全bp) — 排除是否在砍对的股票 ===")
    inw = bv[bv['bkt'] == '窗内(现排除)'].copy()
    inw['year'] = inw['d'].dt.year
    gy = inw.groupby('year').agg(n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    gy['mean5'] = (gy['mean5'] * 100).round(2)
    print(gy.to_string())


if __name__ == '__main__':
    main()
