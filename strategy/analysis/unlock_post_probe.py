#!/usr/bin/env python
"""解禁后行为探针 (2026-09-09): 解禁靴子落地后买点质量 — 排除窗外的未知区

现有机制: 未来30天内有占流通市值>=5%的解禁不入选(unlock_ahead_days=30,
min_ratio=0.05, 供给压力事前排除)。本探针问窗后: 解禁发生后N天内买入信号
的fwd5如何? 两种可能: (a) 解禁后短期仍有供给消化压力 → fwd5低, 排除窗应
向事后延伸; (b) 靴子落地(同yjyg预告效应) → fwd5高, 是反转类优先槽候选。
PIT: unlock_date必须严格早于信号日。
输入: rolling_validation_results/backtest_signals.csv (E-E2消融run)
      data/alternative_data/unlock_schedule.pkl
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
MIN_RATIO = 0.05  # 与组合层解禁排除同口径


def main():
    ul = pd.read_pickle('/mnt/d/quant/data/alternative_data/unlock_schedule.pkl')
    ul = ul[ul['ratio'] >= MIN_RATIO].copy()
    ul['code'] = ul['code'].astype(str).str.zfill(6)
    ul['ud'] = pd.to_datetime(ul['unlock_date'])
    ul = ul.sort_values('ud')
    print(f"解禁事件(ratio>={MIN_RATIO}): {len(ul):,} | "
          f"区间 {ul['ud'].min().date()} → {ul['ud'].max().date()}")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'],
                      low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    b = sig[sig['buy']].copy().reset_index(drop=True)
    b['bp'] = b['chan_buy_point'].astype(int)
    print(f"buy信号: {len(b):,}")

    e3 = ul[['code', 'ud']].rename(columns={'ud': 'ul_d'}).sort_values('ul_d')
    e3['ul_d'] = e3['ul_d'].astype('datetime64[us]')
    m = pd.merge_asof(b.sort_values('d'), e3, left_on='d', right_on='ul_d',
                      by='code', direction='backward',
                      allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=90))
    m['lag'] = (m['d'] - m['ul_d']).dt.days
    m['has_ul'] = m['lag'].notna()
    print(f"90天内解禁覆盖: {m['has_ul'].mean()*100:.1f}%")

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
    f5s = np.full(len(m), np.nan)
    for code, grp in m.groupby('code', sort=False):
        if code not in closes:
            continue
        dts, cl = closes[code]
        d64 = grp['d'].values
        i = np.searchsorted(dts, d64)
        ok = (i >= 0) & (i < len(cl) - 5)
        f5s[grp.index.values[ok]] = cl[i[ok] + 5] / cl[i[ok]] - 1
    m['fwd5'] = f5s
    mv = m[m['fwd5'].notna()].copy()

    print("\n=== ① 解禁后lag桶 × fwd5 (全bp) ===")
    mv['lb'] = pd.cut(mv['lag'], [0, 5, 10, 20, 30, 60, 90],
                      labels=['1-5天', '6-10天', '11-20天', '21-30天',
                              '31-60天', '61-90天'])
    g = mv.groupby(['has_ul', 'lb'], observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g['mean5'] = (g['mean5'] * 100).round(2)
    print(g.to_string())
    none_ = mv[~mv['has_ul']]
    print(f"无解禁基线: n={len(none_):,} mean5 {none_['fwd5'].mean()*100:+.2f}%")

    print("\n=== ② 解禁后30天内 × bp类 × fwd5 ===")
    rec = mv[mv['has_ul'] & (mv['lag'] <= 30)]
    g2 = rec.groupby('bp', observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g2['mean5'] = (g2['mean5'] * 100).round(2)
    print(g2.to_string())
    for bpv in (0, 1, 2, 7):
        sub = rec[rec['bp'] == bpv]
        subn = mv[(mv['bp'] == bpv) & (~mv['has_ul'])]
        if len(sub) >= 30:
            print(f"bp{bpv}: 解禁30天内 n={len(sub)} mean5 {sub['fwd5'].mean()*100:+.2f}% "
                  f"vs 无解禁 n={len(subn):,} {subn['fwd5'].mean()*100:+.2f}%")

    print("\n=== ③ 解禁后30天内 逐年 (全bp) ===")
    rec2 = rec.copy()
    rec2['year'] = rec2['d'].dt.year
    gy = rec2.groupby('year').agg(n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    gy['mean5'] = (gy['mean5'] * 100).round(2)
    print(gy.to_string())

    print("\n=== ④ 解禁type分层 (解禁后30天内) ===")
    e4 = ul[['code', 'ud', 'type']].rename(columns={'ud': 'ul_d'}).copy()
    e4['ul_d'] = e4['ul_d'].astype('datetime64[us]')
    m4 = pd.merge_asof(b.sort_values('d'), e4.sort_values('ul_d'),
                       left_on='d', right_on='ul_d', by='code',
                       direction='backward', allow_exact_matches=False,
                       tolerance=pd.Timedelta(days=30))
    m4['fwd5'] = f5s[m4.index.values]
    m4 = m4[m4['fwd5'].notna() & m4['type'].notna()]
    gt = m4.groupby('type', observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    gt['mean5'] = (gt['mean5'] * 100).round(2)
    print(gt.to_string())


if __name__ == '__main__':
    main()
