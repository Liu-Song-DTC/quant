#!/usr/bin/env python
"""买点类全景普查 (2026-09-09): 找下一个被系统性低估的买点类

E-K1成功模式: bp2类普查发现 hit1 81.8%/mean5+3.02% 但score五档flat →
类被低估 → +0.45加成采纳(四指标全优)。E-N7/N10/N15失败模式: 探针桶≠机制人群。

本普查把9个买点类(0..9)全部过一遍筛选器:
  ① 类质量: hit1/mean5/med5 (信号日fwd1/fwd5)
  ② 稳定性: 逐年6/6年 (E-K1的唯一幸存者=唯一6/6年稳定的类)
  ③ 低估性: 类内score四分档是否flat (flat=评分系统对该类无区分力=类级低估候选)
  ④ 卖点污染: chan_sell_point>0占比 (E-K1加成条件=无卖点)
筛选器目的: 只在"质量高×稳定×score flat"的类上考虑类级机制(加成/专用通道)。
输入: rolling_validation_results/backtest_signals.csv
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_sell_point',
            'signal_level', 'score', 'trend_type']


def load_closes():
    import glob
    closes = {}
    cal = pd.read_csv(os.path.join(DATA_DIR, 'sh000001_qfq.csv'),
                      usecols=['datetime'])['datetime']
    cal = pd.to_datetime(cal).sort_values().reset_index(drop=True)
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))
    return closes, cal


def fwd_from_sig(closes, code, d, horizon):
    if code not in closes:
        return np.nan
    dts, cl = closes[code]
    i = np.searchsorted(dts, np.datetime64(d))
    if i + horizon >= len(cl) or i < 0:
        return np.nan
    base = cl[i]
    return float(cl[i + horizon] / base - 1) if base > 0 else np.nan


def main():
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    b = sig[sig['buy']].copy()
    b = b.reset_index(drop=True)  # 索引回0..n-1 (原sig行号高达3.5M, 不能用于f1s下标)
    b['year'] = b['d'].dt.year
    print(f"buy信号总数: {len(b):,}")

    closes, cal = load_closes()
    print(f"closes载入: {len(closes)} 只")

    # 信号日fwd1/fwd5 (全量, 内存安全: 720k × 2 floats)
    print("计算fwd1/fwd5 (全量)...")
    f1s = np.full(len(b), np.nan)
    f5s = np.full(len(b), np.nan)
    # 按code分组向量化
    for code, grp in b.groupby('code', sort=False):
        if code not in closes:
            continue
        dts, cl = closes[code]
        d64 = grp['d'].values
        i = np.searchsorted(dts, d64)
        ok = (i >= 0) & (i < len(cl) - 5)
        f1s[grp.index.values[ok]] = cl[i[ok] + 1] / cl[i[ok]] - 1
        f5s[grp.index.values[ok]] = cl[i[ok] + 5] / cl[i[ok]] - 1
    b['fwd1'] = f1s
    b['fwd5'] = f5s

    valid = b['fwd5'].notna()
    print(f"fwd5有效: {valid.sum():,}/{len(b)} ({valid.mean()*100:.1f}%)")
    bv = b[valid].copy()
    bv['hit1'] = (bv['fwd1'] > 0).astype(float)
    bv['bp'] = bv['chan_buy_point'].astype(int)

    print("\n=== ① 类质量总览 (信号日fwd5) ===")
    g = bv.groupby('bp').agg(n=('fwd5', 'size'),
                             hit1=('hit1', 'mean'), mean5=('fwd5', 'mean'),
                             med5=('fwd5', 'median'),
                             sell_pt=('chan_sell_point', lambda x: (x > 0).mean()),
                             sl0=('signal_level', lambda x: (x == 0).mean()))
    print(g.round(4).to_string())

    print("\n=== ② 稳定性: 逐年 mean5 (6/6年=E-K1级稳定) ===")
    yg = bv.groupby(['bp', 'year']).agg(n=('fwd5', 'size'), mean5=('fwd5', 'mean'),
                                        hit1=('hit1', 'mean'))
    print(yg.round(4).to_string())

    print("\n=== ③ 低估性: 类内score四分档 × mean5/hit1 (flat=无区分=类级低估候选) ===")
    for bp in sorted(bv['bp'].unique()):
        sub = bv[bv['bp'] == bp]
        if len(sub) < 200:
            continue
        sub = sub.copy()
        try:
            sub['q'] = pd.qcut(sub['score'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        except ValueError:
            continue
        gq = sub.groupby('q', observed=True).agg(
            n=('fwd5', 'size'), hit1=('hit1', 'mean'), mean5=('fwd5', 'mean'))
        spread = gq['mean5'].max() - gq['mean5'].min()
        print(f"\nbp{bp} (n={len(sub):,}): score四分档 spread={spread*100:+.2f}pp")
        print(gq.round(4).to_string())

    print("\n=== ④ 无卖点子集 (E-K1加成条件的类) ===")
    g2 = bv[bv['chan_sell_point'] == 0].groupby('bp').agg(
        n=('fwd5', 'size'), hit1=('hit1', 'mean'), mean5=('fwd5', 'mean'))
    print(g2.round(4).to_string())

    print("\n=== ⑤ 无卖点×逐年 mean5 (筛选器主表) ===")
    yg2 = bv[bv['chan_sell_point'] == 0].groupby(['bp', 'year']).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'), hit1=('hit1', 'mean'))
    print(yg2.round(4).to_string())


if __name__ == '__main__':
    main()
