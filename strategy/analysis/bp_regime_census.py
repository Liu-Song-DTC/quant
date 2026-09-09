#!/usr/bin/env python
"""买点类 × 市场状态 普查 (2026-09-09): 类级加成的最后一个未测维度

背景: 普查(buy_point_census)确认bp7/bp5过不了E-K1第二关(realized), bp7入场人群
逆向选择(fwd5 -0.27% vs 全池+1.14%), 类级加成方向仅剩"regime定向"未被系统测过:
  - bp2_boost_norm_only: false (加成在BEAR/FAST期也生效, 是否该关?)
  - bp1硬编码+0.08: 全regime生效 (一买反转在熊市底部是否该加强? E-N10 b臂norm_only
    已否决, 但bear-only从未测过)
本普查输出: 信号日fwd5 × 类 × 状态桶(NORM/BEAR/FAST/SEVERE) 全期+逐年,
以及已入场realized × 类 × 状态桶。状态标签与回测同源(market_regime_detector)。
输入: rolling_validation_results/backtest_signals.csv (E-E2消融run, bp8行有偏)
      rolling_validation_results/trade_realized.csv (同run)
      data/stock_data/backtrader_data/sh000001_qfq.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.market_regime_detector import MarketRegimeDetector

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_sell_point', 'signal_level']


def load_closes():
    closes = {}
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))
    return closes


def load_regime():
    idx = pd.read_csv(os.path.join(DATA_DIR, 'sh000001_qfq.csv'))
    idx['datetime'] = pd.to_datetime(idx['datetime'])
    det = MarketRegimeDetector()
    det.generate(idx)
    idf = det.index_data[['datetime', 'regime', 'bear_risk', 'bear_risk_fast',
                          'severe_bear']].copy()
    idf['date'] = idf['datetime'].dt.date
    idf = idf.drop_duplicates('date')
    # 状态桶: SEVERE(持续熊) > FAST(急跌) > BEAR > NORM
    idf['bucket'] = 'NORM'
    idf.loc[idf['bear_risk'], 'bucket'] = 'BEAR'
    idf.loc[idf['bear_risk_fast'], 'bucket'] = 'FAST'
    idf.loc[idf['severe_bear'], 'bucket'] = 'SEVERE'
    return idf.set_index('date')['bucket']


def main():
    reg = load_regime()
    print("状态桶分布(交易日):")
    print(reg.value_counts().to_string())

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    b = sig[sig['buy']].copy().reset_index(drop=True)
    b['date'] = b['d'].dt.date
    b['bucket'] = b['date'].map(reg)
    b['bp'] = b['chan_buy_point'].astype(int)
    b['year'] = b['d'].dt.year
    print(f"buy信号: {len(b):,} | 有状态标签: {b['bucket'].notna().sum():,}")

    print("载入全市场收盘价...")
    closes = load_closes()
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

    print("\n=== ① 类 × 状态桶 全期 mean5 (信号日) ===")
    g = bv.groupby(['bp', 'bucket'], observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g['mean5'] = (g['mean5'] * 100).round(2)
    print(g.to_string())

    print("\n=== ② bp2逐年 × 状态 (bp2加成norm_only该不该开) ===")
    b2 = bv[bv['bp'] == 2]
    g2 = b2.groupby(['year', 'bucket'], observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g2['mean5'] = (g2['mean5'] * 100).round(2)
    print(g2.to_string())
    print("\nbp2 非NORM桶(加成在bear期也生效的那部分):")
    nb = b2[b2['bucket'] != 'NORM']
    print(f"  n={len(nb):,} mean5 {(nb['fwd5'].mean())*100:+.2f}% vs bp2全池 "
          f"{(b2['fwd5'].mean())*100:+.2f}%")

    print("\n=== ③ bp1逐年 × 状态 (一买反转bear-only加成候选) ===")
    b1 = bv[bv['bp'] == 1]
    g1 = b1.groupby(['year', 'bucket'], observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g1['mean5'] = (g1['mean5'] * 100).round(2)
    print(g1.to_string())

    print("\n=== ④ bp7/bp5 × 状态桶 (无加成类的状态画像) ===")
    for bpv in (7, 5):
        sub = bv[bv['bp'] == bpv]
        gs = sub.groupby('bucket', observed=True).agg(
            n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
        gs['mean5'] = (gs['mean5'] * 100).round(2)
        print(f"-- bp{bpv} --")
        print(gs.to_string())

    print("\n=== ⑤ 全池逐年 × 状态桶 (基线画像) ===")
    ga = bv.groupby(['year', 'bucket'], observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    ga['mean5'] = (ga['mean5'] * 100).round(2)
    print(ga.to_string())

    # === realized × 状态桶 ===
    trades = pd.read_csv(f'{BASE}/trade_realized.csv')
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    sigb = sig[sig['buy']].copy()  # 只取buy行, 防非buy行bp=0污染类标签
    m = trades.merge(sigb, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=20))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    m = m[m['chan_buy_point'].notna()].copy()
    m['bp'] = m['chan_buy_point'].astype(int)
    m['date'] = m['entry_date'].dt.date
    m['bucket'] = m['date'].map(reg)

    print("\n=== ⑥ 已入场realized × 类 × 状态桶 ===")
    gr = m.groupby(['bp', 'bucket'], observed=True).agg(
        n=('ret', 'size'), winrate=('ret', lambda x: (x > 0).mean()),
        mean_ret=('ret', 'mean'))
    gr['winrate'] = (gr['winrate'] * 100).round(1)
    gr['mean_ret'] = (gr['mean_ret'] * 100).round(2)
    print(gr.to_string())


if __name__ == '__main__':
    main()
