#!/usr/bin/env python
"""信号新鲜度探针 (E-N6, 2026-09-08)

问题: rebalance_days=10 → 入场时触发信号可能已过期最多9天。旧信号入场的
短期准确率是否更低? 若显著 → 加"信号年龄≤N天"接受闸, 且为E-N1快车道
(新鲜bp2次日建仓)提供弹药。

同时测: 入场价相对信号日收盘的漂移(追高)是否伤害5日准确率。

输入: /tmp/tr_EN5bh_0907.csv (bearhard@9/7逐笔平仓)
      rolling_validation_results/backtest_signals.csv.todate0907 (同run信号)
输出: 年龄桶×fwd5/hit1, 年龄×买点类, 漂移桶×fwd5
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
TRADES = '/tmp/tr_off_0904.csv'  # off@9/4基线逐笔平仓 (E-N5对照的基线侧)
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_structure_score',
            'chan_divergence_type', 'signal_level', 'score', 'industry', 'trend_type']

_qfq_cache = {}


def load_qfq(code):
    if code in _qfq_cache:
        return _qfq_cache[code]
    p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        _qfq_cache[code] = None
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    _qfq_cache[code] = df
    return df


def px_on(df_price, day):
    """day当日(或之后首个交易日)收盘"""
    idx = df_price['datetime'].searchsorted(pd.Timestamp(day))
    if idx >= len(df_price):
        return np.nan
    return float(df_price['close'].iloc[idx])


def fwd_ret(df_price, entry_date, horizon):
    idx = df_price['datetime'].searchsorted(pd.Timestamp(entry_date))
    if idx + horizon >= len(df_price) or idx < 0:
        return np.nan
    base = df_price['close'].iloc[idx]
    if base <= 0:
        return np.nan
    return float(df_price['close'].iloc[idx + horizon] / base - 1)


def main():
    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    print(f"trades: {len(trades)} | {trades['entry_date'].min().date()} → {trades['entry_date'].max().date()}")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    print(f"signals(buy=True): {len(sig):,}")

    # 触发信号 = 入场日之前最近一次buy=True (放宽到20天窗口, 全量覆盖)
    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=20))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    m['age'] = (m['entry_date'] - m['d']).dt.days
    matched = m['chan_buy_point'].notna().sum()
    print(f"信号匹配(20天窗): {matched}/{len(trades)} ({matched/len(trades)*100:.1f}%)")

    # fwd + 漂移
    f1, f5, drift = [], [], []
    for _, t in m.iterrows():
        dfp = load_qfq(t['code'])
        if dfp is None:
            f1.append(np.nan); f5.append(np.nan); drift.append(np.nan)
            continue
        f1.append(fwd_ret(dfp, t['entry_date'], 1))
        f5.append(fwd_ret(dfp, t['entry_date'], 5))
        sig_px = px_on(dfp, t['d'])
        cost = t['avg_cost']
        drift.append(cost / sig_px - 1 if sig_px and sig_px > 0 else np.nan)
    m['fwd1'] = f1
    m['fwd5'] = f5
    m['drift'] = drift

    print("\n=== 信号年龄分布 ===")
    print(m['age'].value_counts().sort_index().to_string())

    print("\n=== 年龄桶 × 5日准确率 ===")
    m['age_b'] = pd.cut(m['age'], [-1, 2, 5, 9, 20], labels=['0-2天', '3-5天', '6-9天', '10-20天'])
    g = m.groupby('age_b', observed=True).agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'),
        mean_drift=('drift', 'mean'))
    print(g.round(4).to_string())

    print("\n=== 年龄桶 × 买点类 (bp0 vs bp2 vs 其他) ===")
    m['bp'] = m['chan_buy_point'].map(lambda x: 'bp0' if x == 0 else ('bp2' if x == 2 else f'bp{x}'))
    print(pd.crosstab(m['age_b'], m['bp'], margins=True).to_string())
    print("\n年龄桶内 bp2 的5日准确率:")
    b2 = m[m['bp'] == 'bp2'].groupby('age_b', observed=True)['fwd5'].agg(['size', 'mean'])
    print(b2.round(4).to_string())

    print("\n=== 年龄×结构: 有结构(非bp0)入场随年龄衰减? ===")
    sc = m[m['bp'] != 'bp0'].groupby('age_b', observed=True).agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()), mean5=('fwd5', 'mean'))
    print(sc.round(4).to_string())

    print("\n=== 价格漂移(入场成本 vs 信号日收盘) × 5日准确率 ===")
    m['drift_b'] = pd.cut(m['drift'], [-1, 0.0, 0.02, 0.05, 5],
                          labels=['≤0(低吸)', '0~2%', '2~5%', '>5%(追高)'])
    dg = m.groupby('drift_b', observed=True).agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'))
    print(dg.round(4).to_string())

    print("\n=== 追高(>5%)且年龄>5天的组合桶 ===")
    z = m[(m['drift'] > 0.05) & (m['age'] > 5)]
    print(f"n={len(z)} | hit1={(z['fwd1'] > 0).mean()*100:.1f}% | mean5={z['fwd5'].mean()*100:+.2f}%")
    print(z.groupby('bp', observed=True)['fwd5'].agg(['size', 'mean']).round(4).to_string())


if __name__ == '__main__':
    main()
