#!/usr/bin/env python
"""短持桶(1-10天)买点归因探针 (E-N4, 2026-09-08)

用户指令: 提高持仓短期买入准确率 → 提高资金利用率。
基线解剖(trade_anatomy): 1-10天桶174笔(32%)胜率~49% sum_ret≈+1.06≈白噪音;
钱仓=21-60天桶(+9.84)。本探针回答: 短持桶里什么买点在亏/什么买点在赚,
5天准确率的驱动变量是结构(bp2)还是分数(score五档)。

输入: rolling_validation_results/trade_realized.csv (目标run的逐笔平仓)
      rolling_validation_results/backtest_signals.csv (同run信号, usecols轻载)
      data/stock_data/backtrader_data/{code}_qfq.csv (按需读, 逐笔1/5日forward)
用法: python analysis/short_hold_attribution.py
用法前确认两个CSV属于同一run (mtime + 日志"逐笔平仓审计已保存"对得上)。
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_structure_score',
            'chan_divergence_type', 'score', 'industry', 'trend_type']

_qfq_cache = {}

TRADES_OVERRIDE = os.environ.get('SH_TRADES')  # 可覆盖逐笔平仓来源(A/B对比用)


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


def fwd_ret(df_price, entry_date, horizon):
    """入场日后第horizon个交易日的收盘价相对入场成本的变化"""
    idx = df_price['datetime'].searchsorted(pd.Timestamp(entry_date))
    if idx + horizon >= len(df_price) or idx < 0:
        return np.nan
    base = df_price['close'].iloc[idx]
    if base <= 0:
        return np.nan
    return float(df_price['close'].iloc[idx + horizon] / base - 1)


def main():
    trades = pd.read_csv(TRADES_OVERRIDE or f'{BASE}/trade_realized.csv')
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)

    print(f"trades: {len(trades)} | {trades['entry_date'].min().date()} → {trades['exit_date'].max().date()}")

    # 触发信号匹配: 入场日前3日内最近的buy=True信号 (bp2_trade_audit同款逻辑)
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    print(f"signals(buy=True): {len(sig):,}")

    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=3))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    matched = m['chan_buy_point'].notna().sum()
    print(f"信号匹配: {matched}/{len(trades)} ({matched/len(trades)*100:.1f}%)")

    # 逐笔 forward 1/5日 (用入场成本口径)
    f1, f5 = [], []
    for _, t in m.iterrows():
        dfp = load_qfq(t['code'])
        if dfp is None:
            f1.append(np.nan)
            f5.append(np.nan)
            continue
        f1.append(fwd_ret(dfp, t['entry_date'], 1))
        f5.append(fwd_ret(dfp, t['entry_date'], 5))
    m['fwd1'] = f1
    m['fwd5'] = f5

    # 持仓分桶
    m['hb'] = pd.cut(m['hold_days'], [0, 10, 20, 60, 999],
                     labels=['1-10天', '11-20天', '21-60天', '61+天'])

    print(f"\n=== 入场后5日准确率: 全池 vs 短持桶 ===")
    print(f"全池: hit1={(m['fwd1'] > 0).mean()*100:.1f}% mean5={m['fwd5'].mean()*100:+.2f}% "
          f"(n={m['fwd5'].notna().sum()})")
    s = m[m['hb'] == '1-10天']
    print(f"短持桶: hit1={(s['fwd1'] > 0).mean()*100:.1f}% mean5={s['fwd5'].mean()*100:+.2f}% "
          f"(n={s['fwd5'].notna().sum()})")
    w = m[m['hb'] == '21-60天']
    print(f"钱仓桶: hit1={(w['fwd1'] > 0).mean()*100:.1f}% mean5={w['fwd5'].mean()*100:+.2f}% "
          f"(n={w['fwd5'].notna().sum()})")

    print("\n=== 按买点类: 5日准确率 (全池) ===")
    g = m.groupby('chan_buy_point').agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'),
        mean_score=('score', 'mean'))
    print(g.round(4))

    print("\n=== 短持桶(1-10天)内: 按买点类 ===")
    gs = s.groupby('chan_buy_point').agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), mean_ret_realized=('ret', 'mean'))
    print(gs.round(4))
    print(f"短持桶买点类占比: {s['chan_buy_point'].value_counts().to_dict()}")

    print("\n=== 短持桶 vs 钱仓桶: 结构评分分布 ===")
    for label, sub in [('1-10天', s), ('21-60天', w)]:
        print(f"-- {label} --")
        print(sub['chan_structure_score'].value_counts().head(6).to_string())

    print("\n=== score五档对5日准确率的区分度 ===")
    q = pd.qcut(m['score'], 5, duplicates='drop')
    print(m.groupby(q, observed=True)['fwd5'].agg(['size', 'mean']).round(4).to_string())

    print("\n=== 短持桶逐年: 买点类构成 ===")
    s['y'] = s['exit_date'].dt.year
    print(pd.crosstab(s['y'], s['chan_buy_point']).to_string())

    print("\n=== 近0出场(-2%~+2%)在短持桶中的占比 (熊市翻转切仓proxy) ===")
    z = s[(s['ret'] > -0.02) & (s['ret'] < 0.02)]
    print(f"{len(z)}/{len(s)} = {len(z)/len(s)*100:.1f}% | 中位持有 {z['hold_days'].median():.0f}天")


if __name__ == '__main__':
    main()
