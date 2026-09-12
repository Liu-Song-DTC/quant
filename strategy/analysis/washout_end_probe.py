#!/usr/bin/env python
"""洗盘结束甄别探针 (E-N11, 2026-09-08)

用户问题: 持仓10天内洗盘 → 为什么不洗盘完再买? 有没有洗盘结束的甄别信号?
本探针用历史逐笔入场验证: 入场时点的洗盘状态 (高位未回调 / 回调中未收复 /
洗盘结束(缩量回调+收复短均线) / 深回调底部) 对未来5日收益的区分度。

洗盘结束定义 (三件套):
  1. 回调: 入场价相对入场前20日高点回撤 ≥3%
  2. 缩量: 回调段(近5日)均量 < 20日均量的80% (抛压枯竭)
  3. 收复: 入场日收盘 > MA5 且 > MA10 (多头重新接管)

输入: rolling_validation_results/trade_realized.csv (目标run逐笔平仓, 默认今晚
      9/8态bearhard实盘run)
      rolling_validation_results/backtest_signals.csv (同run信号)
      data/stock_data/backtrader_data/{code}_qfq.csv (OHLCV按需读)
输出: 各洗盘状态桶的 n/hit1/mean5/median5/realized, 以及状态×买点类交叉
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
TRADES = os.environ.get('WASH_TRADES') or f'{BASE}/trade_realized.csv'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'score', 'industry']

_qfq_cache = {}


def load_qfq(code):
    if code in _qfq_cache:
        return _qfq_cache[code]
    p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        _qfq_cache[code] = None
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    _qfq_cache[code] = df
    return df


def fwd_ret(df_price, entry_date, horizon):
    idx = df_price['datetime'].searchsorted(pd.Timestamp(entry_date))
    if idx + horizon >= len(df_price) or idx < 0:
        return np.nan
    base = df_price['close'].iloc[idx]
    if base <= 0:
        return np.nan
    return float(df_price['close'].iloc[idx + horizon] / base - 1)


def washout_state(df_price, entry_date):
    """入场日洗盘状态: 返回(state, 回调幅度, 量比, 是否收复)"""
    idx = df_price['datetime'].searchsorted(pd.Timestamp(entry_date))
    if idx < 25 or idx >= len(df_price):
        return '数据不足', np.nan, np.nan, np.nan
    win = df_price.iloc[idx - 20:idx]          # 入场前20个交易日
    close_e = float(df_price['close'].iloc[idx])
    high20 = float(win['close'].max())
    pullback = close_e / high20 - 1            # 相对20日高点的位置 (<0=回调中)
    vol5 = float(df_price['volume'].iloc[idx - 5:idx].mean())
    vol20 = float(df_price['volume'].iloc[idx - 20:idx].mean())
    vol_ratio = vol5 / vol20 if vol20 > 0 else np.nan
    ma5 = float(df_price['close'].iloc[idx - 5:idx + 1].mean())
    ma10 = float(df_price['close'].iloc[idx - 10:idx + 1].mean())
    reclaim = close_e > ma5 and close_e > ma10
    if pullback > -0.03:
        state = '高位(未回调)'
    elif pullback <= -0.03 and vol_ratio < 0.85 and reclaim:
        state = '洗盘结束(缩量+收复)'
    elif pullback <= -0.15:
        state = '深回调底部'
    else:
        state = '回调中(未收复)'
    return state, pullback, vol_ratio, reclaim


def main():
    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    print(f"trades: {len(trades)} | {trades['entry_date'].min().date()} → {trades['exit_date'].max().date()}")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')

    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=3))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    print(f"信号匹配: {m['chan_buy_point'].notna().sum()}/{len(trades)}")

    states, pbs, vrs, f1, f5 = [], [], [], [], []
    for _, t in m.iterrows():
        dfp = load_qfq(t['code'])
        if dfp is None:
            states.append('无K线'); pbs.append(np.nan); vrs.append(np.nan)
            f1.append(np.nan); f5.append(np.nan)
            continue
        st, pb, vr, _ = washout_state(dfp, t['entry_date'])
        states.append(st); pbs.append(pb); vrs.append(vr)
        f1.append(fwd_ret(dfp, t['entry_date'], 1))
        f5.append(fwd_ret(dfp, t['entry_date'], 5))
    m['ws'] = states
    m['pullback'] = pbs
    m['vol_ratio'] = vrs
    m['fwd1'] = f1
    m['fwd5'] = f5

    order = ['洗盘结束(缩量+收复)', '回调中(未收复)', '深回调底部', '高位(未回调)', '数据不足', '无K线']
    print(f"\n=== 洗盘状态 × 5日收益 (全池) ===")
    g = m.groupby('ws').agg(
        n=('fwd5', 'size'),
        hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'),
        mean_ret_realized=('ret', 'mean'), med_ret_realized=('ret', 'median'),
        mean_pullback=('pullback', 'mean'), mean_vol_ratio=('vol_ratio', 'mean'))
    g = g.reindex([s for s in order if s in g.index])
    print(g.round(4).to_string())

    m['hb'] = pd.cut(m['hold_days'], [0, 10, 20, 60, 999],
                     labels=['1-10天', '11-20天', '21-60天', '61+天'])
    print("\n=== 洗盘状态 × 持仓桶 (realized) ===")
    print(m.pivot_table(index='ws', columns='hb', values='ret',
                        aggfunc=['count', 'mean']).round(4).to_string())

    print("\n=== 洗盘状态 × 买点类 (bp0/bp1/bp2/其他) ===")
    m['bp'] = m['chan_buy_point'].map(
        lambda x: 'bp0' if x == 0 else ('bp1' if x == 1 else ('bp2' if x == 2 else f'bp{x}')))
    ct = pd.crosstab(m['ws'], m['bp'])
    print(ct.to_string())
    print("\n洗盘结束桶内 bp0 的5日收益:")
    z = m[(m['ws'] == '洗盘结束(缩量+收复)')]
    print(z.groupby('bp', observed=True)['fwd5'].agg(['size', 'mean']).round(4).to_string())

    print("\n=== 逐年: 洗盘结束入场占比 ===")
    m['y'] = m['exit_date'].dt.year
    print(pd.crosstab(m['y'], m['ws']).to_string())


if __name__ == '__main__':
    main()
