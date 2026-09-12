#!/usr/bin/env python
"""洗盘延迟入场反事实探针 (E-N11b, 2026-09-08)

用户问题: 持仓10天内洗盘 → 为什么不能洗盘完再买?
上一探针(washout_end_probe)用入场时点状态分桶, 本探针做严格反事实:
  对每笔实际入场, 观察入场后10个交易日内的洗盘过程(先挖坑≥3%再收复MA5&MA10),
  比较三种时点的5日收益:
    A) 实际入场日 (现状)
    B) 洗盘结束日(坑后首个收盘站上MA5且MA10) — "洗盘完再买"
    C) 坑底日(事后最优基准, 可达成上界)
若B显著好于A → 延迟入场机制有据; 若B≈A或更差 → 等待无收益且冒V型踏空风险。

输出: 洗盘特征(频率/坑深/收复耗时) + 三时点fwd5/hit1对比 + 按持仓桶分层
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
TRADES = os.environ.get('WASH_TRADES') or f'{BASE}/trade_realized.csv'

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


def fwd5(df_price, idx):
    """idx日后第5个交易日收盘相对idx日收盘"""
    if idx + 5 >= len(df_price) or idx < 0:
        return np.nan
    base = float(df_price['close'].iloc[idx])
    if base <= 0:
        return np.nan
    return float(df_price['close'].iloc[idx + 5] / base - 1)


def main():
    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    print(f"trades: {len(trades)}")

    # 买点类匹配 (同washout_end_probe)
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'], low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    t2 = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    t2 = t2[(t2['d'] <= t2['entry_date']) & (t2['d'] >= t2['entry_date'] - pd.Timedelta(days=3))]
    t2 = t2.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    trades = t2
    trades['bp'] = trades['chan_buy_point'].map(
        lambda x: 'bp0' if x == 0 else ('bp1' if x == 1 else ('bp2' if x == 2 else f'bp{x}')))

    rows = []
    for _, t in trades.iterrows():
        dfp = load_qfq(t['code'])
        if dfp is None:
            continue
        i = dfp['datetime'].searchsorted(t['entry_date'])
        if i + 21 >= len(dfp) or i < 10:
            continue
        cl = dfp['close']
        entry_close = float(cl.iloc[i])
        # 入场后10个交易日窗口: 找坑
        win = cl.iloc[i + 1:i + 11].values.astype(float)
        trough_rel = win.min() / entry_close - 1
        trough_i = i + 1 + int(np.argmin(win))
        has_washout = trough_rel <= -0.03
        # 坑后首个收复日 (收盘 > MA5 且 > MA10), 观察至坑后10日
        reclaim_i = None
        if has_washout:
            for j in range(trough_i + 1, min(trough_i + 11, len(dfp) - 6)):
                ma5 = float(cl.iloc[j - 4:j + 1].mean())
                ma10 = float(cl.iloc[j - 9:j + 1].mean())
                if float(cl.iloc[j]) > ma5 and float(cl.iloc[j]) > ma10:
                    reclaim_i = j
                    break
        rows.append({
            'entry_idx': i, 'trough_rel': trough_rel, 'trough_i': trough_i,
            'has_washout': has_washout,
            'reclaim_i': reclaim_i,
            'reclaim_days': (reclaim_i - trough_i) if reclaim_i else np.nan,
            'fwd5_actual': fwd5(dfp, i),
            'fwd5_delayed': fwd5(dfp, reclaim_i) if reclaim_i else np.nan,
            'fwd5_trough': fwd5(dfp, trough_i) if has_washout else np.nan,
            'hold_days': t['hold_days'], 'ret': t['ret'], 'bp': t['bp'],
        })
    m = pd.DataFrame(rows)
    print(f"可用: {len(m)} 笔 (有10日以上后续K线)")

    w = m[m['has_washout']]
    print(f"\n=== 洗盘特征 (入场后10日窗口) ===")
    print(f"发生洗盘(坑深≥3%): {len(w)}/{len(m)} = {len(w)/len(m)*100:.1f}%")
    print(f"坑深中位数: {w['trough_rel'].median()*100:.1f}% | 到坑底中位: "
          f"{(w['trough_i'] - w['entry_idx']).median():.0f}天")
    wr = w[w['reclaim_i'].notna()]
    print(f"坑后10日内收复MA5&MA10: {len(wr)}/{len(w)} ({len(wr)/len(w)*100:.0f}%)"
          f" | 坑底→收复中位 {wr['reclaim_days'].median():.0f}天")

    print(f"\n=== 反事实: 三时点5日收益 (仅洗盘笔, 有收复日的) ===")
    z = w[w['reclaim_i'].notna()].copy()
    print(f"n={len(z)}")
    for col, label in [('fwd5_actual', 'A 实际入场日'),
                       ('fwd5_delayed', 'B 洗盘结束日(收复)'),
                       ('fwd5_trough', 'C 坑底日(事后最优)')]:
        v = z[col]
        print(f"{label}: hit1={(v > 0).mean()*100:5.1f}%  mean5={v.mean()*100:+6.2f}%  "
              f"med5={v.median()*100:+6.2f}%  (n={v.notna().sum()})")
    print(f"B-A 均值差: {(z['fwd5_delayed'] - z['fwd5_actual']).mean()*100:+.2f}pp | "
          f"B优于A占比: {(z['fwd5_delayed'] > z['fwd5_actual']).mean()*100:.0f}%")

    print(f"\n=== 分层: 短持(≤10天) vs 中长持 的洗盘笔 ===")
    for label, sub in [('短持≤10天', z[z['hold_days'] <= 10]),
                       ('11-20天', z[(z['hold_days'] > 10) & (z['hold_days'] <= 20)]),
                       ('21天+', z[z['hold_days'] > 20])]:
        if len(sub) == 0:
            continue
        a = sub['fwd5_actual']; b = sub['fwd5_delayed']
        print(f"{label} (n={len(sub)}): A={a.mean()*100:+.2f}%  B={b.mean()*100:+.2f}%  "
              f"B-A={(b-a).mean()*100:+.2f}pp  实际平仓={sub['ret'].mean()*100:+.2f}%")

    print(f"\n=== 直线上涨笔(无洗盘)对照 ===")
    nu = m[~m['has_washout']]
    v = nu['fwd5_actual']
    print(f"n={len(nu)}: hit1={(v > 0).mean()*100:.1f}%  mean5={v.mean()*100:+.2f}%  "
          f"实际平仓={nu['ret'].mean()*100:+.2f}%")

    print(f"\n=== 买点类 × 洗盘率 × A/B对比 ===")
    print(pd.crosstab(m['bp'], m['has_washout'], normalize='index').round(3).to_string())
    for bp in ['bp0', 'bp1', 'bp2']:
        sub = z[z['bp'] == bp] if 'z' in dir() else None
        sub = m[(m['has_washout']) & (m['reclaim_i'].notna()) & (m['bp'] == bp)]
        if len(sub) == 0:
            continue
        a = sub['fwd5_actual']; b = sub['fwd5_delayed']
        print(f"{bp} (n={len(sub)}): A={a.mean()*100:+.2f}%  B={b.mean()*100:+.2f}%  "
              f"B-A={(b-a).mean()*100:+.2f}pp  实际平仓={sub['ret'].mean()*100:+.2f}%")
    # 未收复的洗盘笔 (机制下将-3%止损后永不回补)
    no_rc = w[w['reclaim_i'].isna()]
    print(f"\n未收复洗盘笔(机制下-3%止损离场不再回补) n={len(no_rc)}: "
          f"A fwd5={no_rc['fwd5_actual'].mean()*100:+.2f}%  "
          f"实际平仓={no_rc['ret'].mean()*100:+.2f}%")
    print(pd.crosstab(no_rc['bp'], no_rc['hold_days'].apply(lambda x: '≤10d' if x <= 10 else '11-20d' if x <= 20 else '21d+')).to_string())


if __name__ == '__main__':
    main()
