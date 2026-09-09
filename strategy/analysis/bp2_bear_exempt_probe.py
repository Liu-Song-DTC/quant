#!/usr/bin/env python
"""E-N2前置探针 (2026-09-09): bp2持仓"熊市清仓豁免"的奖品与成本

问题: 系统在深度回撤窗口(HDS -20%熔断强制清仓 / 组合止损)会无差别砍仓。
bp2类(6/6年稳定, V型反弹赢家证据)在深度回撤窗口被砍掉的仓位, 砍错了吗?
豁免(留住bp2)能赚回多少, 代价(裸扛回撤)多大?

证据两侧:
  收益侧: 深度回撤窗口砍掉的bp2仓, 砍后fwd5/10/20收复率 vs 全池对照
  成本侧: 豁免=扛住-20%组合回撤期, 被豁免仓的继续下跌幅度(bp2是否真能扛)
输入: rolling_validation_results/equity_curve.csv (基线净值)
      rolling_validation_results/trade_realized.csv (基线逐笔)
      rolling_validation_results/backtest_signals.csv (买点类)
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
EQ = f'{BASE}/equity_curve.csv'
TRADES = f'{BASE}/trade_realized.csv'

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


def fwd_ret(code, exit_date, horizon):
    """exit当日收盘起的fwd收益 (砍仓后若豁免会拿到的收益)"""
    dfp = load_qfq(code)
    if dfp is None:
        return np.nan
    i = dfp['datetime'].searchsorted(pd.Timestamp(exit_date))
    if i + horizon >= len(dfp) or i < 0:
        return np.nan
    base = dfp['close'].iloc[i]
    if base <= 0:
        return np.nan
    return float(dfp['close'].iloc[i + horizon] / base - 1)


def main():
    eq = pd.read_csv(EQ)
    eq['date'] = pd.to_datetime(eq['date'])
    eq['dd'] = eq['nav'] / eq['nav'].cummax() - 1

    # 深度回撤窗口: dd <= -15% (HDS触发-20%前夜 ~ 恢复-10%之间)
    eq['deep'] = eq['dd'] <= -0.15
    print("深度回撤(dd≤-15%)窗口统计:")
    win = []
    in_w = False
    for d, deep in zip(eq['date'], eq['deep']):
        if deep and not in_w:
            in_w = True
            s = d
        elif not deep and in_w:
            in_w = False
            win.append((s, d))
    if in_w:
        win.append((s, eq['date'].iloc[-1]))
    for s, e in win:
        print(f"  {s.date()} → {e.date()} ({len(eq[(eq['date']>=s)&(eq['date']<=e)]):>4}天)")
    deep_days = set(eq.loc[eq['deep'], 'date'].values)

    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'], low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=20))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    m['bp2'] = m['chan_buy_point'] == 2
    print(f"\nbp2匹配: {m['bp2'].sum()}/{len(m)}")

    # 砍仓发生在深度回撤窗口内?
    def in_deep(d):
        dd_prev = eq.loc[eq['date'] <= d, 'deep']
        return bool(dd_prev.iloc[-1]) if len(dd_prev) else False
    m['cut_deep'] = m['exit_date'].map(in_deep)
    m['year'] = m['exit_date'].dt.year

    # 收益侧: 砍后收复
    f5, f10, f20 = [], [], []
    for code, ed in zip(m['code'], m['exit_date']):
        f5.append(fwd_ret(code, ed, 5))
        f10.append(fwd_ret(code, ed, 10))
        f20.append(fwd_ret(code, ed, 20))
    m['f5'] = f5
    m['f10'] = f10
    m['f20'] = f20

    print("\n=== 深度回撤窗口内砍掉的仓位: bp2 vs 非bp2 ===")
    cd = m[m['cut_deep']]
    for name, sub in [('全部', cd), ('bp2', cd[cd['bp2']]), ('非bp2', cd[~cd['bp2']])]:
        if len(sub) == 0:
            print(f"  {name}: n=0")
            continue
        print(f"  {name}: n={len(sub)} ret中位{sub['ret'].median()*100:+.2f}% | "
              f"f5中位{sub['f5'].median()*100:+.2f}% f10 {sub['f10'].median()*100:+.2f}% "
              f"f20 {sub['f20'].median()*100:+.2f}% | f20>0占比{(sub['f20']>0).mean()*100:.0f}%")

    print("\n=== 对照: 非深度窗口砍掉的仓位 ===")
    nd = m[~m['cut_deep']]
    for name, sub in [('全部', nd), ('bp2', nd[nd['bp2']]), ('非bp2', nd[~nd['bp2']])]:
        print(f"  {name}: n={len(sub)} ret中位{sub['ret'].median()*100:+.2f}% | "
              f"f20中位{sub['f20'].median()*100:+.2f}% | f20>0占比{(sub['f20']>0).mean()*100:.0f}%")

    print("\n=== 深度窗口砍仓逐年 (奖品规模) ===")
    print(cd.groupby('year').agg(n=('ret', 'size'),
                                 ret_med=('ret', 'median'),
                                 bp2_n=('bp2', 'sum')).round(4).to_string())

    print("\n=== 成本侧: 豁免=扛深度窗口, 被豁免仓继续跌多少 ===")
    # 豁免后继续持有: 用exit日之后的路径看最大继续回撤 (fwd min)
    min5, min10, min20 = [], [], []
    for code, ed in zip(cd['code'], cd['exit_date']):
        dfp = load_qfq(code)
        if dfp is None:
            min5.append(np.nan); min10.append(np.nan); min20.append(np.nan)
            continue
        i = dfp['datetime'].searchsorted(pd.Timestamp(ed))
        base = dfp['close'].iloc[i] if i < len(dfp) else np.nan
        if base is None or base <= 0 or i >= len(dfp):
            min5.append(np.nan); min10.append(np.nan); min20.append(np.nan)
            continue
        for h, arr in [(5, min5), (10, min10), (20, min20)]:
            seg = dfp['close'].iloc[i:i + h + 1]
            arr.append(float(seg.min() / base - 1) if len(seg) else np.nan)
    cd['min5'] = min5; cd['min10'] = min10; cd['min20'] = min20
    for name, sub in [('bp2', cd[cd['bp2']]), ('非bp2', cd[~cd['bp2']])]:
        print(f"  {name}: n={len(sub)} 豁免后20日内最深跌至中位 "
              f"{sub['min20'].median()*100:+.2f}% (p25 {sub['min20'].quantile(.25)*100:+.2f}%)")

    print("\n=== 深度窗口内被砍bp2明细 ===")
    cols = ['entry_date', 'exit_date', 'code', 'ret', 'f10', 'f20', 'min20']
    det = cd[cd['bp2']][cols].copy()
    for c in ['ret', 'f10', 'f20', 'min20']:
        det[c] = (det[c] * 100).round(2)
    print(det.to_string(index=False))


if __name__ == '__main__':
    main()
