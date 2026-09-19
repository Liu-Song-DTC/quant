#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N1探针 (2026-09-20): 周内效应 — 基线buy信号逐笔 × fwd5/fwd10 按入场星期几分桶.
问: 周五/周四买入的fwd差是否系统性不同于其他日 (开源文献周内效应: 周一高波动、
周五低收益等)。若某周日fwd显著差 → 可尝试按周日错峰入场/门槛调整。
纪律: 探针只测信号层偏导, 组合层需机制臂另行裁决。
输入: 基线signals(buy子集) + 日K线 (均为只读归档)。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS = os.path.join(BASE, 'arms_20260919', '_baseline_sig', 'backtest_signals.csv')
KLINE_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'

_kcache = {}


def kline(code):
    if code in _kcache:
        return _kcache[code]
    p = os.path.join(KLINE_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        _kcache[code] = None
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')['close'].sort_index()
    _kcache[code] = df
    return df


def fwd_ret(k, d, n):
    """d日close → 之后第n个交易日close的收益; 不足n日用最后一日。"""
    pos = k.index.searchsorted(d)
    seg = k.iloc[pos:pos + n + 1]
    if len(seg) == 0:
        return np.nan
    return seg.iloc[min(n, len(seg) - 1)] / seg.iloc[0] - 1


def main():
    sig = pd.read_csv(SIGNALS, usecols=['date', 'code', 'buy'], dtype={'code': str})
    sig['date'] = pd.to_datetime(sig['date'])
    sig['buy'] = sig['buy'].astype(str).str.lower().map({'true': True, 'false': False})
    buys = sig[sig['buy']].drop_duplicates(['date', 'code'])
    print(f'buy信号行数(去重): {len(buys)}')
    wd = buys['date'].dt.dayofweek  # 0=周一
    print('\n=== buy信号按星期几分布 (可能受调仓日机制约束, 分布本身即信息) ===')
    for i, name in enumerate(['周一', '周二', '周三', '周四', '周五']):
        print(f"  {name}: {(wd == i).sum()}")

    rows = []
    for (d, c) in zip(buys['date'], buys['code']):
        k = kline(str(c).zfill(6))
        if k is None:
            continue
        rows.append({'date': d, 'code': c, 'weekday': d.dayofweek,
                     'fwd5': fwd_ret(k, d, 5), 'fwd10': fwd_ret(k, d, 10)})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=['fwd10'])
    print(f'\n可算fwd笔数: {len(df)}')

    print('\n=== fwd收益按星期几 (均值%, 中位%, n, 与全样本差) ===')
    all5, all10 = df['fwd5'].mean(), df['fwd10'].mean()
    for i, name in enumerate(['周一', '周二', '周三', '周四', '周五']):
        g = df[df['weekday'] == i]
        if len(g) == 0:
            print(f"  {name}: 无样本")
            continue
        d5 = g['fwd5'].mean() * 100 - all5 * 100
        d10 = g['fwd10'].mean() * 100 - all10 * 100
        # 置换检验: 洗牌weekday标签, 测均值差分布 (2000次)
        rng = np.random.default_rng(42)
        obs10 = g['fwd10'].mean() - all10
        cnt = 0
        nperm = 2000
        arr = df['fwd10'].values
        for _ in range(nperm):
            perm = rng.permutation(arr)
            if perm[:len(g)].mean() - perm.mean() >= obs10:
                cnt += 1
        p = cnt / nperm
        print(f"  {name}: fwd5 {g['fwd5'].mean()*100:+.2f}% (Δ{d5:+.2f}) | "
              f"fwd10 {g['fwd10'].mean()*100:+.2f}% (Δ{d10:+.2f}) | "
              f"n={len(g)} | 置换p(单侧高)={p:.3f}")

    # 年度稳定性: 最强的负向周日是否逐年稳定
    worst = df.groupby('weekday')['fwd10'].mean().idxmin()
    print(f'\n=== 最差周日({["周一","周二","周三","周四","周五"][worst]}) 逐年fwd10均值 ===')
    df['yr'] = df['date'].dt.year
    for yr, g in df[df['weekday'] == worst].groupby('yr'):
        rest = df[(df['weekday'] != worst) & (df['yr'] == yr)]
        print(f"  {yr}: 最差周日 {g['fwd10'].mean()*100:+.2f}% (n={len(g)}) vs "
              f"其余 {rest['fwd10'].mean()*100:+.2f}% (n={len(rest)})")

    out = os.path.join(BASE, 'analysis', 'probe_n1_weekday_20260920.csv')
    df.to_csv(out, index=False)
    print(f'\n明细已存: {out}')


if __name__ == '__main__':
    main()
