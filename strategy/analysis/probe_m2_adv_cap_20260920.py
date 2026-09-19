#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2探针 (2026-09-20): ADV感知仓位上限前置检验 — 组合每个持仓的
市值/当日成交额 比分布。若该比在全史任何时点都 << 1%, 则50万账户的流动性
冲击/滑点失真可忽略, M2机制(成交额/ADV>阈值降仓)无收益面 → 关闭。
反之若存在显著越限(≥1%)的仓位-日, 则量化"本应裁剪的市值", 供机制臂设计。
输入: 基线归档 equity_curve/portfolio_selections + 日K线amount (只读)。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = os.path.join(BASE, 'arms_20260919', 'C1_faithful_0_55')
EQ = os.path.join(ARM, 'pre_equity_curve.csv')
SEL = os.path.join(ARM, 'pre_portfolio_selections.csv')
KLINE_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'

_kcache = {}


def amount_on(code, d):
    """d日(或其前最近交易日)成交额; 无→NaN"""
    if code not in _kcache:
        p = os.path.join(KLINE_DIR, f'{code}_qfq.csv')
        if not os.path.exists(p):
            _kcache[code] = None
            return np.nan
        df = pd.read_csv(p, usecols=['datetime', 'amount'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')['amount'].sort_index()
        _kcache[code] = df
    k = _kcache[code]
    if k is None or len(k) == 0:
        return np.nan
    pos = k.index.searchsorted(d, side='right') - 1
    if pos < 0:
        return np.nan
    return float(k.iloc[pos])


def main():
    eq = pd.read_csv(EQ, parse_dates=['date']).set_index('date')['nav']
    sel = pd.read_csv(SEL, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])
    sel = sel[sel['weight'] > 0].copy()
    print(f'选股行(weight>0): {len(sel)}, 日期数: {sel["date"].nunique()}')

    rows = []
    missing = 0
    for _, r in sel.iterrows():
        d, code, w = r['date'], r['code'].zfill(6), r['weight']
        # 选股日净值 = 当日nav(组合层用当日净值做预算)
        try:
            nav = float(eq.loc[d])
        except KeyError:
            missing += 1
            continue
        value = w * nav
        amt = amount_on(code, d)
        if np.isnan(amt) or amt <= 0:
            missing += 1
            continue
        rows.append({'date': d, 'code': code, 'value': value, 'amount': amt,
                     'ratio': value / amt, 'yr': d.year})
    df = pd.DataFrame(rows)
    print(f'可算仓位-日: {len(df)} (缺失: {missing})')

    print('\n=== 市值/成交额 比分布 ===')
    print(f"p50 {df['ratio'].median()*100:.3f}%  p90 {df['ratio'].quantile(0.90)*100:.3f}%  "
          f"p99 {df['ratio'].quantile(0.99)*100:.3f}%  max {df['ratio'].max()*100:.3f}%")
    for th in [0.005, 0.01, 0.02, 0.05]:
        n = (df['ratio'] > th).sum()
        clipped = (df.loc[df['ratio'] > th, 'value'] - df.loc[df['ratio'] > th, 'amount'] * th).sum()
        print(f"  超{th*100:.1f}%: {n} 仓位-日 ({n/len(df)*100:.2f}%), "
              f"本应裁剪市值合计 {clipped/10000:.0f}万")
    print(f"  50万账户持仓市值分布: p50 {df['value'].median()/10000:.1f}万  "
          f"max {df['value'].max()/10000:.1f}万")
    print(f"  成交额分布: p50 {df['amount'].median()/1e8:.2f}亿  "
          f"p5 {df['amount'].quantile(0.05)/1e8:.3f}亿")

    print('\n=== 逐年 ===')
    for yr, g in df.groupby('yr'):
        over1 = (g['ratio'] > 0.01)
        print(f"  {yr}: n={len(g):4d}  p90 {g['ratio'].quantile(0.90)*100:.3f}%  "
              f"max {g['ratio'].max()*100:.3f}%  超1%: {int(over1.sum())}")

    out = os.path.join(BASE, 'analysis', 'probe_m2_advcap_20260920.csv')
    df.to_csv(out, index=False)
    print(f'\n明细已存: {out}')


if __name__ == '__main__':
    main()
