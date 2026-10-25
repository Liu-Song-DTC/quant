"""买点结构探针 v2 (2026-10-25): 完整选股集 (6053 行) 的买点分布。

回答两个问题:
  1. 组合层实际选中的票里, 有多少是"无结构"(bp=0)?
     → 若为 0, 则"无结构必须叠加其他买点"这条改动是空转 (no-op), 不值得开臂。
  2. 各买点桶在**池级**(选股集)的占比与后续收益, 供臂设计参考。
只读离线。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt/d/quant')
RR = ROOT / 'strategy/rolling_validation_results'


def scan_signals(keys):
    want = ['code', 'date', 'buy', 'chan_buy_point', 'signal_level', 'bi_buy_point',
            'score', 'adjusted_score', 'factor_name', 'industry']
    out = []
    for chunk in pd.read_csv(RR / 'backtest_signals.csv', usecols=want, chunksize=400_000,
                             dtype={'code': str, 'factor_name': str, 'industry': str}):
        chunk['code'] = chunk['code'].astype(str).str.zfill(6)
        chunk['date'] = pd.to_datetime(chunk['date'])
        m = pd.Series(False, index=chunk.index)
        for code, day in keys:
            m |= (chunk['code'] == code) & (chunk['date'] == day)
        hit = chunk[m]
        if len(hit):
            out.append(hit)
    return pd.concat(out) if out else pd.DataFrame(columns=want)


def main():
    sel = pd.read_csv(RR / 'portfolio_selections.csv', dtype={'code': str})
    sel['code'] = sel['code'].str.zfill(6)
    sel['date'] = pd.to_datetime(sel['date'])
    print(f'[probe] 选股集 {len(sel)} 行, {sel["date"].nunique()} 个决策日, '
          f'{sel["code"].nunique()} 只')
    keys = set(zip(sel['code'], sel['date']))
    sig = scan_signals(keys)
    print(f'[probe] 命中信号行 {len(sig)}')
    m = sel.merge(sig, left_on=['code', 'date'], right_on=['code', 'date'], how='left')
    m['bp'] = pd.to_numeric(m['chan_buy_point'], errors='coerce').fillna(-1).astype(int)
    m['lvl'] = pd.to_numeric(m['signal_level'], errors='coerce').fillna(-1).astype(int)
    m['bi'] = pd.to_numeric(m['bi_buy_point'], errors='coerce').fillna(-1).astype(int)
    print('\n--- 选股集买点分布 (组合层实际选中的票) ---')
    g = m.groupby('bp').agg(n=('code', 'size'), codes=('code', 'nunique'),
                             mean_score=('score', 'mean'))
    g['pct'] = (g['n'] / len(m) * 100).round(2)
    g['mean_score'] = g['mean_score'].round(4)
    print(g.to_string())
    n0 = int((m['bp'] == 0).sum())
    print(f'\n无结构(bp=0) 占比: {n0}/{len(m)} = {n0/len(m)*100:.2f}%')
    print(f'未命中信号的选股行: {int((m["bp"] == -1).sum())}')
    print('\n--- 无结构行的 score 分布 (若有) ---')
    z = m[m['bp'] == 0]
    if len(z):
        print(z['score'].describe().round(4).to_string())
    m.to_csv(ROOT / '.arms/armout/bp_sel_probe_1025.csv', index=False)
    print('\n→ .arms/armout/bp_sel_probe_1025.csv')


if __name__ == '__main__':
    sys.exit(main())
