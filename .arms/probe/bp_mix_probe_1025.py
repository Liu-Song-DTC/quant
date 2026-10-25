"""买点结构探针 (2026-10-25): 无结构买入是否真的差?

realized-first: 用已有 166 笔实际成交, 按信号层的 chan_buy_point 分类
(0=无结构 / 1-5=B1-B5 / 6-9=BP6-9), 比较各桶的胜率/收益。
只读离线, 不碰生产链。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt/d/quant')
RR = ROOT / 'strategy/rolling_validation_results'


def load_trades():
    t = pd.read_csv(RR / 'trade_realized.csv', dtype={'code': str})
    t['code'] = t['code'].str.zfill(6)
    t['entry_date'] = pd.to_datetime(t['entry_date'])
    t['exit_date'] = pd.to_datetime(t['exit_date'])
    return t


def decision_dates(t, sel):
    """成交日的前一交易日 = 决策日 (T收盘决策 → T+1开盘成交)."""
    cal = pd.DatetimeIndex(sorted(sel['date'].unique()))
    out = []
    for _, r in t.iterrows():
        idx = cal.searchsorted(r['entry_date'])
        if idx <= 0:
            out.append(pd.NaT)
            continue
        prev = cal[idx - 1]
        hit = sel[(sel['code'] == r['code']) & (sel['date'] == prev)]
        out.append(prev if len(hit) else pd.NaT)
    return pd.Series(out, index=t.index)


def load_signal_fields(keys, path):
    """从 3.2GB 信号 CSV 里只取需要的 (code,date) 行."""
    want = {'code', 'date', 'buy', 'chan_buy_point', 'signal_level', 'bi_buy_point',
            'score', 'adjusted_score', 'factor_name', 'industry', 'ml_score'}
    kc = {c: d for c, d in keys.items()}
    out = []
    for chunk in pd.read_csv(path, usecols=list(want), chunksize=400_000,
                             dtype={'code': str, 'factor_name': str, 'industry': str}):
        chunk['code'] = chunk['code'].astype(str).str.zfill(6)
        chunk['date'] = pd.to_datetime(chunk['date'])
        m = pd.Series(False, index=chunk.index)
        for (code, day), idxs in kc.items():
            m |= (chunk['code'] == code) & (chunk['date'] == day)
        hit = chunk[m]
        if len(hit):
            out.append(hit)
    return pd.concat(out) if out else pd.DataFrame(columns=list(want))


def main():
    t = load_trades()
    sel = pd.read_csv(RR / 'portfolio_selections.csv', dtype={'code': str})
    sel['code'] = sel['code'].str.zfill(6)
    sel['date'] = pd.to_datetime(sel['date'])
    t['decision_date'] = decision_dates(t, sel)
    keep = t[t['decision_date'].notna()].copy()
    print(f'[probe] 成交 {len(t)} 笔, 能对上决策日 {len(keep)} 笔')
    keys = {(r['code'], r['decision_date']): [i] for i, r in keep.iterrows()}
    sig = load_signal_fields(keys, RR / 'backtest_signals.csv')
    print(f'[probe] 命中信号行 {len(sig)}')
    m = keep.merge(sig, left_on=['code', 'decision_date'], right_on=['code', 'date'],
                   how='left', suffixes=('', '_sig'))
    m['bp'] = pd.to_numeric(m['chan_buy_point'], errors='coerce').fillna(-1).astype(int)
    m['bi'] = pd.to_numeric(m['bi_buy_point'], errors='coerce').fillna(-1).astype(int)
    m['lvl'] = pd.to_numeric(m['signal_level'], errors='coerce').fillna(-1).astype(int)
    m['win'] = m['ret'] > 0
    m['bucket'] = np.where(m['bp'] == 0, '无结构(0)',
                    np.where(m['bp'] <= 5, 'B1-B5(结构)', 'BP6-9(技术)'))
    print('\n--- 按买点大类 (166笔口径, 当前锚) ---')
    g = m.groupby('bucket').agg(n=('ret', 'size'), wr=('win', 'mean'),
                                 mean=('ret', 'mean'), med=('ret', 'median'),
                                 hold=('hold_days', 'mean'))
    g['wr'] = (g['wr'] * 100).round(1); g['mean'] = (g['mean'] * 100).round(2)
    g['med'] = (g['med'] * 100).round(2); g['hold'] = g['hold'].round(0)
    print(g.to_string())
    print('\n--- 细分 ---')
    g2 = m.groupby('bp').agg(n=('ret', 'size'), wr=('win', 'mean'), mean=('ret', 'mean'))
    g2['wr'] = (g2['wr'] * 100).round(1); g2['mean'] = (g2['mean'] * 100).round(2)
    print(g2.to_string())
    print('\n--- 无结构(0) 里 bi_buy_point 细分 ---')
    sub = m[m['bp'] == 0]
    if len(sub):
        g3 = sub.groupby('bi').agg(n=('ret', 'size'), wr=('win', 'mean'), mean=('ret', 'mean'))
        g3['wr'] = (g3['wr'] * 100).round(1); g3['mean'] = (g3['mean'] * 100).round(2)
        print(g3.to_string())
    print('\n--- 无结构(0) 里 signal_level 细分 ---')
    if len(sub):
        g4 = sub.groupby('lvl').agg(n=('ret', 'size'), wr=('win', 'mean'), mean=('ret', 'mean'))
        g4['wr'] = (g4['wr'] * 100).round(1); g4['mean'] = (g4['mean'] * 100).round(2)
        print(g4.to_string())
    m[['code', 'decision_date', 'entry_date', 'ret', 'hold_days', 'bp', 'bi', 'lvl',
       'score', 'factor_name']].to_csv(
        ROOT / '.arms/armout/bp_mix_probe_1025.csv', index=False)
    print('\n→ .arms/armout/bp_mix_probe_1025.csv')


if __name__ == '__main__':
    sys.exit(main())
