#!/usr/bin/env python3
"""Arm snapshot stats for bp source-gate arms (2026-10-22).

Usage: python arm_stats.py TAG [TAG...]   (reads .arms/armout/bp1022_TAG/)
Baseline = first tag (normally base3). Prints one row per arm + yearly table +
trade source breakdown. All return-like numbers are decimal unless marked %.
"""
import os
import sys
import numpy as np
import pandas as pd

A = '/mnt/d/quant/.arms/armout'


def load(tag):
    d = f'{A}/bp1022_{tag}'
    eq = pd.read_csv(f'{d}/equity_curve.csv', parse_dates=['date']).set_index('date')
    tr = pd.read_csv(f'{d}/trade_realized.csv', dtype={'code': str},
                     parse_dates=['entry_date', 'exit_date'])
    sel = pd.read_csv(f'{d}/portfolio_selections.csv', dtype={'code': str},
                      parse_dates=['date'])
    return eq, tr, sel


def mdd(nav):
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def metrics(eq, tr):
    nav = eq['nav'].astype(float)
    r = nav.pct_change().dropna()
    yrs = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = r.mean() * 252 / downside if downside > 0 else np.nan
    # 5/20日非重叠窗口胜率
    def win_h(h):
        x = nav.iloc[h::h] / nav.shift(h).iloc[h::h] - 1
        return float((x > 0).mean()) if len(x) else np.nan
    # 月/年胜率
    m = nav.resample('ME').last()
    mret = m.pct_change().dropna()
    y = nav.resample('YE').last()
    yret = y.pct_change().dropna()
    # 交易
    n = len(tr)
    wins = tr.loc[tr.ret > 0, 'ret'].sum()
    losses = -tr.loc[tr.ret < 0, 'ret'].sum()
    hold = tr.hold_days.mean() if n else np.nan
    return {
        'n_days': len(nav), 'nav': nav.iloc[-1], 'total': nav.iloc[-1] / nav.iloc[0] - 1,
        'cagr': cagr, 'vol': vol, 'sharpe': sharpe, 'sortino': sortino,
        'mdd': mdd(nav), 'calmar': cagr / abs(mdd(nav)) if mdd(nav) < 0 else np.nan,
        'win_d': float((r > 0).mean()), 'win_5d': win_h(5), 'win_20d': win_h(20),
        'win_m': float((mret > 0).mean()), 'win_y': float((yret > 0).mean()),
        'n_tr': n, 'win_tr': float((tr.ret > 0).mean()) if n else np.nan,
        'mean_tr': tr.ret.mean() if n else np.nan,
        'med_tr': tr.ret.median() if n else np.nan,
        'pf_tr': wins / losses if losses > 0 else np.inf,
        'hold': hold,
        'n_sel': len(sel), 'n_sel_dates': sel.date.nunique(),
        'avg_pos': np.nan,
    }, yret, mret


TAGS = sys.argv[1:]
if not TAGS:
    print(__doc__)
    sys.exit(1)

rows = {}
yearly = {}
for tag in TAGS:
    eq, tr, sel = load(tag)
    m, yret, mret = metrics(eq, tr)
    # 日均持仓数: 从成交区间重建
    dates = eq.index
    cnt = np.zeros(len(dates), dtype=int)
    for _, t in tr.iterrows():
        mask = (dates >= t.entry_date) & (dates <= t.exit_date)
        cnt += mask.astype(int)
    m['avg_pos'] = float(cnt.mean())
    rows[tag] = m
    yearly[tag] = yret

df = pd.DataFrame(rows).T
base = TAGS[0]
for c in ['nav', 'sharpe', 'mdd', 'cagr', 'win_tr', 'win_20d', 'win_m', 'avg_pos']:
    df[f'd_{c}'] = df[c] - df.loc[base, c]
# 选股 Jaccard
sets = {}
for tag in TAGS:
    _, _, sel = load(tag)
    sets[tag] = set(zip(sel.date.dt.strftime('%Y-%m-%d'), sel.code))
base_set = sets[base]
df['jacc_sel'] = [len(sets[t] & base_set) / len(sets[t] | base_set) if sets[t] | base_set else np.nan
                  for t in TAGS]

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 80)
print('=== ARM METRICS ===')
show = ['n_days', 'nav', 'total', 'cagr', 'vol', 'sharpe', 'sortino', 'mdd', 'calmar',
        'win_d', 'win_5d', 'win_20d', 'win_m', 'win_y',
        'n_tr', 'win_tr', 'mean_tr', 'med_tr', 'pf_tr', 'hold',
        'n_sel', 'avg_pos', 'jacc_sel']
print(df[show].to_string(float_format=lambda v: f'{v:.4f}'))
print('\n=== DELTA vs', base, '===')
print(df[[f'd_{c}' for c in ['nav', 'sharpe', 'mdd', 'cagr', 'win_tr', 'win_20d', 'win_m', 'avg_pos']]]
      .to_string(float_format=lambda v: f'{v:+.4f}'))

print('\n=== 逐年收益 (年末对年末; 首年从起点) ===')
yd = pd.DataFrame({t: yearly[t] for t in TAGS})
yd.index = yd.index.year
print(yd.to_string(float_format=lambda v: f'{v*100:+.2f}%'))

print('\n=== 成交源分布 (成交日回溯到选股日) ===')
cells = pd.read_parquet('/mnt/d/quant/.arms/probe/bp_deep_1021/cells.parquet',
                        columns=['date', 'code', 'S1', 'S2', 'S3', 'S4', 'k'])
cells['date'] = pd.to_datetime(cells['date'])
for tag in TAGS:
    eq, tr, sel = load(tag)
    tr = tr.sort_values(['entry_date', 'code'])
    s = sel.sort_values(['date', 'code'])[['date', 'code']]
    t = pd.merge_asof(tr, s.rename(columns={'date': 'sel_date'}),
                      left_on='entry_date', right_on='sel_date', by='code',
                      direction='backward', tolerance=pd.Timedelta('10D'))
    t = t.merge(cells, left_on=['sel_date', 'code'], right_on=['date', 'code'],
                how='left', suffixes=('', '_c'))
    t['subset'] = t[['S1', 'S2', 'S3', 'S4']].fillna(-1).astype(int).astype(str).agg(''.join, axis=1)
    g = t.groupby('subset').apply(lambda x: pd.Series({
        'n': len(x), 'win': (x.ret > 0).mean(), 'mean': x.ret.mean(),
        'pf': x.loc[x.ret > 0, 'ret'].sum() / max(-x.loc[x.ret < 0, 'ret'].sum(), 1e-12),
    }))
    print(f'--- {tag} ---')
    print(g.to_string(float_format=lambda v: f'{v:.4f}'))
