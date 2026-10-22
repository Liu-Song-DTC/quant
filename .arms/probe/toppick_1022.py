#!/usr/bin/env python3
"""每日"权重最大股"与"评分最高股"买入后胜率 (2026-10-23, 只读, anchor_1022).

用户问询: 每次选股中 权重最大的股 和 评分最高的股, 买入后的胜率是多少.
口径:
- selections.date = 决策日; trade_realized.entry_date = 执行日(次日开盘).
  两文件 166↔166 1:1, 按 code + entry_date∈(date, date+6d] 配对.
- realized 胜率 = ret>0 (真实成交到出场); 另报 w5/w20 = 执行日收盘→+5/+20交易日收盘>0.
- 混杂因子: 单日只有 1 笔入场时 top_w == top_s == 唯一腿 ⇒ 分 n_ent=1/2 拆.
用法: python toppick_1022.py
"""
import os
import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'

sel = pd.read_csv(f'{R}/portfolio_selections.csv.anchor_1022',
                  dtype={'code': str}, parse_dates=['date'])
sel['code'] = sel['code'].str.zfill(6)
tr = pd.read_csv(f'{R}/trade_realized.csv.anchor_1022',
                 dtype={'code': str}, parse_dates=['entry_date', 'exit_date'])
tr['code'] = tr['code'].str.zfill(6)

# ---- 配对: sel(date,code) -> tr(code, entry_date in (date, date+6]) ----
tr_by_code = {c: g for c, g in tr.groupby('code')}
rows = []
for r in sel.itertuples():
    g = tr_by_code.get(r.code)
    hit = None
    if g is not None:
        cand = g[(g.entry_date > r.date) & (g.entry_date <= r.date + pd.Timedelta(days=6))]
        if len(cand):
            hit = cand.iloc[0]
    rows.append(hit)
tr_hit = pd.DataFrame(rows, index=sel.index)
miss = tr_hit['ret'].isna().sum()
d = sel.copy()
for c in ['entry_date', 'exit_date', 'hold_days', 'ret', 'exit_px']:
    d[c] = tr_hit[c].values
print(f'配对: {len(d) - miss}/{len(d)} 命中 (miss={miss})')

# ---- 执行日收盘 → w5/w20 ----
codes = sorted(d['code'].unique())
cache = {}
for code in codes:
    f = f'{BT}/{code}_qfq.csv'
    if os.path.exists(f):
        x = pd.read_csv(f, usecols=['datetime', 'close'])
        cache[code] = (pd.to_datetime(x['datetime']).values, x['close'].values.astype(float))

def fwd(code, edate, n):
    cc = cache.get(code)
    if cc is None:
        return np.nan
    dts, cl = cc
    i = int(np.searchsorted(dts, np.datetime64(edate), side='right') - 1)
    if i < 0 or i + n >= len(cl):
        return np.nan
    return cl[i + n] / cl[i] - 1.0

d['w5'] = [fwd(c, e, 5) for c, e in zip(d.code, d.entry_date)]
d['w20'] = [fwd(c, e, 20) for c, e in zip(d.code, d.entry_date)]

# ---- 每日 top_w / top_s ----
d['n_ent'] = d.groupby('date')['code'].transform('count')
d['rank_w'] = d.groupby('date')['weight'].rank(ascending=False, method='first')
d['rank_s'] = d.groupby('date')['score'].rank(ascending=False, method='first')
top_w = d[d['rank_w'] == 1].copy()
top_s = d[d['rank_s'] == 1].copy()
same = int((d.sort_values('date').groupby('date').apply(
    lambda g: g.loc[g['weight'].idxmax(), 'code'] == g.loc[g['score'].idxmax(), 'code'])).sum())
ndates = d['date'].nunique()
print(f'\n{len(d)} 笔入场 / {ndates} 个决策日; 单日1笔={int((d.n_ent==1).groupby(d.date).first().sum())}天, '
      f'2笔={(d.n_ent==2).groupby(d.date).first().sum()}天')
print(f'top_w == top_s 的决策日: {same}/{ndates} ({same/ndates*100:.1f}%)')

def stat(g, name):
    if len(g) == 0:
        return
    print(f'{name:26s} n={len(g):3d}  realized胜率={ (g.ret>0).mean()*100:5.1f}%  '
          f'均值ret={g.ret.mean()*100:+6.2f}%  中位hold={g.hold_days.median():.0f}d  '
          f'w5={(g.w5>0).mean()*100:5.1f}%  w20={(g.w20>0).mean()*100:5.1f}%  '
          f'w20均值={g.w20.mean()*100:+6.2f}%')

print('\n== 全体 / top_w / top_s ==')
stat(d, '全部入场 (基线)')
stat(top_w, '每日权重最大股')
stat(top_s, '每日评分最高股')

print('\n== 按当日入场笔数拆 ==')
for ne in sorted(d.n_ent.unique()):
    stat(top_w[top_w.n_ent == ne], f'top_w (当日{int(ne)}笔)')
    stat(top_s[top_s.n_ent == ne], f'top_s (当日{int(ne)}笔)')

# 非top腿对照 (2笔日里被比下去的那条)
non_top = d[(d.n_ent == 2) & (d['rank_w'] == 2) & (d['rank_s'] == 2)]
stat(non_top, '2笔日的另一腿(双非top)')

print('\n== 年度拆: top_w realized 胜率 ==')
d['yr'] = d.date.dt.year
for y, g in top_w.groupby(top_w.date.dt.year):
    print(f'  {y}: n={len(g):3d}  胜率={(g.ret>0).mean()*100:5.1f}%  均值ret={g.ret.mean()*100:+6.2f}%')
print('== 年度拆: top_s realized 胜率 ==')
for y, g in top_s.groupby(top_s.date.dt.year):
    print(f'  {y}: n={len(g):3d}  胜率={(g.ret>0).mean()*100:5.1f}%  均值ret={g.ret.mean()*100:+6.2f}%')

# 大亏/大赢集中度
print('\n== top_w 尾部 ==')
tw = top_w.sort_values('ret')
print(tw[['date', 'code', 'weight', 'score', 'hold_days', 'ret']].head(5).to_string(index=False))
print(tw[['date', 'code', 'weight', 'score', 'hold_days', 'ret']].tail(5).to_string(index=False))
d.to_csv('/mnt/d/quant/.arms/armout/toppick_1022.csv', index=False)
print('\n落盘: .arms/armout/toppick_1022.csv')
