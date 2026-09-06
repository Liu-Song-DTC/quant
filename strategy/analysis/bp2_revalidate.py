#!/usr/bin/env python
"""bp2(缠论2买)信号新鲜数据再验证: 9/4数据态+当前代码下的edge是否成立

输入: rolling_validation_results/validation_results.csv (signal_validator重跑, future_ret=20日)
      rolling_validation_results/backtest_signals.csv (当前代码, 含chan_buy_point)
合并键: (date, code), buy=True去重
输出: bp2 vs bp0/bp1的20日收益/命中率/分年/score五档区分度
用法: python analysis/bp2_revalidate.py
"""
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'

print('加载 validation_results...')
v = pd.read_csv(f'{BASE}/validation_results.csv',
                usecols=['date', 'code', 'score', 'buy', 'future_ret'])
v = v[v['buy']].drop_duplicates(subset=['date', 'code'])
v['code'] = v['code'].astype(str).str.zfill(6)
v['date'] = v['date'].astype(str)
print(f'  buy行去重后: {len(v):,}')

print('加载 backtest_signals (只取买点列)...')
s = pd.read_csv(f'{BASE}/backtest_signals.csv',
                usecols=['date', 'code', 'buy', 'chan_buy_point',
                         'chan_structure_score', 'chan_divergence_type'],
                low_memory=False)
s = s[s['buy']].drop_duplicates(subset=['date', 'code'])
s['code'] = s['code'].astype(str).str.zfill(6)
s['date'] = s['date'].astype(str)
print(f'  buy行去重后: {len(s):,}')

m = v.merge(s, on=['date', 'code'], how='inner')
m['year'] = pd.to_datetime(m['date']).dt.year
m = m[m['future_ret'].notna()]
print(f'合并后: {len(m):,}')

bp = m['chan_buy_point']
print('\n=== 各类买点 20日future_ret ===')
g = m.groupby(bp).agg(n=('future_ret', 'size'),
                      hit=('future_ret', lambda x: (x > 0).mean()),
                      mean20=('future_ret', 'mean'),
                      med20=('future_ret', 'median'),
                      mean_score=('score', 'mean'))
print(g.round(4))

print('\n=== bp2 分年 ===')
b2 = m[bp == 2]
print(b2.groupby('year').agg(n=('future_ret', 'size'),
                             hit=('future_ret', lambda x: (x > 0).mean()),
                             mean20=('future_ret', 'mean')).round(4).to_string())

print('\n=== bp2 vs bp0 逐年对照 ===')
for yr, grp in m.groupby('year'):
    for bpv in (0, 2):
        sub = grp[grp['chan_buy_point'] == bpv]['future_ret']
        if len(sub) > 20:
            print(f"  {yr} bp{bpv}: n={len(sub):>5} hit={(sub>0).mean()*100:5.1f}% "
                  f"mean20={sub.mean()*100:+5.2f}%")

print('\n=== bp2 score五档 (score对bp2的区分度) ===')
if len(b2) > 100:
    q = pd.qcut(b2['score'], 5, duplicates='drop')
    print(b2.groupby(q, observed=True)['future_ret'].agg(
        ['size', 'mean']).round(4).to_string())

print('\n=== bp2 结构评分分布 ===')
print(b2['chan_structure_score'].value_counts().head(10).to_string())

print('\n=== bp2 按结构评分 ===')
print(b2.groupby('chan_structure_score')['future_ret'].agg(
    ['size', 'mean']).sort_values('size', ascending=False).head(8).round(4).to_string())

print('\n=== bp2 按背驰类型 ===')
print(b2.groupby('chan_divergence_type')['future_ret'].agg(
    ['size', 'mean']).sort_values('size', ascending=False).round(4).to_string())
