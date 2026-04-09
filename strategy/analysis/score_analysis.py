#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('rolling_validation_results/validation_results.csv')

print('=== score绝对值与收益关系分析 ===')

# 按score绝对值分位数分析
df['score_bin'] = pd.qcut(df['score'], 10, duplicates='drop')

print('\n按score分位数:')
for i, (bin_name, sub) in enumerate(df.groupby('score_bin', observed=True)):
    acc = (sub['future_ret'] > 0).mean()
    avg_ret = sub['future_ret'].mean()
    avg_score = sub['score'].mean()
    print(f'Q{i+1}: score均值={avg_score:.3f}, 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')

# 分析正score和负score的表现
print('\n正负score分析:')
for cond, name in [(df['score'] > 0, 'score>0'), (df['score'] < 0, 'score<0'), (df['score'] == 0, 'score=0')]:
    sub = df[cond]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f'{name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')

# 分析极端score
print('\n极端score分析:')
for thresh in [0.5, 1.0, 1.5, 2.0]:
    sub = df[df['score'] > thresh]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f'score>{thresh}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')

# 分析score在特定区间
print('\nscore区间分析:')
ranges = [
    ('score<0', df['score'] < 0),
    ('0<score<0.5', (df['score'] >= 0) & (df['score'] < 0.5)),
    ('0.5<score<1.0', (df['score'] >= 0.5) & (df['score'] < 1.0)),
    ('1.0<score<1.5', (df['score'] >= 1.0) & (df['score'] < 1.5)),
    ('1.5<score<2.0', (df['score'] >= 1.5) & (df['score'] < 2.0)),
    ('score>2.0', df['score'] >= 2.0),
]

for name, cond in ranges:
    sub = df[cond]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f'{name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')