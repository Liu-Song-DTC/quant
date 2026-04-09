#!/usr/bin/env python
import pandas as pd

# 检查选股结果
df = pd.read_csv('rolling_validation_results/portfolio_selections.csv')
print('选股结果统计:')
print(f'总选股次数: {len(df):,}')
print(f'平均分数: {df["score"].mean():.3f}')
print(f'分数范围: {df["score"].min():.3f} - {df["score"].max():.3f}')

# 检查分数分布
print('\n分数分布:')
for thresh in [0, 0.5, 1.0, 1.5, 2.0]:
    cnt = (df['score'] > thresh).sum()
    print(f'  score>{thresh}: {cnt:,} ({cnt/len(df)*100:.1f}%)')

# 检查负分数量
neg_cnt = (df['score'] < 0).sum()
print(f'\n负分股票数量: {neg_cnt:,} ({neg_cnt/len(df)*100:.1f}%)')

# 检查分数>2.0的数量
high_cnt = (df['score'] > 2.0).sum()
print(f'分数>2.0的股票数量: {high_cnt:,} ({high_cnt/len(df)*100:.1f}%)')