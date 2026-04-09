#!/usr/bin/env python
import pandas as pd
import numpy as np

df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 检查buy=True的条件
buy_df = df[df['buy'] == True]

print('=== 买入信号分析 ===')
print(f'买入信号占比: {len(buy_df)/len(df)*100:.1f}%')

# 分析买入信号的score分布
print('\n买入信号score分布:')
print(f'  最小值: {buy_df["score"].min():.3f}')
print(f'  最大值: {buy_df["score"].max():.3f}')
print(f'  均值: {buy_df["score"].mean():.3f}')

# 分析负分买入信号
neg_score_buy = buy_df[buy_df['score'] < 0]
print(f'\n负分买入信号数量: {len(neg_score_buy):,} ({len(neg_score_buy)/len(buy_df)*100:.1f}%)')
if len(neg_score_buy) > 0:
    acc = (neg_score_buy['future_ret'] > 0).mean()
    avg_ret = neg_score_buy['future_ret'].mean()
    print(f'负分买入准确率: {acc*100:.2f}%')
    print(f'负分买入平均收益: {avg_ret*100:.2f}%')

# 分析正分买入信号
pos_score_buy = buy_df[buy_df['score'] >= 0]
print(f'\n正分买入信号数量: {len(pos_score_buy):,} ({len(pos_score_buy)/len(buy_df)*100:.1f}%)')
if len(pos_score_buy) > 0:
    acc = (pos_score_buy['future_ret'] > 0).mean()
    avg_ret = pos_score_buy['future_ret'].mean()
    print(f'正分买入准确率: {acc*100:.2f}%')
    print(f'正分买入平均收益: {avg_ret*100:.2f}%')

# 分析不同score区间的买入准确率
print('\n=== 不同score区间的买入准确率 ===')
for thresh in [-1, -0.5, 0, 0.5, 1.0]:
    sub = buy_df[buy_df['score'] < thresh]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f'score<{thresh}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')