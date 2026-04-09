#!/usr/bin/env python
import pandas as pd

df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 分析如果要求不同score阈值会怎样
buy_df = df[df['buy'] == True]

for thresh in [0, 0.3, 0.5, 0.8, 1.0]:
    pos_buy = buy_df[buy_df['score'] > thresh]
    print(f'=== 如果要求score > {thresh} ===')
    print(f'买入信号数: {len(pos_buy):,}')
    print(f'买入信号占比: {len(pos_buy)/len(df)*100:.1f}%')
    if len(pos_buy) > 0:
        acc = (pos_buy['future_ret'] > 0).mean()
        avg_ret = pos_buy['future_ret'].mean()
        print(f'买入准确率: {acc*100:.2f}%')
        print(f'买入平均收益: {avg_ret*100:.2f}%')
    print()