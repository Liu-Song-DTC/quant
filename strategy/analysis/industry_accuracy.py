#!/usr/bin/env python
"""
行业准确率分析
目标：找出高准确率行业，集中投资
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

df = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/validation_results.csv'))
df['date'] = pd.to_datetime(df['date'])

# 只看_F因子和买入信号
buy_df = df[(df['buy'] == True) & (df['factor_name'].str.endswith('_F', na=False))]

print("=" * 80)
print("行业准确率分析")
print("=" * 80)

print(f"\n买入信号总数: {len(buy_df):,}")

# 1. 按行业统计
print("\n" + "=" * 80)
print("1. 各行业买入准确率")
print("=" * 80)

industry_stats = []
for industry in buy_df['industry'].unique():
    sub = buy_df[buy_df['industry'] == industry]
    if len(sub) > 1000:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        industry_stats.append({
            'industry': industry,
            'count': len(sub),
            'accuracy': acc,
            'avg_ret': avg_ret
        })

industry_df = pd.DataFrame(industry_stats)
industry_df = industry_df.sort_values('accuracy', ascending=False)

print("\n准确率排序:")
for _, row in industry_df.iterrows():
    print(f"  {row['industry']}: 准确率={row['accuracy']*100:.2f}%, 收益={row['avg_ret']*100:.3f}%, n={row['count']:,}")

# 2. 分析高准确率行业的特征
print("\n" + "=" * 80)
print("2. 高准确率行业特征")
print("=" * 80)

high_acc_industries = industry_df[industry_df['accuracy'] > 0.52]['industry'].tolist()
low_acc_industries = industry_df[industry_df['accuracy'] < 0.50]['industry'].tolist()

print(f"\n高准确率行业(>52%): {high_acc_industries}")
print(f"低准确率行业(<50%): {low_acc_industries}")

# 3. 如果只投资高准确率行业
print("\n" + "=" * 80)
print("3. 只投资高准确率行业的效果")
print("=" * 80)

high_acc_signals = buy_df[buy_df['industry'].isin(high_acc_industries)]
low_acc_signals = buy_df[buy_df['industry'].isin(low_acc_industries)]

if len(high_acc_signals) > 0:
    acc = (high_acc_signals['future_ret'] > 0).mean()
    avg_ret = high_acc_signals['future_ret'].mean()
    print(f"\n高准确率行业信号:")
    print(f"  数量: {len(high_acc_signals):,} ({len(high_acc_signals)/len(buy_df)*100:.1f}%)")
    print(f"  准确率: {acc*100:.2f}%")
    print(f"  平均收益: {avg_ret*100:.3f}%")

if len(low_acc_signals) > 0:
    acc = (low_acc_signals['future_ret'] > 0).mean()
    avg_ret = low_acc_signals['future_ret'].mean()
    print(f"\n低准确率行业信号:")
    print(f"  数量: {len(low_acc_signals):,} ({len(low_acc_signals)/len(buy_df)*100:.1f}%)")
    print(f"  准确率: {acc*100:.2f}%")
    print(f"  平均收益: {avg_ret*100:.3f}%")

# 4. 建议
print("\n" + "=" * 80)
print("4. 优化建议")
print("=" * 80)

# 计算如果排除低准确率行业的预期提升
remaining = buy_df[~buy_df['industry'].isin(low_acc_industries)]
if len(remaining) > 0:
    acc = (remaining['future_ret'] > 0).mean()
    avg_ret = remaining['future_ret'].mean()
    print(f"\n排除低准确率行业后:")
    print(f"  信号数量: {len(remaining):,} ({len(remaining)/len(buy_df)*100:.1f}%)")
    print(f"  准确率: {acc*100:.2f}%")
    print(f"  平均收益: {avg_ret*100:.3f}%")