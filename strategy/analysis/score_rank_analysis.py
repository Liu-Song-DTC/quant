#!/usr/bin/env python
"""
Score排名与实际表现分析
目标：理解为什么高分股票表现差
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

print("=" * 80)
print("Score排名与实际表现分析")
print("=" * 80)

# 计算每日Score排名
df['score_rank'] = df.groupby('date')['score'].rank(ascending=False, pct=True)

# 按Score排名分10组看准确率
print("\n按Score排名分位数看准确率和收益:")
for i in range(10):
    low, high = i * 0.1, (i + 1) * 0.1
    sub = df[(df['score_rank'] >= low) & (df['score_rank'] < high)]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        ic, _ = stats.spearmanr(sub['score'], sub['future_ret'])
        print(f"  {i*10:2d}%-{(i+1)*10:2d}%: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.3f}%, IC={ic*100:.2f}%, n={len(sub):,}")

# 只看买入信号
buy_df = df[df['buy'] == True]
buy_df['score_rank'] = buy_df.groupby('date')['score'].rank(ascending=False, pct=True)

print("\n买入信号按Score排名分位数:")
for i in range(5):
    low, high = i * 0.2, (i + 1) * 0.2
    sub = buy_df[(buy_df['score_rank'] >= low) & (buy_df['score_rank'] < high)]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f"  Q{i+1}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.3f}%, n={len(sub):,}")

# 分析Score与factor_value的关系
print("\nScore与factor_value的关系:")
corr = df[['score', 'factor_value']].corr().iloc[0, 1]
print(f"  相关系数: {corr:.3f}")

# 按factor_value分组看Score
print("\n按factor_value分组看Score和准确率:")
df['fv_bin'] = pd.qcut(df['factor_value'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
for bin_name in ['Q1','Q2','Q3','Q4','Q5']:
    sub = df[df['fv_bin'] == bin_name]
    if len(sub) > 100:
        avg_score = sub['score'].mean()
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f"  {bin_name}: Score均值={avg_score:.3f}, 准确率={acc*100:.2f}%, 收益={avg_ret*100:.3f}%")

# 分析为什么高分股票表现差
print("\n" + "=" * 80)
print("为什么高分股票表现差？")
print("=" * 80)

# 高分股票的特征
high_score = df[df['score_rank'] < 0.2]  # Top 20%
low_score = df[df['score_rank'] > 0.8]  # Bottom 20%

print("\n高分(Top20%)vs低分(Bottom20%)特征:")
print(f"  高分 factor_value均值: {high_score['factor_value'].mean():.3f}")
print(f"  低分 factor_value均值: {low_score['factor_value'].mean():.3f}")

# 查看高分股票的因子类型
print("\n高分股票的因子类型分布:")
high_score_f = high_score[high_score['factor_name'].str.endswith('_F', na=False)]
high_score_t = high_score[high_score['factor_name'].str.endswith('_T', na=False)]
print(f"  _F因子: {len(high_score_f):,} ({len(high_score_f)/len(high_score)*100:.1f}%)")
print(f"  _T因子: {len(high_score_t):,} ({len(high_score_t)/len(high_score)*100:.1f}%)")

if len(high_score_f) > 0:
    acc_f = (high_score_f['future_ret'] > 0).mean()
    print(f"  _F准确率: {acc_f*100:.2f}%")
if len(high_score_t) > 0:
    acc_t = (high_score_t['future_ret'] > 0).mean()
    print(f"  _T准确率: {acc_t*100:.2f}%")

# 分析动量因子的影响
print("\n动量因子分析:")
# 检查是否动量因子在高分股票中占比高
# mom_5 > 0.1 或 mom_20 > 0.2 的股票可能有动量过热问题
# 由于数据中没有mom值，我们通过factor_value极端值分析
extreme_fv = df[abs(df['factor_value']) > 1.5]
normal_fv = df[abs(df['factor_value']) <= 1.5]

print(f"  factor_value极端值(>1.5): 准确率={(extreme_fv['future_ret']>0).mean()*100:.2f}%, n={len(extreme_fv):,}")
print(f"  factor_value正常值(<=1.5): 准确率={(normal_fv['future_ret']>0).mean()*100:.2f}%, n={len(normal_fv):,}")