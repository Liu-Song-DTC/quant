#!/usr/bin/env python
"""
参数优化分析
目标：找到最优的factor_value范围和buy_threshold
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

# 只看_F因子
df = df[df['factor_name'].str.endswith('_F', na=False)]

print("=" * 80)
print("参数优化分析（仅_F因子）")
print("=" * 80)

print(f"\n数据量: {len(df):,}")

# 1. 因子值范围优化
print("\n" + "=" * 80)
print("1. 因子值范围优化")
print("=" * 80)

upper_limits = [0.8, 1.0, 1.2, 1.5, 2.0]
lower_thresholds = [0.1, 0.15, 0.18, 0.2, 0.25]

print("\n因子值范围 vs 准确率/收益:")
for upper in upper_limits:
    for lower in lower_thresholds:
        sub = df[(df['factor_value'] > lower) & (df['factor_value'] < upper)]
        if len(sub) > 1000:
            acc = (sub['future_ret'] > 0).mean()
            avg_ret = sub['future_ret'].mean()
            print(f"  {lower:.2f} < fv < {upper:.1f}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.3f}%, n={len(sub):,}")

# 2. 最优范围详细分析
print("\n" + "=" * 80)
print("2. 最优范围详细分析")
print("=" * 80)

# 找到最佳表现的范围
best_acc = 0
best_range = (0, 0)
for lower in np.arange(0.1, 0.3, 0.02):
    for upper in np.arange(0.8, 2.0, 0.1):
        sub = df[(df['factor_value'] > lower) & (df['factor_value'] < upper)]
        if len(sub) > 10000:
            acc = (sub['future_ret'] > 0).mean()
            if acc > best_acc:
                best_acc = acc
                best_range = (lower, upper)

print(f"\n最佳因子值范围: {best_range[0]:.2f} < factor_value < {best_range[1]:.1f}")
print(f"  准确率: {best_acc*100:.2f}%")

# 3. Score范围优化
print("\n" + "=" * 80)
print("3. Score范围优化")
print("=" * 80)

# 在最佳因子值范围内分析Score
sub = df[(df['factor_value'] > best_range[0]) & (df['factor_value'] < best_range[1])]

print(f"\n在最佳因子值范围内({best_range[0]:.2f} < fv < {best_range[1]:.1f})的Score分布:")
print(f"  Score范围: {sub['score'].min():.3f} ~ {sub['score'].max():.3f}")

# 按Score分位数看表现
sub['score_bin'] = pd.qcut(sub['score'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
for bin_name in ['Q1','Q2','Q3','Q4','Q5']:
    bin_sub = sub[sub['score_bin'] == bin_name]
    if len(bin_sub) > 0:
        acc = (bin_sub['future_ret'] > 0).mean()
        avg_ret = bin_sub['future_ret'].mean()
        print(f"  {bin_name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.3f}%")

# 4. 综合优化建议
print("\n" + "=" * 80)
print("4. 综合优化建议")
print("=" * 80)

# 计算最优参数下的预期表现
optimal_sub = df[(df['factor_value'] > best_range[0]) & (df['factor_value'] < best_range[1])]
optimal_acc = (optimal_sub['future_ret'] > 0).mean()
optimal_ret = optimal_sub['future_ret'].mean()

print(f"""
推荐参数:
- factor_value下限: {best_range[0]:.2f}
- factor_value上限: {best_range[1]:.1f}
- 过滤_T因子

预期表现:
- 买入准确率: {optimal_acc*100:.2f}%
- 平均收益: {optimal_ret*100:.3f}%
- 信号数量: {len(optimal_sub):,} ({len(optimal_sub)/len(df)*100:.1f}%)
""")