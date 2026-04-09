#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats

# 加载数据
selections = pd.read_csv('rolling_validation_results/portfolio_selections.csv')
validation = pd.read_csv('rolling_validation_results/validation_results.csv')

print('=== 被选中股票的表现分析 ===')
print(f'选股次数: {len(selections):,}')

# 合并数据看收益
merged = selections.merge(
    validation[['date', 'code', 'future_ret', 'factor_name', 'factor_value']],
    on=['date', 'code'],
    how='left'
)

print(f'合并后记录数: {len(merged):,}')

if len(merged) > 0:
    # 计算准确率
    acc = (merged['future_ret'] > 0).mean()
    avg_ret = merged['future_ret'].mean()
    print(f'\n被选中股票准确率: {acc*100:.2f}%')
    print(f'被选中股票平均收益: {avg_ret*100:.2f}%')

    # 按score分组看准确率
    print('\n按score分位数看准确率:')
    merged['score_bin'] = pd.qcut(merged['score'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    for bin_name in ['Q1','Q2','Q3','Q4']:
        sub = merged[merged['score_bin'] == bin_name]
        if len(sub) > 0:
            acc = (sub['future_ret'] > 0).mean()
            avg_ret = sub['future_ret'].mean()
            print(f'  {bin_name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')

    # 按因子类型看表现
    print('\n按因子类型看表现:')
    for suffix in ['_T', '_F']:
        sub = merged[merged['factor_name'].str.endswith(suffix, na=False)]
        if len(sub) > 100:
            acc = (sub['future_ret'] > 0).mean()
            avg_ret = sub['future_ret'].mean()
            print(f'  {suffix}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, n={len(sub):,}')