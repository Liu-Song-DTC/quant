#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats

# 加载验证数据
df = pd.read_csv('rolling_validation_results/validation_results.csv')

print('=== 买入信号详细分析 ===')
buy_df = df[df['buy'] == True].copy()
print(f'买入信号数量: {len(buy_df):,}')
print(f'买入准确率: {(buy_df["future_ret"] > 0).mean()*100:.2f}%')

# 按factor_value分位数看准确率
buy_df['fv_bin'] = pd.qcut(buy_df['factor_value'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
print('\n按factor_value分位数:')
for bin_name in ['Q1','Q2','Q3','Q4','Q5']:
    sub = buy_df[buy_df['fv_bin'] == bin_name]
    if len(sub) > 0:
        acc = (sub['future_ret'] > 0).mean()
        print(f'  {bin_name}: 准确率={acc*100:.2f}%, 平均收益={sub["future_ret"].mean()*100:.2f}%, n={len(sub):,}')

print('\n=== 按score分位数分析 ===')
buy_df['score_bin'] = pd.qcut(buy_df['score'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
for bin_name in ['Q1','Q2','Q3','Q4','Q5']:
    sub = buy_df[buy_df['score_bin'] == bin_name]
    if len(sub) > 0:
        acc = (sub['future_ret'] > 0).mean()
        print(f'  {bin_name}: 准确率={acc*100:.2f}%, 平均收益={sub["future_ret"].mean()*100:.2f}%, n={len(sub):,}')

print('\n=== 高factor_value买入分析 ===')
for thresh in [2, 5, 10, 20]:
    sub = buy_df[buy_df['factor_value'] > thresh]
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        print(f'  fv>{thresh}: 准确率={acc*100:.2f}%, 平均收益={sub["future_ret"].mean()*100:.2f}%, n={len(sub):,}')

print('\n=== 负IC因子分析 ===')
neg_ic_factors = []
for fn in df['factor_name'].unique():
    if pd.isna(fn) or not str(fn).startswith('DYN_'):
        continue
    sub = df[df['factor_name'] == fn].dropna(subset=['factor_value', 'future_ret'])
    if len(sub) > 100:
        ic, _ = stats.spearmanr(sub['factor_value'], sub['future_ret'])
        if ic < -0.02:
            buy_sub = sub[sub['buy'] == True]
            if len(buy_sub) > 0:
                acc = (buy_sub['future_ret'] > 0).mean()
                neg_ic_factors.append((fn, ic, acc, len(buy_sub)))

neg_ic_factors.sort(key=lambda x: x[1])
for fn, ic, acc, n in neg_ic_factors[:10]:
    print(f'  {fn}: IC={ic*100:.2f}%, 买入准确率={acc*100:.2f}%, n_buy={n:,}')

print('\n=== 极端值来源分析 ===')
extreme_df = df[(df['factor_value'] > 2) | (df['factor_value'] < -2)].copy()
factor_extreme_count = extreme_df.groupby('factor_name').size().sort_values(ascending=False).head(10)
print('极端值最多的因子:')
for fn, cnt in factor_extreme_count.items():
    pct = cnt / len(extreme_df) * 100
    print(f'  {fn}: {cnt:,} ({pct:.1f}%)')

print('\n=== 动量与买入准确率 ===')
# 分析动量与收益关系
if 'mom_5' in df.columns:
    df['mom_bin'] = pd.qcut(df['mom_5'], 5, labels=['Q1跌','Q2','Q3','Q4','Q5涨'], duplicates='drop')
    print('按mom_5分位数:')
    for bin_name in ['Q1跌','Q2','Q3','Q4','Q5涨']:
        sub = df[df['mom_bin'] == bin_name]
        if len(sub) > 0:
            acc = (sub['future_ret'] > 0).mean()
            print(f'  {bin_name}: 未来收益准确率={acc*100:.2f}%, 平均收益={sub["future_ret"].mean()*100:.2f}%')