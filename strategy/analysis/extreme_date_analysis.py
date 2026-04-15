#!/usr/bin/env python
"""
追踪极端值的原始来源：具体是哪个因子产生的？
需要分析因子选择器的数据
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, '/Users/litiancheng01/code/ltc/quant/strategy')

from core.config_loader import load_config
from core.dynamic_factor_selector import DynamicFactorSelector

print("=" * 70)
print("追踪极端值的原始因子来源")
print("=" * 70)

# 1. 加载因子选择器
config = load_config()
selector = DynamicFactorSelector(config)
selector.load_factor_data()

if selector.factor_df is None:
    print("无法加载因子数据")
    sys.exit(1)

print(f"\n因子数据: {len(selector.factor_df):,} 条")
print(f"日期范围: {selector.factor_df['date'].min()} ~ {selector.factor_df['date'].max()}")

# 2. 分析各因子的值分布
print("\n[1] 各因子的值分布 (所有日期)")

factor_cols = [c for c in selector.factor_df.columns if c not in ['date', 'code']]
print(f"\n因子数量: {len(factor_cols)}")

# 统计每个因子的极端值比例
factor_stats = []
for col in factor_cols:
    values = selector.factor_df[col].dropna()
    if len(values) == 0:
        continue

    extreme_pos = (values > 50).sum()
    extreme_neg = (values < -50).sum()
    extreme_ratio = (extreme_pos + extreme_neg) / len(values) * 100

    factor_stats.append({
        'factor': col,
        'count': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'extreme_pos': extreme_pos,
        'extreme_neg': extreme_neg,
        'extreme_ratio': extreme_ratio,
    })

stats_df = pd.DataFrame(factor_stats).sort_values('extreme_ratio', ascending=False)
print("\n极端值比例最高的因子 (Top 15):")
print(stats_df.head(15).to_string(index=False))

# 3. 分析具体股票的因子值
print("\n[2] 分析极端案例股票的原始因子值")

# 688223 和 688390 是之前发现的极端值股票
target_codes = ['688223', '688390']
target_dates = ['2022-09-06', '2022-01-04']

for code, date in zip(target_codes, target_dates):
    print(f"\n股票 {code} 在 {date} 的因子值:")
    stock_data = selector.factor_df[(selector.factor_df['code'] == code) &
                                     (selector.factor_df['date'] == date)]
    if len(stock_data) > 0:
        row = stock_data.iloc[0]
        # 显示所有非空因子值
        for col in factor_cols:
            val = row[col]
            if pd.notna(val) and abs(val) > 1:  # 只显示绝对值>1的
                print(f"  {col}: {val:.4f}")

# 4. 分析基本面因子
print("\n[3] 基本面因子分布分析")

fund_factors = [c for c in factor_cols if c.startswith('fund_')]
print(f"\n基本面因子数量: {len(fund_factors)}")

for col in fund_factors:
    values = selector.factor_df[col].dropna()
    if len(values) == 0:
        continue
    print(f"\n{col}:")
    print(f"  均值: {values.mean():.4f}")
    print(f"  标准差: {values.std():.4f}")
    print(f"  最小值: {values.min():.4f}")
    print(f"  最大值: {values.max():.4f}")
    print(f"  极端值(>50): {(values > 50).sum()} ({(values > 50).sum()/len(values):.2%})")

# 5. 分析技术因子
print("\n[4] 技术因子分布分析")

tech_factors = [c for c in factor_cols if not c.startswith('fund_')]
print(f"\n技术因子数量: {len(tech_factors)}")

# 只显示有极端值的因子
for col in tech_factors:
    values = selector.factor_df[col].dropna()
    if len(values) == 0:
        continue
    extreme = ((values > 50) | (values < -50)).sum()
    if extreme > 0:
        print(f"\n{col}:")
        print(f"  均值: {values.mean():.4f}")
        print(f"  标准差: {values.std():.4f}")
        print(f"  最小值: {values.min():.4f}")
        print(f"  最大值: {values.max():.4f}")
        print(f"  极端值: {extreme} ({extreme/len(values):.2%})")
