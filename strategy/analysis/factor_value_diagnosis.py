#!/usr/bin/env python
"""
分析factor_value的分布和含义
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("Factor_value 分布分析")
print("=" * 70)

df = pd.read_csv('rolling_validation_results/validation_results.csv')

print("\n[1] Factor_value 分布")
print(f"  均值: {df['factor_value'].mean():.4f}")
print(f"  中位数: {df['factor_value'].median():.4f}")
print(f"  标准差: {df['factor_value'].std():.4f}")
print(f"  最小值: {df['factor_value'].min():.4f}")
print(f"  最大值: {df['factor_value'].max():.4f}")

print("\n百分位分布:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}%: {df['factor_value'].quantile(p/100):.4f}")

print("\n[2] Score计算验证")
# 模拟score计算
fv = df['factor_value'].values
base_score = np.clip(fv, -10, 10)
# 假设动量贡献为0
score_min = base_score * 0.7 - 0.5  # 动量最小贡献
score_max = base_score * 0.7 + 0.5  # 动量最大贡献

print(f"\n如果动量贡献=0，score范围:")
print(f"  均值: {base_score.mean() * 0.7:.4f}")
print(f"  中位数: {np.median(base_score) * 0.7:.4f}")

print(f"\n实际score分布:")
print(f"  均值: {df['score'].mean():.4f}")
print(f"  中位数: {df['score'].median():.4f}")

print("\n[3] Factor_value vs Score 关系")
# 采样10000条
sample = df.sample(min(10000, len(df)), random_state=42)
corr = np.corrcoef(sample['factor_value'], sample['score'])[0, 1]
print(f"  相关系数: {corr:.4f}")

print("\n[4] 因子类型 vs Factor_value")
for ftype in ['Dynamic', 'Static', 'Fallback']:
    sub = df[df['factor_name'].apply(lambda x: 'Dynamic' if str(x).startswith('DYN_') else ('Static' if str(x).startswith('IND_') else 'Fallback')) == ftype]
    if len(sub) > 0:
        print(f"\n{ftype}:")
        print(f"  factor_value均值: {sub['factor_value'].mean():.4f}")
        print(f"  factor_value中位数: {sub['factor_value'].median():.4f}")
        print(f"  score均值: {sub['score'].mean():.4f}")

print("\n[5] 动量因子分布（验证是否被正确标准化）")
# 动量通常在 -0.3 到 0.3 范围（-30%到30%）
# 标准化后应该是 -6 到 6
# 但代码中clip到-2到2
print(f"\n假设动量范围:")
print(f"  mom_5 正常范围: -30% 到 +30%")
print(f"  mom_5_norm = mom_5 / 0.05 -> 范围: -6 到 6")
print(f"  但代码clip到: -2 到 2")
print(f"  这意味着动量贡献最多: ±0.4 (±2 * 0.2)")

print("\n[6] 核心问题总结")
print("""
核心问题：
1. factor_value没有被标准化，均值=7.79
2. 代码假设factor_value范围是-10到10，但实际数据偏离严重
3. score计算公式导致几乎所有股票score为正
4. 买入阈值0.18无法筛选（73%的股票score>0.18）

根本原因：
- 因子计算时没有做截面标准化
- 不同因子的量纲不同，直接加权导致偏差

解决方案：
1. factor_value需要做截面标准化（z-score）
2. 或者使用百分位排名
3. 调整score计算公式，使score具有可比性
""")
