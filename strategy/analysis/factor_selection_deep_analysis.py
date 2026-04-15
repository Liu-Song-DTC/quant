#!/usr/bin/env python
"""
深入分析：为什么Dynamic因子IC低但使用比例高？
"""
import pandas as pd
import numpy as np
from scipy import stats

print("=" * 70)
print("因子选择逻辑分析")
print("=" * 70)

df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 1. 分析Dynamic因子选择的因子质量
print("\n[1] Dynamic因子选择的因子质量")

# 按行业分析Dynamic因子的IC
df['industry_abbr'] = df['factor_name'].apply(
    lambda x: str(x).split('_')[1] if str(x).startswith('DYN_') else None
)

def calc_ic(g):
    valid = g.dropna(subset=['factor_value', 'future_ret'])
    if len(valid) < 10:
        return np.nan
    return stats.spearmanr(valid['factor_value'], valid['future_ret'])[0]

print("\n各行业Dynamic因子IC:")
for ind in df['industry_abbr'].dropna().unique():
    sub = df[(df['industry_abbr'] == ind) & (df['factor_name'].astype(str).str.startswith('DYN_'))]
    if len(sub) < 1000:
        continue
    ic = sub.groupby('date').apply(calc_ic)
    buy_signals = sub[sub['buy'] == True]
    buy_acc = (buy_signals['future_ret'] > 0).mean() if len(buy_signals) > 0 else 0
    print(f"  {ind}: IC={ic.mean():.4f}, buy_acc={buy_acc:.2%}, n={len(sub):,}")

# 2. 分析动态因子选择器的质量阈值
print("\n[2] 因子选择质量阈值分析")

# 读取配置
print("""
配置文件中的动态因子选择参数:
- DYN_QUALITY_THRESHOLD = 0.05 (combined_ir < 0.05 认为质量不足)
- min_factor_count = 2 (最少2个因子)
- min_ic_1f = 0.10 (1因子最低IC门槛)

问题分析:
1. 阈值0.05太低，允许了低质量因子
2. 行业间IC差异大，但没有行业特定的阈值
""")

# 3. 分析静态因子配置
print("\n[3] 静态因子配置分析")

static_df = df[df['factor_name'].astype(str).str.startswith('IND_')]
if len(static_df) > 0:
    print(f"静态因子总数: {len(static_df):,}")

    # 解析静态因子名称获取行业
    static_df['static_ind'] = static_df['factor_name'].apply(
        lambda x: str(x).split('_')[1] if '_' in str(x) else None
    )

    for ind in static_df['static_ind'].dropna().unique()[:10]:
        sub = static_df[static_df['static_ind'] == ind]
        if len(sub) < 100:
            continue
        ic = sub.groupby('date').apply(calc_ic)
        print(f"  {ind}: IC={ic.mean():.4f}, n={len(sub):,}")

# 4. 分析为什么Dynamic使用比例高
print("\n[4] Dynamic因子使用比例高的原因")

print("""
根据代码逻辑 (signal_engine.py:702-753):

1. factor_mode = 'both' (动态优先，失败fallback到静态)
2. 动态因子选择流程:
   - 获取DynamicFactorSelector选择的因子
   - 检查dyn_quality (combined_ir) 是否 > 0.05
   - 检查因子数量是否 >= min_factor_count (2)
   - 如果不满足，fallback到静态因子

问题:
- 阈值0.05太低，几乎所有动态因子都能通过
- 但这些因子的实际IC只有2.18%

根本原因:
动态因子选择器在训练时，IC衰减因子(ic_decay_factor=0.8)可能导致
选择了历史IC高但未来IC低的因子（过拟合）
""")

# 5. 验证：分析Dynamic因子的因子数量
print("\n[5] Dynamic因子数量分布")

df['n_factors'] = df['factor_name'].apply(
    lambda x: int(str(x).split('_')[2].replace('F', '')) if str(x).startswith('DYN_') and len(str(x).split('_')) >= 3 else 0
)

for n in sorted(df[df['n_factors'] > 0]['n_factors'].unique()):
    sub = df[df['n_factors'] == n]
    ic = sub.groupby('date').apply(calc_ic)
    buy_signals = sub[sub['buy'] == True]
    buy_acc = (buy_signals['future_ret'] > 0).mean() if len(buy_signals) > 0 else 0
    print(f"  {n}F: IC={ic.mean():.4f}, buy_acc={buy_acc:.2%}, n={len(sub):,}")

# 6. 建议
print("\n" + "=" * 70)
print("优化建议")
print("=" * 70)
print("""
1. 提高动态因子质量阈值:
   - DYN_QUALITY_THRESHOLD: 0.05 → 0.15
   - 只使用高质量因子

2. 增加静态因子使用比例:
   - 降低factor_mode='fixed'的fallback门槛
   - 或直接增加静态因子的权重

3. 按行业设置不同的阈值:
   - 高IC行业(自动化/制造、通信/计算机): 使用动态因子
   - 低IC行业(半导体/光伏、军工): 使用静态因子或降低仓位

4. 改进动态因子选择:
   - 增加正则化，减少过拟合
   - 使用更长的训练窗口
""")
