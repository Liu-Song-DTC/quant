#!/usr/bin/env python
"""
追踪异常值根因：哪些因子产生了极端值？
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("异常值根因追踪")
print("=" * 70)

df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 1. 分析极端值的因子来源
print("\n[1] 极端值 (factor_value > 50 或 < -50) 的因子分布")

extreme_df = df[(df['factor_value'] > 50) | (df['factor_value'] < -50)]
print(f"极端值记录数: {len(extreme_df):,} ({len(extreme_df)/len(df):.2%})")

# 按因子名称统计
print("\n极端值按因子名称分布 (Top 20):")
extreme_by_factor = extreme_df.groupby('factor_name').size().sort_values(ascending=False).head(20)
for fname, count in extreme_by_factor.items():
    pct = count / len(extreme_df) * 100
    print(f"  {fname}: {count:,} ({pct:.1f}%)")

# 2. 分析Dynamic因子的极端值
print("\n[2] Dynamic因子极端值详细分析")

dynamic_df = df[df['factor_name'].astype(str).str.startswith('DYN_')]
print(f"\nDynamic因子总数: {len(dynamic_df):,}")

# 统计各范围的比例
print("\nDynamic因子factor_value范围分布:")
ranges = [
    ('< -100', dynamic_df['factor_value'] < -100),
    ('-100 ~ -50', (dynamic_df['factor_value'] >= -100) & (dynamic_df['factor_value'] < -50)),
    ('-50 ~ -10', (dynamic_df['factor_value'] >= -50) & (dynamic_df['factor_value'] < -10)),
    ('-10 ~ -5', (dynamic_df['factor_value'] >= -10) & (dynamic_df['factor_value'] < -5)),
    ('-5 ~ 0', (dynamic_df['factor_value'] >= -5) & (dynamic_df['factor_value'] < 0)),
    ('0 ~ 5', (dynamic_df['factor_value'] >= 0) & (dynamic_df['factor_value'] < 5)),
    ('5 ~ 10', (dynamic_df['factor_value'] >= 5) & (dynamic_df['factor_value'] < 10)),
    ('10 ~ 50', (dynamic_df['factor_value'] >= 10) & (dynamic_df['factor_value'] < 50)),
    ('50 ~ 100', (dynamic_df['factor_value'] >= 50) & (dynamic_df['factor_value'] < 100)),
    ('> 100', dynamic_df['factor_value'] >= 100),
]
for label, mask in ranges:
    count = mask.sum()
    pct = count / len(dynamic_df) * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")

# 3. 分析极端值对应的具体因子组合
print("\n[3] 极端值对应的行业和因子数量")

extreme_dynamic = extreme_df[extreme_df['factor_name'].astype(str).str.startswith('DYN_')]
if len(extreme_dynamic) > 0:
    # 解析因子名称
    # 格式: DYN_有色/钢_3F_F 或 DYN_电力设备_2F
    def parse_factor_name(name):
        parts = str(name).split('_')
        if len(parts) >= 3:
            industry = parts[1]
            n_factors = parts[2].replace('F', '')
            return industry, n_factors
        return None, None

    extreme_dynamic[['industry_abbr', 'n_factors']] = extreme_dynamic['factor_name'].apply(
        lambda x: pd.Series(parse_factor_name(x))
    )

    print("\n按因子数量统计极端值:")
    print(extreme_dynamic.groupby('n_factors').size().sort_values(ascending=False))

    print("\n按行业统计极端值:")
    print(extreme_dynamic.groupby('industry_abbr').size().sort_values(ascending=False).head(10))

# 4. 分析静态因子是否有极端值
print("\n[4] 静态因子 vs 动态因子极端值对比")

static_df = df[df['factor_name'].astype(str).str.startswith('IND_')]
print(f"\n静态因子总数: {len(static_df):,}")
static_extreme = static_df[(static_df['factor_value'] > 50) | (static_df['factor_value'] < -50)]
print(f"静态因子极端值: {len(static_extreme):,} ({len(static_extreme)/len(static_df):.4%}%)")

print(f"\n动态因子总数: {len(dynamic_df):,}")
dynamic_extreme = dynamic_df[(dynamic_df['factor_value'] > 50) | (dynamic_df['factor_value'] < -50)]
print(f"动态因子极端值: {len(dynamic_extreme):,} ({len(dynamic_extreme)/len(dynamic_df):.2%}%)")

# 5. 追踪具体的极端值案例
print("\n[5] 具体极端值案例分析")

# 找出最极端的案例
most_extreme = df.nlargest(10, 'factor_value')[['date', 'code', 'factor_name', 'factor_value', 'score', 'future_ret']]
print("\n最极端的正值案例 (Top 10):")
print(most_extreme.to_string(index=False))

most_negative = df.nsmallest(10, 'factor_value')[['date', 'code', 'factor_name', 'factor_value', 'score', 'future_ret']]
print("\n最极端的负值案例 (Top 10):")
print(most_negative.to_string(index=False))

# 6. 分析因子值的计算来源
print("\n[6] 可能的根因分析")
print("""
根据分析，极端值主要来自Dynamic因子。

可能的原因：
1. 某些原始因子值没有被标准化（如动量、波动率）
2. 因子加权时某些因子权重过大
3. 基本面因子处理不当（如利润增长率可能>100%）

需要进一步检查：
1. signal_engine._select_factor_dynamic() 中的因子值获取逻辑
2. 各个因子的计算公式
3. 因子加权的权重分布
""")
