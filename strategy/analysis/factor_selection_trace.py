#!/usr/bin/env python
"""
追踪动态因子选择：哪些因子被选中并导致极端值？
"""
import pandas as pd
import numpy as np
import sys
import os

print("=" * 70)
print("追踪动态因子选择的因子")
print("=" * 70)

# 读取验证数据
df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 1. 分析哪些行业-因子数组合产生了极端值
print("\n[1] 极端值按行业-因子数分布")

# 解析因子名称
def parse_factor_name(name):
    """DYN_有色/钢_3F_F -> ('有色/钢', '3')"""
    parts = str(name).split('_')
    if len(parts) >= 3:
        industry = parts[1]
        n_factors = parts[2].replace('F', '')
        return industry, n_factors
    return None, None

# 提取行业和因子数
df['industry_abbr'] = df['factor_name'].apply(lambda x: parse_factor_name(x)[0] if str(x).startswith('DYN_') else None)
df['n_factors'] = df['factor_name'].apply(lambda x: parse_factor_name(x)[1] if str(x).startswith('DYN_') else None)

# 统计极端值
extreme_df = df[(df['factor_value'] > 50) | (df['factor_value'] < -50)]
print(f"\n极端值总数: {len(extreme_df):,}")

# 按因子数统计
print("\n按因子数统计极端值:")
print(extreme_df.groupby('n_factors').size())

# 2. 检查因子选择器选择了哪些因子
print("\n[2] 查看因子选择配置")

# 读取配置文件中的backtest_factors
config_factors = [
    'trend_mom_v41', 'trend_mom_v24', 'trend_mom_v46',
    'mom_x_lowvol_20_20', 'mom_x_lowvol_20_10', 'mom_x_lowvol_10_20', 'mom_x_lowvol_10_10',
    'mom_diff_5_20', 'mom_diff_10_20',
    'rsi_factor', 'volatility', 'volume_ratio', 'bb_width_20',
    'fund_score', 'fund_profit_growth', 'fund_revenue_growth', 'fund_roe', 'fund_cf_to_profit', 'fund_gross_margin',
    'V41_RSI_915', 'tech_fund_combo',
    'rsi_vol_combo', 'bb_rsi_combo',
    'ret_vol_ratio_10', 'ret_vol_ratio_20',
    'momentum_reversal', 'momentum_acceleration'
]

# 分析哪些因子可能产生极端值
print("\n可能产生极端值的因子分析:")
print("""
高风险因子（可能产生极端值）:
1. ret_vol_ratio_10 = mom_10 / volatility_10
   - 当波动率接近0时，值会非常大
   - 典型极端值: 20~100+

2. ret_vol_ratio_20 = mom_20 / volatility_20
   - 同上

3. trend_mom_v41 = m20 * 2.1
   - 当m20>50%时，因子值>100%
   - 涨停后的极端动量

4. volume_ratio
   - 极端放量时可能>10
   - 但通常不会超过100

安全因子（值范围有限）:
1. rsi_factor = (rsi - 50) / 100
   - 范围: [-0.5, 0.5]

2. volatility = -v20
   - 范围: [-1, 0] 左右

3. mom_x_lowvol = m20 * (-v20)
   - 范围: [-0.3, 0.3] 左右
""")

# 3. 验证假设：检查IC加权时哪些因子权重高
print("\n[3] 检查因子权重问题")
print("""
问题分析：
1. IC加权时，权重来自历史IC值
2. 如果某个因子IC高但值极端，加权后仍会产生极端值
3. 基本面因子用tanh压缩了，技术因子没有

核心问题：
- factor_preparer.py 预计算时，技术因子使用原始值
- signal_engine._select_factor_dynamic 获取技术因子时，也没有标准化
- 只有基本面因子在获取时做了tanh压缩

解决方案：
在获取技术因子值时，添加tanh压缩或标准化
""")

# 4. 统计各行业的极端值比例
print("\n[4] 各行业极端值比例")

for industry in df['industry_abbr'].dropna().unique():
    sub = df[df['industry_abbr'] == industry]
    extreme_count = ((sub['factor_value'] > 50) | (sub['factor_value'] < -50)).sum()
    extreme_pct = extreme_count / len(sub) * 100 if len(sub) > 0 else 0
    if extreme_pct > 5:  # 只显示极端值>5%的行业
        print(f"  {industry}: {extreme_pct:.1f}% ({extreme_count:,}/{len(sub):,})")
