#!/usr/bin/env python
"""
分析技术因子的极端值来源
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("技术因子极端值根因分析")
print("=" * 70)

# 读取验证数据
df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 分析极端值股票
print("\n[1] 极端值股票分析")

# 688223 是factor_value=100的股票
code = '688223'
stock_df = df[df['code'] == code].sort_values('date')

print(f"\n股票 {code}:")
print(f"  记录数: {len(stock_df)}")
print(f"  factor_value范围: {stock_df['factor_value'].min():.2f} ~ {stock_df['factor_value'].max():.2f}")

# 找到极端值发生的日期
extreme_dates = stock_df[stock_df['factor_value'] == 100]['date'].unique()
print(f"\n  factor_value=100的日期数: {len(extreme_dates)}")
if len(extreme_dates) > 0:
    print(f"  示例日期: {extreme_dates[:5]}")

# 分析这些日期的行业
extreme_df = stock_df[stock_df['factor_value'] == 100]
print(f"\n  极端值时的因子名称: {extreme_df['factor_name'].value_counts().head(5).to_dict()}")
print(f"  极端值时的行业: {extreme_df['industry'].value_counts().head(5).to_dict()}")

# 分析技术因子的计算公式
print("\n[2] 技术因子计算公式分析")
print("""
根据 factor_calculator.py:

1. trend_mom_v41 = m20 * 2.1
   - 如果m20=50%, 则因子值=105%
   - 极端情况：涨停后动量极大

2. mom_x_lowvol_20_20 = m20 * (-v20)
   - 如果m20=30%, v20=10%, 则因子值=-3%
   - 相对温和

3. ret_vol_ratio_10 = mom_10 / volatility_10
   - 如果mom_10=10%, volatility_10=0.5%
   - 则因子值=20 (极端！)
   - **这是最可能的问题来源**

4. momentum_acceleration = mom_10 - mom_20
   - 通常范围在±30%以内

5. 成交量因子 volume_ratio
   - 可能出现极端放量，ratio>10

**核心问题**:
- ret_vol_ratio_10 等比率因子，当分母接近0时会产生极大值
- 这些极端值没有被标准化或压缩
""")

# 分析score的计算
print("\n[3] Score计算逻辑分析")
print("""
根据 signal_engine.py 第645-667行:

base_score = np.clip(factor_value, -10, 10)
mom_5_norm = np.clip(mom_5 / 0.05, -2, 2)
mom_20_norm = np.clip(mom_20 / 0.10, -2, 2)
score = base_score * 0.7 + mom_5_norm * 0.2 + mom_20_norm * 0.1

问题:
1. factor_value在clip之前已经是±100
2. clip(-10, 10)把100变成10，但仍然太大
3. score = 10 * 0.7 + 动量贡献 ≈ 7
4. 这就是为什么score均值=0.74的原因

但数据中factor_value=100时，score=7.6
这说明clip可能在更后面执行，或者有其他逻辑
""")

# 验证clip的时机
print("\n[4] 验证数据中的极端值")
print(f"\n数据中factor_value=100的记录数: {(df['factor_value'] == 100).sum():,}")
print(f"数据中factor_value=-100的记录数: {(df['factor_value'] == -100).sum():,}")
print(f"数据中factor_value>10的记录数: {(df['factor_value'] > 10).sum():,}")
print(f"数据中factor_value<-10的记录数: {(df['factor_value'] < -10).sum():,}")

# 分析factor_value>10的情况
high_fv = df[df['factor_value'] > 10]
print(f"\nfactor_value>10时的因子分布:")
print(high_fv['factor_name'].value_counts().head(10))
