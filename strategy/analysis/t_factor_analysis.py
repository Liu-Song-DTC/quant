#!/usr/bin/env python
"""
_T因子分析
目标：理解为什么_T因子IC为负，以及如何处理
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
print("_T因子分析")
print("=" * 80)

# 分离_T和_F因子
t_factors = df[df['factor_name'].str.endswith('_T', na=False)]
f_factors = df[df['factor_name'].str.endswith('_F', na=False)]

print(f"\n数据量:")
print(f"  _T因子: {len(t_factors):,} ({len(t_factors)/len(df)*100:.1f}%)")
print(f"  _F因子: {len(f_factors):,} ({len(f_factors)/len(df)*100:.1f}%)")

# IC对比
print(f"\nIC对比:")
ic_t, _ = stats.spearmanr(t_factors['factor_value'], t_factors['future_ret'])
ic_f, _ = stats.spearmanr(f_factors['factor_value'], f_factors['future_ret'])
print(f"  _T因子IC: {ic_t*100:.2f}%")
print(f"  _F因子IC: {ic_f*100:.2f}%")

# 买入信号准确率
buy_t = t_factors[t_factors['buy'] == True]
buy_f = f_factors[f_factors['buy'] == True]

print(f"\n买入信号数量:")
print(f"  _T: {len(buy_t):,} ({len(buy_t)/len(t_factors)*100:.1f}%)")
print(f"  _F: {len(buy_f):,} ({len(buy_f)/len(f_factors)*100:.1f}%)")

print(f"\n买入准确率:")
acc_t = (buy_t['future_ret'] > 0).mean()
acc_f = (buy_f['future_ret'] > 0).mean()
print(f"  _T: {acc_t*100:.2f}%")
print(f"  _F: {acc_f*100:.2f}%")

print(f"\n买入平均收益:")
ret_t = buy_t['future_ret'].mean()
ret_f = buy_f['future_ret'].mean()
print(f"  _T: {ret_t*100:.3f}%")
print(f"  _F: {ret_f*100:.3f}%")

# 分析_T因子为什么表现差
print("\n" + "=" * 80)
print("_T因子为什么表现差？")
print("=" * 80)

# 查看_T因子的具体名称
print("\n_T因子类型:")
t_factor_names = t_factors['factor_name'].unique()
print(f"  共 {len(t_factor_names)} 种_T因子")
for name in list(t_factor_names)[:10]:
    sub = t_factors[t_factors['factor_name'] == name]
    ic, _ = stats.spearmanr(sub['factor_value'], sub['future_ret'])
    print(f"    {name}: IC={ic*100:.2f}%, n={len(sub):,}")

# 分析_T因子的factor_value分布
print("\n_T因子factor_value分布:")
print(f"  最小值: {t_factors['factor_value'].min():.3f}")
print(f"  最大值: {t_factors['factor_value'].max():.3f}")
print(f"  均值: {t_factors['factor_value'].mean():.3f}")
print(f"  标准差: {t_factors['factor_value'].std():.3f}")

# 建议
print("\n" + "=" * 80)
print("建议:")
print("=" * 80)
print("""
_T因子是纯技术因子（没有基本面数据），IC为负(-0.59%)。
建议：
1. 完全禁用_T因子
2. 或者在买入条件中过滤掉_T因子的信号
3. 这应该能进一步提升Sharpe
""")