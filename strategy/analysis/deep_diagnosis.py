#!/usr/bin/env python
"""
深度诊断：为什么因子没有预测能力？
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys

print("=" * 70)
print("深度诊断：因子预测能力失效原因")
print("=" * 70)

df = pd.read_csv('rolling_validation_results/validation_results.csv')

# 1. 分析score和factor_value的关系
print("\n[1] Score vs Factor_value 关系")

# score是如何计算的？
print("\n按factor_value分组的统计:")
df['fv_bin'] = pd.qcut(df['factor_value'], q=10, duplicates='drop')
fv_stats = df.groupby('fv_bin').agg({
    'future_ret': ['mean', 'std', 'count'],
    'score': 'mean'
}).round(4)
fv_stats.columns = ['avg_ret', 'ret_std', 'count', 'avg_score']
fv_stats['pos_rate'] = df.groupby('fv_bin').apply(lambda x: (x['future_ret'] > 0).mean()).values
print(fv_stats.to_string())

# 2. 分析动态因子的问题
print("\n[2] 动态因子详细分析")

dynamic_df = df[df['factor_name'].astype(str).str.startswith('DYN_')]
print(f"动态因子数量: {dynamic_df['factor_name'].nunique()}")

# 按因子名称分析
print("\n最常用的动态因子 (Top 15):")
top_factors = dynamic_df['factor_name'].value_counts().head(15)
for fname, count in top_factors.items():
    sub = dynamic_df[dynamic_df['factor_name'] == fname]
    ic = sub.groupby('date').apply(
        lambda g: stats.spearmanr(g['factor_value'].dropna(), g['future_ret'].dropna())[0]
        if len(g.dropna(subset=['factor_value', 'future_ret'])) > 10 else np.nan
    ).mean()
    buy_signals = sub[sub['buy'] == True]
    buy_acc = (buy_signals['future_ret'] > 0).mean() if len(buy_signals) > 0 else 0
    print(f"  {fname}: n={count:,}, IC={ic:.4f}, buy_acc={buy_acc:.2%}")

# 3. 分析买入阈值的合理性问题
print("\n[3] 买入阈值分析 (buy_threshold=0.18)")

print("\n当前买入条件: score > 0.18 且 score < 5.0")
print(f"满足条件的记录比例: {(df['score'] > 0.18).mean():.2%}")

# 不同阈值下的准确率
print("\n不同score阈值下的买入准确率:")
for threshold in [0.0, 0.1, 0.15, 0.18, 0.2, 0.3, 0.4, 0.5]:
    signals = df[df['score'] > threshold]
    acc = (signals['future_ret'] > 0).mean() if len(signals) > 0 else 0
    avg_ret = signals['future_ret'].mean() if len(signals) > 0 else 0
    print(f"  score > {threshold}: n={len(signals):,}, acc={acc:.2%}, avg_ret={avg_ret:.4f}")

# 4. 分析分数计算过程
print("\n[4] 分数分布异常分析")

print(f"\nscore分布:")
print(f"  均值: {df['score'].mean():.4f}")
print(f"  中位数: {df['score'].median():.4f}")
print(f"  标准差: {df['score'].std():.4f}")
print(f"  最小值: {df['score'].min():.4f}")
print(f"  最大值: {df['score'].max():.4f}")

# 分数分布百分位
print(f"\nscore百分位分布:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}%: {df['score'].quantile(p/100):.4f}")

# 5. 分析factor_value的预测能力
print("\n[5] Factor_value 预测能力分析")

def calc_ic(g):
    valid = g.dropna(subset=['factor_value', 'future_ret'])
    if len(valid) < 10:
        return np.nan
    return stats.spearmanr(valid['factor_value'], valid['future_ret'])[0]

# 全局IC
global_ic = df.groupby('date').apply(calc_ic)
print(f"\n全局IC (基于factor_value): {global_ic.mean():.4f}")

# 分年度IC
print("\n年度IC趋势:")
df['year'] = pd.to_datetime(df['date']).dt.year
for year in sorted(df['year'].unique()):
    sub = df[df['year'] == year]
    ic = sub.groupby('date').apply(calc_ic).mean()
    print(f"  {year}: IC={ic:.4f}")

# 6. 分析股票选择后的实际收益
print("\n[6] 实际买入后的收益分析")

buy_signals = df[df['buy'] == True]
print(f"\n买入信号总数: {len(buy_signals):,}")
print(f"平均未来收益: {buy_signals['future_ret'].mean()*100:.2f}%")
print(f"收益标准差: {buy_signals['future_ret'].std()*100:.2f}%")
print(f"夏普比率 (假设持有20天): {buy_signals['future_ret'].mean() / buy_signals['future_ret'].std() * np.sqrt(252/20):.4f}")

# 正收益 vs 负收益分布
pos_ret = buy_signals[buy_signals['future_ret'] > 0]
neg_ret = buy_signals[buy_signals['future_ret'] <= 0]
print(f"\n正收益信号: {len(pos_ret):,} ({len(pos_ret)/len(buy_signals):.2%})")
print(f"  平均收益: {pos_ret['future_ret'].mean()*100:.2f}%")
print(f"负收益信号: {len(neg_ret):,} ({len(neg_ret)/len(buy_signals):.2%})")
print(f"  平均亏损: {neg_ret['future_ret'].mean()*100:.2f}%")

# 7. 关键发现总结
print("\n" + "=" * 70)
print("关键发现")
print("=" * 70)

print("""
[发现1] Score分布异常偏高
  - score均值0.74，中位数0.79，说明几乎所有股票都有正分数
  - 这是信号生成逻辑的系统性偏差

[发现2] Factor_value IC只有2.02%
  - 远低于预期的5%+
  - 说明因子组合方法或因子本身质量有问题

[发现3] 买入阈值形同虚设
  - buy_threshold=0.18，但score>0.18的比例高达63%
  - 阈值没有起到筛选作用

[发现4] 分数与收益关系弱
  - score从负值到正值，准确率只提高4%
  - 说明score计算公式有问题

[根本原因]
  需要检查signal_engine中score的计算逻辑
  - score是否正确反映因子预测能力？
  - 是否存在系统性偏差？
""")
