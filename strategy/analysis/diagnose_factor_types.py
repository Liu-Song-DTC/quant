"""
分析: 技术因子 vs 基本面因子的预测能力对比
以及factor_value的截面分布
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

validation = pd.read_csv(os.path.join(results_dir, 'validation_results.csv'))
signals = pd.read_csv(os.path.join(results_dir, 'backtest_signals.csv'))

validation['date'] = pd.to_datetime(validation['date'])
signals['date'] = pd.to_datetime(signals['date'])

print("="*60)
print("1. factor_name分布和各自IC")
print("="*60)

# 合并signals和validation
# signals有factor_name, validation有future_ret
merged = signals.merge(validation[['date', 'code', 'future_ret']], on=['date', 'code'], how='inner')
print(f"合并数据: {len(merged)} 条")

# 按factor_name计算IC
factor_stats = []
for fn, group in merged.groupby('factor_name'):
    if len(group) < 1000:
        continue
    ic, _ = spearmanr(group['factor_value'], group['future_ret'])
    factor_stats.append({
        'factor_name': fn,
        'count': len(group),
        'ic': ic,
        'fv_mean': group['factor_value'].mean(),
        'fv_std': group['factor_value'].std(),
        'accuracy': (group[group['factor_value'] > 0]['future_ret'] > 0).mean() if (group['factor_value'] > 0).any() else 0,
    })

factor_df = pd.DataFrame(factor_stats).sort_values('ic', ascending=False)
print(f"\n{'FactorName':<35} {'Count':>8} {'IC':>8} {'FV_Mean':>8} {'FV_Std':>8} {'Acc':>6}")
print("-"*80)
for _, row in factor_df.iterrows():
    marker = " ★" if row['ic'] > 0.05 else (" ⚠" if row['ic'] < 0 else "")
    print(f"{row['factor_name']:<35} {row['count']:>8} {row['ic']:>8.4f} {row['fv_mean']:>8.4f} {row['fv_std']:>8.4f} {row['accuracy']:>5.1%}{marker}")

# 区分技术因子和基本面因子
print("\n" + "="*60)
print("2. 技术因子 vs 基本面因子")
print("="*60)

tech_factors = factor_df[~factor_df['factor_name'].str.contains('fund_|F_|_F|DYN_', regex=True)]
fund_factors = factor_df[factor_df['factor_name'].str.contains('fund_', regex=True)]
industry_factors = factor_df[factor_df['factor_name'].str.startswith('IND_')]
dynamic_factors = factor_df[factor_df['factor_name'].str.startswith('DYN_')]

print(f"\n技术因子 ({len(tech_factors)}个):")
print(f"  IC均值: {tech_factors['ic'].mean():.4f}")
print(f"  IC>0: {(tech_factors['ic'] > 0).sum()}/{len(tech_factors)}")
print(f"  IC>5%: {(tech_factors['ic'] > 0.05).sum()}/{len(tech_factors)}")

if len(fund_factors) > 0:
    print(f"\n基本面因子 ({len(fund_factors)}个):")
    print(f"  IC均值: {fund_factors['ic'].mean():.4f}")
    print(f"  IC>0: {(fund_factors['ic'] > 0).sum()}/{len(fund_factors)}")
    print(f"  IC>5%: {(fund_factors['ic'] > 0.05).sum()}/{len(fund_factors)}")

if len(industry_factors) > 0:
    print(f"\n行业因子 ({len(industry_factors)}个):")
    print(f"  IC均值: {industry_factors['ic'].mean():.4f}")
    print(f"  IC>0: {(industry_factors['ic'] > 0).sum()}/{len(industry_factors)}")
    print(f"  IC>5%: {(industry_factors['ic'] > 0.05).sum()}/{len(industry_factors)}")

# 核心问题: factor_value的截面标准差太小?
print("\n" + "="*60)
print("3. factor_value截面标准差分析")
print("="*60)

daily_fv_std = []
for date, group in merged.groupby('date'):
    fv_std = group['factor_value'].std()
    daily_fv_std.append({'date': date, 'fv_std': fv_std, 'n_stocks': len(group)})

fv_std_df = pd.DataFrame(daily_fv_std)
print(f"每日截面factor_value标准差:")
print(f"  mean: {fv_std_df['fv_std'].mean():.4f}")
print(f"  median: {fv_std_df['fv_std'].median():.4f}")
print(f"  min: {fv_std_df['fv_std'].min():.4f}")
print(f"  max: {fv_std_df['fv_std'].max():.4f}")

# 如果截面std很小(比如<0.5), 说明factor_value区分度不够
# tanh压缩后值域(-1, 1), 但如果大部分在(-0.3, 0.3), 区分度很低
if fv_std_df['fv_std'].mean() < 0.5:
    print(f"\n[问题] 截面std={fv_std_df['fv_std'].mean():.4f} < 0.5")
    print(f"factor_value区分度不够!")
    print(f"tanh压缩后大量股票集中在0附近")

# 检查: factor_value在(-0.3, 0.3)范围内的比例
fv_in_range = ((merged['factor_value'] > -0.3) & (merged['factor_value'] < 0.3)).mean()
print(f"\nfactor_value在(-0.3, 0.3)范围内: {fv_in_range:.1%}")

fv_in_range2 = ((merged['factor_value'] > -0.1) & (merged['factor_value'] < 0.1)).mean()
print(f"factor_value在(-0.1, 0.1)范围内: {fv_in_range2:.1%}")

# 按行业分析factor_value的区分度
print("\n" + "="*60)
print("4. 各行业factor_value区分度")
print("="*60)

if 'industry' in merged.columns:
    for ind in sorted(merged['industry'].unique()):
        group = merged[merged['industry'] == ind]
        if len(group) < 1000:
            continue
        fv_std = group['factor_value'].std()
        ic, _ = spearmanr(group['factor_value'], group['future_ret'])
        print(f"  {ind:<20} std={fv_std:.4f}, IC={ic:.4f}")

# 建议
print("\n" + "="*60)
print("5. 优化方向建议")
print("="*60)

print(f"""
当前问题:
- factor_value截面std={fv_std_df['fv_std'].mean():.4f}, 区分度不够
- {(fv_in_range*100):.0f}%的股票factor_value在(-0.3, 0.3)
- 买入准确率50.9% (勉强>50%)
- 20日spread=1.49%, 扣除交易成本后利润微薄

根本原因:
- 基本面因子压缩后值域很窄(fund_score ~(-0.5, 0.5))
- 压缩后的基本面因子与技术因子混合，拉低了整体区分度
- 很多行业因子以基本面为主(IC高但区分度低)

优化方向:
1. 不混合: 技术因子选股，基本面因子只做过滤(排除基本面差的)
2. 提高区分度: 技术因子不做tanh压缩，保留原始值域
3. 更严格的选股: rank_pct>0.7(只选top30%)而非>0.5
4. 更激进的行业选择: 只选IC>5%的行业
5. 降低交易频率: 减少换手降低成本侵蚀
""")
