#!/usr/bin/env python
"""
系统性分析框架 - 找出IC到Sharpe转化的核心瓶颈
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys

print("=" * 70)
print("系统性分析：IC → Sharpe 转化瓶颈")
print("=" * 70)

# 1. 加载数据
print("\n[1] 加载数据...")
df = pd.read_csv('rolling_validation_results/validation_results.csv')
print(f"总记录数: {len(df):,}")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
print(f"股票数: {df['code'].nunique()}")

# 2. 核心指标计算
print("\n[2] 核心指标计算...")

# 全样本IC
def calc_ic(group):
    """计算IC (Spearman相关系数)"""
    if len(group) < 10:
        return np.nan
    valid = group.dropna(subset=['factor_value', 'future_ret'])
    if len(valid) < 10:
        return np.nan
    corr, _ = stats.spearmanr(valid['factor_value'], valid['future_ret'])
    return corr

ic_by_date = df.groupby('date').apply(calc_ic)
print(f"\n全样本IC:")
print(f"  IC均值: {ic_by_date.mean():.4f} ({ic_by_date.mean()*100:.2f}%)")
print(f"  IC标准差: {ic_by_date.std():.4f}")
print(f"  IR: {ic_by_date.mean() / ic_by_date.std():.4f}")
print(f"  IC>0比例: {(ic_by_date > 0).mean():.2%}")

# 3. 按因子类型分析
print("\n[3] 因子类型分析...")

# 区分动态因子、静态因子、fallback因子
df['factor_type'] = df['factor_name'].apply(
    lambda x: 'Dynamic' if str(x).startswith('DYN_') else
              ('Static' if str(x).startswith('IND_') else 'Fallback')
)

for ftype in ['Dynamic', 'Static', 'Fallback']:
    sub = df[df['factor_type'] == ftype]
    if len(sub) < 100:
        continue
    ic = sub.groupby('date').apply(calc_ic)
    buy_signals = sub[sub['buy'] == True]
    buy_acc = (buy_signals['future_ret'] > 0).mean() if len(buy_signals) > 0 else 0

    print(f"\n{ftype}因子:")
    print(f"  记录数: {len(sub):,} ({len(sub)/len(df):.1%})")
    print(f"  IC均值: {ic.mean():.4f} ({ic.mean()*100:.2f}%)")
    print(f"  IR: {ic.mean() / ic.std() if ic.std() > 0 else 0:.4f}")
    print(f"  买入信号数: {len(buy_signals):,}")
    print(f"  买入准确率: {buy_acc:.2%}")

# 4. 行业分析
print("\n[4] 行业IC分析...")

industry_stats = []
for ind in df['industry'].dropna().unique():
    sub = df[df['industry'] == ind]
    if len(sub) < 100:
        continue
    ic = sub.groupby('date').apply(calc_ic)
    buy_signals = sub[sub['buy'] == True]
    buy_acc = (buy_signals['future_ret'] > 0).mean() if len(buy_signals) > 0 else 0
    avg_ret = buy_signals['future_ret'].mean() if len(buy_signals) > 0 else 0

    industry_stats.append({
        'industry': ind,
        'count': len(sub),
        'ic_mean': ic.mean(),
        'ir': ic.mean() / ic.std() if ic.std() > 0 else 0,
        'buy_count': len(buy_signals),
        'buy_acc': buy_acc,
        'avg_ret': avg_ret,
    })

ind_df = pd.DataFrame(industry_stats).sort_values('ic_mean', ascending=False)
print("\n行业IC排名 (Top 10):")
print(ind_df.head(10).to_string(index=False))
print("\n行业IC排名 (Bottom 10):")
print(ind_df.tail(10).to_string(index=False))

# 5. 信号质量分析
print("\n[5] 信号质量分析...")

buy_signals = df[df['buy'] == True]
print(f"\n买入信号分析:")
print(f"  总信号数: {len(buy_signals):,}")
print(f"  准确率 (ret>0): {(buy_signals['future_ret'] > 0).mean():.2%}")
print(f"  平均未来收益: {buy_signals['future_ret'].mean():.4f}")
print(f"  收益标准差: {buy_signals['future_ret'].std():.4f}")

# 按分数分组
print("\n按分数分组的收益表现:")
df['score_bin'] = pd.cut(df['score'], bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, np.inf],
                         labels=['<0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>0.8'])
for bin_label in ['<0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>0.8']:
    sub = df[df['score_bin'] == bin_label]
    if len(sub) < 100:
        continue
    avg_ret = sub['future_ret'].mean()
    pos_rate = (sub['future_ret'] > 0).mean()
    print(f"  {bin_label}: n={len(sub):,}, avg_ret={avg_ret:.4f}, pos_rate={pos_rate:.2%}")

# 6. IC衰减分析
print("\n[6] 时间维度IC稳定性...")

df['year'] = pd.to_datetime(df['date']).dt.year
for year in sorted(df['year'].unique()):
    sub = df[df['year'] == year]
    ic = sub.groupby('date').apply(calc_ic)
    print(f"  {year}: IC={ic.mean():.4f}, IR={ic.mean()/ic.std():.4f}, n={len(sub):,}")

# 7. 核心问题诊断
print("\n" + "=" * 70)
print("核心问题诊断")
print("=" * 70)

# 问题1: IC到收益的转化效率
print("\n[问题1] IC→收益转化效率")
print(f"  当前IC: {ic_by_date.mean()*100:.2f}%")
print(f"  买入信号平均收益: {buy_signals['future_ret'].mean()*100:.2f}%")
print(f"  理论期望收益 (IC * vol * sqrt(period)): 需要进一步计算")

# 问题2: 信号分布
print("\n[问题2] 信号分布问题")
print(f"  buy=True比例: {len(buy_signals)/len(df):.2%}")
print(f"  score>0比例: {(df['score'] > 0).mean():.2%}")
print(f"  score均值: {df['score'].mean():.4f}")
print(f"  factor_value均值: {df['factor_value'].mean():.4f}")

# 问题3: 行业偏差
print("\n[问题3] 行业偏差")
low_ic_industries = ind_df[ind_df['ic_mean'] < 0.03]
print(f"  低IC行业数 (IC<3%): {len(low_ic_industries)}")
print(f"  这些行业的记录占比: {low_ic_industries['count'].sum() / len(df):.2%}")

# 问题4: Fallback因子使用比例
print("\n[问题4] Fallback因子依赖")
fallback_ratio = (df['factor_type'] == 'Fallback').mean()
print(f"  Fallback因子使用比例: {fallback_ratio:.2%}")
if fallback_ratio > 0.3:
    print(f"  ⚠️ 警告: Fallback因子占比过高，说明动态/静态因子覆盖不足")

# 8. 保存详细报告
ind_df.to_csv('analysis_output/industry_analysis.csv', index=False)
print(f"\n详细行业分析已保存到 analysis_output/industry_analysis.csv")
