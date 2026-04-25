"""
深入诊断: 为什么选股数只有6只，权重只有6.9%?
"""

import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

selections = pd.read_csv(os.path.join(results_dir, 'portfolio_selections.csv'))
signals = pd.read_csv(os.path.join(results_dir, 'backtest_signals.csv'))

print("="*60)
print("1. 选股详情分析")
print("="*60)

selections['date'] = pd.to_datetime(selections['date'])

# 每日选股数分布
daily = selections.groupby('date').size()
print(f"每日选股数分布:")
print(f"  mean: {daily.mean():.1f}")
print(f"  median: {daily.median():.0f}")
print(f"  min: {daily.min()}")
print(f"  max: {daily.max()}")
print(f"  分布:")
for n in sorted(daily.unique()):
    cnt = (daily == n).sum()
    print(f"    {n}只: {cnt}天 ({100*cnt/len(daily):.1f}%)")

# 每日总权重
daily_weight = selections.groupby('date')['weight'].sum()
print(f"\n每日总权重:")
print(f"  mean: {daily_weight.mean():.4f}")
print(f"  median: {daily_weight.median():.4f}")
print(f"  min: {daily_weight.min():.4f}")
print(f"  max: {daily_weight.max():.4f}")
print(f"  <0.5的天数: {(daily_weight < 0.5).sum()}")
print(f"  <0.8的天数: {(daily_weight < 0.8).sum()}")

# 检查单个权重
print(f"\n单个权重分布:")
print(f"  mean: {selections['weight'].mean():.4f}")
print(f"  median: {selections['weight'].median():.4f}")
print(f"  max: {selections['weight'].max():.4f}")
print(f"  =0.10的数量: {(selections['weight'] == 0.10).sum()}")
print(f"  >0.10的数量: {(selections['weight'] > 0.10).sum()}")

print("\n" + "="*60)
print("2. 权重为什么这么低？")
print("="*60)

# portfolio.py中的权重计算逻辑:
# rank_weight = c['rank_pct']  (0.5-1.0)
# risk_vol = max(0.01, min(1.0, c['risk_vol']))
# vol_factor = min(1.0 / risk_vol, 2.0)
# extreme_factor = 0.7 if risk_extreme else 1.0
# c['position'] = rank_weight * vol_factor * extreme_factor
# 然后归一化: raw_weights[c] = (c['position'] / total_position) * max_gross_exposure
# 然后限制: raw_weights[code] = min(raw_weights[code], max_single_weight=0.10)
# 然后重新归一化

# 检查score对权重的影响
print(f"score分布 (选股):")
print(f"  mean: {selections['score'].mean():.4f}")
print(f"  min: {selections['score'].min():.4f}")
print(f"  max: {selections['score'].max():.4f}")
print(f"  <0: {(selections['score'] < 0).sum()}/{len(selections)}")

# score<0的选股仍然存在!?
neg_score_sel = selections[selections['score'] < 0]
if len(neg_score_sel) > 0:
    print(f"\n[问题] score<0的选股: {len(neg_score_sel)} 条!")
    print(f"  权重: mean={neg_score_sel['weight'].mean():.4f}")
    print(f"  这些股票不应该被选中!")

print("\n" + "="*60)
print("3. rank_pct在选股中的分布")
print("="*60)

if 'rank_pct' in selections.columns:
    rp = selections['rank_pct'].dropna()
    print(f"rank_pct 分布:")
    print(f"  mean: {rp.mean():.4f}")
    print(f"  median: {rp.median():.4f}")
    print(f"  min: {rp.min():.4f}")
    print(f"  max: {rp.max():.4f}")
    print(f"  <0.5: {(rp < 0.5).sum()}/{len(rp)}")
    print(f"  <0.7: {(rp < 0.7).sum()}/{len(rp)}")
    print(f"  >0.9: {(rp > 0.9).sum()}/{len(rp)}")
else:
    print("rank_pct 列不存在")

print("\n" + "="*60)
print("4. 关键问题: 权重归一化后 < 0.10 per stock")
print("="*60)

# 理论上: 10只股票, max_gross_exposure=1.0
# 等权: 每只 0.10
# 但max_single_weight=0.10, 加上行业折扣和风险调整, 实际<0.10
# 6只股票: 等权0.167, 但被0.10截断 -> 重新归一化后只有0.10

# 实际计算: 如果只有6只, max_single_weight=0.10
# 6 * 0.10 = 0.60 总权重 -> 只用了60%资金!
print(f"""
权重问题分析:
- max_single_weight = 0.10 (单只上限)
- 6只股票 × 0.10 = 0.60 (只用了60%资金)
- 10只股票 × 0.10 = 1.00 (100%资金)
- 选股只有6只 → 40%资金闲置!
- 10万资金只用了6万 → 收益大打折扣

为什么只选6只?
- 行业均衡: industry_cap=2 (每行业最多2只)
- 有色/钢铁/煤炭/建材 186/897=20.7% → 每期2只
- 电子 123/897=13.7% → 每期2只
- 但4个行业就占了8个位子（每行业2只）
- 加上其他行业的限制, 总共只能选6-10只
- 实际6只意味着很多行业选不满2只

解决方案:
1. 提高 industry_cap 到 3
2. 或增大 max_position 到 15
3. 或去掉行业均衡限制, 纯按rank_pct选
""")

print("\n" + "="*60)
print("5. 离线模拟选股 vs 实际选股")
print("="*60)

# 模拟: 对每个rebalance日, 按factor_value排序top10(无行业限制)
signals['date'] = pd.to_datetime(signals['date'])
sel_dates = set(selections['date'].unique())

# 只看rebalance日
rebalance_signals = signals[signals['date'].isin(sel_dates)]
print(f"再平衡日: {len(sel_dates)} 天")

# 模拟纯排名选股
sim_pure = []
for date, group in rebalance_signals.groupby('date'):
    valid = group.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue
    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)
    qualified = valid[valid['rank_pct'] > 0.5]
    if len(qualified) == 0:
        qualified = valid[valid['rank_pct'] > 0.3]
    top10 = qualified.nlargest(10, 'factor_value')
    for _, row in top10.iterrows():
        sim_pure.append({
            'date': date,
            'code': row['code'],
            'factor_value': row['factor_value'],
            'rank_pct': row['rank_pct'],
        })

sim_pure_df = pd.DataFrame(sim_pure)
print(f"\n纯排名选股(无行业限制):")
print(f"  每日选股: {sim_pure_df.groupby('date').size().mean():.1f}")
print(f"  rank_pct mean: {sim_pure_df['rank_pct'].mean():.4f}")
print(f"  factor_value mean: {sim_pure_df['factor_value'].mean():.4f}")

# 对比
actual_fv_mean = selections['score'].mean()  # score ≈ factor_value related
print(f"\n实际选股:")
print(f"  score mean: {actual_fv_mean:.4f}")

# 再看看行业均衡的影响
sim_with_industry = []
for date, group in rebalance_signals.groupby('date'):
    valid = group.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue
    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)
    qualified = valid[valid['rank_pct'] > 0.5]
    if len(qualified) == 0:
        qualified = valid[valid['rank_pct'] > 0.3]
    qualified = qualified.sort_values('factor_value', ascending=False)

    # 行业均衡
    industry_count = {}
    selected = []
    for _, row in qualified.iterrows():
        ind = row.get('industry', 'default')
        if pd.isna(ind):
            ind = 'default'
        if industry_count.get(ind, 0) >= 2:
            continue
        selected.append(row)
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(selected) >= 10:
            break

    for row in selected:
        sim_with_industry.append({
            'date': date,
            'code': row['code'],
            'factor_value': row['factor_value'],
            'rank_pct': row['rank_pct'],
            'industry': row.get('industry', 'default'),
        })

sim_ind_df = pd.DataFrame(sim_with_industry)
print(f"\n行业均衡选股(industry_cap=2):")
print(f"  每日选股: {sim_ind_df.groupby('date').size().mean():.1f}")
print(f"  rank_pct mean: {sim_ind_df['rank_pct'].mean():.4f}")
print(f"  factor_value mean: {sim_ind_df['factor_value'].mean():.4f}")
print(f"  行业数: {sim_ind_df['industry'].nunique()}")

# 看行业均衡选股时，哪些行业占满了配额
print(f"\n行业均衡选股的行业分布:")
ind_counts = sim_ind_df['industry'].value_counts()
for ind, cnt in ind_counts.head(15).items():
    print(f"  {ind}: {cnt}")

print("\n" + "="*60)
print("6. 资金利用率")
print("="*60)

# 每日选股数 vs 资金利用率
daily_stats = selections.groupby('date').agg(
    n_selected=('code', 'count'),
    total_weight=('weight', 'sum'),
).reset_index()

print(f"选股数 vs 总权重:")
for n in sorted(daily_stats['n_selected'].unique()):
    subset = daily_stats[daily_stats['n_selected'] == n]
    print(f"  {n}只股票: {len(subset)}天, 平均总权重={subset['total_weight'].mean():.4f}")

# 资金利用率 = 总权重 (理想=1.0)
utilization = daily_stats['total_weight'].mean()
print(f"\n平均资金利用率: {utilization:.1%}")
print(f"资金浪费: {(1-utilization)*100:.1f}%")
print(f"10万资金中，实际使用: {100000*utilization:.0f}元")
print(f"闲置资金: {100000*(1-utilization):.0f}元")
