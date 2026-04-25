"""
深度分析: rank_pct选股的实际预测能力
对比不同rank_pct分组的未来收益
"""

import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

validation = pd.read_csv(os.path.join(results_dir, 'validation_results.csv'))
signals = pd.read_csv(os.path.join(results_dir, 'backtest_signals.csv'))

print("="*60)
print("1. 按factor_value十分位的未来收益")
print("="*60)

validation['date'] = pd.to_datetime(validation['date'])

# 按日期分组，计算截面rank
daily_groups = validation.groupby('date')

results = []
for date, group in daily_groups:
    if len(group) < 50:
        continue
    group = group.copy()
    group['rank_pct'] = group['factor_value'].rank(pct=True)
    group['decile'] = pd.qcut(group['factor_value'], 10, labels=False, duplicates='drop')
    results.append(group)

all_data = pd.concat(results, ignore_index=True)
print(f"分析数据: {len(all_data)} 条")

# 按decile统计未来收益
decile_stats = all_data.groupby('decile').agg(
    count=('future_ret', 'count'),
    future_ret_mean=('future_ret', 'mean'),
    future_ret_median=('future_ret', 'median'),
    positive_rate=('future_ret', lambda x: (x > 0).mean()),
).reset_index()

print(f"\nfactor_value十分位 → 20日未来收益:")
print(f"{'Decile':>7} {'Count':>8} {'MeanRet':>10} {'MedRet':>10} {'PosRate':>10}")
print("-"*50)
for _, row in decile_stats.iterrows():
    d = int(row['decile'])
    print(f"D{d:<6} {row['count']:>8.0f} {row['future_ret_mean']:>10.4f} {row['future_ret_median']:>10.4f} {row['positive_rate']:>10.1%}")

# 顶部decile vs 底部decile
top_ret = decile_stats.iloc[-1]['future_ret_mean']
bot_ret = decile_stats.iloc[0]['future_ret_mean']
print(f"\nTop D9 vs Bottom D0: {top_ret:.4f} vs {bot_ret:.4f}, spread={top_ret-bot_ret:.4f}")

# 按rank_pct分组
print("\n" + "="*60)
print("2. 按rank_pct分组的未来收益")
print("="*60)

rank_groups = [
    ('0-0.2', (all_data['rank_pct'] >= 0) & (all_data['rank_pct'] < 0.2)),
    ('0.2-0.4', (all_data['rank_pct'] >= 0.2) & (all_data['rank_pct'] < 0.4)),
    ('0.4-0.5', (all_data['rank_pct'] >= 0.4) & (all_data['rank_pct'] < 0.5)),
    ('0.5-0.6', (all_data['rank_pct'] >= 0.5) & (all_data['rank_pct'] < 0.6)),
    ('0.6-0.8', (all_data['rank_pct'] >= 0.6) & (all_data['rank_pct'] < 0.8)),
    ('0.8-1.0', (all_data['rank_pct'] >= 0.8) & (all_data['rank_pct'] <= 1.0)),
]

print(f"{'RankPct':>10} {'Count':>8} {'MeanRet':>10} {'PosRate':>10}")
print("-"*45)
for name, mask in rank_groups:
    subset = all_data[mask]
    if len(subset) == 0:
        continue
    print(f"{name:>10} {len(subset):>8} {subset['future_ret'].mean():>10.4f} {(subset['future_ret'] > 0).mean():>10.1%}")

# 按行业分析IC
print("\n" + "="*60)
print("3. 按行业的IC和准确率")
print("="*60)

if 'industry' in all_data.columns:
    industry_stats = []
    for ind, group in all_data.groupby('industry'):
        if len(group) < 1000:
            continue
        # Spearman IC
        from scipy.stats import spearmanr
        ic, _ = spearmanr(group['factor_value'], group['future_ret'])
        # 顶部50%的准确率
        top50 = group[group['rank_pct'] > 0.5]
        accuracy = (top50['future_ret'] > 0).mean() if len(top50) > 0 else 0
        # 顶部20%的准确率
        top20 = group[group['rank_pct'] > 0.8]
        accuracy_top20 = (top20['future_ret'] > 0).mean() if len(top20) > 0 else 0

        industry_stats.append({
            'industry': ind,
            'count': len(group),
            'ic': ic,
            'top50_accuracy': accuracy,
            'top20_accuracy': accuracy_top20,
            'top50_ret': top50['future_ret'].mean() if len(top50) > 0 else 0,
        })

    ind_df = pd.DataFrame(industry_stats).sort_values('ic', ascending=False)
    print(f"{'Industry':<20} {'IC':>8} {'Top50%Acc':>10} {'Top20%Acc':>10} {'Top50%Ret':>10}")
    print("-"*65)
    for _, row in ind_df.iterrows():
        marker = " ★" if row['ic'] > 0.05 else (" ⚠" if row['ic'] < 0 else "")
        print(f"{row['industry']:<20} {row['ic']:>8.4f} {row['top50_accuracy']:>10.1%} {row['top20_accuracy']:>10.1%} {row['top50_ret']:>10.4f}{marker}")

# 按市场状态分析
print("\n" + "="*60)
print("4. 按市场状态的IC和收益")
print("="*60)

if 'regime' in all_data.columns:
    for regime, group in all_data.groupby('regime'):
        from scipy.stats import spearmanr
        ic, _ = spearmanr(group['factor_value'], group['future_ret'])
        top50 = group[group['rank_pct'] > 0.5]
        accuracy = (top50['future_ret'] > 0).mean() if len(top50) > 0 else 0
        regime_name = {1: 'Bull', 0: 'Neutral', -1: 'Bear'}.get(regime, f'Regime{regime}')
        print(f"{regime_name:<10} count={len(group):>8}, IC={ic:.4f}, Top50%Acc={accuracy:.1%}, Top50%Ret={top50['future_ret'].mean():.4f}")

# 核心结论
print("\n" + "="*60)
print("5. 核心结论")
print("="*60)

# 如果Top50%准确率<50%, 说明因子排序方向可能有问题
top50_all = all_data[all_data['rank_pct'] > 0.5]
top50_acc = (top50_all['future_ret'] > 0).mean()
top50_ret = top50_all['future_ret'].mean()

bot50_all = all_data[all_data['rank_pct'] <= 0.5]
bot50_acc = (bot50_all['future_ret'] > 0).mean()
bot50_ret = bot50_all['future_ret'].mean()

print(f"Top50%: 准确率={top50_acc:.1%}, 平均收益={top50_ret:.4f}")
print(f"Bot50%: 准确率={bot50_acc:.1%}, 平均收益={bot50_ret:.4f}")
print(f"Spread: {(top50_ret-bot50_ret)*100:.2f}%")

if top50_acc < 0.50:
    print(f"\n[严重问题] Top50%准确率={top50_acc:.1%} < 50%!")
    print(f"因子排序方向与实际收益不一致!")
    print(f"可能原因:")
    print(f"  1. 因子值压缩(tanh)后区分度不够")
    print(f"  2. 基本面因子(压缩后)覆盖了技术因子的信号")
    print(f"  3. 熊市中因子反转(正IC变负IC)")
else:
    print(f"\nTop50%准确率={top50_acc:.1%} > 50%, 因子有效")
    print(f"但Spread={((top50_ret-bot50_ret)*100):.2f}%可能不够大")

# 关键: 选股策略能否从spread中获利?
# 如果top10只的rank_pct>0.9, 她们的收益如何?
top10pct = all_data[all_data['rank_pct'] > 0.9]
top10_acc = (top10pct['future_ret'] > 0).mean()
top10_ret = top10pct['future_ret'].mean()
print(f"\nTop10%: 准确率={top10_acc:.1%}, 平均收益={top10_ret:.4f}")

# 最关键的: top10(等权) vs 全市场等权
print(f"\n如果策略只选rank_pct>0.9的top10%, 年化收益估算:")
# 20天forward_period, 假设每年250交易日, 约12.5个周期
periods_per_year = 250 / 20
annual_ret_top10 = (1 + top10_ret) ** periods_per_year - 1
annual_ret_bot50 = (1 + bot50_ret) ** periods_per_year - 1
print(f"  Top10%: 20日收益{top10_ret:.4f} → 年化{annual_ret_top10:.1%}")
print(f"  Bot50%: 20日收益{bot50_ret:.4f} → 年化{annual_ret_bot50:.1%}")
