"""
量化回测vs理想模拟的差距来源
"""
import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

selections = pd.read_csv(os.path.join(results_dir, 'portfolio_selections.csv'))
validation = pd.read_csv(os.path.join(results_dir, 'validation_results.csv'))

selections['date'] = pd.to_datetime(selections['date'])
validation['date'] = pd.to_datetime(validation['date'])

print("=" * 60)
print("1. 实际选股的离线收益验证")
print("=" * 60)

# 把selections和validation合并，看实际选中的股票的future_ret
merged = selections.merge(
    validation[['date', 'code', 'future_ret']],
    on=['date', 'code'],
    how='left'
)

# 计算每日等权组合的future_ret
daily_port = merged.groupby('date').agg(
    n=('code', 'count'),
    ret=('future_ret', 'mean'),
    total_weight=('weight', 'sum'),
).reset_index()

daily_port = daily_port.dropna(subset=['ret'])
print(f"有future_ret的再平衡日: {len(daily_port)}")
print(f"平均20日组合收益: {daily_port['ret'].mean():.4f}")
print(f"收益标准差: {daily_port['ret'].std():.4f}")

# 年化Sharpe (简化: 每期20天, 每年12.5期)
sharpe = daily_port['ret'].mean() / daily_port['ret'].std() * np.sqrt(12.5)
print(f"理论年化Sharpe(基于选股): {sharpe:.4f}")

# 交易成本: 每次换仓50% * 0.3% (佣金+滑点)
sel_dates = sorted(selections['date'].unique())
turnover_costs = []
for i in range(1, len(sel_dates)):
    prev = set(selections[selections['date'] == sel_dates[i-1]]['code'])
    curr = set(selections[selections['date'] == sel_dates[i]]['code'])
    total = len(prev) + len(curr)
    if total == 0:
        continue
    turnover = len(prev.symmetric_difference(curr)) / total
    cost = turnover * 0.003  # 0.3% round-trip
    turnover_costs.append(cost)

avg_cost = np.mean(turnover_costs) if turnover_costs else 0

# 简单计算换手率
turnover_list = []
for i in range(1, len(sel_dates)):
    prev_codes = set(selections[selections['date'] == sel_dates[i-1]]['code'])
    curr_codes = set(selections[selections['date'] == sel_dates[i]])
    if len(prev_codes) == 0 and len(curr_codes) == 0:
        continue
    total = len(prev_codes) + len(curr_codes)
    changed = len(prev_codes.symmetric_difference(curr_codes))
    turnover_list.append(changed / total if total > 0 else 0)

avg_turnover = np.mean(turnover_list)
cost_per_rebal = avg_turnover * 0.003
print(f"平均换手率: {avg_turnover:.1%}")
print(f"每次再平衡交易成本: {cost_per_rebal:.4f}")

# 扣除成本后的Sharpe
net_ret = daily_port['ret'].mean() - cost_per_rebal
net_sharpe = net_ret / daily_port['ret'].std() * np.sqrt(12.5)
print(f"\n扣除成本后20日ret: {net_ret:.4f}")
print(f"扣除成本后年化Sharpe: {net_sharpe:.4f}")

# 对比理想模拟
print(f"\n=== 差距分解 ===")
print(f"理想模拟Sharpe(全量数据rank_pct选股): 0.773")
print(f"实际选股Sharpe(selections+validation):  {sharpe:.3f}")
print(f"扣除成本后Sharpe:                       {net_sharpe:.3f}")
print(f"实际回测Sharpe:                         0.528")

# 差距1: 理想vs实际选股
gap1 = 0.773 - sharpe
print(f"\n差距1(理想→实际选股): {gap1:.3f}")
print(f"  原因: 理想用全量validation数据, 实际用backtest_signals")
print(f"  可能: signals中的factor_value与validation不完全一致")

# 差距2: 选股→扣除成本
gap2 = sharpe - net_sharpe
print(f"差距2(选股→扣成本): {gap2:.3f}")
print(f"  原因: 交易成本(佣金+滑点)")

# 差距3: 扣成本→实际回测
gap3 = net_sharpe - 0.528
print(f"差距3(扣成本→实际回测): {gap3:.3f}")
print(f"  原因: 100股限制、渐进进出、非再平衡日止损、执行价格差异")

# 年度分析
print("\n" + "=" * 60)
print("2. 年度收益对比")
print("=" * 60)

daily_port['year'] = daily_port['date'].dt.year
yearly = daily_port.groupby('year').agg(
    mean_ret=('ret', 'mean'),
    std_ret=('ret', 'std'),
    count=('ret', 'count'),
)
yearly['sharpe'] = yearly['mean_ret'] / yearly['std_ret'] * np.sqrt(12.5)

# 对比实际回测的年度收益
actual_annual = {
    2015: 1.131, 2016: -0.308, 2017: 0.478, 2018: -0.262,
    2019: 1.031, 2020: 0.319, 2021: 0.142, 2022: -0.201,
    2023: -0.051, 2024: 0.007, 2025: 0.590
}

print(f"{'Year':>6} {'理论ret':>10} {'理论Sharpe':>12} {'实际ret':>10} {'差距':>8}")
for year, row in yearly.iterrows():
    actual = actual_annual.get(year, 0)
    diff = row['mean_ret'] * 12.5 - actual  # 粗略年化
    print(f"{year:>6} {row['mean_ret']:>10.4f} {row['sharpe']:>12.4f} {actual:>10.3f} {diff:>8.3f}")
