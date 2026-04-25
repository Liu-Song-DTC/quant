"""
最终诊断: IC=6.47%但Sharpe=0.35, 问题出在哪?
模拟理想策略(无交易成本) vs 实际回测
"""

import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

validation = pd.read_csv(os.path.join(results_dir, 'validation_results.csv'))
selections = pd.read_csv(os.path.join(results_dir, 'portfolio_selections.csv'))

validation['date'] = pd.to_datetime(validation['date'])
selections['date'] = pd.to_datetime(selections['date'])

print("=" * 60)
print("1. 理想策略模拟: 每期选rank_pct>0.5的top10等权")
print("=" * 60)

rebalance_period = 20
dates = sorted(validation['date'].unique())
rebalance_dates = dates[::rebalance_period]

portfolio_returns = []
for rb_date in rebalance_dates:
    day_data = validation[validation['date'] == rb_date].copy()
    if len(day_data) < 50:
        continue
    day_data['rank_pct'] = day_data['factor_value'].rank(pct=True)
    qualified = day_data[day_data['rank_pct'] > 0.5]
    if len(qualified) == 0:
        continue
    top10 = qualified.nlargest(10, 'factor_value')
    port_ret = top10['future_ret'].mean()
    portfolio_returns.append({'date': rb_date, 'ret': port_ret})

port_df = pd.DataFrame(portfolio_returns)
if len(port_df) == 0:
    print("No data")
    exit(1)

mean_ret = port_df['ret'].mean()
std_ret = port_df['ret'].std()
sharpe_20 = mean_ret / std_ret if std_ret > 0 else 0
annual_sharpe = sharpe_20 * np.sqrt(12.5)
print(f"再平衡次数: {len(port_df)}")
print(f"20日均收益: {mean_ret:.4f}")
print(f"年化Sharpe: {annual_sharpe:.4f}")

# 年度分析
port_df['year'] = port_df['date'].dt.year
yearly = port_df.groupby('year').agg(
    mean_ret=('ret', 'mean'),
    std_ret=('ret', 'std'),
    count=('ret', 'count'),
)
yearly['sharpe'] = yearly['mean_ret'] / yearly['std_ret']

print(f"\n年度收益:")
for year, row in yearly.iterrows():
    print(f"  {year}: ret={row['mean_ret']:.4f}, sharpe={row['sharpe']:.4f}")

# 对比: 更严格的选股
print("\n" + "=" * 60)
print("2. 不同选股门槛对比")
print("=" * 60)

for threshold_name, threshold in [("rank>0.3", 0.3), ("rank>0.5", 0.5), ("rank>0.7", 0.7)]:
    rets = []
    for rb_date in rebalance_dates:
        day_data = validation[validation['date'] == rb_date].copy()
        if len(day_data) < 50:
            continue
        day_data['rank_pct'] = day_data['factor_value'].rank(pct=True)
        qualified = day_data[day_data['rank_pct'] > threshold]
        if len(qualified) == 0:
            continue
        top_n = qualified.nlargest(10, 'factor_value')
        port_ret = top_n['future_ret'].mean()
        rets.append(port_ret)
    ret_arr = np.array(rets)
    sharpe = ret_arr.mean() / ret_arr.std() * np.sqrt(12.5) if ret_arr.std() > 0 else 0
    print(f"  {threshold_name}: 20日ret={ret_arr.mean():.4f}, sharpe={sharpe:.4f}, n={len(rets)}")

# 对比: top5 vs top10 vs top20
print("\n" + "=" * 60)
print("3. 不同持仓数对比 (rank>0.5)")
print("=" * 60)

for n_stock in [5, 10, 15, 20]:
    rets = []
    for rb_date in rebalance_dates:
        day_data = validation[validation['date'] == rb_date].copy()
        if len(day_data) < 50:
            continue
        day_data['rank_pct'] = day_data['factor_value'].rank(pct=True)
        qualified = day_data[day_data['rank_pct'] > 0.5]
        if len(qualified) == 0:
            continue
        top_n = qualified.nlargest(n_stock, 'factor_value')
        port_ret = top_n['future_ret'].mean()
        rets.append(port_ret)
    ret_arr = np.array(rets)
    sharpe = ret_arr.mean() / ret_arr.std() * np.sqrt(12.5) if ret_arr.std() > 0 else 0
    print(f"  top{n_stock}: 20日ret={ret_arr.mean():.4f}, sharpe={sharpe:.4f}")

# 交易成本分析
print("\n" + "=" * 60)
print("4. 交易成本影响")
print("=" * 60)

sel_dates_list = sorted(selections['date'].unique())
turnover_rates = []
for i in range(1, len(sel_dates_list)):
    prev_codes = set(selections[selections['date'] == sel_dates_list[i-1]]['code'])
    curr_codes = set(selections[selections['date'] == sel_dates_list[i]]['code'])
    total = len(prev_codes) + len(curr_codes)
    if total == 0:
        continue
    turnover = len(prev_codes.symmetric_difference(curr_codes)) / total
    turnover_rates.append(turnover)

avg_turnover = np.mean(turnover_rates) if turnover_rates else 0
cost_per_rebalance = avg_turnover * 0.003  # 0.3% round-trip
annual_cost = cost_per_rebalance * 12.5  # ~12.5 rebalances per year

print(f"  平均换手率: {avg_turnover:.1%}")
print(f"  每次再平衡成本: {cost_per_rebalance:.4f}")
print(f"  年化交易成本: {annual_cost:.4f}")

# 扣除交易成本后的Sharpe
net_ret = mean_ret - cost_per_rebalance
net_sharpe = net_ret / std_ret * np.sqrt(12.5) if std_ret > 0 else 0
print(f"  扣除成本后20日ret: {net_ret:.4f}")
print(f"  扣除成本后年化Sharpe: {net_sharpe:.4f}")

# 检查: 是否可以通过减仓来提高Sharpe?
# 牛市全仓, 熊市半仓
print("\n" + "=" * 60)
print("5. 市场状态调整: 牛市全仓, 熊市半仓")
print("=" * 60)

# 从validation获取市场状态
# 用future_ret的符号作为简化判断
port_df['is_positive_market'] = port_df['ret'] > 0
bull_rets = port_df[port_df['is_positive_market']]['ret']
bear_rets = port_df[~port_df['is_positive_market']]['ret']

print(f"  正收益期: {len(bull_rets)}次, 均收益={bull_rets.mean():.4f}")
print(f"  负收益期: {len(bear_rets)}次, 均收益={bear_rets.mean():.4f}")

# 如果在负收益期半仓
adjusted_rets = port_df['ret'].copy()
adjusted_rets[~port_df['is_positive_market']] *= 0.5
adj_sharpe = adjusted_rets.mean() / adjusted_rets.std() * np.sqrt(12.5)
print(f"  半仓调整后Sharpe: {adj_sharpe:.4f}")

# 关键: 不用未来信息, 用factor_value sign来判断
print("\n" + "=" * 60)
print("6. 用截面factor_value均值判断市场")
print("=" * 60)

smart_returns = []
for rb_date in rebalance_dates:
    day_data = validation[validation['date'] == rb_date].copy()
    if len(day_data) < 50:
        continue
    day_data['rank_pct'] = day_data['factor_value'].rank(pct=True)

    # 截面factor_value均值 > 0: 整体看好
    fv_mean = day_data['factor_value'].mean()

    qualified = day_data[day_data['rank_pct'] > 0.5]
    if len(qualified) == 0:
        continue
    top10 = qualified.nlargest(10, 'factor_value')
    port_ret = top10['future_ret'].mean()

    # 如果截面因子均值为负, 降低仓位
    if fv_mean < 0:
        port_ret *= 0.5

    smart_returns.append(port_ret)

smart_arr = np.array(smart_returns)
smart_sharpe = smart_arr.mean() / smart_arr.std() * np.sqrt(12.5) if smart_arr.std() > 0 else 0
print(f"  智能仓位调整后20日ret: {smart_arr.mean():.4f}")
print(f"  智能仓位调整后Sharpe: {smart_sharpe:.4f}")
