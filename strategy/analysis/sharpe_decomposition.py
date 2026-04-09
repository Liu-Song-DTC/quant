#!/usr/bin/env python
"""
Sharpe分解分析
目标：理解为什么Sharpe只有0.554，以及如何提升到1.0
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

# 加载验证数据
df = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/validation_results.csv'))
df['date'] = pd.to_datetime(df['date'])

# 加载选股结果
selections = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/portfolio_selections.csv'))
selections['date'] = pd.to_datetime(selections['date'])

print("=" * 80)
print("Sharpe分解分析")
print("=" * 80)

print(f"\n当前Sharpe: 0.554")
print(f"目标Sharpe: 1.0")
print(f"需要提升: {(1.0/0.554 - 1)*100:.1f}%")

# 1. 分析被选中股票的表现
print("\n" + "=" * 80)
print("1. 被选中股票表现分析")
print("=" * 80)

# 合并数据
merged = selections.merge(
    df[['date', 'code', 'future_ret', 'score', 'factor_value', 'factor_name']],
    on=['date', 'code'],
    how='left'
)

print(f"\n选股数量: {len(selections):,}")
print(f"合并成功: {merged['future_ret'].notna().sum():,}")

if merged['future_ret'].notna().sum() > 0:
    valid = merged[merged['future_ret'].notna()]
    acc = (valid['future_ret'] > 0).mean()
    avg_ret = valid['future_ret'].mean()
    print(f"\n被选中股票准确率: {acc*100:.2f}%")
    print(f"被选中股票平均收益: {avg_ret*100:.3f}%")

# 2. 日收益分析
print("\n" + "=" * 80)
print("2. 日收益分布分析")
print("=" * 80)

# 按日期聚合
daily = df[df['buy'] == True].groupby('date').agg({
    'future_ret': ['mean', 'std', 'count']
}).reset_index()
daily.columns = ['date', 'avg_ret', 'ret_std', 'n_signals']

print(f"\n交易日数: {len(daily)}")
print(f"日均收益: {daily['avg_ret'].mean()*100:.3f}%")
print(f"日收益标准差: {daily['avg_ret'].std()*100:.3f}%")

# 计算日度Sharpe
daily_sharpe = daily['avg_ret'].mean() / daily['avg_ret'].std() * np.sqrt(252)
print(f"日度Sharpe估算: {daily_sharpe:.3f}")

# 3. 分析收益分布
print("\n" + "=" * 80)
print("3. 收益分布分析")
print("=" * 80)

buy_df = df[df['buy'] == True]
pos_ret = buy_df[buy_df['future_ret'] > 0]
neg_ret = buy_df[buy_df['future_ret'] <= 0]

print(f"\n正收益信号:")
print(f"  数量: {len(pos_ret):,} ({len(pos_ret)/len(buy_df)*100:.1f}%)")
print(f"  平均收益: {pos_ret['future_ret'].mean()*100:.3f}%")
print(f"  总收益: {pos_ret['future_ret'].sum()*100:.1f}%")

print(f"\n负收益信号:")
print(f"  数量: {len(neg_ret):,} ({len(neg_ret)/len(buy_df)*100:.1f}%)")
print(f"  平均亏损: {neg_ret['future_ret'].mean()*100:.3f}%")
print(f"  总亏损: {neg_ret['future_ret'].sum()*100:.1f}%")

# 盈亏比
profit_loss_ratio = pos_ret['future_ret'].mean() / abs(neg_ret['future_ret'].mean())
print(f"\n盈亏比: {profit_loss_ratio:.3f}")

# 4. Sharpe提升路径分析
print("\n" + "=" * 80)
print("4. Sharpe提升路径分析")
print("=" * 80)

current_acc = (buy_df['future_ret'] > 0).mean()
avg_pos = pos_ret['future_ret'].mean()
avg_neg = neg_ret['future_ret'].mean()

print(f"\n当前状态:")
print(f"  准确率: {current_acc*100:.2f}%")
print(f"  盈亏比: {profit_loss_ratio:.3f}")
print(f"  平均收益: {buy_df['future_ret'].mean()*100:.3f}%")

# 模拟不同准确率下的收益
print(f"\n如果准确率提升到不同水平:")
for target_acc in [0.52, 0.55, 0.60]:
    n_signals = len(buy_df)
    n_pos = int(n_signals * target_acc)
    n_neg = n_signals - n_pos
    avg_ret = (n_pos * avg_pos + n_neg * avg_neg) / n_signals
    print(f"  准确率{target_acc*100:.0f}%: 平均收益={avg_ret*100:.3f}%")

# 模拟不同盈亏比
print(f"\n如果盈亏比提升（保持准确率不变）:")
for target_pl in [0.6, 0.7, 0.8]:
    # 提高正收益，保持负收益不变
    new_avg_pos = target_pl * abs(avg_neg)
    avg_ret = (current_acc * new_avg_pos + (1-current_acc) * avg_neg)
    print(f"  盈亏比{target_pl:.1f}: 平均收益={avg_ret*100:.3f}%")

# 5. 关键发现
print("\n" + "=" * 80)
print("5. 关键发现与建议")
print("=" * 80)
print(f"""
关键发现:
1. 盈亏比只有{profit_loss_ratio:.3f}（正收益平均{avg_pos*100:.2f}%，负收益平均{avg_neg*100:.2f}%）
2. 准确率51%，略高于随机
3. 负收益信号的绝对值大于正收益信号

提升路径:
1. 提高准确率: 从51%提升到55% → 收益提升约{(0.55-0.51)*avg_pos*100 + (0.45-0.49)*avg_neg*100:.2f}%
2. 提高盈亏比: 从{profit_loss_ratio:.2f}提升到0.7 → 收益提升约{(current_acc * 0.7 * abs(avg_neg) + (1-current_acc) * avg_neg - buy_df['future_ret'].mean())*100:.2f}%
3. 减少极端亏损: 过滤掉亏损最大的信号
""")