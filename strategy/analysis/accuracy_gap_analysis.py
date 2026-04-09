#!/usr/bin/env python
"""
准确率差距分析
目标：理解为什么准确率只有48-51%，而不是55%
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('rolling_validation_results/validation_results.csv')
df['date'] = pd.to_datetime(df['date'])

print("=" * 80)
print("准确率差距分析")
print("=" * 80)

# 1. 市场环境分析
print("\n1. 市场环境分析:")
# 计算每日市场收益
daily_ret = df.groupby('date')['future_ret'].mean()
market_up_days = (daily_ret > 0).sum()
print(f"  上涨天数占比: {market_up_days/len(daily_ret)*100:.1f}%")
print(f"  市场平均日收益: {daily_ret.mean()*100:.2f}%")

# 2. 如果随机选股，准确率是多少？
print("\n2. 随机选股基准:")
acc_random = (df['future_ret'] > 0).mean()
print(f"  随机选股准确率: {acc_random*100:.2f}%")

# 3. 不同市场环境下的准确率
print("\n3. 不同市场环境下的准确率:")
df['market_ret'] = df.groupby('date')['future_ret'].transform('mean')
df['market_up'] = df['market_ret'] > 0

buy_df = df[df['buy'] == True]
for market_cond, name in [(True, '市场上涨日'), (False, '市场下跌日')]:
    sub = buy_df[buy_df['market_up'] == market_cond]
    if len(sub) > 0:
        acc = (sub['future_ret'] > 0).mean()
        print(f"  {name}: 准确率={acc*100:.2f}%, n={len(sub):,}")

# 4. 准确率提升需要什么？
print("\n4. 准确率提升分析:")
current_acc = 0.4855
target_acc = 0.55
print(f"  当前准确率: {current_acc*100:.2f}%")
print(f"  目标准确率: {target_acc*100:.2f}%")
print(f"  需要提升: {(target_acc - current_acc)*100:.2f}个百分点")

# 分析：如果准确率提升到55%，收益会如何变化？
buy_df_pos = buy_df[buy_df['future_ret'] > 0]
buy_df_neg = buy_df[buy_df['future_ret'] <= 0]
avg_pos_ret = buy_df_pos['future_ret'].mean()
avg_neg_ret = buy_df_neg['future_ret'].mean()

print(f"\n  正收益信号平均收益: {avg_pos_ret*100:.2f}%")
print(f"  负收益信号平均收益: {avg_neg_ret*100:.2f}%")

# 当前平均收益
current_avg_ret = buy_df['future_ret'].mean()
print(f"  当前平均收益: {current_avg_ret*100:.2f}%")

# 如果准确率55%的平均收益
n_signals = len(buy_df)
n_pos = int(n_signals * target_acc)
n_neg = n_signals - n_pos
target_avg_ret = (n_pos * avg_pos_ret + n_neg * avg_neg_ret) / n_signals
print(f"  如果准确率55%, 平均收益: {target_avg_ret*100:.2f}%")

# 5. 分析信号质量提升空间
print("\n5. 信号质量提升空间:")
# 如果只用score > 1的信号
high_score = buy_df[buy_df['score'] > 1.0]
if len(high_score) > 0:
    acc_high = (high_score['future_ret'] > 0).mean()
    avg_ret_high = high_score['future_ret'].mean()
    print(f"  score>1信号准确率: {acc_high*100:.2f}%, 收益: {avg_ret_high*100:.2f}%")

# 如果只用_F因子
f_only = buy_df[buy_df['factor_name'].str.endswith('_F', na=False)]
if len(f_only) > 0:
    acc_f = (f_only['future_ret'] > 0).mean()
    avg_ret_f = f_only['future_ret'].mean()
    print(f"  _F因子信号准确率: {acc_f*100:.2f}%, 收益: {avg_ret_f*100:.2f}%")

# 如果只用3F因子
f3_only = buy_df[buy_df['factor_name'].str.contains('3F', na=False)]
if len(f3_only) > 0:
    acc_f3 = (f3_only['future_ret'] > 0).mean()
    avg_ret_f3 = f3_only['future_ret'].mean()
    print(f"  3F因子信号准确率: {acc_f3*100:.2f}%, 收益: {avg_ret_f3*100:.2f}%")