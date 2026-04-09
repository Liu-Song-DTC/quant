#!/usr/bin/env python
"""
市场状态检测效果验证
目标：检查regime=-1是否正确识别了市场下跌日
"""
import os
import sys
import pandas as pd
import numpy as np

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

from core.market_regime_detector import MarketRegimeDetector

# 加载验证结果
df = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/validation_results.csv'))
df['date'] = pd.to_datetime(df['date'])

# 加载指数数据
DATA_DIR = os.path.join(os.path.dirname(STRATEGY_DIR), 'data/stock_data/backtrader_data')
index_file = os.path.join(DATA_DIR, 'sh000001_qfq.csv')

index_df = pd.read_csv(index_file)
index_df['datetime'] = pd.to_datetime(index_df['datetime'])

print("=" * 80)
print("市场状态检测效果验证")
print("=" * 80)

# 创建并初始化检测器
detector = MarketRegimeDetector()
detector.generate(index_df)

print(f"\n指数数据日期范围: {index_df['datetime'].min()} ~ {index_df['datetime'].max()}")
print(f"总交易日: {len(index_df)}")

# 查看检测到的regime分布
print("\n检测器输出的市场状态分布:")
for regime_val, name in [(1, '牛市'), (0, '震荡'), (-1, '熊市')]:
    count = (detector.index_data['regime'] == regime_val).sum()
    print(f"  {name}(regime={regime_val}): {count}天 ({count/len(detector.index_data)*100:.1f}%)")

# 计算每日市场收益
daily_market = df.groupby('date').agg({
    'future_ret': 'mean'
}).reset_index()
daily_market['market_up'] = daily_market['future_ret'] > 0

print(f"\n验证数据日期范围: {daily_market['date'].min()} ~ {daily_market['date'].max()}")
print(f"总交易日: {len(daily_market)}")
print(f"上涨日: {daily_market['market_up'].sum()} ({daily_market['market_up'].mean()*100:.1f}%)")
print(f"下跌日: {(~daily_market['market_up']).sum()} ({(~daily_market['market_up']).mean()*100:.1f}%)")

# 将regime映射到daily_market
regime_lookup = detector.index_data.set_index('datetime')['regime']
daily_market['regime'] = daily_market['date'].map(
    lambda d: regime_lookup.get(pd.Timestamp(d), 0) if pd.Timestamp(d) in regime_lookup.index else 0
)

# 核心问题：regime=-1时，实际市场是下跌的比例是多少？
print("\n" + "=" * 80)
print("市场状态与实际市场表现对应关系:")
print("=" * 80)

for regime_val, name in [(1, '牛市'), (0, '震荡'), (-1, '熊市')]:
    sub = daily_market[daily_market['regime'] == regime_val]
    if len(sub) > 0:
        down_rate = (~sub['market_up']).mean()
        avg_ret = sub['future_ret'].mean()
        print(f"  {name}: 下跌日占比={down_rate*100:.1f}%, 平均市场收益={avg_ret*100:.3f}%, n={len(sub)}")

# 反过来：市场实际下跌时，regime=-1的比例是多少？
print("\n" + "=" * 80)
print("市场下跌时的状态识别:")
print("=" * 80)

down_days = daily_market[~daily_market['market_up']]
up_days = daily_market[daily_market['market_up']]

print(f"\n市场下跌日 (n={len(down_days)}):")
for regime_val, name in [(1, '牛市'), (0, '震荡'), (-1, '熊市')]:
    count = (down_days['regime'] == regime_val).sum()
    print(f"  被识别为{name}: {count}天 ({count/len(down_days)*100:.1f}%)")

print(f"\n市场上涨日 (n={len(up_days)}):")
for regime_val, name in [(1, '牛市'), (0, '震荡'), (-1, '熊市')]:
    count = (up_days['regime'] == regime_val).sum()
    print(f"  被识别为{name}: {count}天 ({count/len(up_days)*100:.1f}%)")

# 计算检测准确率
print("\n" + "=" * 80)
print("检测效果量化:")
print("=" * 80)

# 下跌日被识别为熊市的比例
down_as_bear = (down_days['regime'] == -1).mean() if len(down_days) > 0 else 0
# 上涨日被识别为牛市的比例
up_as_bull = (up_days['regime'] == 1).mean() if len(up_days) > 0 else 0

print(f"  下跌日被识别为熊市的比例: {down_as_bear*100:.1f}%")
print(f"  上涨日被识别为牛市的比例: {up_as_bull*100:.1f}%")

# 合并到原始数据看不同regime下的准确率
print("\n" + "=" * 80)
print("不同市场状态下的买入信号准确率:")
print("=" * 80)

df['regime'] = df['date'].map(
    lambda d: regime_lookup.get(pd.Timestamp(d), 0) if pd.Timestamp(d) in regime_lookup.index else 0
)
buy_df = df[df['buy'] == True]

for regime_val, name in [(1, '牛市'), (0, '震荡'), (-1, '熊市')]:
    sub = buy_df[buy_df['regime'] == regime_val]
    if len(sub) > 0:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        print(f"  {name}: 准确率={acc*100:.2f}%, 平均收益={avg_ret*100:.3f}%, n={len(sub):,}")

# 关键问题：regime!=-1时的下跌日准确率是多少？
print("\n" + "=" * 80)
print("关键问题：非熊市状态下的下跌日表现:")
print("=" * 80)

# 合并市场涨跌信息
df['market_up'] = df['date'].map(
    lambda d: daily_market.set_index('date').loc[pd.Timestamp(d), 'market_up'] if pd.Timestamp(d) in daily_market['date'].values else True
)

buy_with_market = buy_df.copy()
buy_with_market['market_up'] = buy_with_market['date'].map(
    lambda d: daily_market.set_index('date').loc[pd.Timestamp(d), 'market_up'] if pd.Timestamp(d) in daily_market['date'].values else True
)

# 非熊市状态 + 市场下跌日
non_bear_down = buy_with_market[(buy_with_market['regime'] != -1) & (~buy_with_market['market_up'])]
if len(non_bear_down) > 0:
    acc = (non_bear_down['future_ret'] > 0).mean()
    avg_ret = non_bear_down['future_ret'].mean()
    print(f"  非熊市+下跌日: 准确率={acc*100:.2f}%, 平均收益={avg_ret*100:.3f}%, n={len(non_bear_down):,}")

# 熊市状态 + 市场下跌日
bear_down = buy_with_market[(buy_with_market['regime'] == -1) & (~buy_with_market['market_up'])]
if len(bear_down) > 0:
    acc = (bear_down['future_ret'] > 0).mean()
    avg_ret = bear_down['future_ret'].mean()
    print(f"  熊市+下跌日: 准确率={acc*100:.2f}%, 平均收益={avg_ret*100:.3f}%, n={len(bear_down):,}")

# 分析动量指标分布
print("\n" + "=" * 80)
print("动量指标分析 (为什么检测不准?):")
print("=" * 80)

# 查看检测器的阈值
print(f"\n当前检测阈值:")
print(f"  熊市阈值: mom5_bear={detector.mom5_bear}, mom_bear={detector.mom_bear}")
print(f"  牛市阈值: mom_bull={detector.mom_bull}, mom60_bull={detector.mom60_bull}")

# 查看下跌日的动量分布
down_dates = down_days['date'].values
detector_index = detector.index_data.set_index('datetime')

down_momentums = []
for d in down_dates:
    ts = pd.Timestamp(d)
    if ts in detector_index.index:
        row = detector_index.loc[ts]
        down_momentums.append({
            'momentum_5': row.get('momentum_5', row.get('momentum_score', 0)) if isinstance(row, pd.Series) else 0,
            'momentum_20': row.get('momentum', row.get('momentum_score', 0)) if isinstance(row, pd.Series) else 0,
        })

if down_momentums:
    mom_df = pd.DataFrame(down_momentums)
    print(f"\n下跌日的动量分布:")
    if 'momentum_5' in mom_df.columns:
        print(f"  5日动量: 均值={mom_df['momentum_5'].mean()*100:.2f}%, < -2%占比={ (mom_df['momentum_5'] < -0.02).mean()*100:.1f}%")
    if 'momentum_20' in mom_df.columns:
        print(f"  20日动量: 均值={mom_df['momentum_20'].mean()*100:.2f}%, < 0占比={ (mom_df['momentum_20'] < 0).mean()*100:.1f}%")

# 建议
print("\n" + "=" * 80)
print("分析结论与建议:")
print("=" * 80)
print("""
如果下跌日被识别为熊市的比例低，说明市场状态检测不够灵敏。
建议：
1. 调整market_regime_detector的阈值，使其更早识别市场下跌
2. 或者在信号层加入市场动量因子，降低下跌日的买入信号
3. 或者在组合层更激进地降低熊市仓位
""")