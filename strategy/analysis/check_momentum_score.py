#!/usr/bin/env python
"""
检查market_info中的momentum_score值
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

from core.market_regime_detector import MarketRegimeDetector

# 加载指数数据
DATA_DIR = os.path.join(os.path.dirname(STRATEGY_DIR), 'data/stock_data/backtrader_data')
index_file = os.path.join(DATA_DIR, 'sh000001_qfq.csv')

index_df = pd.read_csv(index_file)
index_df['datetime'] = pd.to_datetime(index_df['datetime'])

# 创建并初始化检测器
detector = MarketRegimeDetector()
result = detector.generate(index_df)

print("=" * 80)
print("momentum_score 分析")
print("=" * 80)

# 查看momentum_score分布
print("\nmomentum_score分布:")
print(f"  最小值: {result['momentum_score'].min():.3f}")
print(f"  最大值: {result['momentum_score'].max():.3f}")
print(f"  均值: {result['momentum_score'].mean():.3f}")
print(f"  标准差: {result['momentum_score'].std():.3f}")

# 查看不同范围的分布
print("\nmomentum_score范围分布:")
print(f"  < -0.5: {(result['momentum_score'] < -0.5).sum()} ({(result['momentum_score'] < -0.5).mean()*100:.1f}%)")
print(f"  -0.5 ~ -0.2: {((result['momentum_score'] >= -0.5) & (result['momentum_score'] < -0.2)).sum()}")
print(f"  -0.2 ~ 0: {((result['momentum_score'] >= -0.2) & (result['momentum_score'] < 0)).sum()}")
print(f"  0 ~ 0.2: {((result['momentum_score'] >= 0) & (result['momentum_score'] < 0.2)).sum()}")
print(f"  0.2 ~ 0.5: {((result['momentum_score'] >= 0.2) & (result['momentum_score'] < 0.5)).sum()}")
print(f"  > 0.5: {(result['momentum_score'] >= 0.5).sum()}")

# 查看regime与momentum_score的关系
print("\nregime与momentum_score的关系:")
for regime_val, name in [(1, '牛市'), (0, '震荡'), (-1, '熊市')]:
    sub = result[result['regime'] == regime_val]
    if len(sub) > 0:
        print(f"  {name}: momentum_score均值={sub['momentum_score'].mean():.3f}, n={len(sub)}")

# 计算实际的市场动量（20日收益）
print("\n实际市场动量（20日收益）:")
result['actual_mom_20'] = result['close'].pct_change(20)
print(f"  最小值: {result['actual_mom_20'].min()*100:.1f}%")
print(f"  最大值: {result['actual_mom_20'].max()*100:.1f}%")
print(f"  均值: {result['actual_mom_20'].mean()*100:.2f}%")

# 比较momentum_score与实际动量
print("\nmomentum_score与实际动量相关性:")
corr = result[['momentum_score', 'actual_mom_20']].corr().iloc[0, 1]
print(f"  相关系数: {corr:.3f}")

# 查看最近一年的数据
recent = result[result['datetime'] > result['datetime'].max() - pd.Timedelta(days=365)]
print(f"\n最近一年数据 (n={len(recent)}):")
print(f"  momentum_score均值: {recent['momentum_score'].mean():.3f}")
print(f"  momentum_score < 0: {(recent['momentum_score'] < 0).sum()} ({(recent['momentum_score'] < 0).mean()*100:.1f}%)")

# 显示一些具体日期的值
print("\n样本数据（最近10个交易日）:")
sample = result.tail(10)[['datetime', 'close', 'momentum_score', 'regime', 'actual_mom_20']]
print(sample.to_string())