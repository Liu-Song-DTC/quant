"""
系统性熊市/回调风险分析
目标: 用数据确定趋势阈值、最优暴露、因子有效性
"""
import pandas as pd
import numpy as np
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# 1. 加载市场状态数据
from core.market_regime_detector import MarketRegimeDetector
import backtrader as bt

# 加载上证指数数据
idx_path = os.path.join(os.path.dirname(BASE), 'data/stock_data/backtrader_data/sh000001_qfq.csv')
if not os.path.exists(idx_path):
    print(f"ERROR: index data not found at {idx_path}")
    sys.exit(1)

idx_df = pd.read_csv(idx_path)
idx_df['datetime'] = pd.to_datetime(idx_df['datetime'])
idx_df = idx_df.sort_values('datetime')

# 创业板指
gem_path = os.path.join(os.path.dirname(BASE), 'data/stock_data/backtrader_data/sz399006_qfq.csv')
gem_df = None
if os.path.exists(gem_path):
    gem_df = pd.read_csv(gem_path)
    gem_df['datetime'] = pd.to_datetime(gem_df['datetime'])
    gem_df = gem_df.sort_values('datetime')

print(f"上证指数: {len(idx_df)} 天, {idx_df['datetime'].min().date()} ~ {idx_df['datetime'].max().date()}")

# 生成市场状态
detector = MarketRegimeDetector()
regime_df = detector.generate(idx_df, growth_df=gem_df)
if regime_df is None:
    print("ERROR: regime generation failed")
    sys.exit(1)

regime_df['date'] = pd.to_datetime(regime_df['datetime']).dt.date
print(f"市场状态: {len(regime_df)} 条")
print(f"state分布: bear={sum(regime_df['regime']==-1)}, neutral={sum(regime_df['regime']==0)}, bull={sum(regime_df['regime']==1)}")
print(f"bear_risk分布: {sum(regime_df['bear_risk'])} 天")
print(f"bear_risk_fast分布: {sum(regime_df['bear_risk_fast'])} 天")
print(f"trend_score范围: [{regime_df['trend_score'].min():.3f}, {regime_df['trend_score'].max():.3f}]")

# 2. 计算前向收益
close = idx_df['close'].values
dates = pd.to_datetime(idx_df['datetime']).dt.date.values

# 各周期的前向收益
for horizon_days in [5, 10, 20, 60]:
    horizon_steps = horizon_days  # approximate
    fwd_ret = np.full(len(close), np.nan)
    for i in range(len(close) - horizon_steps):
        fwd_ret[i] = (close[i + horizon_steps] / close[i] - 1)
    regime_df[f'fwd_ret_{horizon_days}d'] = fwd_ret[:len(regime_df)]

print("\n========== Trend Score 与 前向收益 分析 ==========")

# 3. 按 trend_score 分桶分析
bins = [-2.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 2.0]
labels = ['≤-0.5', '(-0.5,-0.3]', '(-0.3,-0.2]', '(-0.2,-0.1]', '(-0.1,-0.05]', '(-0.05,0]',
          '(0,0.05]', '(0.05,0.1]', '(0.1,0.2]', '(0.2,0.3]', '(0.3,0.5]', '>0.5']
regime_df['ts_bucket'] = pd.cut(regime_df['trend_score'], bins=bins, labels=labels)

print(f"\n{'Bucket':<16} {'天数':>6} {'fwd5d_均值':>10} {'fwd5d_胜率':>10} {'fwd10d_均值':>10} {'fwd10d_胜率':>11} {'fwd20d_均值':>10} {'fwd20d_胜率':>11} {'fwd60d_均值':>10} {'fwd60d_胜率':>11}")
print("-" * 120)

yearly_data = {}
for bucket in labels:
    mask = regime_df['ts_bucket'] == bucket
    n = mask.sum()
    if n == 0:
        continue
    row_data = []
    for h in [5, 10, 20, 60]:
        col = f'fwd_ret_{h}d'
        vals = regime_df.loc[mask, col].dropna()
        row_data.append(f"{vals.mean():+.4f}")
        row_data.append(f"{100*(vals>0).mean():.1f}%")
    print(f"{bucket:<16} {n:>6} {row_data[0]:>10} {row_data[1]:>10} {row_data[2]:>10} {row_data[3]:>11} {row_data[4]:>10} {row_data[5]:>11} {row_data[6]:>10} {row_data[7]:>11}")

# 4. 按年分析 trend_score 分布
print("\n========== 各年份 Trend Score 分布 ==========")
regime_df['year'] = pd.to_datetime(regime_df['datetime']).dt.year
for year in sorted(regime_df['year'].unique()):
    yr = regime_df[regime_df['year'] == year]
    neg_mask = yr['trend_score'] < 0
    neg_days = neg_mask.sum()
    total = len(yr)
    print(f"  {year}: {neg_days}/{total} 天 trend<0 ({100*neg_days/total:.0f}%), "
          f"trend均值={yr['trend_score'].mean():+.3f}, "
          f"bear_risk={yr['bear_risk'].sum()}天, bear_fast={yr['bear_risk_fast'].sum()}天")

# 5. 关键在于: 什么趋势下买入信号有效?
print("\n========== 关键阈值分析 ==========")
for threshold in [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0]:
    below = regime_df[regime_df['trend_score'] < threshold]
    above = regime_df[regime_df['trend_score'] >= threshold]
    if len(below) > 0 and len(above) > 0:
        for h in [10, 20]:
            below_ret = below[f'fwd_ret_{h}d'].dropna().mean()
            above_ret = above[f'fwd_ret_{h}d'].dropna().mean()
            below_wr = 100 * (below[f'fwd_ret_{h}d'].dropna() > 0).mean()
            above_wr = 100 * (above[f'fwd_ret_{h}d'].dropna() > 0).mean()
            print(f"  trend < {threshold:+.2f}: fwd{h}d={below_ret:+.4f} WR={below_wr:.1f}% "
                  f"| trend >= {threshold:+.2f}: fwd{h}d={above_ret:+.4f} WR={above_wr:.1f}% "
                  f"| 差值={above_ret-below_ret:+.4f}")

# 6. 建议: 基于数据分析的最优趋势阈值
print("\n========== 建议 ==========")
# 找趋势转负且前向收益明显转负的临界点
for threshold in np.arange(-0.5, 0.05, 0.05):
    below = regime_df[regime_df['trend_score'] < threshold]
    if len(below) < 10:
        continue
    ret20 = below['fwd_ret_20d'].dropna().mean()
    wr20 = 100 * (below['fwd_ret_20d'].dropna() > 0).mean()
    n = len(below)
    if ret20 < -0.005:  # 前向20日收益<-0.5%才值得降仓
        print(f"  trend<{threshold:+.2f}: {n}天, fwd20d={ret20:+.4f}, WR={wr20:.1f}% ← 建议降仓")
