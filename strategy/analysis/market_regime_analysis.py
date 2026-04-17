#!/usr/bin/env python
"""
市场状态检测分析

分析当前市场状态检测器的效果，以及熊市期间策略的表现
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.market_regime_detector import MarketRegimeDetector


def load_index_data():
    """加载指数数据"""
    index_path = Path(__file__).parent.parent.parent / 'data' / 'stock_data' / 'raw_data' / 'sh000001'

    files = list(index_path.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"未找到指数数据: {index_path}")

    # 合并所有文件
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # 列名是 'date' 不是 'datetime'
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


def analyze_regime_accuracy():
    """分析市场状态检测的准确性"""

    # 加载指数数据
    index_df = load_index_data()
    print(f"指数数据: {len(index_df)} 条, {index_df['datetime'].min()} ~ {index_df['datetime'].max()}")

    # 生成市场状态
    detector = MarketRegimeDetector()
    regime_df = detector.generate(index_df)

    # 加载验证结果
    validation_path = Path(__file__).parent.parent / 'rolling_validation_results' / 'validation_results.csv'
    if not validation_path.exists():
        print(f"验证结果文件不存在: {validation_path}")
        return

    val_df = pd.read_csv(validation_path)
    val_df['date'] = pd.to_datetime(val_df['date'])

    # 合并市场状态
    regime_df['date'] = regime_df['datetime'].dt.date
    val_df['date_only'] = val_df['date'].dt.date

    # 按市场状态分组分析
    print("\n========== 市场状态统计 ==========")
    stats = detector.get_regime_stats()
    for year, s in stats.items():
        print(f"{year}: 牛市 {s['bull_ratio']:.1%}, 熊市 {s['bear_days']/250:.1%}天")

    # === 新增：bear_risk 分析 ===
    print("\n========== bear_risk 信号统计 ==========")
    bear_risk_days = regime_df['bear_risk'].sum()
    print(f"bear_risk触发天数: {bear_risk_days} ({100*bear_risk_days/len(regime_df):.1f}%)")

    # bear_risk 按年份统计
    regime_df['year'] = regime_df['datetime'].dt.year
    bear_risk_by_year = regime_df.groupby('year')['bear_risk'].sum()
    print("\n各年份bear_risk天数:")
    for year, days in bear_risk_by_year.items():
        if days > 0:
            print(f"  {year}: {days}天")

    # 分析各市场状态下的因子准确率
    print("\n========== 各市场状态下的因子表现 ==========")

    # 合并数据
    merged = val_df.merge(
        regime_df[['date', 'regime', 'momentum_score', 'trend_score', 'volatility', 'bear_risk']],
        left_on='date_only',
        right_on='date',
        how='left'
    )

    merged['regime'] = merged['regime'].fillna(0)
    merged['bear_risk'] = merged['bear_risk'].fillna(False)

    # 计算各状态下的准确率
    for regime in [-1, 0, 1]:
        regime_name = {-1: '熊市', 0: '震荡', 1: '牛市'}[regime]
        subset = merged[merged['regime'] == regime]

        if len(subset) == 0:
            continue

        # 买入准确率
        buy_signals = subset[subset['buy'] == True]
        if len(buy_signals) > 0:
            accuracy = (buy_signals['future_ret'] > 0).mean()
            avg_ret = buy_signals['future_ret'].mean()
            print(f"{regime_name}: 买入信号 {len(buy_signals)} 条, 准确率 {accuracy:.2%}, 平均收益 {avg_ret:.2%}")

    # === 新增：bear_risk 下的因子表现 ===
    print("\n========== bear_risk 下的因子表现 ==========")
    bear_risk_df = merged[merged['bear_risk'] == True]
    if len(bear_risk_df) > 0:
        from scipy import stats as sp_stats
        ic = sp_stats.spearmanr(bear_risk_df['factor_value'], bear_risk_df['future_ret'])[0]
        print(f"bear_risk期间因子IC: {ic:.2%}")

        buy_signals = bear_risk_df[bear_risk_df['buy'] == True]
        if len(buy_signals) > 0:
            accuracy = (buy_signals['future_ret'] > 0).mean()
            avg_ret = buy_signals['future_ret'].mean()
            print(f"bear_risk期间买入准确率: {accuracy:.2%}, 平均收益: {avg_ret:.2%}")
    else:
        print("bear_risk期间无数据")

    # 分析熊市期间的因子IC
    print("\n========== 熊市期间的因子IC ==========")
    bear_df = merged[merged['regime'] == -1]
    if len(bear_df) > 0:
        from scipy import stats as sp_stats
        ic = sp_stats.spearmanr(bear_df['factor_value'], bear_df['future_ret'])[0]
        print(f"熊市因子IC: {ic:.2%}")

    bull_df = merged[merged['regime'] == 1]
    if len(bull_df) > 0:
        ic = sp_stats.spearmanr(bull_df['factor_value'], bull_df['future_ret'])[0]
        print(f"牛市因子IC: {ic:.2%}")

    # 按动量分数分组分析
    print("\n========== 按动量分数分组的因子表现 ==========")
    merged['mom_group'] = pd.cut(merged['momentum_score'], bins=[-1, -0.5, -0.2, 0, 0.2, 0.5, 1])

    for group in merged['mom_group'].unique():
        if pd.isna(group):
            continue
        subset = merged[merged['mom_group'] == group]
        if len(subset) > 100:
            buy_signals = subset[subset['buy'] == True]
            if len(buy_signals) > 0:
                accuracy = (buy_signals['future_ret'] > 0).mean()
                avg_ret = buy_signals['future_ret'].mean()
                print(f"动量 {group}: 买入准确率 {accuracy:.2%}, 平均收益 {avg_ret:.2%}")

    return merged


def suggest_improvements():
    """提出改进建议"""
    print("\n========== 改进建议 ==========")
    print("""
1. 熊市检测优化:
   - 当前: 5日动量 < -2% 或 20日动量 < 0
   - 建议: 使用月度收益MA + 回撤双重判断
   - 熊市信号: 过去5个月平均收益 < 0 且 回撤 > 5%

2. 动态仓位调整:
   - 当前: 熊市只是减少持仓数量
   - 建议: 熊市降低总仓位到30%以下

3. 因子选择:
   - 熊市使用防御性因子(低波动、高股息)
   - 牛市使用进攻性因子(动量、成长)
    """)


if __name__ == '__main__':
    merged = analyze_regime_accuracy()
    suggest_improvements()
