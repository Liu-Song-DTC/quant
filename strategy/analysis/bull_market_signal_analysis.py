#!/usr/bin/env python
"""
牛市信号分析

分析牛市中什么样的信号表现好
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# 添加路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.market_regime_detector import MarketRegimeDetector


def load_data():
    """加载数据"""
    # 加载验证结果
    val_path = Path(__file__).parent.parent / 'rolling_validation_results' / 'validation_results.csv'
    val_df = pd.read_csv(val_path)
    val_df['date'] = pd.to_datetime(val_df['date'])

    # 加载指数数据
    index_path = Path(__file__).parent.parent.parent / 'data' / 'stock_data' / 'raw_data' / 'sh000001'
    files = list(index_path.glob('*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    index_df = pd.concat(dfs, ignore_index=True)
    index_df['datetime'] = pd.to_datetime(index_df['date'])
    index_df = index_df.sort_values('datetime').reset_index(drop=True)

    # 生成市场状态
    detector = MarketRegimeDetector()
    regime_df = detector.generate(index_df)

    return val_df, regime_df


def analyze_bull_signals():
    """分析牛市信号"""
    val_df, regime_df = load_data()

    # 合并市场状态
    regime_df['date_only'] = regime_df['datetime'].dt.date
    val_df['date_only'] = val_df['date'].dt.date

    merged = val_df.merge(
        regime_df[['date_only', 'regime', 'momentum_score', 'trend_score']],
        left_on='date_only',
        right_on='date_only',
        how='left'
    )
    merged['regime'] = merged['regime'].fillna(0)

    print("=" * 70)
    print("牛市信号深度分析")
    print("=" * 70)

    # 1. 牛市中不同factor_value区间的表现
    print("\n【1. 牛市中factor_value区间分析】")
    bull_df = merged[merged['regime'] == 1].copy()
    print(f"牛市总信号数: {len(bull_df)}")

    if len(bull_df) > 0:
        # 按factor_value分组
        bull_df['fv_bin'] = pd.cut(bull_df['factor_value'], bins=[-1, 0, 0.3, 0.5, 0.7, 0.9, 1.5])

        print("\nfactor_value区间表现:")
        for bin_label in bull_df['fv_bin'].unique():
            if pd.isna(bin_label):
                continue
            subset = bull_df[bull_df['fv_bin'] == bin_label]
            if len(subset) > 10:
                buy_signals = subset[subset['buy'] == True]
                if len(buy_signals) > 0:
                    accuracy = (buy_signals['future_ret'] > 0).mean()
                    avg_ret = buy_signals['future_ret'].mean()
                    print(f"  {bin_label}: 信号{len(buy_signals):4d}, 准确率{accuracy:.1%}, 平均收益{avg_ret:.2%}")

    # 2. 牛市中不同行业的表现
    print("\n【2. 牛市中行业表现】")
    if len(bull_df) > 0:
        industry_stats = bull_df.groupby('industry').agg({
            'buy': 'sum',
            'future_ret': ['mean', 'std', 'count']
        }).round(4)
        industry_stats.columns = ['buy_count', 'avg_ret', 'ret_std', 'total']

        # 只看有买入信号的行业
        active_industries = industry_stats[industry_stats['buy_count'] > 0].sort_values('avg_ret', ascending=False)

        print("\n行业买入信号数 & 平均收益:")
        for ind, row in active_industries.head(10).iterrows():
            if row['buy_count'] > 0:
                subset = bull_df[(bull_df['industry'] == ind) & (bull_df['buy'] == True)]
                if len(subset) > 0:
                    accuracy = (subset['future_ret'] > 0).mean()
                    print(f"  {ind}: 买入{int(row['buy_count']):3d}, 准确率{accuracy:.1%}, 平均收益{row['avg_ret']:.2%}")

    # 3. 牛市中不同因子名的表现
    print("\n【3. 牛市中因子表现】")
    if len(bull_df) > 0 and 'factor_name' in bull_df.columns:
        factor_stats = bull_df.groupby('factor_name').agg({
            'buy': 'sum',
            'future_ret': 'mean'
        }).sort_values('future_ret', ascending=False)

        print("\n因子买入信号数 & 平均收益 (Top10):")
        for fn, row in factor_stats.head(10).iterrows():
            if row['buy'] > 0:
                subset = bull_df[(bull_df['factor_name'] == fn) & (bull_df['buy'] == True)]
                if len(subset) > 0:
                    accuracy = (subset['future_ret'] > 0).mean()
                    avg_ret = subset['future_ret'].mean()
                    print(f"  {fn}: 买入{int(row['buy']):3d}, 准确率{accuracy:.1%}, 平均收益{avg_ret:.2%}")

    # 4. 牛市 vs 熊市 vs 震荡对比
    print("\n【4. 不同市场状态对比】")
    for regime in [-1, 0, 1]:
        regime_name = {-1: '熊市', 0: '震荡', 1: '牛市'}[regime]
        subset = merged[merged['regime'] == regime]

        if len(subset) > 0:
            buy_signals = subset[subset['buy'] == True]
            if len(buy_signals) > 0:
                accuracy = (buy_signals['future_ret'] > 0).mean()
                avg_ret = buy_signals['future_ret'].mean()
                median_ret = buy_signals['future_ret'].median()

                # 计算IC
                ic = stats.spearmanr(subset['factor_value'], subset['future_ret'])[0]

                print(f"\n{regime_name}:")
                print(f"  信号数: {len(buy_signals)}")
                print(f"  准确率: {accuracy:.1%}")
                print(f"  平均收益: {avg_ret:.2%}")
                print(f"  中位收益: {median_ret:.2%}")
                print(f"  因子IC: {ic:.2%}")

    # 5. 寻找牛市有效信号特征
    print("\n【5. 牛市有效信号特征分析】")
    if len(bull_df) > 0:
        buy_signals = bull_df[bull_df['buy'] == True]
        if len(buy_signals) > 0:
            # 按收益分组
            buy_signals['is_winner'] = buy_signals['future_ret'] > 0

            winners = buy_signals[buy_signals['is_winner']]
            losers = buy_signals[~buy_signals['is_winner']]

            print(f"\n赢家的特征 (n={len(winners)}):")
            if len(winners) > 0:
                print(f"  factor_value均值: {winners['factor_value'].mean():.3f}")
                print(f"  factor_value中位: {winners['factor_value'].median():.3f}")
                print(f"  score均值: {winners['score'].mean():.3f}")

            print(f"\n输家的特征 (n={len(losers)}):")
            if len(losers) > 0:
                print(f"  factor_value均值: {losers['factor_value'].mean():.3f}")
                print(f"  factor_value中位: {losers['factor_value'].median():.3f}")
                print(f"  score均值: {losers['score'].mean():.3f}")

    # 6. 牛市动量分析
    print("\n【6. 牛市动量分数分析】")
    if len(bull_df) > 0 and 'momentum_score' in bull_df.columns:
        bull_df['mom_bin'] = pd.cut(bull_df['momentum_score'], bins=[-1, -0.3, 0, 0.3, 0.6, 1.0])

        print("\n动量分数区间表现:")
        for bin_label in bull_df['mom_bin'].unique():
            if pd.isna(bin_label):
                continue
            subset = bull_df[bull_df['mom_bin'] == bin_label]
            buy_signals = subset[subset['buy'] == True]
            if len(buy_signals) > 0:
                accuracy = (buy_signals['future_ret'] > 0).mean()
                avg_ret = buy_signals['future_ret'].mean()
                print(f"  动量{bin_label}: 买入{len(buy_signals):3d}, 准确率{accuracy:.1%}, 平均收益{avg_ret:.2%}")

    return merged


if __name__ == '__main__':
    merged = analyze_bull_signals()
