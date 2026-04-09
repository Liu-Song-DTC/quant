#!/usr/bin/env python
"""
模块化定量分析脚本
目标：Sharpe > 1.0，拆解到各模块的可量化指标
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)


def load_data():
    """加载数据"""
    df = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/validation_results.csv'))
    df['date'] = pd.to_datetime(df['date'])
    return df


def analyze_factor_layer(df):
    """因子层分析"""
    print("=" * 80)
    print("【因子层分析】")
    print("=" * 80)

    # 1. 整体IC
    ic, _ = stats.spearmanr(df['factor_value'], df['future_ret'])
    print(f"\n1. 整体IC: {ic*100:.2f}%")

    # 2. 按因子类型分IC
    print("\n2. 按因子类型分IC:")
    for suffix in ['_T', '_F']:
        sub = df[df['factor_name'].str.endswith(suffix, na=False)]
        if len(sub) > 100:
            ic, _ = stats.spearmanr(sub['factor_value'], sub['future_ret'])
            print(f"   {suffix}: IC={ic*100:.2f}%, n={len(sub):,}")

    # 3. 按因子数量分IC
    print("\n3. 按因子数量分IC:")
    for n_f in ['1F', '2F', '3F']:
        sub = df[df['factor_name'].str.contains(f'_{n_f}_', na=False) | df['factor_name'].str.endswith(f'_{n_f}', na=False)]
        if len(sub) > 100:
            ic, _ = stats.spearmanr(sub['factor_value'], sub['future_ret'])
            print(f"   {n_f}: IC={ic*100:.2f}%, n={len(sub):,}")

    # 4. IR分析（近5年）
    print("\n4. IR分析（近5年）:")
    recent = df[df['date'] > df['date'].max() - pd.Timedelta(days=365*5)]
    ic_series = []
    for date, group in recent.groupby('date'):
        if len(group) > 10:
            ic, _ = stats.spearmanr(group['factor_value'], group['future_ret'])
            ic_series.append(ic)
    if ic_series:
        mean_ic = np.mean(ic_series)
        std_ic = np.std(ic_series)
        ir = mean_ic / std_ic
        print(f"   IC均值: {mean_ic*100:.2f}%")
        print(f"   IC标准差: {std_ic*100:.2f}%")
        print(f"   IR: {ir:.2f}")

    return ic


def analyze_signal_layer(df):
    """信号层分析"""
    print("\n" + "=" * 80)
    print("【信号层分析】")
    print("=" * 80)

    buy_df = df[df['buy'] == True]
    sell_df = df[df['sell'] == True]

    # 1. 买入信号分析
    print(f"\n1. 买入信号:")
    print(f"   数量: {len(buy_df):,}")
    print(f"   占比: {len(buy_df)/len(df)*100:.1f}%")

    # 买入准确率
    buy_acc = (buy_df['future_ret'] > 0).mean()
    print(f"   准确率: {buy_acc*100:.2f}%")

    # 买入信号IC
    ic_buy, _ = stats.spearmanr(buy_df['factor_value'], buy_df['future_ret'])
    print(f"   IC: {ic_buy*100:.2f}%")

    # 买入平均收益
    avg_ret = buy_df['future_ret'].mean()
    print(f"   平均收益: {avg_ret*100:.2f}%")

    # 2. 卖出信号分析
    print(f"\n2. 卖出信号:")
    print(f"   数量: {len(sell_df):,}")
    sell_acc = (sell_df['future_ret'] < 0).mean()
    print(f"   准确率: {sell_acc*100:.2f}%")

    # 3. 按factor_value分位数看买入准确率
    print("\n3. 买入准确率按factor_value分位数:")
    buy_df['fv_bin'] = pd.qcut(buy_df['factor_value'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
    for bin_name in ['Q1','Q2','Q3','Q4','Q5']:
        sub = buy_df[buy_df['fv_bin'] == bin_name]
        if len(sub) > 0:
            acc = (sub['future_ret'] > 0).mean()
            avg_ret = sub['future_ret'].mean()
            print(f"   {bin_name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%")

    # 4. 按score分位数看买入准确率
    print("\n4. 买入准确率按score分位数:")
    buy_df['score_bin'] = pd.qcut(buy_df['score'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
    for bin_name in ['Q1','Q2','Q3','Q4','Q5']:
        sub = buy_df[buy_df['score_bin'] == bin_name]
        if len(sub) > 0:
            acc = (sub['future_ret'] > 0).mean()
            avg_ret = sub['future_ret'].mean()
            print(f"   {bin_name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%")

    return buy_acc, ic_buy


def analyze_portfolio_layer(df):
    """组合层分析"""
    print("\n" + "=" * 80)
    print("【组合层分析】")
    print("=" * 80)

    # 模拟组合层选股：每天选score最高的N只股票
    print("\n1. 模拟选股效果（每天选Top N股票）:")

    for n in [6, 8, 10]:
        # 按日期计算score排名
        df['score_rank'] = df.groupby('date')['score'].rank(ascending=False)

        # 选Top N
        top_n = df[df['score_rank'] <= n]

        # 计算准确率和收益
        acc = (top_n['future_ret'] > 0).mean()
        avg_ret = top_n['future_ret'].mean()
        ic, _ = stats.spearmanr(top_n['score'], top_n['future_ret'])

        print(f"   Top{n}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, IC={ic*100:.2f}%")

    # 分析不同排名区间的表现
    print("\n2. 不同排名区间的表现:")
    df['score_rank_pct'] = df.groupby('date')['score'].rank(ascending=False, pct=True)

    ranges = [
        ('Top10%', 'score_rank_pct <= 0.1'),
        ('10-30%', 'score_rank_pct > 0.1 and score_rank_pct <= 0.3'),
        ('30-50%', 'score_rank_pct > 0.3 and score_rank_pct <= 0.5'),
        ('50-70%', 'score_rank_pct > 0.5 and score_rank_pct <= 0.7'),
        ('Bottom30%', 'score_rank_pct > 0.7'),
    ]

    for name, cond in ranges:
        sub = df.query(cond)
        if len(sub) > 100:
            acc = (sub['future_ret'] > 0).mean()
            avg_ret = sub['future_ret'].mean()
            ic, _ = stats.spearmanr(sub['score'], sub['future_ret'])
            print(f"   {name}: 准确率={acc*100:.2f}%, 收益={avg_ret*100:.2f}%, IC={ic*100:.2f}%")


def analyze_sharpe_breakdown(df):
    """Sharpe分解分析"""
    print("\n" + "=" * 80)
    print("【Sharpe分解分析】")
    print("=" * 80)

    # Sharpe = 收益均值 / 收益标准差
    # 收益 = 信号强度 × 持仓数量 × 换手率

    buy_df = df[df['buy'] == True]

    # 1. 信号强度分析
    print("\n1. 信号强度:")
    signal_strength = buy_df['future_ret'].mean()
    print(f"   买入信号平均收益: {signal_strength*100:.2f}%")

    # 2. 信号稳定性
    print("\n2. 信号稳定性:")
    daily_ret = []
    for date, group in buy_df.groupby('date'):
        if len(group) > 0:
            daily_ret.append(group['future_ret'].mean())

    if daily_ret:
        mean_ret = np.mean(daily_ret)
        std_ret = np.std(daily_ret)
        print(f"   日均收益: {mean_ret*100:.2f}%")
        print(f"   收益标准差: {std_ret*100:.2f}%")
        print(f"   日度Sharpe: {mean_ret/std_ret:.3f}")

    # 3. 分析为什么Sharpe只有0.516
    print("\n3. Sharpe差距分析:")
    print(f"   当前Sharpe: 0.516")
    print(f"   目标Sharpe: 1.0")
    print(f"   差距: {1.0 - 0.516:.3f}")
    print(f"   需要提升: {(1.0/0.516 - 1)*100:.1f}%")


def main():
    df = load_data()
    print(f"数据量: {len(df):,} 条")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    analyze_factor_layer(df)
    analyze_signal_layer(df)
    analyze_portfolio_layer(df)
    analyze_sharpe_breakdown(df)

    # 目标拆解
    print("\n" + "=" * 80)
    print("【目标拆解】Sharpe > 1.0")
    print("=" * 80)
    print("""
根据量化策略理论，Sharpe = IC × sqrt(N) × 胜率提升
- IC: 因子预测能力
- N: 持仓数量
- 胜率提升: 选股带来的额外收益

当前状态:
- IC = 5.08% (目标 > 5%)
- 买入准确率 = 51% (目标 > 55%)
- IR = 1.93 (目标 > 0.5)

关键瓶颈:
1. 买入信号IC ≈ 0 (筛选破坏了预测能力)
2. 买入准确率51% < 目标55%
3. Score高分股票表现反而差
""")


if __name__ == '__main__':
    main()