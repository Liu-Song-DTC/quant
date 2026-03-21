# analysis/system_evaluator.py
"""
系统评估脚本 - 统一评估流程

运行方式:
    python system_evaluator.py

输出:
    1. 因子层分析 - IC/IR/因子分布
    2. 信号层分析 - 买卖信号质量
    3. 组合层分析 - 行业分布/选股统计
    4. 回测分析 - 与大盘对比/年度收益
    5. 异常检测 - 异常值/潜在问题
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(strategy_dir)
sys.path.insert(0, strategy_dir)

RESULTS_DIR = os.path.join(strategy_dir, 'rolling_validation_results')


def load_data():
    """加载数据"""
    validation_path = os.path.join(RESULTS_DIR, 'validation_results.csv')
    selections_path = os.path.join(RESULTS_DIR, 'portfolio_selections.csv')

    if not os.path.exists(validation_path):
        print(f"错误: 找不到 {validation_path}")
        print("请先运行: python bt_execution.py")
        return None, None

    df = pd.read_csv(validation_path)
    selections = pd.read_csv(selections_path) if os.path.exists(selections_path) else None

    return df, selections


def analyze_factors(df):
    """因子层分析"""
    print("\n" + "=" * 70)
    print("【一、因子层分析】")
    print("=" * 70)

    # 基本统计
    print("\n1.1 因子值统计:")
    print(f"   总记录数: {len(df):,}")
    print(f"   因子值均值: {df['factor_value'].mean():.6f}")
    print(f"   因子值标准差: {df['factor_value'].std():.6f}")
    print(f"   因子值偏度: {df['factor_value'].skew():.4f}")
    print(f"   因子值峰度: {df['factor_value'].kurtosis():.4f}")

    # 异常值检测
    print("\n1.2 异常值检测:")
    q1 = df['factor_value'].quantile(0.01)
    q99 = df['factor_value'].quantile(0.99)
    extreme_low = (df['factor_value'] < q1).sum()
    extreme_high = (df['factor_value'] > q99).sum()
    print(f"   1%分位数: {q1:.4f}, 99%分位数: {q99:.4f}")
    print(f"   极端低值占比: {extreme_low/len(df)*100:.2f}%")
    print(f"   极端高值占比: {extreme_high/len(df)*100:.2f}%")

    # 因子类型分布
    df['factor_type'] = df['factor_name'].apply(lambda x:
        'INDUSTRY' if 'IND_' in str(x) else
        'DYNAMIC' if 'DYN_' in str(x) else
        'DEFAULT' if 'DEFAULT' in str(x) else 'OTHER')

    print("\n1.3 因子类型IC分析:")
    for ft in ['INDUSTRY', 'DYNAMIC', 'DEFAULT', 'OTHER']:
        subset = df[df['factor_type'] == ft].dropna(subset=['factor_value', 'future_ret'])
        if len(subset) > 100:
            ic, p = stats.spearmanr(subset['factor_value'], subset['future_ret'])
            print(f"   {ft}: IC={ic:.4f}, p={p:.4f}, n={len(subset):,}")


def analyze_signals(df):
    """信号层分析"""
    print("\n" + "=" * 70)
    print("【二、信号层分析】")
    print("=" * 70)

    total = len(df)
    buy_n = (df['buy'] == True).sum()
    sell_n = (df['sell'] == True).sum()

    print("\n2.1 信号分布:")
    print(f"   买入信号: {buy_n:,} ({buy_n/total*100:.1f}%)")
    print(f"   卖出信号: {sell_n:,} ({sell_n/total*100:.1f}%)")
    print(f"   无信号: {total - buy_n - sell_n:,} ({(total-buy_n-sell_n)/total*100:.1f}%)")

    # 买入信号质量
    print("\n2.2 买入信号质量:")
    buy_df = df[df['buy'] == True].dropna(subset=['future_ret'])
    if len(buy_df) > 0:
        print(f"   总数: {len(buy_df):,}")
        print(f"   准确率: {(buy_df['future_ret'] > 0).mean()*100:.1f}%")
        print(f"   平均收益: {buy_df['future_ret'].mean()*100:.2f}%")
        print(f"   中位数收益: {buy_df['future_ret'].median()*100:.2f}%")

    # 卖出信号质量
    print("\n2.3 卖出信号质量:")
    sell_df = df[df['sell'] == True].dropna(subset=['future_ret'])
    if len(sell_df) > 0:
        print(f"   总数: {len(sell_df):,}")
        print(f"   准确率: {(sell_df['future_ret'] < 0).mean()*100:.1f}%")
        print(f"   平均收益: {sell_df['future_ret'].mean()*100:.2f}%")

    # IC分析
    print("\n2.4 IC分析:")
    valid = df.dropna(subset=['factor_value', 'future_ret'])
    if len(valid) > 0:
        ic, p = stats.spearmanr(valid['factor_value'], valid['future_ret'])
        print(f"   整体IC: {ic:.4f} (p={p:.4f})")


def analyze_portfolio(df, selections):
    """组合层分析"""
    print("\n" + "=" * 70)
    print("【三、组合层分析】")
    print("=" * 70)

    if selections is None:
        print("\n   (选股结果文件不存在)")
        return

    print(f"\n3.1 选股统计:")
    print(f"   总选股次数: {len(selections):,}")
    print(f"   覆盖日期: {selections['date'].min()} 到 {selections['date'].max()}")
    print(f"   平均持仓: {len(selections)/selections['date'].nunique():.1f} 只/次")

    # 行业分布
    print("\n3.2 行业分布:")
    ind_dist = selections['industry'].value_counts()
    for ind, cnt in ind_dist.head(10).items():
        pct = cnt/len(selections)*100
        print(f"   {ind}: {cnt} ({pct:.1f}%)")

    # 行业集中度
    top3_pct = ind_dist.head(3).sum() / len(selections) * 100
    print(f"\n   前3行业集中度: {top3_pct:.1f}%")


def analyze_anomalies(df):
    """异常检测"""
    print("\n" + "=" * 70)
    print("【四、异常检测】")
    print("=" * 70)

    issues = []

    # 检查1: 因子值分布异常
    if df['factor_value'].skew() < -5:
        issues.append(f"因子值严重左偏 (偏度={df['factor_value'].skew():.2f})")

    if df['factor_value'].kurtosis() > 100:
        issues.append(f"因子值存在极端尖峰 (峰度={df['factor_value'].kurtosis():.2f})")

    # 检查2: 买入准确率低于50%
    buy_df = df[df['buy'] == True].dropna(subset=['future_ret'])
    if len(buy_df) > 0:
        buy_acc = (buy_df['future_ret'] > 0).mean()
        if buy_acc < 0.50:
            issues.append(f"买入准确率偏低 ({buy_acc*100:.1f}%)")

    # 检查3: 行业集中度
    if 'industry' in df.columns:
        ind_counts = df[df['industry'].notna()]['industry'].value_counts()
        if len(ind_counts) > 0:
            top1_pct = ind_counts.iloc[0] / len(df) * 100
            if top1_pct > 40:
                issues.append(f"单一行业占比过高 ({top1_pct:.1f}%: {ind_counts.index[0]})")

    # 检查4: 未来收益分布
    if df['future_ret'].mean() < 0:
        issues.append(f"未来收益均值为负 ({df['future_ret'].mean()*100:.2f}%)")

    if issues:
        print("\n⚠️ 发现以下问题:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ 未发现明显异常")


def print_summary():
    """打印汇总"""
    print("\n" + "=" * 70)
    print("【五、优化建议汇总】")
    print("=" * 70)
    print("""
优先级建议:
  🔴 P0: 提高买入阈值 (当前0.30, 建议0.25-0.35)
  🔴 P0: 限制单一行业权重 (当前单一行业可达100%)
  🟡 P1: 因子值用tanh压缩极端值
  🟡 P1: 排除负IC行业 (交运、通信/计算机)
  🟢 P2: 调整买卖阈值比例
""")


def main():
    print("=" * 70)
    print("系统评估脚本")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df, selections = load_data()
    if df is None:
        return

    analyze_factors(df)
    analyze_signals(df)
    analyze_portfolio(df, selections)
    analyze_anomalies(df)
    print_summary()


if __name__ == "__main__":
    main()
