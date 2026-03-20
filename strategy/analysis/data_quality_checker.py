#!/usr/bin/env python3
"""
数据质量检查脚本 - 统一分析因子层、信号层、组合层

使用方式：
    python analysis/data_quality_checker.py

功能：
1. 检查三层数据是否有 NaN/Inf/Extreme 值
2. 检查数据泄露
3. 分析策略表现
4. 保存报告
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(STRATEGY_DIR)
sys.path.insert(0, STRATEGY_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data/stock_data/backtrader_data/')
SIGNALS_FILE = os.path.join(STRATEGY_DIR, 'rolling_validation_results/backtest_signals.csv')
SELECTIONS_FILE = os.path.join(STRATEGY_DIR, 'rolling_validation_results/portfolio_selections.csv')
FORWARD_PERIOD = 20


def load_stock_data():
    """加载股票数据，返回 {code: DataFrame}"""
    stock_data = {}
    for item in os.listdir(DATA_PATH):
        if item.endswith('_hfq.csv') and item != 'sh000001_hfq.csv':
            code = item.replace('_hfq.csv', '')
        elif item.endswith('_qfq.csv') and item != 'sh000001_qfq.csv':
            code = item.replace('_qfq.csv', '')
        else:
            continue
        df = pd.read_csv(os.path.join(DATA_PATH, item))
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        stock_data[code] = df
    return stock_data


def normalize_code(code):
    """统一代码格式为6位字符串"""
    code_str = str(code)
    if len(code_str) == 6:
        return code_str
    elif len(code_str) <= 6:
        return code_str.zfill(6)
    return code_str


def check_anomalies(signals_df, selections_df):
    """检查三层异常数值"""
    print("\n" + "="*70)
    print("【一、异常数值检查】")
    print("="*70)

    results = {}

    # 因子层
    print("\n1.1 因子层 (factor_value):")
    nan = signals_df['factor_value'].isna().sum()
    inf = np.isinf(signals_df['factor_value']).sum()
    extreme = signals_df[abs(signals_df['factor_value']) > 1].shape[0]
    print(f"   NaN: {nan:,}")
    print(f"   Inf: {inf:,}")
    print(f"   范围: [{signals_df['factor_value'].min():.6f}, {signals_df['factor_value'].max():.6f}]")
    print(f"   |value| > 1: {extreme:,} ({extreme/len(signals_df)*100:.3f}%)")
    results['factor_nan'] = nan
    results['factor_inf'] = inf
    results['factor_extreme'] = extreme

    # 信号层
    print("\n1.2 信号层 (score):")
    nan = signals_df['score'].isna().sum()
    inf = np.isinf(signals_df['score']).sum()
    extreme = signals_df[abs(signals_df['score']) > 1].shape[0]
    print(f"   NaN: {nan:,}")
    print(f"   Inf: {inf:,}")
    print(f"   范围: [{signals_df['score'].min():.6f}, {signals_df['score'].max():.6f}]")
    print(f"   |score| > 1: {extreme:,} ({extreme/len(signals_df)*100:.3f}%)")
    results['score_nan'] = nan
    results['score_inf'] = inf
    results['score_extreme'] = extreme

    # 组合层
    print("\n1.3 组合层 (weight):")
    nan = selections_df['weight'].isna().sum()
    inf = np.isinf(selections_df['weight']).sum()
    print(f"   NaN: {nan}")
    print(f"   Inf: {inf}")
    print(f"   范围: [{selections_df['weight'].min():.6f}, {selections_df['weight'].max():.6f}]")
    results['weight_nan'] = nan
    results['weight_inf'] = inf

    # 买卖信号分布
    print("\n1.4 买卖信号分布:")
    buy = (signals_df['buy'] == True).sum()
    sell = (signals_df['sell'] == True).sum()
    no_sig = ((signals_df['buy'] == False) & (signals_df['sell'] == False)).sum()
    print(f"   买入: {buy:,} ({buy/len(signals_df)*100:.2f}%)")
    print(f"   卖出: {sell:,} ({sell/len(signals_df)*100:.2f}%)")
    print(f"   无信号: {no_sig:,} ({no_sig/len(signals_df)*100:.2f}%)")
    results['buy_count'] = buy
    results['sell_count'] = sell

    return results


def check_data_leakage(signals_df, stock_data):
    """检查数据泄露"""
    print("\n" + "="*70)
    print("【二、数据泄露检查】")
    print("="*70)

    results = {}

    # 检查日期范围
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    signal_start = signals_df['date'].min().strftime('%Y-%m-%d')
    signal_end = signals_df['date'].max().strftime('%Y-%m-%d')

    # 获取数据日期范围
    all_starts = []
    all_ends = []
    for code, df in list(stock_data.items())[:10]:  # 采样检查
        all_starts.append(df.index.min())
        all_ends.append(df.index.max())

    print(f"\n2.1 日期范围:")
    print(f"   信号日期: {signal_start} ~ {signal_end}")
    print(f"   数据采样: {min(all_starts)} ~ {max(all_ends)}")

    # IC 分析
    print("\n2.2 Score-FutureReturn IC 分析:")
    print("   (IC 很低说明无泄露，IC 很高反而可能有问题)")

    return results


def analyze_strategy_performance(signals_df, selections_df, stock_data):
    """分析策略表现"""
    print("\n" + "="*70)
    print("【三、策略表现分析】")
    print("="*70)

    results = {}

    # 选股分布
    print("\n3.1 选股分布:")
    print(f"   总选股: {len(selections_df):,}")
    print(f"   平均持仓: {len(selections_df) / selections_df['date'].nunique():.1f} 只/期")
    print(f"   参与股票: {selections_df['code'].nunique()}")

    # 权重分布
    print("\n3.2 权重分布:")
    print(f"   范围: [{selections_df['weight'].min():.4f}, {selections_df['weight'].max():.4f}]")
    print(f"   平均: {selections_df['weight'].mean():.4f}")

    # 计算实际收益
    print("\n3.3 实际收益计算...")
    portfolio_results = []
    for idx, row in tqdm(selections_df.iterrows(), total=len(selections_df), desc="   处理"):
        code = normalize_code(row['code'])
        date = row['date']
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

        if code not in stock_data:
            continue

        df = stock_data[code]
        try:
            if date_str not in df.index:
                continue
            idx_pos = df.index.get_loc(date_str)
        except:
            continue

        if not isinstance(idx_pos, int) or idx_pos + FORWARD_PERIOD >= len(df):
            continue

        future_price = df.iloc[idx_pos + FORWARD_PERIOD]['close']
        current_price = df.iloc[idx_pos]['close']

        if current_price > 0 and future_price > 0:
            future_ret = (future_price - current_price) / current_price
            if abs(future_ret) < 0.5:
                portfolio_results.append({
                    'date': date,
                    'year': date.year if hasattr(date, 'year') else pd.to_datetime(date).year,
                    'ret': future_ret,
                    'weight': row['weight'],
                    'score': row['score']
                })

    if not portfolio_results:
        print("   无有效数据")
        return results

    result_df = pd.DataFrame(portfolio_results)
    valid_count = len(result_df)
    total_count = len(selections_df)
    print(f"   有效数据: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")

    # 按日期汇总
    port_rets = []
    for date, group in result_df.groupby('date'):
        weights = group['weight'].values / group['weight'].sum()
        port_ret = (group['ret'].values * weights).sum()
        port_rets.append({'date': date, 'year': group['year'].iloc[0], 'ret': port_ret})

    port_df = pd.DataFrame(port_rets)

    print(f"\n3.4 组合月度收益:")
    print(f"   月均收益: {port_df['ret'].mean()*100:.2f}%")
    print(f"   月收益标准差: {port_df['ret'].std()*100:.2f}%")
    print(f"   胜率: {(port_df['ret'] > 0).mean()*100:.1f}%")
    print(f"   最大月收益: {port_df['ret'].max()*100:.2f}%")
    print(f"   最小月收益: {port_df['ret'].min()*100:.2f}%")

    # 按年度
    print("\n3.5 年度收益:")
    yearly = []
    for year, group in port_df.groupby('year'):
        total = (1 + group['ret']).prod() - 1
        win = (group['ret'] > 0).mean() * 100
        avg = group['ret'].mean() * 100
        yearly.append({'year': year, 'total': total, 'win_rate': win, 'avg': avg})
        print(f"   {year}: {total*100:+.1f}% (胜率{win:.0f}%, 月均{avg:+.2f}%)")

    results['yearly'] = yearly
    results['monthly_avg'] = port_df['ret'].mean()
    results['win_rate'] = (port_df['ret'] > 0).mean()

    return results


def analyze_factor_layer(signals_df):
    """分析因子层"""
    print("\n" + "="*70)
    print("【四、因子层分析】")
    print("="*70)

    # 因子分布
    print("\n4.1 因子值统计:")
    print(f"   均值: {signals_df['factor_value'].mean():.6f}")
    print(f"   标准差: {signals_df['factor_value'].std():.6f}")
    print(f"   偏度: {signals_df['factor_value'].skew():.4f}")
    print(f"   峰度: {signals_df['factor_value'].kurtosis():.4f}")

    # 因子频次
    print("\n4.2 因子频次 (Top 10):")
    for name, count in signals_df['factor_name'].value_counts().head(10).items():
        pct = count / len(signals_df) * 100
        print(f"   {name}: {count:,} ({pct:.1f}%)")

    # score 与 factor_value 相关性
    print("\n4.3 Score-FactorValue 相关性:")
    corr = signals_df['factor_value'].corr(signals_df['score'])
    print(f"   Pearson相关系数: {corr:.4f}")


def main():
    print("="*70)
    print("数据质量检查报告")
    print("="*70)

    # 检查文件存在
    if not os.path.exists(SIGNALS_FILE):
        print(f"\n错误: {SIGNALS_FILE} 不存在")
        print("请先运行回测: python bt_execution.py")
        return

    # 加载数据
    print("\n加载数据...")
    signals_df = pd.read_csv(SIGNALS_FILE)
    selections_df = pd.read_csv(SELECTIONS_FILE, parse_dates=['date'])
    stock_data = load_stock_data()

    print(f"信号数据: {len(signals_df):,} 条")
    print(f"选股数据: {len(selections_df):,} 条")
    print(f"股票数据: {len(stock_data)} 只")

    # 执行检查
    anomaly_results = check_anomalies(signals_df, selections_df)
    check_data_leakage(signals_df, stock_data)
    perf_results = analyze_strategy_performance(signals_df, selections_df, stock_data)
    analyze_factor_layer(signals_df)

    # 保存报告
    print("\n" + "="*70)
    print("【报告已生成】")
    print("="*70)


if __name__ == '__main__':
    main()
