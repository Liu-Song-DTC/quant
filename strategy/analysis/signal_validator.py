# analysis/signal_validator.py
"""
信号系统验证脚本 - 只做评估，不包含策略逻辑

设计原则：
1. 只做评估，不包含策略逻辑
2. 读取回测输出的选股结果进行验证
3. 计算信号对应的未来收益进行验证

三层评估：
1. 因子层 - IC/IR分析
2. 信号层 - 买卖信号分析
3. 组合层 - 实际选股结果分析
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(strategy_dir)
sys.path.insert(0, strategy_dir)

from core.config_loader import load_config

config = load_config(os.path.join(project_root, 'strategy/config/factor_config.yaml'))

# 配置
DATA_PATH = os.path.join(project_root, 'data/stock_data/backtrader_data/')
RESULTS_DIR = os.path.join(strategy_dir, 'rolling_validation_results')
FORWARD_PERIOD = config.config.get('dynamic_factor', {}).get('forward_period', 20)


def normalize_code(code):
    """统一代码格式为6位字符串"""
    code_str = str(code)
    if len(code_str) == 6:
        return code_str
    elif len(code_str) <= 6:
        return code_str.zfill(6)
    return code_str


def load_stock_data():
    """加载股票价格数据"""
    all_items = os.listdir(DATA_PATH)
    stock_data = {}
    for item in tqdm(all_items, desc="加载股票数据"):
        # 支持 _qfq.csv 和 _hfq.csv 两种格式
        if item.endswith('_qfq.csv') and item != 'sh000001_qfq.csv':
            code = item.replace('_qfq.csv', '')
        elif item.endswith('_hfq.csv') and item != 'sh000001_hfq.csv':
            code = item.replace('_hfq.csv', '')
        else:
            continue
        df = pd.read_csv(os.path.join(DATA_PATH, item))
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        stock_data[code] = df
    print(f"加载 {len(stock_data)} 只股票")
    return stock_data


def calculate_future_returns(signals_df, stock_data):
    """计算信号对应的未来收益"""
    print("\n计算未来收益...")
    results = []

    for idx, row in tqdm(signals_df.iterrows(), total=len(signals_df), desc="处理信号"):
        code = normalize_code(row['code'])
        date_str = str(row['date'])[:10]

        if code not in stock_data:
            continue

        df = stock_data[code]

        try:
            if date_str not in df.index:
                continue
            idx_pos = df.index.get_loc(date_str)
        except:
            continue

        if not isinstance(idx_pos, int):
            continue
        if idx_pos + FORWARD_PERIOD >= len(df):
            continue

        future_price = df.iloc[idx_pos + FORWARD_PERIOD]['close']
        current_price = df.iloc[idx_pos]['close']

        if current_price > 0 and future_price > 0:
            future_ret = (future_price - current_price) / current_price
            if abs(future_ret) < 0.5:
                results.append({
                    'date': row['date'],
                    'code': code,
                    'score': row.get('score', 0),
                    'factor_value': row.get('factor_value', 0),
                    'buy': row.get('buy', False),
                    'sell': row.get('sell', False),
                    'industry': row.get('industry', ''),
                    'factor_name': row.get('factor_name', ''),
                    'weight': row.get('weight', 1.0),
                    'future_ret': future_ret
                })

    result_df = pd.DataFrame(results)
    print(f"有效数据: {len(result_df)} 条")
    return result_df


# ==================== 因子层评估 ====================

def evaluate_factor_layer(results_df):
    """因子层评估：分析不同因子的表现"""
    print("\n" + "=" * 70)
    print("【因子层评估】")
    print("=" * 70)

    # 检查是否有因子数据
    has_factor_data = ('factor_name' in results_df.columns and not results_df['factor_name'].isna().all()) or \
                      ('factor_value' in results_df.columns and not results_df['factor_value'].isna().all())

    if not has_factor_data:
        print("因子名称/因子值数据不可用，跳过因子层分析")
        return

    # 还需要factor_value列
    if 'factor_value' not in results_df.columns:
        print("因子值数据不可用，跳过因子层分析")
        return

    # 按因子分组分析
    factor_stats = []
    for factor_name, group in results_df.groupby('factor_name'):
        if pd.isna(factor_name) or len(group) < 100:
            continue

        # 过滤NaN
        valid_data = group.dropna(subset=['factor_value', 'future_ret'])
        if len(valid_data) < 100:
            continue

        ic, p_value = stats.spearmanr(valid_data['factor_value'], valid_data['future_ret'])
        if not np.isnan(ic):
            factor_stats.append({
                'factor': factor_name,
                'ic': ic,
                'p_value': p_value,
                'n': len(group),
                'avg_ret': group['future_ret'].mean() * 100,
                'win_rate': (group['future_ret'] > 0).mean() * 100
            })

    if not factor_stats:
        print("没有足够的因子数据进行分析")
        return

    factor_df = pd.DataFrame(factor_stats)
    factor_df = factor_df.sort_values('ic', ascending=False)

    print(f"\n因子数量: {len(factor_df)}")
    print("\n有效因子 (IC > 0, p < 0.1):")
    valid = factor_df[(factor_df['ic'] > 0) & (factor_df['p_value'] < 0.1)]
    if len(valid) > 0:
        for _, row in valid.head(10).iterrows():
            print(f"  {row['factor']}: IC={row['ic']:.4f}, p={row['p_value']:.3f}, 胜率={row['win_rate']:.1f}%")
    else:
        print("  无")

    print("\n负IC因子 (可能需要排除):")
    invalid = factor_df[factor_df['ic'] < 0].head(5)
    for _, row in invalid.iterrows():
        print(f"  {row['factor']}: IC={row['ic']:.4f}, p={row['p_value']:.3f}")


# ==================== 信号层评估 ====================

def evaluate_signal_layer(results_df):
    """信号层评估：分析买卖信号的表现"""
    print("\n" + "=" * 70)
    print("【信号层评估】")
    print("=" * 70)

    total = len(results_df)
    buy_signals = results_df[results_df['buy'] == True]
    sell_signals = results_df[results_df['sell'] == True]
    no_signal = results_df[(results_df['buy'] == False) & (results_df['sell'] == False)]

    print(f"\n信号分布:")
    print(f"  总观测: {total:,}")
    print(f"  买入信号: {len(buy_signals):,} ({len(buy_signals)/total*100:.1f}%)")
    print(f"  卖出信号: {len(sell_signals):,} ({len(sell_signals)/total*100:.1f}%)")
    print(f"  无信号: {len(no_signal):,} ({len(no_signal)/total*100:.1f}%)")

    # 买入信号分析
    print(f"\n买入信号质量:")
    if len(buy_signals) > 0:
        accuracy = (buy_signals['future_ret'] > 0).mean()
        avg_ret = buy_signals['future_ret'].mean()
        print(f"  数量: {len(buy_signals):,}")
        print(f"  准确率: {accuracy*100:.1f}%")
        print(f"  平均收益: {avg_ret*100:.2f}%")
        print(f"  中位数收益: {buy_signals['future_ret'].median()*100:.2f}%")

    # 卖出信号分析
    print(f"\n卖出信号质量:")
    if len(sell_signals) > 0:
        accuracy = (sell_signals['future_ret'] < 0).mean()
        avg_ret = sell_signals['future_ret'].mean()
        print(f"  数量: {len(sell_signals):,}")
        print(f"  准确率: {accuracy*100:.1f}%")
        print(f"  平均收益: {avg_ret*100:.2f}%")

    # 信号IC分析
    print(f"\n信号IC分析:")
    valid = results_df.dropna(subset=['score', 'future_ret'])
    if len(valid) > 100:
        ic, p_value = stats.spearmanr(valid['score'], valid['future_ret'])
        print(f"  整体IC: {ic:.4f} (p={p_value:.4f})")

        # 按日期汇总IC
        ic_by_date = []
        for date, group in valid.groupby('date'):
            if len(group) >= 10:
                ic_d, _ = stats.spearmanr(group['score'], group['future_ret'])
                if not np.isnan(ic_d):
                    ic_by_date.append(ic_d)

        if ic_by_date:
            mean_ic = np.mean(ic_by_date)
            std_ic = np.std(ic_by_date)
            ir = mean_ic / (std_ic + 1e-10)
            print(f"  日度IC均值: {mean_ic:.4f}")
            print(f"  日度IC标准差: {std_ic:.4f}")
            print(f"  IR: {ir:.4f}")
            print(f"  胜率: {np.mean([1 if x > 0 else 0 for x in ic_by_date])*100:.1f}%")


# ==================== 组合层评估 ====================

def evaluate_portfolio_layer(results_df):
    """组合层评估：分析实际选股策略的表现"""
    print("\n" + "=" * 70)
    print("【组合层评估】")
    print("=" * 70)

    # 检查是否有实际选股数据
    has_weight = 'weight' in results_df.columns and results_df['weight'].notna().any()

    if not has_weight:
        print("没有实际选股数据，跳过组合层分析")
        return

    # 按日期计算组合收益
    print(f"\n实际选股结果分析:")
    portfolio_rets = []
    for date, group in results_df.groupby('date'):
        if len(group) == 0:
            continue
        # 加权收益
        weights = group['weight'].values
        weights = weights / weights.sum()  # 归一化
        ret = (group['future_ret'].values * weights).sum()
        portfolio_rets.append({
            'date': date,
            'ret': ret,
            'n_stocks': len(group)
        })

    if not portfolio_rets:
        print("没有有效的组合收益数据")
        return

    portfolio_df = pd.DataFrame(portfolio_rets)

    # 统计指标
    avg_ret = portfolio_df['ret'].mean()
    std_ret = portfolio_df['ret'].std()
    sharpe = avg_ret / (std_ret + 1e-10) * np.sqrt(12) if std_ret > 0 else 0
    win_rate = (portfolio_df['ret'] > 0).mean()
    avg_n = portfolio_df['n_stocks'].mean()

    print(f"  调仓次数: {len(portfolio_df)}")
    print(f"  平均持仓: {avg_n:.1f} 只")
    print(f"  月均收益: {avg_ret*100:.2f}%")
    print(f"  月收益标准差: {std_ret*100:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  胜率: {win_rate*100:.1f}%")

    # 累计收益
    cumulative = (1 + portfolio_df['ret']).cumprod()
    total_ret = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    print(f"  累计收益: {total_ret*100:.2f}%")

    # 行业分布分析
    if 'industry' in results_df.columns:
        print(f"\n行业分布:")
        industry_counts = results_df['industry'].value_counts()
        for ind, count in industry_counts.head(10).items():
            pct = count / len(results_df) * 100
            print(f"  {ind}: {count} ({pct:.1f}%)")


def main():
    print("=" * 70)
    print("信号系统验证报告")
    print("=" * 70)

    # 加载股票数据
    stock_data = load_stock_data()

    # 1. 加载选股结果（组合层评估用）
    selections_file = os.path.join(RESULTS_DIR, 'portfolio_selections.csv')
    signals_file = os.path.join(RESULTS_DIR, 'backtest_signals.csv')

    # 加载信号数据（因子层和信号层评估用）
    if os.path.exists(signals_file):
        print(f"\n加载信号数据...")
        signals_df = pd.read_csv(signals_file, parse_dates=['date'], low_memory=False)
        print(f"信号数据: {len(signals_df)} 条")
        # 计算未来收益
        signals_results = calculate_future_returns(signals_df, stock_data)
    else:
        signals_results = pd.DataFrame()

    # 加载选股结果（组合层评估用）
    if os.path.exists(selections_file):
        print(f"\n加载选股结果...")
        selections_df = pd.read_csv(selections_file, parse_dates=['date'])
        print(f"选股结果: {len(selections_df)} 条")
        # 计算未来收益
        portfolio_results = calculate_future_returns(selections_df, stock_data)
    else:
        portfolio_results = pd.DataFrame()

    if len(signals_results) == 0 and len(portfolio_results) == 0:
        print("没有有效数据")
        return

    # 4. 三层评估
    # 因子层：使用信号数据
    if len(signals_results) > 0:
        evaluate_factor_layer(signals_results)
        evaluate_signal_layer(signals_results)
    else:
        print("\n" + "=" * 70)
        print("【因子层评估】")
        print("=" * 70)
        print("无信号数据，跳过")
        print("\n" + "=" * 70)
        print("【信号层评估】")
        print("=" * 70)
        print("无信号数据，跳过")

    # 组合层：使用选股结果
    if len(portfolio_results) > 0:
        evaluate_portfolio_layer(portfolio_results)
    else:
        print("\n" + "=" * 70)
        print("【组合层评估】")
        print("=" * 70)
        print("无选股结果，跳过")

    # 5. 保存结果
    if len(signals_results) > 0:
        output_file = os.path.join(RESULTS_DIR, 'validation_results.csv')
        signals_results.to_csv(output_file, index=False)
        print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
