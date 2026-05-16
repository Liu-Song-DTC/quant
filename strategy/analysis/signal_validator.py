# analysis/signal_validator.py
"""
信号数据准备脚本 - 计算未来收益

功能：
1. 读取 backtest_signals.csv
2. 加载股票价格数据
3. 计算每条信号的 future_ret
4. 保存到 validation_results.csv

使用方式：
    python analysis/signal_validator.py

注意：此脚本只做数据准备，分析工作由 analysis_framework.py 完成
"""

import os
import sys
import numpy as np
import pandas as pd
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
        except Exception:
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


def main():
    """主函数"""
    print("=" * 70)
    print("信号数据准备 - 计算未来收益")
    print("=" * 70)

    # 检查输入文件
    signals_file = os.path.join(RESULTS_DIR, 'backtest_signals.csv')
    if not os.path.exists(signals_file):
        print(f"错误: 找不到 {signals_file}")
        print("请先运行回测: python bt_execution.py")
        return

    # 加载信号数据
    print(f"\n加载信号数据...")
    signals_df = pd.read_csv(signals_file, parse_dates=['date'], low_memory=False)
    print(f"信号数据: {len(signals_df):,} 条")

    # 加载股票数据
    stock_data = load_stock_data()

    # 计算未来收益
    results_df = calculate_future_returns(signals_df, stock_data)

    # 保存结果
    if len(results_df) > 0:
        output_file = os.path.join(RESULTS_DIR, 'validation_results.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n结果已保存: {output_file}")
        print(f"  - 总记录: {len(results_df):,}")
        print(f"  - 日期范围: {results_df['date'].min()} ~ {results_df['date'].max()}")
    else:
        print("\n警告: 没有有效数据")


if __name__ == '__main__':
    main()
