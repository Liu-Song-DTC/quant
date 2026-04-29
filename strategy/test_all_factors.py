# test_all_factors.py
"""
完整因子IC测试脚本
测试所有技术面和基本面因子的IC效果
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import os

from core.factors import FactorRegistry, calc_base_indicators
from core.fundamental import FundamentalData

DATA_PATH = '../data/stock_data/backtrader_data/'
FUNDAMENTAL_PATH = '../data/stock_data/fundamental_data/'


def load_stock_data(max_stocks=None):
    """加载全部股票数据"""
    if max_stocks is None:
        all_items = [f for f in os.listdir(DATA_PATH)
                    if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv']
    else:
        all_items = [f for f in os.listdir(DATA_PATH)
                    if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv'][:max_stocks]

    all_data = []
    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            if len(df) < 120:
                continue
            df['code'] = code
            all_data.append(df)
        except:
            continue

    return pd.concat(all_data, ignore_index=True)


def load_fundamental_data(stock_codes):
    """加载基本面数据"""
    return FundamentalData(FUNDAMENTAL_PATH, stock_codes)


def calc_ic(values, returns):
    """计算IC"""
    valid = (~np.isnan(values)) & (~np.isnan(returns)) & (np.abs(returns) < 1)
    unique_vals = np.unique(values[valid])
    if len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0}):
        pass
    else:
        valid = valid & (np.abs(values) < 10)
    if valid.sum() < 100:
        return 0
    return np.corrcoef(values[valid], returns[valid])[0, 1]


def test_technical_factors(df):
    """测试技术面因子"""
    print("\n" + "="*60)
    print("技术面因子IC测试")
    print("="*60)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    base_ind = calc_base_indicators(close, high, low, volume)
    future_ret = np.roll(close, -20) / close - 1

    results = []
    for name in FactorRegistry.list_factors():
        func = FactorRegistry.get_factor(name)
        if func is None:
            continue

        try:
            sig = list(func.__code__.co_varnames)[:func.__code__.co_argcount]

            # 根据参数名准备输入数据
            kwargs = {}
            for param in sig:
                if param in base_ind:
                    kwargs[param] = base_ind[param]
                elif param == 'close':
                    kwargs[param] = close
                elif param == 'volume':
                    kwargs[param] = volume
                elif param == 'high':
                    kwargs[param] = high
                elif param == 'low':
                    kwargs[param] = low
                elif param == 'fundamental_score_val':
                    kwargs[param] = 0.5  # 默认值
                elif param in ['roe', 'profit_growth', 'revenue_growth', 'eps', 'debt_ratio', 'gross_margin']:
                    continue  # 跳过基本面参数
                else:
                    continue  # 跳过不支持的参数

            values = func(**kwargs)
            ic = calc_ic(values, future_ret)
            results.append({'name': name, 'ic': ic})
        except Exception as e:
            continue

    # 按IC排序
    results.sort(key=lambda x: abs(x['ic']), reverse=True)

    print(f'{"排名":<4} {"因子名称":<25} {"IC":>10}')
    print('-'*60)
    for i, r in enumerate(results, 1):
        ic = r['ic']
        status = '✓' if abs(ic) > 0.04 else ' '
        print(f'{status}{i:<3} {r["name"]:<25} {ic*100:>9.2f}%')

    return results


def test_fundamental_factors(df, fund_data):
    """测试基本面因子"""
    print("\n" + "="*60)
    print("基本面因子IC测试")
    print("="*60)

    # 计算未来收益
    df = df.sort_values(['code', 'datetime'])
    df['future_ret'] = df.groupby('code')['close'].shift(-20) / df['close'] - 1

    # 提取基本面因子
    fund_factors = []
    for idx, row in df.iterrows():
        code = row['code']
        date = row['datetime']

        fund_factors.append({
            'code': code,
            'date': date,
            'roe': fund_data.get_roe(code, date),
            'profit_growth': fund_data.get_profit_growth(code, date),
            'revenue_growth': fund_data.get_revenue_growth(code, date),
            'eps': fund_data.get_eps(code, date),
            'future_ret': row['future_ret']
        })

    fund_df = pd.DataFrame(fund_factors)

    # 测试各基本面因子
    factors = {
        'ROE': fund_df['roe'].values,
        '净利润增长': fund_df['profit_growth'].values,
        '营收增长': fund_df['revenue_growth'].values,
        'EPS': fund_df['eps'].values,
    }

    results = []
    for name, values in factors.items():
        ic = calc_ic(values, fund_df['future_ret'].values)
        results.append({'name': name, 'ic': ic})

    results.sort(key=lambda x: abs(x['ic']), reverse=True)

    print(f'{"因子名称":<25} {"IC":>10}')
    print('-'*50)
    for r in results:
        ic = r['ic']
        status = '✓' if abs(ic) > 0.02 else ' '
        print(f'{status} {r["name"]:<23} {ic*100:>9.2f}%')

    return results


def test_tech_fund_combo(df, fund_data):
    """测试技术+基本面组合因子"""
    print("\n" + "="*60)
    print("技术面 + 基本面 组合因子IC测试")
    print("="*60)

    df = df.sort_values(['code', 'datetime'])
    df['future_ret'] = df.groupby('code')['close'].shift(-20) / df['close'] - 1
    df['mom_10'] = df.groupby('code')['close'].pct_change(10)
    df['mom_20'] = df.groupby('code')['close'].pct_change(20)

    # 技术因子
    df['tech_factor'] = np.where(
        df['mom_10'] > 0,
        df['mom_20'] * 2.1,
        df['mom_20'] * 0
    )

    # 获取基本面数据
    fund_factors = []
    for idx, row in df.iterrows():
        code = row['code']
        date = row['datetime']
        roe = fund_data.get_roe(code, date)
        fund_factors.append({'roe': roe if roe else 0})

    fund_series = pd.Series([f['roe'] for f in fund_factors])
    df['roe_factor'] = fund_series.values

    # 归一化
    tech_norm = df['tech_factor'].fillna(0)
    fund_norm = df['roe_factor'].fillna(0).clip(0, 30) / 30

    results = []
    for tech_w in [0.6, 0.7, 0.8, 0.9]:
        fund_w = 1 - tech_w
        combo = tech_norm * tech_w + fund_norm * fund_w
        ic = calc_ic(combo.values, df['future_ret'].values)
        results.append({'name': f'技术{tech_w*100:.0f}%+基本面{fund_w*100:.0f}%', 'ic': ic})

    print(f'{"组合":<25} {"IC":>10}')
    print('-'*50)
    for r in results:
        ic = r['ic']
        status = '✓' if abs(ic) > 0.02 else ' '
        print(f'{status} {r["name"]:<23} {ic*100:>9.2f}%')

    return results


if __name__ == "__main__":
    print("加载股票数据...")
    df = load_stock_data()  # 加载全部股票
    print(f"总数据量: {len(df)}")
    stock_codes = df['code'].unique().tolist()
    print(f"股票数: {len(stock_codes)}")

    print("\n加载基本面数据...")
    fund_data = load_fundamental_data(stock_codes)

    # 测试技术面因子
    tech_results = test_technical_factors(df)

    # 测试基本面因子
    fund_results = test_fundamental_factors(df, fund_data)

    # 测试组合因子
    combo_results = test_tech_fund_combo(df, fund_data)

    # 汇总
    print("\n" + "="*60)
    print("汇总: Top 10 因子")
    print("="*60)

    all_results = []
    for r in tech_results:
        all_results.append((r['name'], r['ic'], '技术面'))
    for r in fund_results:
        all_results.append((r['name'], r['ic'], '基本面'))
    for r in combo_results:
        all_results.append((r['name'], r['ic'], '组合'))

    all_results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f'{"排名":<4} {"因子名称":<30} {"IC":>10} {"类型":>10}')
    print('-'*60)
    for i, (name, ic, ftype) in enumerate(all_results[:10], 1):
        status = '✓' if abs(ic) > 0.04 else ' '
        print(f'{status}{i:<3} {name:<30} {ic*100:>9.2f}% {ftype:>10}')
