#!/usr/bin/env python
"""
静态因子离线标定 - 基于因子预计算数据

使用factor_preparer预计算的因子数据进行IC分析
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

from core.config_loader import load_config
from core.industry_mapping import INDUSTRY_KEYWORDS


def load_precomputed_factors():
    """加载预计算的因子数据（调用factor_preparer）"""
    from core.fundamental import FundamentalData
    from core.factor_preparer import prepare_factor_data

    # 数据路径
    data_path = os.path.join(os.path.dirname(STRATEGY_DIR), 'data/stock_data/backtrader_data')
    fundamental_path = os.path.join(os.path.dirname(STRATEGY_DIR), 'data/stock_data/fundamental_data')

    print("加载股票数据...")
    stock_data = {}
    for f in os.listdir(data_path):
        if not f.endswith('_qfq.csv'):
            continue
        code = f.replace('_qfq.csv', '')
        df = pd.read_csv(os.path.join(data_path, f))
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        stock_data[code] = df

    print(f"  加载 {len(stock_data)} 只股票")

    print("加载基本面数据...")
    fd = FundamentalData(fundamental_path, list(stock_data.keys()))

    print("预计算因子数据...")
    factor_df, industry_codes, all_dates = prepare_factor_data(
        stock_data, fd, INDUSTRY_KEYWORDS, num_workers=8
    )

    return factor_df, industry_codes


def calc_ic_for_factor(df, factor_col):
    """计算单个因子的IC"""
    def calc_ic(g):
        valid = g[[factor_col, 'future_ret']].dropna()
        if len(valid) < 10:
            return np.nan
        try:
            ic, _ = spearmanr(valid[factor_col], valid['future_ret'])
            return ic
        except:
            return np.nan

    ic_values = df.groupby('date').apply(calc_ic).dropna()
    if len(ic_values) < 5:
        return None

    return {
        'mean_ic': ic_values.mean(),
        'std_ic': ic_values.std(),
        'ir': ic_values.mean() / ic_values.std() if ic_values.std() > 0 else 0,
        'ic_positive_rate': (ic_values > 0).mean(),
        'n_dates': len(ic_values)
    }


def standardize_factor(df, factor_col):
    """截面标准化"""
    def zscore(x):
        if len(x) < 5 or x.std() < 1e-10:
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - x.mean()) / x.std()

    df = df.copy()
    df[factor_col] = df.groupby('date')[factor_col].transform(zscore)
    return df


def analyze_industry(industry_name, factor_df, stocks, factor_list):
    """分析单个行业的最优因子组合"""
    if len(stocks) < 10:
        return None

    # 筛选该行业数据
    ind_df = factor_df[factor_df['code'].isin(stocks)].copy()
    if len(ind_df) < 500:
        return None

    # 获取可用因子
    available = [f for f in factor_list if f in ind_df.columns and ind_df[f].notna().sum() > 100]
    if len(available) < 3:
        return None

    results = []

    # 单因子分析
    for factor_name in available:
        # 截面标准化
        test_df = standardize_factor(ind_df, factor_name)
        ic_result = calc_ic_for_factor(test_df, factor_name)

        if ic_result and abs(ic_result['mean_ic']) > 0.01:
            results.append({
                'factors': [factor_name],
                'n_factors': 1,
                **ic_result
            })

    # 2因子组合
    for combo in combinations(available, 2):
        f1, f2 = combo
        combo_name = f'{f1}+{f2}'

        # 等权组合
        test_df = ind_df[[f1, f2, 'future_ret', 'date', 'code']].dropna()
        if len(test_df) < 500:
            continue

        test_df['combo'] = (test_df[f1] + test_df[f2]) / 2
        test_df = standardize_factor(test_df, 'combo')

        ic_result = calc_ic_for_factor(test_df, 'combo')
        if ic_result and abs(ic_result['mean_ic']) > 0.01:
            results.append({
                'factors': [f1, f2],
                'n_factors': 2,
                **ic_result
            })

    # 3因子组合
    for combo in combinations(available, 3):
        f1, f2, f3 = combo

        test_df = ind_df[[f1, f2, f3, 'future_ret', 'date', 'code']].dropna()
        if len(test_df) < 500:
            continue

        test_df['combo'] = (test_df[f1] + test_df[f2] + test_df[f3]) / 3
        test_df = standardize_factor(test_df, 'combo')

        ic_result = calc_ic_for_factor(test_df, 'combo')
        if ic_result and abs(ic_result['mean_ic']) > 0.01:
            results.append({
                'factors': [f1, f2, f3],
                'n_factors': 3,
                **ic_result
            })

    if not results:
        return None

    # 按IC排序
    results.sort(key=lambda x: -x['mean_ic'])
    return results


def main():
    print("=" * 80)
    print("静态因子离线标定 - 基于因子预计算数据")
    print("=" * 80)

    # 加载因子数据
    factor_df, industry_codes = load_precomputed_factors()

    if factor_df is None or len(factor_df) == 0:
        print("无法加载因子数据")
        return

    print(f"\n因子数据: {len(factor_df):,} 条")

    # 获取因子列表
    config = load_config()
    factor_list = config.get('backtest_factors', [])

    # 过滤掉不是列名的因子
    factor_list = [f for f in factor_list if f in factor_df.columns]
    print(f"可用因子: {len(factor_list)}")

    # 分析各行业
    print("\n" + "=" * 80)
    print("分析各行业因子组合")
    print("=" * 80)

    all_results = {}

    for industry, stocks in tqdm(industry_codes.items(), desc="标定行业"):
        if len(stocks) < 10:
            continue

        results = analyze_industry(industry, factor_df, stocks, factor_list)
        if results:
            all_results[industry] = results

            # 输出Top 5
            print(f"\n{industry} ({len(stocks)} 只股票):")
            print(f"  {'因子':<50} {'IC':>8} {'IR':>6}")
            print(f"  {'-'*70}")
            for r in results[:5]:
                print(f"  {'+'.join(r['factors']):<50} {r['mean_ic']:>+8.4f} {r['ir']:>6.2f}")

    # 输出配置
    print("\n" + "=" * 80)
    print("推荐配置 - 可直接复制到 factor_config.yaml")
    print("=" * 80)

    print("\nindustry_factors:")
    for industry in INDUSTRY_KEYWORDS.keys():
        if industry in all_results and all_results[industry]:
            best = all_results[industry][0]
            factors_str = ', '.join([f"'{f}'" for f in best['factors']])
            print(f"  {industry}:")
            print(f"    factors: [{factors_str}]  # IC={best['mean_ic']:.4f}, IR={best['ir']:.2f}")
        else:
            # 使用默认配置
            print(f"  {industry}:")
            print(f"    factors: ['fund_score', 'fund_profit_growth', 'mom_x_lowvol_20_20']  # 默认")

    # 保存详细结果
    output_file = os.path.join(SCRIPT_DIR, 'static_calibration_results.csv')
    rows = []
    for industry, results in all_results.items():
        for r in results[:10]:
            rows.append({
                'industry': industry,
                'factors': '+'.join(r['factors']),
                'n_factors': r['n_factors'],
                'mean_ic': r['mean_ic'],
                'ir': r['ir'],
                'ic_positive_rate': r['ic_positive_rate']
            })

    if rows:
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
