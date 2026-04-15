#!/usr/bin/env python
"""
静态因子离线标定 - 使用已有回测数据快速标定

优化思路：
1. 直接使用factor_preparer预计算的因子数据
2. 不重新计算因子，只做IC分析和因子选择
3. 按行业计算每个因子的IC，选择最优组合
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

from core.config_loader import load_config
from core.industry_mapping import INDUSTRY_KEYWORDS


def load_factor_data():
    """加载预计算的因子数据"""
    from core.factor_preparer import prepare_factor_data
    from core.fundamental import FundamentalData
    from core import data_loader

    print("加载股票数据...")
    stock_data = data_loader.load_stock_data()

    print("加载基本面数据...")
    fd = FundamentalData(os.path.join(os.path.dirname(STRATEGY_DIR), 'data/stock_data/fundamental_data'))

    print("准备因子数据...")
    config = load_config()
    detailed_industries = INDUSTRY_KEYWORDS

    factor_df, industry_codes, all_dates = prepare_factor_data(
        stock_data, fd, detailed_industries, num_workers=8
    )

    return factor_df, industry_codes


def load_cached_factor_data():
    """尝试加载缓存的因子数据"""
    cache_path = os.path.join(STRATEGY_DIR, 'rolling_validation_results/backtest_signals.csv')
    if os.path.exists(cache_path):
        print(f"加载缓存因子数据: {cache_path}")
        return pd.read_csv(cache_path)
    return None


def calc_ic(factor_values, future_returns):
    """计算IC"""
    valid = pd.DataFrame({
        'factor': factor_values,
        'ret': future_returns
    }).dropna()

    if len(valid) < 20:
        return np.nan

    try:
        ic, _ = spearmanr(valid['factor'], valid['ret'])
        return ic
    except:
        return np.nan


def cross_section_standardize(df, factor_col):
    """截面标准化"""
    def zscore(x):
        if len(x) < 5 or x.std() < 1e-10:
            return x * 0
        return (x - x.mean()) / x.std()

    df = df.copy()
    df[factor_col] = df.groupby('date')[factor_col].transform(zscore)
    return df


def analyze_single_factor(df, factor_name, industry_stocks):
    """分析单个因子在各行业的IC"""
    results = {}

    for industry, stocks in industry_stocks.items():
        if len(stocks) < 10:
            continue

        sub = df[df['code'].isin(stocks) & df[factor_name].notna()].copy()
        if len(sub) < 500:
            continue

        # 截面标准化
        sub = cross_section_standardize(sub, factor_name)

        # 计算IC
        ic_values = sub.groupby('date').apply(
            lambda g: calc_ic(g[factor_name], g['future_ret'])
        ).dropna()

        if len(ic_values) < 10:
            continue

        results[industry] = {
            'mean_ic': ic_values.mean(),
            'std_ic': ic_values.std(),
            'ir': ic_values.mean() / ic_values.std() if ic_values.std() > 0 else 0,
            'ic_positive_rate': (ic_values > 0).mean(),
            'n': len(sub)
        }

    return results


def analyze_factor_combination(df, factors, industry_stocks):
    """分析因子组合在各行业的IC"""
    results = {}

    # 创建组合因子
    combo_name = '+'.join(factors)
    valid_mask = df[factors[0]].notna()
    for f in factors[1:]:
        valid_mask &= df[f].notna()

    df = df[valid_mask].copy()
    df[combo_name] = df[factors].mean(axis=1)

    for industry, stocks in industry_stocks.items():
        if len(stocks) < 10:
            continue

        sub = df[df['code'].isin(stocks)].copy()
        if len(sub) < 500:
            continue

        # 截面标准化
        sub = cross_section_standardize(sub, combo_name)

        # 计算IC
        ic_values = sub.groupby('date').apply(
            lambda g: calc_ic(g[combo_name], g['future_ret'])
        ).dropna()

        if len(ic_values) < 10:
            continue

        results[industry] = {
            'factors': factors,
            'mean_ic': ic_values.mean(),
            'std_ic': ic_values.std(),
            'ir': ic_values.mean() / ic_values.std() if ic_values.std() > 0 else 0,
            'ic_positive_rate': (ic_values > 0).mean(),
            'n': len(sub)
        }

    return results


def get_industry_mapping_from_fundamental():
    """从基本面数据获取行业映射"""
    from core.fundamental import FundamentalData
    import os

    data_path = os.path.join(os.path.dirname(STRATEGY_DIR), 'data/stock_data/fundamental_data')
    fd = FundamentalData(data_path)

    industry_stocks = {ind: set() for ind in INDUSTRY_KEYWORDS.keys()}

    # 获取所有股票的行业 - 使用fd.stock_codes或扫描目录
    if hasattr(fd, 'stock_codes') and fd.stock_codes:
        all_codes = fd.stock_codes
    else:
        # 扫描目录获取所有股票代码
        all_codes = [f.replace('.csv', '') for f in os.listdir(data_path) if f.endswith('.csv')]
        all_codes = [c.zfill(6) if len(c) < 6 else c for c in all_codes]

    sample_date = '2024-01-01'

    for code in tqdm(all_codes, desc="获取行业映射"):
        try:
            industry = fd.get_industry(code, sample_date)
            if not industry:
                continue

            for ind_name, keywords in INDUSTRY_KEYWORDS.items():
                for kw in keywords:
                    if kw in str(industry):
                        industry_stocks[ind_name].add(code)
                        break
        except:
            continue

    return industry_stocks


def main():
    print("=" * 80)
    print("静态因子离线标定 - 使用已有数据快速标定")
    print("=" * 80)

    # 获取因子列表
    config = load_config()
    factor_list = config.get('backtest_factors', [])
    print(f"\n因子库: {len(factor_list)} 个因子")

    # 获取行业映射
    print("\n获取行业映射...")
    industry_stocks = get_industry_mapping_from_fundamental()
    for ind, stocks in industry_stocks.items():
        if len(stocks) > 0:
            print(f"  {ind}: {len(stocks)} 只股票")

    # 尝试加载缓存数据
    print("\n尝试加载缓存因子数据...")
    factor_df = load_cached_factor_data()

    if factor_df is None:
        print("无缓存数据，需要重新计算因子...")
        factor_df, _ = load_factor_data()

    if factor_df is None or len(factor_df) == 0:
        print("无法加载因子数据，退出")
        return

    print(f"因子数据: {len(factor_df):,} 条")
    print(f"可用因子: {[c for c in factor_df.columns if c in factor_list]}")

    # 检查是否有future_ret列
    if 'future_ret' not in factor_df.columns:
        print("数据中没有future_ret列，无法计算IC")
        return

    # 分析各因子
    print("\n" + "=" * 80)
    print("分析各因子在各行业的IC")
    print("=" * 80)

    # 找出数据中可用的因子
    available_factors = [f for f in factor_list if f in factor_df.columns]
    print(f"\n可用因子: {len(available_factors)}")

    # 存储每个因子的行业IC结果
    factor_results = {}

    for factor_name in tqdm(available_factors, desc="分析因子"):
        results = analyze_single_factor(factor_df, factor_name, industry_stocks)
        factor_results[factor_name] = results

    # 输出每个行业的最佳因子组合
    print("\n" + "=" * 80)
    print("各行业最佳因子分析")
    print("=" * 80)

    industry_best_factors = {}

    for industry in INDUSTRY_KEYWORDS.keys():
        if industry not in industry_stocks or len(industry_stocks[industry]) < 10:
            continue

        # 收集该行业所有因子的IC
        industry_factor_ics = {}
        for factor_name, results in factor_results.items():
            if industry in results:
                industry_factor_ics[factor_name] = results[industry]

        if not industry_factor_ics:
            continue

        # 按IC排序
        sorted_factors = sorted(industry_factor_ics.items(), key=lambda x: -x[1]['mean_ic'])

        print(f"\n{industry}:")
        print(f"  {'因子':<30} {'IC':>8} {'IR':>6} {'IC>0%':>6}")
        print(f"  {'-'*55}")
        for fname, stats in sorted_factors[:5]:
            print(f"  {fname:<30} {stats['mean_ic']:>+8.4f} {stats['ir']:>6.2f} {stats['ic_positive_rate']*100:>5.1f}%")

        # 选择Top-3因子作为推荐配置
        if len(sorted_factors) >= 3:
            industry_best_factors[industry] = {
                'factors': [f[0] for f in sorted_factors[:3]],
                'ic': sorted_factors[0][1]['mean_ic'],
                'ir': sorted_factors[0][1]['ir']
            }
        elif len(sorted_factors) >= 1:
            industry_best_factors[industry] = {
                'factors': [f[0] for f in sorted_factors],
                'ic': sorted_factors[0][1]['mean_ic'],
                'ir': sorted_factors[0][1]['ir']
            }

    # 输出最终配置
    print("\n" + "=" * 80)
    print("推荐配置 - 可直接复制到 factor_config.yaml")
    print("=" * 80)

    print("\nindustry_factors:")
    for industry in INDUSTRY_KEYWORDS.keys():
        if industry in industry_best_factors:
            best = industry_best_factors[industry]
            factors_str = ', '.join([f"'{f}'" for f in best['factors']])
            print(f"  {industry}:")
            print(f"    factors: [{factors_str}]  # IC={best['ic']:.4f}, IR={best['ir']:.2f}")
        else:
            print(f"  {industry}:")
            print(f"    factors: ['fund_score', 'fund_profit_growth', 'mom_x_lowvol_20_20']  # 默认")

    # 保存结果
    output_file = os.path.join(SCRIPT_DIR, 'static_calibration_results.csv')
    rows = []
    for industry, best in industry_best_factors.items():
        rows.append({
            'industry': industry,
            'factors': '+'.join(best['factors']),
            'best_ic': best['ic'],
            'best_ir': best['ir']
        })

    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
