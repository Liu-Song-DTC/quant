#!/usr/bin/env python
"""
离线因子标定脚本 - 使用滚动窗口验证，防止过拟合

对每个行业的因子组合进行 walk-forward IC 验证
分析 DYN 和 IND 因子的 IC 表现，推荐更好的 IND 因子组合
"""
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

from utils.utils import compute_rolling_ic, compute_ic_stats


def load_validation_data():
    """加载验证数据"""
    df = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/validation_results.csv'))
    df['date'] = pd.to_datetime(df['date'])
    print(f"验证数据: {len(df):,} 条, 日期范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def analyze_factor_combinations(df, industries, factor_pool, min_ic=0.03, min_periods=100):
    """分析因子组合的滚动IC表现

    Args:
        df: 验证数据
        industries: 要分析的行业列表
        factor_pool: 可选的因子列表 (从backtest_factors获取)
        min_ic: 最低平均IC门槛
        min_periods: 最少样本数
    """
    results = {}

    for industry in industries:
        print(f"\n{'='*60}")
        print(f"行业: {industry}")
        print('='*60)

        industry_df = df[df['industry'] == industry].copy()

        if len(industry_df) < min_periods:
            print(f"  数据不足 (n={len(industry_df)})")
            continue

        # 获取该行业所有可用的因子名称 (DYN和IND)
        available_factors = industry_df['factor_name'].unique()

        # 按因子类型分组
        dyn_factors = [f for f in available_factors if f.startswith('DYN_')]
        ind_factors = [f for f in available_factors if f.startswith('IND_')]

        # 分析现有DYN因子的IC (按年度)
        print(f"\n  现有DYN因子IC (年度):")
        dyn_results = []
        for fn in dyn_factors:
            subset = industry_df[industry_df['factor_name'] == fn]
            rolling_ics = compute_rolling_ic(subset, min_periods=min_periods, window='year')
            if rolling_ics:
                mean_ic = np.mean([r['ic'] for r in rolling_ics])
                std_ic = np.std([r['ic'] for r in rolling_ics])
                ir = mean_ic / (std_ic + 1e-10) if std_ic > 0 else 0
                n_years = len(rolling_ics)

                # 提取因子信息
                factor_info = parse_factor_name(fn)

                dyn_results.append({
                    'name': fn,
                    'mean_ic': mean_ic,
                    'std_ic': std_ic,
                    'ir': ir,
                    'n_years': n_years,
                    'factors': factor_info['factors'],
                    'factor_count': factor_info['factor_count'],
                    'has_fundamental': factor_info['has_fundamental']
                })

        # 按IC排序
        dyn_results.sort(key=lambda x: -x['mean_ic'])

        print(f"  {'因子名':<25} {'IC均值':>8} {'IC标准差':>8} {'IR':>6} {'年数':>4} {'因子数':>4}")
        print(f"  {'-'*60}")
        for r in dyn_results[:10]:
            print(f"  {r['name']:<25} {r['mean_ic']:>+8.4f} {r['std_ic']:>8.4f} {r['ir']:>6.2f} {r['n_years']:>4} {r['factor_count']:>4}")

        # 分析现有IND因子
        print(f"\n  现有IND因子IC (年度):")
        ind_results = []
        for fn in ind_factors:
            subset = industry_df[industry_df['factor_name'] == fn]
            rolling_ics = compute_rolling_ic(subset, min_periods=min_periods, window='year')
            if rolling_ics:
                mean_ic = np.mean([r['ic'] for r in rolling_ics])
                std_ic = np.std([r['ic'] for r in rolling_ics])
                ir = mean_ic / (std_ic + 1e-10) if std_ic > 0 else 0
                n_years = len(rolling_ics)

                factor_info = parse_factor_name(fn)

                ind_results.append({
                    'name': fn,
                    'mean_ic': mean_ic,
                    'std_ic': std_ic,
                    'ir': ir,
                    'n_years': n_years,
                    'factors': factor_info['factors'],
                    'factor_count': factor_info['factor_count']
                })

        ind_results.sort(key=lambda x: -x['mean_ic'])

        print(f"  {'因子名':<25} {'IC均值':>8} {'IC标准差':>8} {'IR':>6} {'年数':>4}")
        print(f"  {'-'*60}")
        for r in ind_results:
            print(f"  {r['name']:<25} {r['mean_ic']:>+8.4f} {r['std_ic']:>8.4f} {r['ir']:>6.2f} {r['n_years']:>4}")

        results[industry] = {
            'dyn_factors': dyn_results,
            'ind_factors': ind_results
        }

    return results


def parse_factor_name(fn):
    """解析因子名称，提取因子信息"""
    info = {
        'factors': [],
        'factor_count': 0,
        'has_fundamental': False,
        'factor_type': 'unknown'
    }

    # 解析 DYN_化工_1F_F 格式
    if fn.startswith('DYN_'):
        parts = fn.split('_')
        if len(parts) >= 3:
            # 获取因子数量和类型
            factor_part = parts[2]  # e.g., "1F" or "2F"
            if 'F' in factor_part:
                info['has_fundamental'] = 'F' in factor_part
                info['factor_count'] = int(factor_part[0])
                info['factor_type'] = 'F' if 'F' in factor_part else 'T'

    return info


def recommend_factors(results):
    """基于分析结果推荐因子组合"""
    print("\n" + "="*80)
    print("因子推荐")
    print("="*80)

    recommendations = {}

    for industry, data in sorted(results.items()):
        print(f"\n{industry}:")

        # 获取最佳DYN因子
        if data['dyn_factors']:
            best_dyn = data['dyn_factors'][0]
            print(f"  最佳DYN: {best_dyn['name']} (IC={best_dyn['mean_ic']:+.4f}, IR={best_dyn['ir']:.2f})")

        # 获取最佳IND因子
        if data['ind_factors']:
            best_ind = data['ind_factors'][0]
            print(f"  当前IND: {best_ind['name']} (IC={best_ind['mean_ic']:+.4f})")

            # 检查是否需要更新
            if data['dyn_factors'] and best_dyn['mean_ic'] - best_ind['mean_ic'] > 0.05:
                print(f"  ⚠️ DYN优于IND超过5%，建议检查IND配置")

        recommendations[industry] = {
            'best_dyn': data['dyn_factors'][0] if data['dyn_factors'] else None,
            'best_ind': data['ind_factors'][0] if data['ind_factors'] else None
        }

    return recommendations


def main():
    print("="*80)
    print("离线因子标定 - 滚动窗口验证")
    print("="*80)

    # 加载数据
    df = load_validation_data()

    # 分析的行业
    industries = sorted([x for x in df['industry'].unique() if isinstance(x, str)])

    # 分析因子组合
    results = analyze_factor_combinations(
        df,
        industries,
        factor_pool=None,
        min_ic=0.03,
        min_periods=100
    )

    # 推荐因子
    recommendations = recommend_factors(results)

    return results, recommendations


if __name__ == '__main__':
    main()