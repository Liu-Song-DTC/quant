#!/usr/bin/env python
"""
静态因子挖掘脚本 - 基于验证数据分析
分析各行业表现最好的因子组合，更新静态因子配置
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
sys.path.insert(0, STRATEGY_DIR)


def load_validation_data():
    """加载验证数据"""
    print("加载验证数据...")
    df = pd.read_csv(os.path.join(STRATEGY_DIR, 'rolling_validation_results/validation_results.csv'))
    print(f"验证数据: {len(df):,} 条")
    return df


def analyze_industry_factors(df):
    """分析各行业因子表现"""
    print("\n=== 分析各行业因子表现 ===")

    results = {}

    for industry in tqdm(df['industry'].unique(), desc="分析行业"):
        if pd.isna(industry) or industry == '':
            continue

        industry_df = df[df['industry'] == industry]

        if len(industry_df) < 1000:
            continue

        industry_results = {}

        # 分析每个因子名称
        for fn in industry_df['factor_name'].unique():
            if pd.isna(fn):
                continue

            sub = industry_df[industry_df['factor_name'] == fn]

            if len(sub) < 100:
                continue

            # 计算IC
            sub_clean = sub[['factor_value', 'future_ret']].dropna()
            if len(sub_clean) > 50:
                ic, _ = stats.spearmanr(sub_clean['factor_value'], sub_clean['future_ret'])

                if not np.isnan(ic):
                    # 计算买入准确率
                    buy_sub = sub_clean[sub['buy'] == True] if 'buy' in sub.columns else sub_clean
                    buy_acc = (buy_sub['future_ret'] > 0).mean() if len(buy_sub) > 0 else 0

                    industry_results[fn] = {
                        'ic': ic,
                        'accuracy': buy_acc,
                        'n': len(sub_clean)
                    }

        if industry_results:
            results[industry] = industry_results

    return results


def select_best_factors(results, min_ic=0.03):
    """选择每个行业的最优因子组合"""
    print("\n=== 选择最优因子组合 ===")

    best_factors = {}

    for industry, factor_results in results.items():
        # 过滤：IC > min_ic
        valid_factors = {
            fn: r for fn, r in factor_results.items()
            if r['ic'] > min_ic
        }

        if not valid_factors:
            # 降低门槛
            valid_factors = {
                fn: r for fn, r in factor_results.items()
                if r['ic'] > 0.01
            }

        # 按IC排序
        sorted_factors = sorted(valid_factors.items(), key=lambda x: x[1]['ic'], reverse=True)

        # 取Top-3
        best_factors[industry] = {
            'factors': [fn for fn, r in sorted_factors[:3]],
            'details': {fn: r for fn, r in sorted_factors[:3]}
        }

    return best_factors


def print_results(best_factors):
    """打印结果"""
    print("\n" + "=" * 80)
    print("各行业最优因子组合")
    print("=" * 80)

    for industry, data in sorted(best_factors.items()):
        factors = data['factors']
        details = data['details']

        print(f"\n{industry}:")
        for fn in factors:
            r = details[fn]
            print(f"  {fn}: IC={r['ic']*100:.2f}%, 准确率={r['accuracy']*100:.2f}%, n={r['n']:,}")


def main():
    # 加载验证数据
    df = load_validation_data()

    # 分析各行业因子
    results = analyze_industry_factors(df)

    # 选择最优因子
    best_factors = select_best_factors(results)

    # 打印结果
    print_results(best_factors)

    # 统计各行业因子类型偏好
    print("\n" + "=" * 80)
    print("因子类型偏好分析")
    print("=" * 80)

    for industry, data in sorted(best_factors.items()):
        factors = data['factors']
        has_f = any('_F' in f for f in factors)
        has_t = any('_T' in f for f in factors)
        has_3f = any('3F' in f for f in factors)

        pref = []
        if has_f:
            pref.append('基本面因子')
        if has_t:
            pref.append('纯技术因子')
        if has_3f:
            pref.append('3因子组合')

        print(f"{industry}: {', '.join(pref) if pref else '无偏好'}")

    return best_factors


if __name__ == '__main__':
    main()