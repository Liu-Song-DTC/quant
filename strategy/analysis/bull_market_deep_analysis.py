#!/usr/bin/env python
"""
牛市信号深度分析 - 找出有效的牛市策略
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.market_regime_detector import MarketRegimeDetector


def load_data():
    """加载数据"""
    val_path = Path(__file__).parent.parent / 'rolling_validation_results' / 'validation_results.csv'
    val_df = pd.read_csv(val_path)
    val_df['date'] = pd.to_datetime(val_df['date'])

    index_path = Path(__file__).parent.parent.parent / 'data' / 'stock_data' / 'raw_data' / 'sh000001'
    files = list(index_path.glob('*.csv'))
    index_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    index_df['datetime'] = pd.to_datetime(index_df['date'])
    index_df = index_df.sort_values('datetime').reset_index(drop=True)

    detector = MarketRegimeDetector()
    regime_df = detector.generate(index_df)
    return val_df, regime_df


def main():
    val_df, regime_df = load_data()

    regime_df['date_only'] = regime_df['datetime'].dt.date
    val_df['date_only'] = val_df['date'].dt.date

    merged = val_df.merge(
        regime_df[['date_only', 'regime', 'momentum_score', 'trend_score']],
        left_on='date_only',
        right_on='date_only',
        how='left'
    )
    merged['regime'] = merged['regime'].fillna(0)

    print("=" * 70)
    print("牛市策略挖掘")
    print("=" * 70)

    bull_df = merged[merged['regime'] == 1].copy()
    print(f"牛市总信号数: {len(bull_df)}")

    # ===== 策略1: 高动量 + 中等因子值 =====
    print("\n【策略1: 高动量(>0.6) + 中等因子值(0.7-0.95)】")
    s1 = bull_df[(bull_df['momentum_score'] > 0.6) &
                 (bull_df['factor_value'] > 0.7) &
                 (bull_df['factor_value'] < 0.95) &
                 (bull_df['buy'] == True)]
    if len(s1) > 0:
        print(f"  信号数: {len(s1)}")
        print(f"  准确率: {(s1['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s1['future_ret'].mean():.2%}")

    # ===== 策略2: 高动量 + 特定行业 =====
    print("\n【策略2: 高动量(>0.5) + 牛市友好行业】")
    bull_friendly = ['半导体/光伏', '电子', '通信/计算机', '有色/钢铁/煤炭/建材']
    s2 = bull_df[(bull_df['momentum_score'] > 0.5) &
                 (bull_df['industry'].isin(bull_friendly)) &
                 (bull_df['buy'] == True)]
    if len(s2) > 0:
        print(f"  信号数: {len(s2)}")
        print(f"  准确率: {(s2['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s2['future_ret'].mean():.2%}")

    # ===== 策略3: 特定因子 =====
    print("\n【策略3: 使用牛市有效因子】")
    bull_factors = ['DYN_电子_1F_F', 'DYN_半导体/_1F_F', 'DYN_通信/计_1F_F']
    s3 = bull_df[(bull_df['factor_name'].isin(bull_factors)) &
                 (bull_df['buy'] == True)]
    if len(s3) > 0:
        print(f"  信号数: {len(s3)}")
        print(f"  准确率: {(s3['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s3['future_ret'].mean():.2%}")

    # ===== 策略4: 中等因子值(避免过热) =====
    print("\n【策略4: 中等因子值(0.5-0.85)避免过热】")
    s4 = bull_df[(bull_df['factor_value'] > 0.5) &
                 (bull_df['factor_value'] < 0.85) &
                 (bull_df['buy'] == True)]
    if len(s4) > 0:
        print(f"  信号数: {len(s4)}")
        print(f"  准确率: {(s4['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s4['future_ret'].mean():.2%}")

    # ===== 策略5: 行业+因子组合 =====
    print("\n【策略5: 牛市友好行业 + 中等因子值(0.6-0.9)】")
    s5 = bull_df[(bull_df['industry'].isin(bull_friendly)) &
                 (bull_df['factor_value'] > 0.6) &
                 (bull_df['factor_value'] < 0.9) &
                 (bull_df['buy'] == True)]
    if len(s5) > 0:
        print(f"  信号数: {len(s5)}")
        print(f"  准确率: {(s5['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s5['future_ret'].mean():.2%}")

    # ===== 策略6: 排除差的行业 =====
    print("\n【策略6: 排除牛市差的行业（自动化、化工、新能源车）】")
    bad_industries = ['自动化/制造', '化工', '新能源车/风电', '电力设备']
    s6 = bull_df[(~bull_df['industry'].isin(bad_industries)) &
                 (bull_df['factor_value'] > 0.5) &
                 (bull_df['factor_value'] < 0.95) &
                 (bull_df['buy'] == True)]
    if len(s6) > 0:
        print(f"  信号数: {len(s6)}")
        print(f"  准确率: {(s6['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s6['future_ret'].mean():.2%}")

    # ===== 策略7: 综合策略 =====
    print("\n【策略7: 综合策略 - 高动量 + 牛市行业 + 中等因子值】")
    s7 = bull_df[(bull_df['momentum_score'] > 0.3) &
                 (bull_df['industry'].isin(bull_friendly)) &
                 (bull_df['factor_value'] > 0.6) &
                 (bull_df['factor_value'] < 0.95) &
                 (bull_df['buy'] == True)]
    if len(s7) > 0:
        print(f"  信号数: {len(s7)}")
        print(f"  准确率: {(s7['future_ret'] > 0).mean():.1%}")
        print(f"  平均收益: {s7['future_ret'].mean():.2%}")

    # ===== 最佳策略总结 =====
    print("\n" + "=" * 70)
    print("最佳策略对比")
    print("=" * 70)

    strategies = [
        ("原始牛市", bull_df[bull_df['buy'] == True]),
        ("策略1: 高动量+中等因子", s1),
        ("策略2: 高动量+牛市行业", s2),
        ("策略3: 牛市有效因子", s3),
        ("策略4: 中等因子值", s4),
        ("策略5: 牛市行业+中等因子", s5),
        ("策略6: 排除差行业", s6),
        ("策略7: 综合策略", s7),
    ]

    results = []
    for name, df in strategies:
        if len(df) > 0:
            results.append({
                '策略': name,
                '信号数': len(df),
                '准确率': (df['future_ret'] > 0).mean(),
                '平均收益': df['future_ret'].mean(),
            })

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # 按准确率排序
    print("\n按准确率排序:")
    results_df_sorted = results_df.sort_values('准确率', ascending=False)
    print(results_df_sorted.to_string(index=False))


if __name__ == '__main__':
    main()
