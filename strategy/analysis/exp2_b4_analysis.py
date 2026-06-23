#!/usr/bin/env python3
"""实验2: B4回调占比分析 — 按年份和市场状态分析B4 vs 非B4信号收益分布"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
import pandas as pd
import numpy as np
from market_regime_detector import MarketRegimeDetector

# ── 1. 加载数据 ─────────────────────────────────────────────
print("=" * 70)
print("实验2: B4回调信号收益分析")
print("=" * 70)

bs = pd.read_csv('strategy/rolling_validation_results/backtest_signals.csv')
val = pd.read_csv('strategy/rolling_validation_results/validation_results.csv')
bs['date'] = pd.to_datetime(bs['date'])
val['date'] = pd.to_datetime(val['date'])

# 只取买入信号并合并
buy_bs = bs[bs['buy'] == True][['code', 'date', 'chan_buy_point', 'gate_quality', 'score',
                                  'mom_60d', 'dist_ma60', 'trend_type', 'signal_level']]
buy_val = val[val['buy'] == True][['code', 'date', 'future_ret', 'industry']]

merged = buy_bs.merge(buy_val, on=['code', 'date'], how='inner')
merged['year'] = merged['date'].dt.year
merged['is_b4'] = merged['chan_buy_point'] == 4

print(f"\n买入信号总数: {len(merged):,}")
print(f"B4 买入信号: {merged['is_b4'].sum():,} ({merged['is_b4'].mean()*100:.1f}%)")
print(f"非B4 买入信号: {(~merged['is_b4']).sum():,}")

# ── 2. 市场状态检测 ──────────────────────────────────────────
index_df = pd.read_csv('data/stock_data/backtrader_data/sh000001_qfq.csv',
                        parse_dates=['datetime'])
detector = MarketRegimeDetector()
regime_df = detector.generate(index_df)
regime_df['date'] = regime_df['datetime'].dt.date
regime_map = dict(zip(regime_df['date'], regime_df['regime']))

merged['regime'] = merged['date'].apply(lambda d: regime_map.get(d.date(), 0))
regime_names = {-1: '熊市', 0: '震荡', 1: '牛市'}

# ── 辅助函数 ─────────────────────────────────────────────────
def analyze_group(df, label):
    b4 = df[df['is_b4']]
    non_b4 = df[~df['is_b4']]
    hit = lambda x: (x > 0).mean() * 100

    results = {
        '信号数': len(df),
        'B4信号数': len(b4),
        'B4占比': f"{len(b4)/len(df)*100:.1f}%" if len(df) > 0 else 'N/A',
        'B4均值收益': f"{b4['future_ret'].mean()*100:.2f}%" if len(b4) > 0 else 'N/A',
        '非B4均值收益': f"{non_b4['future_ret'].mean()*100:.2f}%" if len(non_b4) > 0 else 'N/A',
        'B4胜率': f"{hit(b4['future_ret']):.1f}%" if len(b4) > 0 else 'N/A',
        '非B4胜率': f"{hit(non_b4['future_ret']):.1f}%" if len(non_b4) > 0 else 'N/A',
        'B4中位收益': f"{b4['future_ret'].median()*100:.2f}%" if len(b4) > 0 else 'N/A',
        '非B4中位收益': f"{non_b4['future_ret'].median()*100:.2f}%" if len(non_b4) > 0 else 'N/A',
        'B4_std': f"{b4['future_ret'].std()*100:.2f}%" if len(b4) > 0 else 'N/A',
    }
    return results

def print_table(title, groups):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    cols = ['信号数', 'B4占比', 'B4均值收益', '非B4均值收益', 'B4胜率', '非B4胜率',
            'B4中位收益', '非B4中位收益', 'B4_std']
    header = f"{'分组':<16}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in groups.items():
        row = f"{name:<16}" + "".join(f"{r.get(c, 'N/A'):>12}" for c in cols)
        print(row)

# ── 3. 分析 2a: 按年份 ───────────────────────────────────────
print("\n\n【2a】B4 vs 非B4 各年份收益分布")
print("─" * 50)
year_groups = {str(y): analyze_group(merged[merged['year'] == y], str(y))
               for y in sorted(merged['year'].unique())}
year_groups['全时段'] = analyze_group(merged, '全时段')
print_table("按年份", year_groups)

# ── 4. 分析 2b: 按市场状态 ───────────────────────────────────
print("\n\n【2b】B4 vs 非B4 不同市场状态下收益分布")
print("─" * 50)
regime_groups = {}
for r in [-1, 0, 1]:
    subset = merged[merged['regime'] == r]
    regime_groups[regime_names[r]] = analyze_group(subset, regime_names[r])
regime_groups['全时段'] = analyze_group(merged, '全时段')
print_table("按市场状态", regime_groups)

# ── 5. 交叉分析: 年份 × 市场状态 ──────────────────────────────
print("\n\n【2b-扩展】年份 × 市场状态 交叉分析")
print("─" * 50)
for year in sorted(merged['year'].unique()):
    print(f"\n  ▸ {year}")
    for r in [-1, 0, 1]:
        subset = merged[(merged['year'] == year) & (merged['regime'] == r)]
        if len(subset) < 50:
            continue
        b4 = subset[subset['is_b4']]
        non_b4 = subset[~subset['is_b4']]
        b4_ret = b4['future_ret'].mean() * 100 if len(b4) > 0 else 0
        non_ret = non_b4['future_ret'].mean() * 100 if len(non_b4) > 0 else 0
        diff = b4_ret - non_ret
        marker = "⚠️ " if diff < -1 else "✅" if diff > 0 else "  "
        print(f"     {marker} {regime_names[r]:<4} | 信号:{len(subset):>6,} | "
              f"B4占比:{len(b4)/len(subset)*100:.0f}% | "
              f"B4均值:{b4_ret:>6.2f}% | 非B4均值:{non_ret:>6.2f}% | "
              f"差值:{diff:>+5.2f}%")

# ── 6. B4 内部细分: 按 trend_type 和 signal_level ────────────
print("\n\n【2c-诊断】B4信号内部分析")
print("─" * 50)
b4_signals = merged[merged['is_b4']]

# 按 trend_type 分
trend_map = {2: '上升', 1: '横盘', 0: '无', -2: '下降'}
print("\nB4 按趋势类型 (trend_type):")
for t in [2, 1, 0, -2]:
    sub = b4_signals[b4_signals['trend_type'] == t]
    if len(sub) == 0:
        continue
    hit = (sub['future_ret'] > 0).mean() * 100
    print(f"  {trend_map.get(t, str(t)):<6} | 信号:{len(sub):>7,} ({len(sub)/len(b4_signals)*100:.0f}%) | "
          f"均值:{sub['future_ret'].mean()*100:>6.2f}% | 胜率:{hit:.1f}%")

# 按 signal_level 分
print("\nB4 按信号级别 (signal_level):")
for sl in [3, 2, 1, 0]:
    sub = b4_signals[b4_signals['signal_level'] == sl]
    if len(sub) == 0:
        continue
    hit = (sub['future_ret'] > 0).mean() * 100
    print(f"  level={sl} | 信号:{len(sub):>7,} ({len(sub)/len(b4_signals)*100:.0f}%) | "
          f"均值:{sub['future_ret'].mean()*100:>6.2f}% | 胜率:{hit:.1f}%")

# 按 gate_quality 分位
print("\nB4 按 gate_quality 分位:")
try:
    b4_signals['gate_q'] = pd.qcut(b4_signals['gate_quality'], 4, labels=['Q1(低)', 'Q2', 'Q3', 'Q4(高)'], duplicates='drop')
except ValueError:
    n_bins = pd.qcut(b4_signals['gate_quality'], 4, duplicates='drop').nunique()
    labels = [f'Q{i+1}' for i in range(n_bins)]
    b4_signals['gate_q'] = pd.qcut(b4_signals['gate_quality'], 4, labels=labels, duplicates='drop')
for q in sorted(b4_signals['gate_q'].cat.categories):
    sub = b4_signals[b4_signals['gate_q'] == q]
    hit = (sub['future_ret'] > 0).mean() * 100
    print(f"  {q:<8} | 信号:{len(sub):>7,} | 均值:{sub['future_ret'].mean()*100:>6.2f}% | "
          f"胜率:{hit:.1f}% | gate范围:[{sub['gate_quality'].min():.2f}, {sub['gate_quality'].max():.2f}]")

# ── 7. B4 vs 各买点类型对比 ──────────────────────────────────
print("\n\n【2c-对比】所有买点类型收益对比")
print("─" * 50)
bp_names = {0:'无结构',1:'B1一买',2:'B2二买',3:'B3三买',4:'B4四买',
            5:'B5五买',6:'B6六买',7:'B7七买',8:'B8八买'}
print(f"{'买点':<10} {'信号数':>8} {'占比':>7} {'均值收益':>10} {'胜率':>8} {'中位收益':>10} {'std':>8}")
print("-" * 65)
for bp in sorted(merged['chan_buy_point'].unique()):
    sub = merged[merged['chan_buy_point'] == bp]
    hit = (sub['future_ret'] > 0).mean() * 100
    print(f"{bp_names.get(bp, str(bp)):<10} {len(sub):>8,} {len(sub)/len(merged)*100:>6.1f}% "
          f"{sub['future_ret'].mean()*100:>9.2f}% {hit:>7.1f}% "
          f"{sub['future_ret'].median()*100:>9.2f}% {sub['future_ret'].std()*100:>7.2f}%")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
