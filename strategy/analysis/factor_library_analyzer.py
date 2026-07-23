#!/usr/bin/env python
# analysis/factor_library_analyzer.py
"""
因子库审计分析器 — 读取 factor_library.parquet, 输出因子质量报告.

用法:
    python analysis/factor_library_analyzer.py
    python analysis/factor_library_analyzer.py --parquet cache/factor_library.parquet --top 20
    python analysis/factor_library_analyzer.py --industry 电子 --json
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# 项目路径
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STRATEGY_DIR = os.path.dirname(_ANALYSIS_DIR)
sys.path.insert(0, _STRATEGY_DIR)

from core.factor_library import FactorStore, FactorLibrary
from core.dynamic_factor_selector import FACTOR_FAMILIES, get_factor_family


def _suffix(factor_name: str) -> str:
    """提取因子后缀: _F, _FLA, _T, _V, _SM, 或 other"""
    for s in ['_FLA', '_SM', '_F', '_T', '_V']:
        if factor_name.endswith(s):
            return s
    return 'other'


def load(parquet_path: str):
    store = FactorStore(parquet_path)
    lib = FactorLibrary(store)
    return store, lib


def report_general(store: FactorStore, lib: FactorLibrary):
    """总体概况."""
    df = store.df
    if df.empty:
        print("因子库为空, 请先运行回测生成数据.\n")
        return

    n_factors = len(store.get_all_factors())
    n_industries = len(store.get_industries())
    n_records = store.size
    date_range = f"{df['eval_date'].min().date()} ~ {df['eval_date'].max().date()}"
    n_windows = df['eval_date'].nunique()

    print("=" * 70)
    print("  因子库总体概况")
    print("=" * 70)
    print(f"  评估记录数:    {n_records:,}")
    print(f"  因子数:        {n_factors}")
    print(f"  行业数:        {n_industries}")
    print(f"  评估窗口数:    {n_windows}")
    print(f"  日期范围:      {date_range}")
    print()

    # 各 lifecycle 状态分布
    lib.update_lifecycles()
    status_counts = {}
    for fn in store.get_all_factors():
        s = lib.get_status(fn)
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  因子状态分布:")
    for s in ['active', 'candidate', 'decaying', 'retired']:
        print(f"    {s:12s}: {status_counts.get(s, 0)}")
    print()


def report_top_factors(store: FactorStore, lib: FactorLibrary, top_n: int = 15):
    """各行业 top-N 因子."""
    df = store.df
    if df.empty:
        return

    latest_date = store.get_latest_eval_date()
    industries = store.get_industries()

    print("=" * 70)
    print(f"  各行业 Top-{top_n} 因子 (时变加权IC, as_of={latest_date.date()})")
    print("=" * 70)

    for ind in industries:
        selected = lib.select(ind, latest_date + pd.Timedelta(days=1),
                              top_n=top_n, exclude_decaying=False)
        if not selected:
            continue
        print(f"\n  [{ind}]")
        print(f"  {'因子':<35s} {'时变IC':>8s} {'最新IC':>8s} {'斜率':>8s} {'衰减分':>7s} {'状态':>10s} {'家族':>12s}")
        print(f"  {'-'*88}")
        for s in selected[:top_n]:
            print(f"  {s['factor_name']:<35s} {s['time_weighted_ic']:>8.4f} "
                  f"{s['raw_latest_ic']:>8.4f} {s['ic_trend_slope']:>8.4f} "
                  f"{s['decay_score']:>7.3f} {s['status']:>10s} {s['family']:>12s}")
    print()


def report_decay(store: FactorStore, lib: FactorLibrary):
    """衰减因子报告."""
    df = store.df
    if df.empty:
        return

    latest_date = store.get_latest_eval_date()
    all_factors = store.get_all_factors()
    industries = store.get_industries()

    decaying = []
    for fn in all_factors:
        for ind in industries:
            q = lib.get_quality(fn, ind, latest_date + pd.Timedelta(days=1))
            if q is None:
                continue
            if q['status'] in ('decaying', 'retired') or q['decay_score'] < -0.1:
                decaying.append({
                    'factor_name': fn,
                    'industry': ind,
                    'time_weighted_ic': q['time_weighted_ic'],
                    'raw_latest_ic': q['raw_latest_ic'],
                    'ic_trend_slope': q['ic_trend_slope'],
                    'decay_score': q['decay_score'],
                    'status': q['status'],
                    'n_windows': q['n_windows'],
                })

    if not decaying:
        print("  无衰减因子.\n")
        return

    decaying.sort(key=lambda x: x['decay_score'])

    print("=" * 70)
    print(f"  衰减/退役因子 (decay_score < -0.1, 按衰减严重度排序)")
    print("=" * 70)
    print(f"  {'因子':<35s} {'行业':<10s} {'时变IC':>8s} {'最新IC':>8s} {'斜率':>8s} {'衰减分':>7s} {'状态':>10s}")
    print(f"  {'-'*88}")
    for d in decaying[:30]:
        print(f"  {d['factor_name']:<35s} {d['industry']:<10s} {d['time_weighted_ic']:>8.4f} "
              f"{d['raw_latest_ic']:>8.4f} {d['ic_trend_slope']:>8.4f} "
              f"{d['decay_score']:>7.3f} {d['status']:>10s}")
    print(f"\n  共 {len(decaying)} 个因子处于衰减/退役状态.\n")


def report_suffix_analysis(store: FactorStore):
    """按后缀聚合 IC 分布."""
    df = store.df
    if df.empty:
        return

    df = df.copy()
    df['suffix'] = df['factor_name'].apply(_suffix)

    print("=" * 70)
    print("  因子后缀质量分布 (全部评估窗口聚合)")
    print("=" * 70)
    print(f"  {'后缀':<8s} {'因子数':>6s} {'记录数':>8s} {'平均IC':>8s} {'中位IC':>8s} "
          f"{'IC>0.03':>8s} {'ret_spread':>10s}")
    print(f"  {'-'*60}")

    for s in ['_FLA', '_F', '_T', '_V', '_SM', 'other']:
        sub = df[df['suffix'] == s]
        if sub.empty:
            continue
        n_factors = sub['factor_name'].nunique()
        n_records = len(sub)
        avg_ic = sub['ic_mean'].mean()
        med_ic = sub['ic_mean'].median()
        ic_above_03 = (sub['ic_mean'] > 0.03).sum()
        avg_spread = sub['ret_spread'].mean()
        print(f"  {s:<8s} {n_factors:>6d} {n_records:>8,d} {avg_ic:>8.4f} {med_ic:>8.4f} "
              f"{ic_above_03:>8,d} {avg_spread:>10.4f}")
    print()


def report_family_analysis(store: FactorStore):
    """按因子家族聚合 IC."""
    df = store.df
    if df.empty:
        return

    df = df.copy()
    df['family'] = df['factor_name'].apply(get_factor_family)

    print("=" * 70)
    print("  因子家族质量分布")
    print("=" * 70)
    print(f"  {'家族':<16s} {'因子数':>6s} {'记录数':>8s} {'平均IC':>8s} {'中位IC':>8s} "
          f"{'IC>0.03%':>9s} {'ret_spread':>10s}")
    print(f"  {'-'*68}")

    for fam in ['momentum', 'lowvol', 'value', 'quality', 'alpha', 'volume_price', 'sentiment', 'other']:
        sub = df[df['family'] == fam]
        if sub.empty:
            continue
        n_factors = sub['factor_name'].nunique()
        n_records = len(sub)
        avg_ic = sub['ic_mean'].mean()
        med_ic = sub['ic_mean'].median()
        ic_above_03_pct = (sub['ic_mean'] > 0.03).sum() / max(n_records, 1) * 100
        avg_spread = sub['ret_spread'].mean()
        print(f"  {fam:<16s} {n_factors:>6d} {n_records:>8,d} {avg_ic:>8.4f} {med_ic:>8.4f} "
              f"{ic_above_03_pct:>8.1f}% {avg_spread:>10.4f}")
    print()


def report_industry_coverage(store: FactorStore, lib: FactorLibrary):
    """行业覆盖分析: 哪些行业缺好因子."""
    df = store.df
    if df.empty:
        return

    latest_date = store.get_latest_eval_date()
    industries = store.get_industries()

    print("=" * 70)
    print(f"  行业因子覆盖 (as_of={latest_date.date()})")
    print("=" * 70)
    print(f"  {'行业':<12s} {'活跃因子':>8s} {'候选因子':>8s} {'最佳因子':<30s} {'最佳IC':>8s}")
    print(f"  {'-'*70}")

    for ind in sorted(industries):
        selected = lib.select(ind, latest_date + pd.Timedelta(days=1),
                              top_n=10, exclude_decaying=False)
        active = [s for s in selected if s['status'] == 'active']
        n_active = len(active)
        n_candidate = len([s for s in selected if s['status'] == 'candidate'])
        best = selected[0] if selected else None
        best_name = best['factor_name'] if best else '-'
        best_ic = best['raw_latest_ic'] if best else 0
        flag = '⚠' if n_active < 2 else '✓'
        print(f"  {ind:<12s} {n_active:>8d} {n_candidate:>8d} {best_name:<30s} {best_ic:>8.4f}  {flag}")
    print()


def report_json(store: FactorStore, lib: FactorLibrary):
    """JSON 格式输出 (供程序化使用)."""
    df = store.df
    if df.empty:
        print(json.dumps({"error": "empty store"}))
        return

    latest_date = store.get_latest_eval_date()
    result = {
        "store_path": store.store_path,
        "n_records": store.size,
        "n_factors": len(store.get_all_factors()),
        "n_industries": len(store.get_industries()),
        "date_range": [str(df['eval_date'].min().date()), str(df['eval_date'].max().date())],
        "latest_eval_date": str(latest_date.date()) if latest_date else None,
    }

    # 每个因子在各行业的最新评分
    factors = []
    for fn in store.get_all_factors():
        for ind in store.get_industries():
            q = lib.get_quality(fn, ind, latest_date + pd.Timedelta(days=1))
            if q is None:
                continue
            factors.append({
                'factor_name': fn,
                'industry': ind,
                'family': lib.registry.get(fn, {}).get('family', 'other'),
                'suffix': _suffix(fn),
                'status': q['status'],
                'time_weighted_ic': round(q['time_weighted_ic'], 5),
                'raw_latest_ic': round(q['raw_latest_ic'], 5),
                'ic_trend_slope': round(q['ic_trend_slope'], 5),
                'decay_score': round(q['decay_score'], 3),
                'n_windows': q['n_windows'],
            })
    result['factors'] = factors
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    default_parquet = os.path.join(_STRATEGY_DIR, 'cache', 'factor_library.parquet')
    parser = argparse.ArgumentParser(description='因子库审计分析器')
    parser.add_argument('--parquet', default=default_parquet,
                        help=f'parquet 文件路径 (默认: {default_parquet})')
    parser.add_argument('--top', type=int, default=15,
                        help='显示 top-N 因子 (默认: 15)')
    parser.add_argument('--industry', type=str, default=None,
                        help='仅分析指定行业')
    parser.add_argument('--json', action='store_true',
                        help='JSON 格式输出')
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"错误: 因子库文件不存在: {args.parquet}")
        print("请先运行回测生成数据.")
        sys.exit(1)

    store, lib = load(args.parquet)
    if store.df.empty:
        print("因子库为空.")
        sys.exit(0)

    if args.industry:
        # 过滤到指定行业 (创建临时 view)
        pass  # 后续版本支持

    if args.json:
        report_json(store, lib)
        return

    report_general(store, lib)
    report_top_factors(store, lib, args.top)
    report_decay(store, lib)
    report_suffix_analysis(store)
    report_family_analysis(store)
    report_industry_coverage(store, lib)


if __name__ == '__main__':
    main()
