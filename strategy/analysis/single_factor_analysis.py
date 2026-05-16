#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单因子IC分析 - 评估每个因子的独立预测能力

目标：从15个backtest_factors中识别出2-3个真正有预测力的核心因子

方法：
1. 对每个因子独立计算截面Spearman IC（每日）
2. 报告IC均值、ICIR、稳定性、胜率
3. 因子相关性矩阵（识别冗余）
4. 滚动IC序列（识别衰减）
5. 按市场状态分层IC

输出：
- 单因子IC排名表
- 因子相关性矩阵
- 核心因子推荐
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.factor_preparer import prepare_factor_data
from core.fundamental import FundamentalData
from core.stock_pool import get_stock_pool


def safe_spearmanr(x, y):
    """安全的Spearman相关计算"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        return 0.0, 1.0
    x_clean = x[mask]
    y_clean = y[mask]
    if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
        return 0.0, 1.0
    r, p = stats.spearmanr(x_clean, y_clean)
    return r if not np.isnan(r) else 0.0, p if not np.isnan(p) else 1.0


def load_stock_data(bt_dir, stock_codes):
    """加载backtrader格式股票数据"""
    stock_data = {}
    for code in stock_codes:
        fpath = os.path.join(bt_dir, f'{code}_qfq.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            if len(df) >= 250:
                stock_data[code] = df
    return stock_data


def compute_single_factor_ic(factor_df, factor_name, forward_period=20):
    """计算单个因子的截面IC

    Args:
        factor_df: DataFrame with columns [code, date, {factor_name}, future_ret]
        factor_name: 要评估的因子名
        forward_period: 前向收益周期

    Returns:
        dict with IC statistics
    """
    df = factor_df[['code', 'date', factor_name, 'future_ret']].copy()
    df = df.dropna(subset=[factor_name, 'future_ret'])

    if len(df) < 100:
        return None

    dates = sorted(df['date'].unique())
    ic_list = []

    for d in dates:
        day_df = df[df['date'] == d]
        if len(day_df) < 20:  # 至少20只股票才有效
            continue

        ic, _ = safe_spearmanr(day_df[factor_name].values, day_df['future_ret'].values)
        if not np.isnan(ic):
            ic_list.append(ic)

    if len(ic_list) < 30:
        return None

    ic_arr = np.array(ic_list)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr) + 1e-10
    icir = ic_mean / ic_std
    hit_rate = np.mean(ic_arr > 0)
    stability = abs(np.sum(np.sign(ic_arr))) / len(ic_arr)
    n_dates = len(ic_arr)
    t_stat = ic_mean / (ic_std / np.sqrt(n_dates))
    p_value = stats.norm.sf(abs(t_stat))

    # 滚动IC窗口
    window = min(60, n_dates // 3)
    rolling_ic = pd.Series(ic_arr).rolling(window).mean().dropna().values
    ic_decay = rolling_ic[-1] - rolling_ic[0] if len(rolling_ic) > 0 else 0

    return {
        'factor': factor_name,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'hit_rate': hit_rate,
        'stability': stability,
        'n_dates': n_dates,
        't_stat': t_stat,
        'p_value': p_value,
        'ic_decay': ic_decay,
        'ic_series': ic_arr,
    }


def compute_factor_correlation(factor_df, factor_names):
    """计算因子间的截面相关性矩阵"""
    # 对每个日期计算截面Spearman，然后取均值
    corr_matrices = []
    dates = sorted(factor_df['date'].unique())

    for d in dates[::5]:  # 每5天采样一次
        day_df = factor_df[factor_df['date'] == d][['code'] + factor_names].dropna()
        if len(day_df) < 20:
            continue
        corr = day_df[factor_names].corr(method='spearman')
        corr_matrices.append(corr.values)

    if not corr_matrices:
        return pd.DataFrame(index=factor_names, columns=factor_names)

    avg_corr = np.mean(corr_matrices, axis=0)
    return pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)


def analyze_by_regime(factor_df, factor_name, regime_df):
    """按市场状态分层IC"""
    if regime_df is None:
        return None

    results = {}
    df = factor_df[['code', 'date', factor_name, 'future_ret']].dropna()

    for regime in ['bull', 'neutral', 'bear']:
        regime_dates = set(regime_df[regime_df['regime'] == regime]['date'])
        regime_data = df[df['date'].isin(regime_dates)]

        if len(regime_data) < 100:
            continue

        ic_list = []
        for d in sorted(regime_data['date'].unique()):
            day_df = regime_data[regime_data['date'] == d]
            if len(day_df) < 10:
                continue
            ic, _ = safe_spearmanr(day_df[factor_name].values, day_df['future_ret'].values)
            if not np.isnan(ic):
                ic_list.append(ic)

        if len(ic_list) >= 10:
            ic_arr = np.array(ic_list)
            results[regime] = {
                'ic_mean': np.mean(ic_arr),
                'icir': np.mean(ic_arr) / (np.std(ic_arr) + 1e-10),
                'hit_rate': np.mean(ic_arr > 0),
                'n_dates': len(ic_arr),
            }

    return results


def main():
    config = load_config()

    # 路径
    base_dir = Path(__file__).parent.parent.parent
    bt_dir = str(base_dir / 'data' / 'stock_data' / 'backtrader_data')
    fund_dir = str(base_dir / 'data' / 'stock_data' / 'fundamental_data')

    # 从配置获取因子列表
    backtest_factors = config.get('backtest_factors', [])
    if not backtest_factors:
        print("ERROR: backtest_factors is empty in config")
        return

    print("=" * 70)
    print("单因子IC分析")
    print("=" * 70)
    print(f"待分析因子数: {len(backtest_factors)}")
    print(f"因子列表: {backtest_factors}")
    print()

    # 获取股票池：按质量筛选，全部通过筛选的股票纳入
    print("构建股票池...")
    pool_codes = list(get_stock_pool())
    print(f"股票池: {len(pool_codes)} 只")

    # 加载数据
    print("加载股票数据...")
    stock_data = load_stock_data(bt_dir, pool_codes)
    print(f"有效股票: {len(stock_data)} 只")

    # 加载基本面
    fd = FundamentalData(fund_dir, list(stock_data.keys())) if os.path.exists(fund_dir) else None

    # 行业映射（简化版）
    from core.industry_mapping import INDUSTRY_KEYWORDS, get_industry_category
    detailed_industries = INDUSTRY_KEYWORDS

    # 计算因子数据
    print("\n预计算因子数据...")
    forward_period = config.get('dynamic_factor', {}).get('forward_period', 20)
    train_window = config.get('dynamic_factor', {}).get('train_window_days', 250)

    factor_df, industry_codes, all_dates = prepare_factor_data(
        stock_data, fd, detailed_industries, num_workers=6
    )

    if factor_df.empty:
        print("ERROR: factor_df is empty")
        return

    print(f"因子数据: {len(factor_df)} 行, {len(factor_df['date'].unique())} 个日期")

    # 找出factor_df中存在的因子
    available_factors = [f for f in backtest_factors if f in factor_df.columns]
    missing = [f for f in backtest_factors if f not in factor_df.columns]
    if missing:
        print(f"缺失因子（跳过）: {missing}")
    print(f"可用因子: {len(available_factors)}")

    # ============================================================
    # 1. 单因子IC分析
    # ============================================================
    print("\n" + "=" * 70)
    print("【1. 单因子IC排名】")
    print("=" * 70)

    factor_stats = []
    for fn in available_factors:
        stats_result = compute_single_factor_ic(factor_df, fn, forward_period)
        if stats_result:
            factor_stats.append(stats_result)

    # 按ICIR排序
    factor_stats.sort(key=lambda x: x['icir'], reverse=True)

    print(f"\n{'因子名':<25} {'IC均值':>8} {'ICIR':>8} {'胜率':>8} {'稳定性':>8} {'T值':>8} {'日期':>6} {'衰减':>8}")
    print("-" * 95)
    for s in factor_stats:
        print(f"{s['factor']:<25} {s['ic_mean']:>8.4f} {s['icir']:>8.3f} {s['hit_rate']:>7.1%} "
              f"{s['stability']:>7.1%} {s['t_stat']:>8.2f} {s['n_dates']:>6d} {s['ic_decay']:>+8.4f}")

    # ============================================================
    # 2. 因子相关性矩阵
    # ============================================================
    print("\n" + "=" * 70)
    print("【2. 因子截面相关性矩阵（Spearman）】")
    print("=" * 70)

    # 只取top 10因子来保持矩阵可读
    top_factors = [s['factor'] for s in factor_stats[:10]]
    corr_mat = compute_factor_correlation(factor_df, top_factors)

    # 打印相关性
    print(f"\n{'':>20}", end='')
    for f in top_factors:
        print(f"{f[:18]:>20}", end='')
    print()
    for f1 in top_factors:
        print(f"{f1:>20}", end='')
        for f2 in top_factors:
            corr_val = corr_mat.loc[f1, f2] if f1 in corr_mat.index and f2 in corr_mat.columns else np.nan
            print(f"{corr_val:>20.3f}", end='')
        print()

    # ============================================================
    # 3. 冗余检测
    # ============================================================
    print("\n" + "=" * 70)
    print("【3. 冗余因子对（相关性 > 0.7）】")
    print("=" * 70)

    redundant_pairs = []
    for i, f1 in enumerate(top_factors):
        for f2 in top_factors[i+1:]:
            corr = corr_mat.loc[f1, f2] if f1 in corr_mat.index and f2 in corr_mat.columns else 0
            if abs(corr) > 0.7:
                redundant_pairs.append((f1, f2, corr))

    if redundant_pairs:
        redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for f1, f2, corr in redundant_pairs:
            # 标记应该保留哪个
            s1 = next((s for s in factor_stats if s['factor'] == f1), None)
            s2 = next((s for s in factor_stats if s['factor'] == f2), None)
            keep = f1 if (s1 and s2 and s1['icir'] >= s2['icir']) else f2
            drop = f2 if keep == f1 else f1
            print(f"  {f1} <-> {f2}: r={corr:.3f} → 保留 {keep} (ICIR更高), 建议删除 {drop}")
    else:
        print("  无高冗余因子对")

    # ============================================================
    # 4. 核心因子推荐
    # ============================================================
    print("\n" + "=" * 70)
    print("【4. 核心因子推荐】")
    print("=" * 70)

    # 策略: 选ICIR最高且相关性低的2-3个因子
    selected = []
    selected_names = set()

    for s in factor_stats:
        fn = s['factor']
        if fn in selected_names:
            continue
        # 检查与已选因子的相关性
        too_correlated = False
        for sel in selected:
            corr = corr_mat.loc[fn, sel['factor']] if fn in corr_mat.index and sel['factor'] in corr_mat.columns else 0
            if abs(corr) > 0.5:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(s)
            selected_names.add(fn)
        if len(selected) >= 3:
            break

    print("\n推荐核心因子组合（高ICIR + 低相关性）:")
    print()
    for i, s in enumerate(selected):
        fam = '?'
        from core.dynamic_factor_selector import get_factor_family as _get_factor_family
        fam = _get_factor_family(s['factor'])
        print(f"  {i+1}. {s['factor']} (家族: {fam})")
        print(f"     IC={s['ic_mean']:.4f}, ICIR={s['icir']:.3f}, 胜率={s['hit_rate']:.1%}")

    if len(selected) >= 2:
        # 等权组合IC
        combined_ic = []
        dates = sorted(factor_df['date'].unique())
        sel_names = [s['factor'] for s in selected]
        for d in dates:
            day_df = factor_df[factor_df['date'] == d][['code'] + sel_names + ['future_ret']].dropna()
            if len(day_df) < 20:
                continue
            # 等权组合
            combo = day_df[sel_names].mean(axis=1).values
            ic, _ = safe_spearmanr(combo, day_df['future_ret'].values)
            if not np.isnan(ic):
                combined_ic.append(ic)

        if combined_ic:
            combo_arr = np.array(combined_ic)
            combo_ic = np.mean(combo_arr)
            combo_icir = combo_ic / (np.std(combo_arr) + 1e-10)
            combo_hit = np.mean(combo_arr > 0)
            print(f"\n  组合IC: {combo_ic:.4f}, 组合ICIR: {combo_icir:.3f}, 组合胜率: {combo_hit:.1%}")

    # ============================================================
    # 5. IC时间序列稳定性
    # ============================================================
    print("\n" + "=" * 70)
    print("【5. IC滚动稳定性（60日窗口）】")
    print("=" * 70)

    for s in factor_stats[:5]:
        ic_series = s['ic_series']
        if len(ic_series) < 60:
            continue
        rolling = pd.Series(ic_series).rolling(60).mean()
        recent_ic = rolling.dropna().iloc[-1] if len(rolling.dropna()) > 0 else np.nan
        min_ic = rolling.min()
        status = "稳定" if not np.isnan(recent_ic) and recent_ic > 0.02 else "衰减中" if not np.isnan(recent_ic) else "?"
        print(f"  {s['factor']:<25}: 最近60日均IC={recent_ic:.4f}, 最低={min_ic:.4f}, [{status}]")

    # ============================================================
    # 6. 总结
    # ============================================================
    print("\n" + "=" * 70)
    print("总结建议")
    print("=" * 70)

    good_factors = [s for s in factor_stats if s['icir'] > 0.3 and s['ic_mean'] > 0.02]
    medium_factors = [s for s in factor_stats if 0.15 < s['icir'] <= 0.3]
    weak_factors = [s for s in factor_stats if s['icir'] <= 0.15]

    print(f"\n  强因子 (ICIR>0.3, IC>2%): {len(good_factors)} 个")
    print(f"  中等因子 (0.15<ICIR<=0.3): {len(medium_factors)} 个")
    print(f"  弱因子 (ICIR<=0.15): {len(weak_factors)} 个")

    if weak_factors:
        print(f"\n  建议删除的弱因子:")
        for s in weak_factors:
            print(f"    - {s['factor']}: IC={s['ic_mean']:.4f}, ICIR={s['icir']:.3f}")

    if len(selected) >= 2:
        print(f"\n  建议保留的核心因子组合 ({len(selected)}个):")
        for i, s in enumerate(selected):
            print(f"    {i+1}. {s['factor']}")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
