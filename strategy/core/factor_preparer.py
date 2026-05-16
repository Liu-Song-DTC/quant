# core/factor_preparer.py
"""
因子数据预计算模块

用于动态因子选择前的数据准备

注意：当 factor_mode='fixed' 时，此模块不参与计算，可跳过以加速回测
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import multiprocessing
from tqdm import tqdm

from .config_loader import load_config
from .fundamental import FundamentalData
from .factor_calculator import calculate_indicators, compute_composite_factors, get_default_params, compress_fundamental_factor


# 全局变量用于 worker 进程 - 每个 worker 创建一个 FundamentalData 实例供所有股票复用
_worker_fd = None


def _init_factor_worker(fundamental_path, stock_codes):
    """Worker 进程初始化函数 - 每个 worker 只创建一次 FundamentalData"""
    global _worker_fd
    if fundamental_path and os.path.exists(fundamental_path):
        _worker_fd = FundamentalData(fundamental_path, stock_codes)
    else:
        _worker_fd = None


def _preload_fundamental_cache(fd, code, dates):
    """预加载股票基本面数据缓存 — 指针推进法，每季度只计算一次而非每个日期查询一次"""
    cache = {}
    if fd is None or code not in fd.stock_data:
        return cache, None

    df = fd.stock_data[code]
    if '数据可用日期' not in df.columns or len(df) == 0:
        return cache, None

    df = df.copy()
    df['数据可用日期_str'] = df['数据可用日期'].astype(str)
    df = df.sort_values('数据可用日期_str').reset_index(drop=True)

    sorted_dates = sorted(dates)
    report_idx = -1
    n_reports = len(df)
    stock_industry = None

    for d in sorted_dates:
        d_ts = pd.Timestamp(d)
        d_str = d_ts.strftime('%Y%m%d')

        while report_idx + 1 < n_reports and df.iloc[report_idx + 1]['数据可用日期_str'] <= d_str:
            report_idx += 1

        if report_idx >= 0:
            row = df.iloc[report_idx]
            cache_key = d
            if cache_key not in cache:
                # 提取基本面值
                roe = _parse_pct(row.get('净资产收益率'))
                profit_growth = _parse_pct(row.get('净利润-同比增长'))
                revenue_growth = _parse_pct(row.get('营业总收入-同比增长'))
                gross_margin = _parse_pct(row.get('销售毛利率'))

                # 综合评分
                fund_score = 0.0
                if roe is not None:
                    fund_score += min(roe * 100, 30)
                if profit_growth is not None:
                    if profit_growth > 0.5: fund_score += 25
                    elif profit_growth > 0.2: fund_score += 15
                    elif profit_growth > 0: fund_score += 5
                eps = row.get('每股收益')
                if eps is not None:
                    try:
                        eps = float(eps)
                        if eps > 0: fund_score += min(eps * 10, 20)
                    except (ValueError, TypeError): pass
                if revenue_growth is not None:
                    if revenue_growth > 0.3: fund_score += 15
                    elif revenue_growth > 0.1: fund_score += 10

                # cf_to_profit
                cf_to_profit = None
                operating_cf = row.get('xjll_经营性现金流-现金流量净额')
                profit = row.get('lrb_净利润')
                if operating_cf is not None and profit is not None:
                    try:
                        if float(profit) > 0:
                            cf_to_profit = float(operating_cf) / float(profit)
                    except (ValueError, TypeError): pass

                cache[cache_key] = {
                    'roe': roe,
                    'profit_growth': profit_growth,
                    'revenue_growth': revenue_growth,
                    'fund_score': fund_score,
                    'gross_margin': gross_margin,
                    'cf_to_profit': cf_to_profit,
                    'industry': row.get('所处行业'),
                }

        if stock_industry is None and report_idx >= 0:
            stock_industry = df.iloc[report_idx].get('所处行业')

    return cache, stock_industry


def _parse_pct(val):
    """解析百分比值"""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            return float(val.strip('%')) / 100.0
        return float(val)
    except (ValueError, TypeError):
        return None


def _compute_stock_factors_worker(args):
    """多进程 worker: 计算单只股票的因子数据（使用统一的factor_calculator）"""
    global _worker_fd
    code, df, factor_dates, lookback, forward_period = args

    # 使用 datetime 列而非 index
    if 'datetime' in df.columns:
        stock_dates = sorted(df['datetime'].tolist())
    else:
        stock_dates = sorted(df.index.tolist())

    n = len(stock_dates)
    if n < lookback + forward_period:
        return []

    # 一次性提取价格数据
    close_arr = df['close'].values
    high_arr = df['high'].values if 'high' in df.columns else close_arr
    low_arr = df['low'].values if 'low' in df.columns else close_arr
    vol_arr = df['volume'].values if 'volume' in df.columns else np.ones(n)
    open_arr = df['open'].values if 'open' in df.columns else close_arr

    # === 使用统一的因子计算器计算所有基础指标 ===
    params = get_default_params()
    ind = calculate_indicators(close_arr, high_arr, low_arr, vol_arr, params, open_arr=open_arr)

    # === 构建日期到索引的映射（O(1) 查找）===
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    # === 预加载基本面缓存（指针推进法，避免每日期查询DataFrame）===
    fund_cache, stock_industry = _preload_fundamental_cache(_worker_fd, code, factor_dates)

    # === 批量向量化构建结果 ===
    results = []
    for sample_date in factor_dates:
        # O(1) 查找索引
        idx = date_to_idx.get(sample_date)
        if idx is None or idx < lookback:
            continue

        # 使用统一的因子计算器计算组合因子
        fund_data = fund_cache.get(sample_date, {})

        # 先获取压缩后的基本面评分，传给 compute_composite_factors
        compressed_fund_score = 0.0
        if fund_data:
            raw_fund_score = fund_data.get('fund_score', 0) or 0
            if isinstance(raw_fund_score, (int, float)):
                compressed_fund_score = compress_fundamental_factor(raw_fund_score, 'fund_score')

        # 使用 factor_calculator 计算所有组合因子（含 tech_fund_combo）
        combo_factors = compute_composite_factors(ind, idx, fund_score=compressed_fund_score)

        row = {'code': code, 'date': sample_date, 'industry': stock_industry}
        row.update(combo_factors)

        # 基本面因子 - 使用统一压缩函数（与signal_engine一致）
        if fund_data:
            raw_roe = fund_data.get('roe')
            raw_profit_growth = fund_data.get('profit_growth')
            raw_revenue_growth = fund_data.get('revenue_growth')
            raw_fund_score = fund_data.get('fund_score')
            raw_gross_margin = fund_data.get('gross_margin')
            raw_cf_to_profit = fund_data.get('cf_to_profit')

            if raw_roe is not None and isinstance(raw_roe, (int, float)):
                row['fund_roe'] = compress_fundamental_factor(raw_roe, 'fund_roe')
            if raw_profit_growth is not None and isinstance(raw_profit_growth, (int, float)):
                row['fund_profit_growth'] = compress_fundamental_factor(raw_profit_growth, 'fund_profit_growth')
            if raw_revenue_growth is not None and isinstance(raw_revenue_growth, (int, float)):
                row['fund_revenue_growth'] = compress_fundamental_factor(raw_revenue_growth, 'fund_revenue_growth')
            if raw_fund_score is not None and isinstance(raw_fund_score, (int, float)):
                row['fund_score'] = compress_fundamental_factor(raw_fund_score, 'fund_score')
            if raw_gross_margin is not None and isinstance(raw_gross_margin, (int, float)):
                row['fund_gross_margin'] = compress_fundamental_factor(raw_gross_margin, 'fund_gross_margin')
            if raw_cf_to_profit is not None and isinstance(raw_cf_to_profit, (int, float)):
                row['fund_cf_to_profit'] = compress_fundamental_factor(raw_cf_to_profit, 'fund_cf_to_profit')

        # 计算未来收益
        if idx + forward_period < n:
            future_price = close_arr[idx + forward_period]
            current_price = close_arr[idx]
            if current_price > 0:
                row['future_ret'] = (future_price - current_price) / current_price
                results.append(row)

    return results


def prepare_factor_data(stock_data: dict, fd,
                       detailed_industries: dict,
                       num_workers: int = 8) -> Tuple[pd.DataFrame, dict, list]:
    """预计算所有股票的因子数据（用于动态因子选择）

    Args:
        stock_data: {code: DataFrame} 股票历史数据
        fd: FundamentalData 实例
        detailed_industries: 行业分类配置
        num_workers: 并行进程数

    Returns:
        tuple: (factor_data, industry_codes, all_dates)
            - factor_data: 所有股票在所有日期的因子值 DataFrame
            - industry_codes: {category: [codes]} 行业映射
            - all_dates: 所有交易日期列表
    """
    config_loader = load_config()
    lookback = config_loader.get('industry_factor_config.lookback_days', 250)
    forward_period = config_loader.get('dynamic_factor.forward_period', 20)

    # 构建行业映射
    industry_codes = {cat: [] for cat in detailed_industries.keys()}
    unmatched_count = 0
    all_dates = set()
    for df in stock_data.values():
        if 'datetime' in df.columns:
            all_dates.update(df['datetime'].tolist())
        else:
            all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    for code in stock_data.keys():
        matched = False
        try:
            # 从日期范围中部采样（避免太早导致基本面数据未发布，也太晚导致已退市）
            # 行业分类稳定，使用中期日期最大化匹配率
            sample_date = all_dates[len(all_dates) // 2]
            ind = fd.get_industry(code, sample_date) if fd else None
            # 如果中期日期无数据，尝试用更晚的日期
            if ind is None:
                for late_date in reversed(all_dates[-200:]):  # 尝试最后200个日期
                    ind = fd.get_industry(code, late_date) if fd else None
                    if ind:
                        break
            if ind:
                for cat, keywords in detailed_industries.items():
                    if any(kw in str(ind) for kw in keywords):
                        industry_codes[cat].append(code)
                        matched = True
                        break
        except Exception:
            pass
        if not matched:
            unmatched_count += 1

    # 日期采样：每N个交易日采样1次（减少计算量）
    date_step = config_loader.get('dynamic_factor.date_sample_step', 3)
    all_factor_dates = all_dates[lookback:-forward_period]
    factor_dates = all_factor_dates[::date_step]
    del all_factor_dates  # 释放中间列表
    print(f"预计算因子数据: {len(factor_dates)} 个时间点 (每{date_step}日采样), {len(stock_data)} 只股票")
    print(f"行业映射: 未匹配 {unmatched_count}/{len(stock_data)} 只股票")
    for cat, codes in industry_codes.items():
        if codes:
            print(f"  {cat}: {len(codes)} 只")

    # 并行计算因子 - 使用 initializer 让每个 worker 只创建一次 FundamentalData
    fundamental_path = fd.data_path if fd is not None else None
    stock_codes = list(stock_data.keys())
    args_list = [
        (code, stock_data[code], factor_dates, lookback, forward_period)
        for code in stock_data.keys()
    ]

    all_factor_data = []
    # 使用 initializer，每个 worker 创建一个 FundamentalData 实例供所有股票复用
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(num_workers, initializer=_init_factor_worker, initargs=(fundamental_path, stock_codes)) as pool:
        for res in tqdm(pool.imap(_compute_stock_factors_worker, args_list, chunksize=10),
                       total=len(args_list), desc="计算因子"):
            all_factor_data.extend(res)

    del args_list  # 释放参数列表（含DataFrame引用）

    factor_data = pd.DataFrame(all_factor_data) if all_factor_data else pd.DataFrame()
    del all_factor_data  # 释放中间列表内存

    # 数据清洗：过滤极端未来收益
    if 'future_ret' in factor_data.columns:
        original_len = len(factor_data)
        factor_data = factor_data[
            (factor_data['future_ret'] > -0.5) &
            (factor_data['future_ret'] < 0.5)
        ]
        print(f"因子数据: {original_len} 条 -> {len(factor_data)} 条 (过滤极端值 {original_len - len(factor_data)} 条)")

    # 因子中性化（行业+市值剥离）
    # 缠论/结构字段具有绝对含义（非截面相对值），不参与中性化
    _CHAN_STRUCTURAL_FIELDS = {
        # 中枢位置
        'pivot_position', 'pivot_present', 'pivot_zg', 'pivot_zd', 'pivot_zz',
        'pivot_level', 'pivot_count',
        'chan_pivot_present', 'chan_pivot_zg', 'chan_pivot_zd', 'chan_pivot_zz', 'chan_pivot_level',
        'breakout_above_pivot', 'breakout_below_pivot', 'consolidation_zone',
        # 走势结构
        'zhongyin', 'structure_complete', 'alignment_score',
        'trend_type', 'trend_strength',
        # 分型/笔/线段
        'top_fractals', 'bottom_fractals', 'fractal_type',
        'stroke_direction', 'stroke_id', 'stroke_count',
        'segment_direction', 'segment_id', 'segment_count',
        # 买卖点
        'buy_point', 'sell_point', 'buy_confidence', 'sell_confidence',
        'bi_buy_point', 'bi_sell_point', 'bi_buy_confidence', 'bi_sell_confidence', 'bi_td',
        'confirmed_buy', 'confirmed_sell', 'signal_level', 'buy_strength', 'sell_strength',
        'second_buy_point', 'second_buy_confidence', 'second_buy_b1_ref',
        # 背离
        'top_divergence', 'bottom_divergence', 'hidden_top_divergence',
        'hidden_bottom_divergence', 'divergence_active',
        # 分型质量
        'bottom_fractal_quality', 'bottom_fractal_strength',
        'bottom_fractal_vol_ratio', 'bottom_fractal_vol_spike', 'bottom_fractal_ema_dist',
        # 缠论信号强度 + 结构止损
        'chan_buy_score', 'chan_sell_score', 'structure_stop_price',
        # 独立系统 (资金流/情绪)
        'capital_flow_score', 'capital_flow_direction',
        'news_sentiment_score', 'news_sentiment_direction',
        'smart_money_flow',
    }
    neu_config = config_loader.get('factor_neutralization', {})
    if neu_config.get('enabled', False) and len(factor_data) > 0:
        from .factor_neutralizer import neutralize_factor_df
        factor_cols = [c for c in factor_data.columns
                       if c not in ('code', 'date', 'future_ret', 'industry')
                       and c not in _CHAN_STRUCTURAL_FIELDS]
        skipped_chan = len([c for c in factor_data.columns if c in _CHAN_STRUCTURAL_FIELDS])
        print(f"因子中性化: {len(factor_cols)} 个因子列 (跳过{skipped_chan}个缠论/结构字段), "
              f"行业={'✓' if neu_config.get('neutralize_industry') else '✗'}, "
              f"市值={'✓' if neu_config.get('neutralize_market_cap') else '✗'}")
        factor_data = neutralize_factor_df(
            factor_data, factor_cols, industry_col='industry',
            market_cap_col=None, date_col='date',
        )
        # 用中性化后的列替换原始列
        for fc in factor_cols:
            neu_col = f'{fc}_neu'
            if neu_col in factor_data.columns:
                factor_data[fc] = factor_data[neu_col]
        factor_data.drop(columns=[c for c in factor_data.columns if c.endswith('_neu')],
                        inplace=True)

    return factor_data, industry_codes, all_dates