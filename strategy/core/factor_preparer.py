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
from multiprocessing import Pool
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

    # === 使用统一的因子计算器计算所有基础指标 ===
    params = get_default_params()
    ind = calculate_indicators(close_arr, high_arr, low_arr, vol_arr, params)

    # === 构建日期到索引的映射（O(1) 查找）===
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    # 获取股票的行业分类
    stock_industry = None
    if _worker_fd is not None:
        try:
            if len(factor_dates) > 0:
                stock_industry = _worker_fd.get_industry(code, factor_dates[0])
        except:
            pass

    # === 批量获取基本面数据 ===
    fund_cache = {}
    if _worker_fd is not None:
        for eval_date in factor_dates:
            try:
                fund_cache[eval_date] = {
                    'roe': _worker_fd.get_roe(code, eval_date),
                    'profit_growth': _worker_fd.get_profit_growth(code, eval_date),
                    'revenue_growth': _worker_fd.get_revenue_growth(code, eval_date),
                    'fund_score': _worker_fd.get_fundamental_score(code, eval_date),
                    'gross_margin': _worker_fd.get_gross_margin(code, eval_date),
                    'cf_to_profit': None
                }
                operating_cf = _worker_fd.get_operating_cash_flow(code, eval_date)
                profit = _worker_fd.get_profit(code, eval_date)
                if operating_cf is not None and profit is not None and profit > 0:
                    fund_cache[eval_date]['cf_to_profit'] = operating_cf / profit
            except:
                fund_cache[eval_date] = {}

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
    lookback = config_loader.get('industry_factor_config.lookback_days', 120)
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
            sample_date = all_dates[100]  # 用较早的日期获取行业
            ind = fd.get_industry(code, sample_date) if fd else None
            for cat, keywords in detailed_industries.items():
                if ind and any(kw in str(ind) for kw in keywords):
                    industry_codes[cat].append(code)
                    matched = True
                    break
        except:
            pass
        if not matched:
            unmatched_count += 1

    # 使用所有日期（不采样，确保动态因子覆盖所有交易日）
    factor_dates = all_dates[lookback:-forward_period]
    print(f"预计算因子数据: {len(factor_dates)} 个时间点, {len(stock_data)} 只股票")
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
    with Pool(num_workers, initializer=_init_factor_worker, initargs=(fundamental_path, stock_codes)) as pool:
        for res in tqdm(pool.imap(_compute_stock_factors_worker, args_list, chunksize=10),
                       total=len(args_list), desc="计算因子"):
            all_factor_data.extend(res)

    factor_data = pd.DataFrame(all_factor_data) if all_factor_data else pd.DataFrame()

    # 数据清洗：过滤极端未来收益
    if 'future_ret' in factor_data.columns:
        original_len = len(factor_data)
        factor_data = factor_data[
            (factor_data['future_ret'] > -0.5) &
            (factor_data['future_ret'] < 0.5)
        ]
        print(f"因子数据: {original_len} 条 -> {len(factor_data)} 条 (过滤极端值 {original_len - len(factor_data)} 条)")

    return factor_data, industry_codes, all_dates