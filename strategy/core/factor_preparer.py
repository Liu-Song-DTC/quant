# core/factor_preparer.py
"""
因子数据预计算模块

用于动态因子选择前的数据准备
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from multiprocessing import Pool
from tqdm import tqdm

from .config_loader import load_config
from .factors import calc_all_factors_for_validation


def _compute_stock_factors_worker(args):
    """多进程 worker: 计算单只股票的因子数据"""
    code, df, sample_dates, lookback, forward_period = args
    # 使用 datetime 列而非 index
    if 'datetime' in df.columns:
        stock_dates = sorted(df['datetime'].tolist())
    else:
        stock_dates = sorted(df.index.tolist())

    results = []
    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < lookback:
            continue

        eval_date = valid_dates[-1]
        idx = stock_dates.index(eval_date)

        if idx < lookback:
            continue

        history = df.iloc[:idx+1].iloc[-lookback:]
        if len(history) < 60:
            continue

        # 计算因子
        factors = calc_all_factors_for_validation(
            history['close'].values,
            history['high'].values if 'high' in history.columns else history['close'].values,
            history['low'].values if 'low' in history.columns else history['close'].values,
            history['volume'].values if 'volume' in history.columns else np.ones(len(history)),
            fundamental_data=None,  # worker 中不传递 fd
            code=code,
            eval_date=eval_date
        )

        row = {'code': code, 'date': eval_date}
        for fn, vals in factors.items():
            if hasattr(vals, '__len__') and len(vals) > 0:
                val = vals[-1]
            else:
                val = vals
            if val is not None and not np.isnan(val):
                row[fn] = float(val)

        # 计算未来收益
        if idx + forward_period < len(df):
            future_price = df.iloc[idx + forward_period]['close']
            current_price = df.iloc[idx]['close']
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

    # 采样日期（每20天采样一次，减少计算量）
    sample_dates = all_dates[lookback:-forward_period:20]
    print(f"预计算因子数据: {len(sample_dates)} 个时间点, {len(stock_data)} 只股票")
    print(f"行业映射: 未匹配 {unmatched_count}/{len(stock_data)} 只股票")
    for cat, codes in industry_codes.items():
        if codes:
            print(f"  {cat}: {len(codes)} 只")

    # 并行计算因子
    args_list = [
        (code, stock_data[code], sample_dates, lookback, forward_period)
        for code in stock_data.keys()
    ]

    all_factor_data = []
    with Pool(num_workers) as pool:
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
