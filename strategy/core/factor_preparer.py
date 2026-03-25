# core/factor_preparer.py
"""
因子数据预计算模块

用于动态因子选择前的数据准备
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from multiprocessing import Pool
from tqdm import tqdm

from .config_loader import load_config
from .fundamental import FundamentalData


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
    """多进程 worker: 计算单只股票的因子数据（彻底向量化优化版）"""
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
    vol_arr = df['volume'].values if 'volume' in df.columns else np.ones(n)

    # === 一次性向量化计算所有基础指标 ===
    # 动量指标（完全向量化）
    mom_5 = np.full(n, 0.0)
    mom_10 = np.full(n, 0.0)
    mom_20 = np.full(n, 0.0)
    for i in range(5, n):
        mom_5[i] = close_arr[i] / close_arr[i-5] - 1 if close_arr[i-5] > 0 else 0
    for i in range(10, n):
        mom_10[i] = close_arr[i] / close_arr[i-10] - 1 if close_arr[i-10] > 0 else 0
    for i in range(20, n):
        mom_20[i] = close_arr[i] / close_arr[i-20] - 1 if close_arr[i-20] > 0 else 0
    mom_5 = np.clip(mom_5, -1, 1)
    mom_10 = np.clip(mom_10, -1, 1)
    mom_20 = np.clip(mom_20, -1, 1)

    # 波动率指标（完全向量化）
    returns = np.zeros(n)
    returns[1:] = (close_arr[1:] - close_arr[:-1]) / (close_arr[:-1] + 1e-10)
    volatility_10 = np.full(n, 0.0)
    volatility_20 = np.full(n, 0.0)
    for i in range(10, n):
        volatility_10[i] = np.std(returns[i-10:i], ddof=0)
    for i in range(20, n):
        volatility_20[i] = np.std(returns[i-20:i], ddof=0)

    # RSI（向量化）
    delta = np.diff(close_arr, prepend=close_arr[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        avg_gain[i] = np.mean(gain[i-14:i])
        avg_loss[i] = np.mean(loss[i-14:i])
    rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    # 成交量比率（向量化）
    vol_ma20 = np.full(n, 1.0)
    for i in range(20, n):
        vol_ma20[i] = np.mean(vol_arr[i-20:i])
    volume_ratio = np.clip(vol_arr / (vol_ma20 + 1e-10), 0, 10)

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

        # 使用预计算的技术指标（直接索引取值）
        fund_data = fund_cache.get(sample_date, {})

        row = {'code': code, 'date': sample_date, 'industry': stock_industry}

        # 趋势动量因子
        m10 = mom_10[idx]
        m20 = mom_20[idx]
        if m10 > 0:
            row['trend_mom_v41'] = m20 * 2.1
            row['trend_mom_v24'] = m20 * 2.1
            row['trend_mom_v46'] = m20 * 2.05
        else:
            row['trend_mom_v41'] = 0.0
            row['trend_mom_v24'] = m20 * 0.04
            row['trend_mom_v46'] = 0.0

        # 动量×低波动
        v10 = volatility_10[idx]
        v20 = volatility_20[idx]
        row['mom_x_lowvol_20_20'] = m20 * (-v20)
        row['mom_x_lowvol_20_10'] = m20 * (-v10)
        row['mom_x_lowvol_10_20'] = m10 * (-v20)
        row['mom_x_lowvol_10_10'] = m10 * (-v10)

        # 动量差异
        row['mom_diff_5_20'] = mom_5[idx] - m20
        row['mom_diff_10_20'] = m10 - m20

        # RSI因子
        row['rsi_factor'] = (rsi[idx] - 50) / 100

        # 波动率因子
        row['volatility'] = -v20

        # 成交量因子
        row['volume_ratio'] = volume_ratio[idx]

        # 基本面因子
        if fund_data:
            row['fund_roe'] = fund_data.get('roe')
            row['fund_profit_growth'] = fund_data.get('profit_growth')
            row['fund_revenue_growth'] = fund_data.get('revenue_growth')
            row['fund_score'] = fund_data.get('fund_score')
            row['fund_gross_margin'] = fund_data.get('gross_margin')
            row['fund_cf_to_profit'] = fund_data.get('cf_to_profit')

        # 组合因子
        trend_mom = row.get('trend_mom_v41', 0)
        rsi_f = row.get('rsi_factor', 0)
        fund_score_val = row.get('fund_score', 0) or 0
        row['V41_RSI_915'] = trend_mom * 0.915 + rsi_f * 0.085
        row['tech_fund_combo'] = trend_mom * 0.7 + rsi_f * 0.1 + fund_score_val * 0.2

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