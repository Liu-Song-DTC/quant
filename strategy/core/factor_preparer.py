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
import ctypes

# 全局题材热度计算器 (fork 前初始化, worker 进程继承)
_worker_concept_calc = None

from .config_loader import load_config
from .fundamental import FundamentalData
from .factor_calculator import calculate_indicators, compute_composite_factors, get_default_params, compress_fundamental_factor
from scipy.special import erfinv


# 全局变量用于 worker 进程 - 每个 worker 创建一个 FundamentalData 实例供所有股票复用
_worker_fd = None


def _init_factor_worker(fd, concept_calc=None):
    """Worker 进程初始化函数 - 共享父进程已加载的 FundamentalData + ConceptHeatCalculator

    使用 fork 后 COW 共享父进程的 FundamentalData，避免每个 worker 从磁盘重新加载
    4000+ 只股票的基本面数据（~8 workers × 4000 CSVs = 32000+ 次文件读取）。
    """
    global _worker_fd, _worker_concept_calc
    _worker_fd = fd
    if concept_calc is not None:
        _worker_concept_calc = concept_calc


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
    """多进程 worker: 计算单只股票的因子数据（从文件读取，避免主进程加载全量数据）"""
    global _worker_fd
    code, filepath, factor_dates, lookback, forward_period = args

    # 从文件读取数据（每个worker只持有一只股票的数据，内存可控）
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
    except Exception:
        return []

    # 使用 datetime 列而非 index
    if 'datetime' in df.columns:
        stock_dates = sorted(df['datetime'].tolist())
    else:
        stock_dates = sorted(df.index.tolist())

    n = len(stock_dates)
    if n < lookback + forward_period:
        return []

    # 一次性提取价格数据（float64 → 计算精度优于 float32）
    close_arr = df['close'].values.astype(np.float64, copy=False)
    high_arr = df['high'].values.astype(np.float64, copy=False) if 'high' in df.columns else close_arr
    low_arr = df['low'].values.astype(np.float64, copy=False) if 'low' in df.columns else close_arr
    vol_arr = df['volume'].values.astype(np.float64, copy=False) if 'volume' in df.columns else np.ones(n)
    open_arr = df['open'].values.astype(np.float64, copy=False) if 'open' in df.columns else close_arr
    turnover_arr = df['turnover_rate'].values.astype(np.float64, copy=False) if 'turnover_rate' in df.columns else None
    del df  # 释放DataFrame，仅保留numpy数组

    # === 使用统一的因子计算器计算所有基础指标 ===
    params = get_default_params()
    ind = calculate_indicators(close_arr, high_arr, low_arr, vol_arr, params,
                               open_arr=open_arr, turnover_rate=turnover_arr)

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

        row = {'code': code, 'date': sample_date, 'industry': stock_industry or ''}
        row.update(combo_factors)

        # 市值因子 (size): A股小盘效应, 流通市值≈成交额/换手率
        if idx < n:
            amt = float(ind['amount'][idx]) if 'amount' in ind and idx < len(ind['amount']) else 0
            to_rate = float(ind['turnover_rate'][idx]) if 'turnover_rate' in ind and idx < len(ind['turnover_rate']) else 0
            if amt > 0 and to_rate > 0.001:
                est_circ_mv = amt / (to_rate / 100.0)
                row['ln_cap'] = float(np.tanh(np.log(max(est_circ_mv, 1e6)) / 10.0 - 0.5))
            else:
                row['ln_cap'] = 0.0

        # 题材热度因子 - 从真实概念数据计算
        global _worker_concept_calc
        if _worker_concept_calc is not None:
            try:
                _worker_concept_calc.set_daily_data(sample_date)
                row['concept_heat'] = _worker_concept_calc.get_concept_heat(code)
            except Exception:
                row['concept_heat'] = 0.5
        elif 'concept_heat' in ind and idx < len(ind['concept_heat']):
            row['concept_heat'] = float(ind['concept_heat'][idx])

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


def prepare_factor_data(stock_file_map: dict, fd,
                       detailed_industries: dict,
                       all_dates: list,
                       num_workers: int = 8) -> Tuple[pd.DataFrame, dict, list]:
    """预计算所有股票的因子数据（用于动态因子选择）

    Args:
        stock_file_map: {code: filepath} 股票文件路径映射（worker从磁盘读取，避免主进程加载全量数据）
        fd: FundamentalData 实例
        detailed_industries: 行业分类配置
        all_dates: 全市场交易日列表（可从sh000001获取，避免遍历所有股票）
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

    # 构建行业映射（使用fundamental_data，无需加载价格数据）
    industry_codes = {cat: [] for cat in detailed_industries.keys()}
    unmatched_count = 0
    mid_date = all_dates[len(all_dates) // 2] if all_dates else None

    for code in stock_file_map.keys():
        matched = False
        try:
            ind = fd.get_industry(code, mid_date) if fd else None
            # 如果中期日期无数据，尝试用更晚的日期
            if ind is None and len(all_dates) > 200:
                for late_date in reversed(all_dates[-200:]):
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
    print(f"预计算因子数据: {len(factor_dates)} 个时间点 (每{date_step}日采样), {len(stock_file_map)} 只股票")

    # === 因子数据缓存: 避免重复计算 ===
    from .cache_manager import load_factor_cache, save_factor_cache
    import hashlib
    _n_stocks = len(stock_file_map)
    _n_dates = len(factor_dates)
    _cache_key_str = ','.join(sorted(stock_file_map.keys())[:100]) + str(_n_stocks) + \
                     str(factor_dates[0]) + str(factor_dates[-1]) + str(date_step) + str(lookback)
    _cache_hash = hashlib.md5(_cache_key_str.encode()).hexdigest()[:8]
    _cached = load_factor_cache(_n_stocks, _n_dates, _cache_hash)
    if _cached is not None and len(_cached) > 0:
        print(f"使用因子缓存，跳过因子计算")
        if _cached['code'].dtype != object:
            _cached['code'] = _cached['code'].astype(str).str.zfill(6)
        return _cached, industry_codes, all_dates

    # 并行计算因子 - worker从文件读取数据，避免主进程加载全量stock_data_dict
    # fork + COW: 每个 worker 通过 _worker_fd 访问父进程已加载的基本面数据，无需重新从磁盘读取
    args_list = [
        (code, stock_file_map[code], factor_dates, lookback, forward_period)
        for code in stock_file_map.keys()
    ]

    # 分批构建 DataFrame：避免 list-of-dicts（~2GB 峰值内存）与 DataFrame 同时存在导致 OOM
    # chunksize=50 减少 IPC RPC 次数（4000只 / 50 = 80次，vs 10=400次）
    BATCH_SIZE = 20000  # 降低batch size以减少内存峰值
    batch_data = []
    total_results = 0

    # 使用临时CSV文件流式写入，避免df_chunks在内存中累积导致OOM
    import tempfile
    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp_factor')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv_path = os.path.join(tmp_dir, 'factor_data_tmp.csv')
    tmp_csv_file = open(tmp_csv_path, 'w', encoding='utf-8')
    header_written = False

    # 加载题材热度计算器（fork前, COW共享）
    concept_calc = None
    try:
        from .concept_heat import ConceptHeatCalculator
        concept_calc = ConceptHeatCalculator()
        concept_calc.load()
        print(f"题材热度计算器已加载: {len(concept_calc._concept_hist)} 概念板块历史")
    except Exception as e:
        print(f"题材热度计算器加载失败(fallback): {e}")

    def _flush_batch(batch):
        """将一批数据写入临时CSV，避免在内存中累积DataFrame"""
        nonlocal header_written, total_results
        if not batch:
            return
        df = pd.DataFrame(batch)
        total_results += len(df)
        df.to_csv(tmp_csv_file, header=not header_written, index=False)
        header_written = True
        del df
        batch.clear()

    import platform
    if platform.system() == 'Windows':
        print("Windows: single-process factor computation (spawn overhead too high)")
        _init_factor_worker(fd, concept_calc)
        for args in tqdm(args_list, desc="计算因子"):
            res = _compute_stock_factors_worker(args)
            batch_data.extend(res)
            if len(batch_data) >= BATCH_SIZE:
                _flush_batch(batch_data)
    else:
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(num_workers, initializer=_init_factor_worker, initargs=(fd, concept_calc)) as pool:
            for res in tqdm(pool.imap(_compute_stock_factors_worker, args_list, chunksize=50),
                           total=len(args_list), desc="计算因子"):
                batch_data.extend(res)
                if len(batch_data) >= BATCH_SIZE:
                    _flush_batch(batch_data)

    # 最后一批
    _flush_batch(batch_data)
    tmp_csv_file.close()

    del args_list, batch_data  # 释放参数列表和临时数据

    # 从临时CSV读取合并后的因子数据（一次性加载，内存可控）
    print(f"从临时CSV加载因子数据: {tmp_csv_path}")
    factor_data = pd.read_csv(tmp_csv_path, parse_dates=['date'], dtype={'code': str, 'industry': str}) if os.path.getsize(tmp_csv_path) > 0 else pd.DataFrame()
    # 确保code保持为字符串（CSV读写可能转为int64导致与concept_map的isin不匹配）
    if len(factor_data) > 0 and factor_data['code'].dtype != object:
        factor_data['code'] = factor_data['code'].astype(str).str.zfill(6)
    # 清理临时文件
    try:
        os.remove(tmp_csv_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass
    print(f"因子数据: {total_results} 条原始记录 → {len(factor_data)} 行")

    # === 内存优化：float64 → float32 (精度足够，内存减半) ===
    if len(factor_data) > 0:
        for col in factor_data.columns:
            if col in ('code', 'date', 'industry'):
                continue
            if factor_data[col].dtype == 'float64':
                factor_data[col] = pd.to_numeric(factor_data[col], downcast='float')
            elif factor_data[col].dtype == 'int64':
                factor_data[col] = pd.to_numeric(factor_data[col], downcast='integer')
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_int(0))
    except Exception:
        pass

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

        # 删除噪声因子（|IC| < 0.01，纯噪声无预测力）
        _NOISE_FACTORS = {
            'turnover', 'turnover_ratio', 'turnover_5d_avg', 'turnover_20d_avg',
            'amplitude', 'amplitude_5d_avg', 'amplitude_20d_avg',
            'chan_divergence_strength', 'chan_buy_point',
            'avg_up_ret', 'up_capture',
            'max_gap_20d', 'max_ret_20d', 'gap_up_count_20d',
            'consecutive_up', 'pos_in_range_pos_20d',
        }
        dropped_noise = [c for c in _NOISE_FACTORS if c in factor_data.columns]
        if dropped_noise:
            factor_data.drop(columns=dropped_noise, inplace=True)
            print(f"删除噪声因子: {len(dropped_noise)} 个 (|IC|<0.01)")

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

    # === 截面排名标准化：每个因子按日期 groupby → rank_pct → Normal变换 → [-1,1] ===
    # 解决原始因子量级差异导致的加权组合失效问题（IC权重被量级差异淹没）
    # 参考 Qlib/WorldQuant: cross-sectional ranking before factor combination
    if len(factor_data) > 0:
        _RANK_FACTORS = [c for c in factor_data.columns
                         if c not in ('code', 'date', 'future_ret', 'industry')
                         and c not in _CHAN_STRUCTURAL_FIELDS
                         and not c.endswith('_rank')]
        _rank_cols = [f'{fc}_rank' for fc in _RANK_FACTORS]
        for rc in _rank_cols:
            if rc not in factor_data.columns:
                factor_data[rc] = np.nan

        for date_key, grp in tqdm(factor_data.groupby('date'), desc="截面排名标准化"):
            valid_mask = grp[_RANK_FACTORS].notna().all(axis=1)
            n_valid = valid_mask.sum()
            if n_valid < 5:
                continue
            for fc in _RANK_FACTORS:
                vals = grp.loc[valid_mask, fc].values.astype(np.float64)
                rank_pct = pd.Series(vals).rank(pct=True).values
                clipped = np.clip(rank_pct, 0.001, 0.999)
                normal_rank = np.sqrt(2.0) * erfinv(2.0 * clipped - 1.0)
                factor_data.loc[grp.index[valid_mask], f'{fc}_rank'] = np.clip(normal_rank, -3.0, 3.0)
        if _rank_cols:
            print(f"截面排名标准化: {len(_RANK_FACTORS)} 个因子 → {len(_rank_cols)} 个 _rank 列 ([-3,3])")

    # 保存到磁盘缓存（首次 ~5s parquet 写入，后续回测跳过 ~3h 计算）
    try:
        save_factor_cache(factor_data, _n_stocks, _n_dates, _cache_hash)
    except Exception:
        pass  # 缓存写入失败不影响主流程

    return factor_data, industry_codes, all_dates