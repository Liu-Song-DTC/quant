#!/usr/bin/env python3
"""
离线标定脚本 - 标定即配置

核心流程：
1. 加载数据（复用bt_execution的数据加载逻辑）
2. 运行MarketRegimeDetector（与回测完全一致）
3. 对每只股票每个日期：
   - calculate_indicators() 计算技术指标
   - 获取基本面数据 → compress_fundamental_factor() 压缩
   - compute_composite_factors(ind, idx, fund_score) 计算组合因子
   - 计算 future_ret (forward_period=20)
4. 按概念板块分组（stock_concept_map.pkl，与信号引擎对齐）
5. 对每个行业 × 每个市场状态：
   - 计算每个候选因子的截面Spearman IC
   - 汇总 IC_mean, IC_std, IR, stability
   - 选择最优因子组合（combined_ir排序）
6. 输出标定结果 → 更新factor_config.yaml

关键对齐：
- 因子计算：与factor_calculator一致
- 基本面压缩：与compress_fundamental_factor一致
- future_ret：与factor_preparer一致
- 市场状态：与MarketRegimeDetector一致
"""

import sys
import os
import gc
import pickle
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing
from tqdm import tqdm
import yaml
from collections import defaultdict

# 防止numpy在worker进程中开启多线程导致内存膨胀
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.fundamental import FundamentalData
from core.factor_calculator import (
    calculate_indicators, compute_composite_factors,
    compress_fundamental_factor, get_default_params,
    compute_fundamental_score
)
from core.market_regime_detector import MarketRegimeDetector
from core.industry_mapping import INDUSTRY_KEYWORDS


# ========== 全局变量用于worker进程 ==========
_worker_fd = None
_worker_factor_dates = None
_worker_lookback = 120
_worker_forward_period = 20
_worker_concept_map = {}  # {code: [concept_names]} — 概念板块标定

# ========== 全局变量用于IC并行标定（fork继承） ==========
_calib_factor_df = None
_calib_concept_to_codes = None
_calib_candidates = None


def _log_memory(tag=""):
    """记录当前进程内存使用（仅Linux）"""
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"  [MEM:{tag}] {rss_mb:.0f} MB")
    except Exception:
        pass


def _init_calib_worker(fundamental_path, stock_codes, factor_dates, lookback, forward_period,
                       concept_map=None):
    """Worker进程初始化（spawn模式：每个worker独立加载FundamentalData）"""
    global _worker_fd, _worker_factor_dates, _worker_lookback, _worker_forward_period, _worker_concept_map
    if fundamental_path and os.path.exists(fundamental_path):
        _worker_fd = FundamentalData(fundamental_path, stock_codes)
    else:
        _worker_fd = None
    _worker_factor_dates = factor_dates
    _worker_lookback = lookback
    _worker_forward_period = forward_period
    _worker_concept_map = concept_map or {}


def _calibrate_stock_worker(args):
    """计算单只股票的因子数据（用于标定）

    与factor_preparer逻辑完全一致，使用统一的压缩函数。
    每个worker自行加载CSV，通过共享全局变量获取参数。
    """
    global _worker_fd, _worker_factor_dates, _worker_lookback, _worker_forward_period, _worker_concept_map

    code, file_path = args
    factor_dates = _worker_factor_dates
    lookback = _worker_lookback
    forward_period = _worker_forward_period

    # 每个worker自行加载股票CSV（raw_data格式: 日期/开盘/收盘/最高/最低/成交量/换手率）
    try:
        df = pd.read_csv(file_path, parse_dates=['日期'])
    except Exception:
        return []
    if df.empty or '日期' not in df.columns:
        return []
    # 统一列名为英文
    col_map = {'日期': 'datetime', '开盘': 'open', '收盘': 'close', '最高': 'high',
               '最低': 'low', '成交量': 'volume', '换手率': 'turnover_rate'}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    if 'datetime' in df.columns:
        stock_dates = sorted(df['datetime'].tolist())
    else:
        stock_dates = sorted(df.index.tolist())

    n = len(stock_dates)
    if n < lookback + forward_period:
        return []

    close_arr = df['close'].values
    high_arr = df['high'].values if 'high' in df.columns else close_arr
    low_arr = df['low'].values if 'low' in df.columns else close_arr
    vol_arr = df['volume'].values if 'volume' in df.columns else np.ones(n)
    turnover_arr = df['turnover_rate'].values if 'turnover_rate' in df.columns else None

    # 使用统一的因子计算器
    params = get_default_params()
    ind = calculate_indicators(close_arr, high_arr, low_arr, vol_arr, params, turnover_rate=turnover_arr)

    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    # 获取概念板块（与信号引擎对齐 — 使用stock_concept_map.pkl）
    stock_concepts = _worker_concept_map.get(code, [])
    stock_industry = stock_concepts[0] if stock_concepts else '其他'

    # 批量获取基本面数据: 预加载+searchsorted替代逐日期filter+sort
    fund_cache = {}
    if _worker_fd is not None:
        # 确保股票基本面数据已加载（与_get_latest的懒加载行为一致）
        if code not in _worker_fd.stock_data:
            try:
                _worker_fd._load_stock(code)
            except Exception:
                pass
    if _worker_fd is not None and code in _worker_fd.stock_data:
        # 预加载基本面DataFrame, 一次性排序
        fund_df = _worker_fd.stock_data[code].copy()
        if '数据可用日期' in fund_df.columns and len(fund_df) > 0:
            fund_df = fund_df.sort_values(['数据可用日期', '报告期'])
            available_dates = fund_df['数据可用日期'].values

            valid_dates = [d for d in factor_dates if d in date_to_idx and date_to_idx[d] >= lookback]
            for eval_date in valid_dates:
                try:
                    date_str = eval_date.strftime('%Y%m%d') if hasattr(eval_date, 'strftime') else str(eval_date).replace('-', '')
                    pos = np.searchsorted(available_dates, date_str, side='right') - 1
                    if pos < 0:
                        fund_cache[eval_date] = {}
                        continue

                    row = fund_df.iloc[pos]

                    def _pct(v):
                        if v is None: return None
                        try:
                            if isinstance(v, str): return float(v.strip('%')) / 100
                            return float(v)
                        except Exception: return None

                    roe = _pct(row.get('净资产收益率'))
                    pg = _pct(row.get('净利润-同比增长'))
                    rg = _pct(row.get('营业总收入-同比增长'))
                    eps = row.get('每股收益')
                    gm = _pct(row.get('销售毛利率'))
                    operating_cf = row.get('xjll_经营性现金流-现金流量净额')
                    profit = row.get('净利润-净利润')
                    bps_val = row.get('每股净资产')

                    fund_score = compute_fundamental_score(
                        roe=roe, profit_growth=pg, revenue_growth=rg, eps=eps)

                    # pg_improve / rg_improve (上一季度)
                    pg_improve = None
                    rg_improve = None
                    if pos > 0:
                        prev_row = fund_df.iloc[pos - 1]
                        prev_pg = _pct(prev_row.get('净利润-同比增长'))
                        prev_rg = _pct(prev_row.get('营业总收入-同比增长'))
                        if pg is not None and prev_pg is not None:
                            pg_improve = pg - prev_pg
                        if rg is not None and prev_rg is not None:
                            rg_improve = rg - prev_rg

                    cf_to_profit = None
                    if operating_cf is not None and profit is not None and profit > 0:
                        cf_to_profit = operating_cf / profit

                    try: eps_val = float(eps) if eps is not None else None
                    except (ValueError, TypeError): eps_val = None
                    try: bps_val = float(bps_val) if bps_val is not None else None
                    except (ValueError, TypeError): bps_val = None

                    fund_cache[eval_date] = {
                        'roe': roe, 'profit_growth': pg, 'revenue_growth': rg,
                        'fund_score': fund_score, 'gross_margin': gm,
                        'pg_improve': pg_improve, 'rg_improve': rg_improve,
                        'cf_to_profit': cf_to_profit, 'eps': eps_val, 'bps': bps_val,
                    }
                except Exception:
                    fund_cache[eval_date] = {}
        else:
            valid_dates = [d for d in factor_dates if d in date_to_idx and date_to_idx[d] >= lookback]
    else:
        valid_dates = [d for d in factor_dates if d in date_to_idx and date_to_idx[d] >= lookback]

    results = []
    for sample_date in valid_dates:  # 复用上面过滤后的有效日期
        idx = date_to_idx[sample_date]  # 已验证存在且>=lookback

        fund_data = fund_cache.get(sample_date, {})

        # 压缩后的基本面评分
        compressed_fund_score = 0.0
        if fund_data:
            raw_fund_score = fund_data.get('fund_score', 0) or 0
            if isinstance(raw_fund_score, (int, float)):
                compressed_fund_score = compress_fundamental_factor(raw_fund_score, 'fund_score')

        # 使用统一因子计算器（含tech_fund_combo）
        combo_factors = compute_composite_factors(ind, idx, fund_score=compressed_fund_score)

        row = {'code': code, 'date': sample_date, 'industry': stock_industry}
        # combo_factors values already float, cast to float32 to save memory
        for k, v in combo_factors.items():
            row[k] = np.float32(v) if isinstance(v, (int, float)) else v

        # 基本面因子 - 使用统一压缩（结果转float32）
        if fund_data:
            fund_fields = [
                ('roe', 'fund_roe'), ('profit_growth', 'fund_profit_growth'),
                ('revenue_growth', 'fund_revenue_growth'), ('fund_score', 'fund_score'),
                ('gross_margin', 'fund_gross_margin'), ('cf_to_profit', 'fund_cf_to_profit'),
                ('pg_improve', 'fund_pg_improve'), ('rg_improve', 'fund_rg_improve'),
            ]
            for raw_key, row_key in fund_fields:
                val = fund_data.get(raw_key)
                if val is not None and isinstance(val, (int, float)):
                    row[row_key] = np.float32(compress_fundamental_factor(val, row_key))

        # 估值因子: PE/PB (Fix#1 — 用当前价格计算)
        if fund_data:
            current_price = close_arr[idx]
            eps = fund_data.get('eps')
            bps = fund_data.get('bps')
            if eps and eps > 0 and current_price > 0:
                pe = current_price / eps
                row['fund_pe'] = np.float32(compress_fundamental_factor(pe, 'fund_pe'))
            if bps and bps > 0 and current_price > 0:
                pb = current_price / bps
                row['fund_pb'] = np.float32(compress_fundamental_factor(pb, 'fund_pb'))

        # 计算未来收益
        if idx + forward_period < n:
            future_price = close_arr[idx + forward_period]
            current_price = close_arr[idx]
            if current_price > 0:
                row['future_ret'] = np.float32((future_price - current_price) / current_price)
                results.append(row)

    # 清理本只股票在worker中的基本面缓存（防止worker处理多只股票后缓存膨胀到GB级）
    if _worker_fd is not None and code in _worker_fd.stock_data:
        del _worker_fd.stock_data[code]

    # 释放本只股票的大对象
    del ind, df, fund_cache, date_to_idx, close_arr, high_arr, low_arr, vol_arr

    return results


def prepare_calibration_data(start_date=None, end_date=None):
    """准备标定所需的元数据（不预加载全部股票数据，节省内存）

    只加载：
    1. 文件路径映射 (code → file_path) — 使用raw_data全量历史数据
    2. 所有日期的并集（通过只读日期列轻量扫描）
    3. 指数数据用于市场状态检测
    4. 基本面数据（FundamentalData内部加载）
    5. 概念板块映射（stock_concept_map.pkl，与信号引擎对齐）

    Args:
        start_date: 标定起始日期 (str or pd.Timestamp), 默认 2016-01-01
        end_date: 标定结束日期 (str or pd.Timestamp), 默认 2020-12-31
    """
    if start_date is None:
        start_date = pd.Timestamp('2016-01-01')
    else:
        start_date = pd.Timestamp(start_date)
    if end_date is None:
        end_date = pd.Timestamp('2020-12-31')
    else:
        end_date = pd.Timestamp(end_date)

    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = os.path.join(project_root, 'data/stock_data/raw_data/')
    fundamental_path = os.path.join(project_root, 'data/stock_data/fundamental_data/')

    # ---- 扫描raw_data目录: 每个股票一个子目录 ----
    stock_file_map = {}  # {code: file_path}
    all_dates = set()
    index_df = None

    for code in tqdm(os.listdir(raw_data_path), desc="scanning files"):
        code_dir = os.path.join(raw_data_path, code)
        if not os.path.isdir(code_dir):
            continue
        qfq_file = os.path.join(code_dir, 'qfq.csv')
        if not os.path.exists(qfq_file):
            continue
        stock_file_map[code] = qfq_file
        # 轻量扫描：只读日期列
        try:
            dates = pd.read_csv(qfq_file, usecols=['日期'], parse_dates=['日期'])
            all_dates.update(dates['日期'].tolist())
        except Exception:
            continue

    stock_codes = [c for c in stock_file_map.keys() if c != "sh000001"]

    # 标定窗口过滤
    all_dates = sorted(d for d in all_dates
                       if d >= start_date and d <= end_date)
    print(f"交易日期: {len(all_dates)} 天 ({start_date.date()} ~ {end_date.date()})")

    # ---- 只验证基本面数据路径存在（worker各自加载） ----
    fund_path_ok = fundamental_path and os.path.exists(fundamental_path)
    if not fund_path_ok:
        print("警告: 基本面数据路径不存在，将仅使用技术因子")
    else:
        print(f"基本面数据路径: {fundamental_path}（各worker独立加载）")

    # ---- 生成市场状态（需要完整指数数据） ----
    regime_lookup = None
    if "sh000001" in stock_file_map:
        index_df = pd.read_csv(stock_file_map["sh000001"], parse_dates=['date'])
        if 'date' in index_df.columns:
            index_df.rename(columns={'date': 'datetime'}, inplace=True)
        detector = MarketRegimeDetector()
        regime_result = detector.generate(index_df)
        regime_lookup = {}
        if regime_result is not None and 'datetime' in regime_result.columns:
            for _, row in regime_result.iterrows():
                dt = pd.to_datetime(row['datetime'])
                regime_lookup[dt] = int(row.get('regime', 0))
        print(f"市场状态数据: {len(regime_lookup)} 天, "
              f"牛市={sum(1 for v in regime_lookup.values() if v == 1)}, "
              f"震荡={sum(1 for v in regime_lookup.values() if v == 0)}, "
              f"熊市={sum(1 for v in regime_lookup.values() if v == -1)}")
        del index_df, regime_result

    # ---- 加载概念板块映射（与信号引擎对齐） ----
    concept_map = {}
    concept_map_path = os.path.join(project_root, 'data', 'stock_concept_map.pkl')
    if os.path.exists(concept_map_path):
        with open(concept_map_path, 'rb') as f:
            raw_concept_map = pickle.load(f)
        # 过滤宽泛标签概念
        STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
                     '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
                     '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
                     '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
                     '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
                     '转债标的', '长江三角', '深圳特区', '破净股', '创投']
        for code, concepts in raw_concept_map.items():
            filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
            if filtered:
                concept_map[code] = filtered
        print(f"概念板块映射: {len(concept_map)} 只股票, "
              f"{len(set(c for clist in concept_map.values() for c in clist))} 个概念")
    else:
        print("警告: 未找到概念板块映射文件，回退到行业关键词")

    _log_memory("after prepare")

    return stock_file_map, fundamental_path if fund_path_ok else None, regime_lookup, stock_codes, sorted(all_dates), concept_map


def _calc_max_workers():
    """根据可用内存估算最大worker数

    每个worker内存占用：
    - FundamentalData惰性加载（虽然处理完清理缓存，但峰值仍需考虑）
    - 技术指标数组（每只股票~几十MB）
    - Python进程基础开销
    保守估计每worker ~500MB，保留1.5GB给系统和因子数据合并。
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    avail_kb = int(line.split()[1])
                    avail_gb = avail_kb / (1024 * 1024)
                    # 保守：每worker 500MB，保留1.5GB给主进程合并factor_df和标定计算
                    max_w = max(1, int((avail_gb - 1.0) / 0.4))
                    max_w = min(max_w, 8)  # 最多8个
                    print(f"可用内存: {avail_gb:.1f} GB → 最大workers: {max_w}")
                    return max_w
    except Exception:
        pass
    return 2


def compute_factor_data(stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
                         concept_map=None):
    """计算所有股票的因子数据（spawn模式，支持概念板块标定）

    核心优化：
    1. spawn上下文：每个worker独立启动，不继承父进程内存（避免COW复制导致OOM）
    2. 共享参数通过initargs传递，per-stock arg仅(code, file_path)
    3. 结果流式写入磁盘分块，避免内存中同时持有全部结果


    核心优化：
    1. spawn上下文：每个worker独立启动，不继承父进程内存（避免COW复制导致OOM）
    2. 共享参数(factor_dates等)通过initargs传递，per-stock arg仅(code, file_path)
    3. 结果流式写入磁盘分块，避免内存中同时持有全部结果
    4. auto-cap workers基于可用内存
    """
    config = load_config()
    lookback = config.get('industry_factor_config.lookback_days', 250)
    forward_period = config.get('dynamic_factor.forward_period', 20)
    configured_workers = config.get('backtest.num_workers', 4)

    factor_dates = all_dates[lookback:-forward_period]

    # 根据可用内存自动限制worker数（每个worker加载FundamentalData约需300-500MB）
    max_workers_by_mem = _calc_max_workers()
    num_workers = min(configured_workers, max_workers_by_mem)
    print(f"workers: {num_workers} (配置={configured_workers}, 内存限制={max_workers_by_mem})")

    print(f"标定范围: {len(factor_dates)} 个时间点, {len(stock_codes)} 只股票, {num_workers} workers")

    # 精简参数列表：共享参数通过initargs传递，每stock仅传(code, file_path)
    args_list = [
        (code, stock_file_map[code])
        for code in stock_codes if code in stock_file_map
    ]

    # 流式并行计算 + 分块写入磁盘
    temp_dir = tempfile.mkdtemp(prefix='calib_factors_')
    temp_files = []
    chunk_buffer = []
    chunk_size = 150000  # 每15万行写入一次
    total_rows = 0

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(num_workers,
                  initializer=_init_calib_worker,
                  initargs=(fundamental_path, stock_codes, factor_dates, lookback, forward_period,
                            concept_map)) as pool:
        for res in tqdm(pool.imap(_calibrate_stock_worker, args_list, chunksize=10),
                       total=len(args_list), desc="计算因子"):
            chunk_buffer.extend(res)
            total_rows += len(res)
            if len(chunk_buffer) >= chunk_size:
                _flush_buffer(chunk_buffer, temp_dir, temp_files)
                gc.collect()

    # 写入尾部
    if chunk_buffer:
        _flush_buffer(chunk_buffer, temp_dir, temp_files)
        del chunk_buffer

    # 显式关闭pool（确保worker释放内存）
    pool.join()
    _log_memory("after workers (streamed)")

    # 读取所有chunk并合并
    if not temp_files:
        print("警告: 无因子数据生成")
        return pd.DataFrame()

    factor_df = _read_and_merge_chunks(temp_files)
    print(f"合并后数据: {len(factor_df)} 行, {factor_df['code'].nunique()} 只股票")

    # 降精度：所有float64 → float32，int64 → int32（大幅减少内存占用）
    _downcast_df(factor_df)

    _log_memory("after df build")

    # 过滤极端未来收益
    if 'future_ret' in factor_df.columns:
        original_len = len(factor_df)
        factor_df = factor_df[
            (factor_df['future_ret'] > -0.5) & (factor_df['future_ret'] < 0.5)
        ]
        print(f"因子数据: {original_len} -> {len(factor_df)} (过滤极端值 {original_len - len(factor_df)})")

    # 合并市场状态
    if regime_lookup and not factor_df.empty and 'date' in factor_df.columns:
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        regime_arr = np.zeros(len(factor_df), dtype=np.int8)
        dates_series = factor_df['date']
        for dt, reg in regime_lookup.items():
            mask = dates_series == dt
            regime_arr[mask] = np.int8(reg)
        factor_df['regime'] = regime_arr
        regime_counts = factor_df['regime'].value_counts().to_dict()
        print(f"市场状态分布: {regime_counts}")

    gc.collect()
    _log_memory("after factor_df done")

    return factor_df


def _flush_buffer(buffer, temp_dir, temp_files):
    """将buffer写入临时pickle文件并清空"""
    if not buffer:
        return
    chunk_df = pd.DataFrame(buffer)
    temp_path = os.path.join(temp_dir, f'chunk_{len(temp_files)}.pkl')
    chunk_df.to_pickle(temp_path)
    temp_files.append(temp_path)
    buffer.clear()


def _read_and_merge_chunks(temp_files):
    """逐文件读取chunk并迭代合并，读取完立即删除临时文件

    迭代concat避免同时持有所有chunk DataFrame + 最终合并结果（峰值内存翻倍）。
    """
    result = None
    for f in temp_files:
        chunk = pd.read_pickle(f)
        if result is None:
            result = chunk
        else:
            result = pd.concat([result, chunk], ignore_index=True)
        del chunk
        try:
            os.remove(f)
        except Exception:
            pass
    try:
        os.rmdir(os.path.dirname(temp_files[0]))
    except Exception:
        pass
    return result if result is not None else pd.DataFrame()


def _downcast_df(df):
    """原地降精度DataFrame以节省内存"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


def _cross_sectional_ic(df, factor_name, value_col='future_ret', min_samples=10):
    """计算单个因子的截面Spearman IC（按日期groupby，避免pivot内存爆炸）

    与pivot方法数学等价，但只遍历日期分组计算rank-IC，
    不需要创建 dates×stocks 宽矩阵，大幅降低内存占用。

    Returns:
        list of float: 每个日期的IC值
    """
    ic_values = []
    for _, group in df.groupby('date', sort=False):
        valid = group[[factor_name, value_col]].dropna()
        if len(valid) >= min_samples:
            if valid[factor_name].nunique() <= 1:
                continue
            ic, _ = stats.spearmanr(valid[factor_name], valid[value_col])
            if not np.isnan(ic):
                ic_values.append(ic)
    return ic_values


def _cross_sectional_ic_batch(regime_df, candidate_factors, value_col='future_ret', min_samples=10):
    """批量计算所有候选因子的截面IC — 直接Spearman公式，避免scipy开销

    核心优化：
    1. 一次groupby完成所有因子IC计算（而非每个因子重复groupby）
    2. 使用 Spearman 简化公式: rs = 1 - 6*sum(d²)/(n*(n²-1))
       避免 scipy.stats.spearmanr 的重复调用开销（~50x speedup per call）
    3. 预计算 future_ret rank（所有因子共用）

    Returns:
        dict: {factor_name: [ic_values]}
    """
    factor_ic_lists = {fn: [] for fn in candidate_factors if fn in regime_df.columns}
    if not factor_ic_lists:
        return {}

    factor_cols = list(factor_ic_lists.keys())

    for _, group in regime_df.groupby('date', sort=False):
        n_total = len(group)
        if n_total < min_samples:
            continue

        ret_mask = group[value_col].notna()
        if ret_mask.sum() < min_samples:
            continue
        ret_rank = group.loc[ret_mask, value_col].rank()

        for fn in factor_cols:
            fn_mask = group[fn].notna()
            common = ret_mask & fn_mask
            n = common.sum()
            if n < min_samples:
                continue

            fn_vals = group.loc[common, fn]
            if fn_vals.nunique() <= 1:
                continue

            fn_rank = fn_vals.rank()
            # Align ret_rank to same index
            cr = ret_rank.reindex(fn_rank.index)

            # Spearman = Pearson on ranks.
            # Use np.corrcoef which handles ties correctly (unlike the simplified
            # formula rs=1-6*sum(d²)/(n*(n²-1)) which assumes no ties).
            # Still much faster than scipy.stats.spearmanr per call.
            corr = np.corrcoef(fn_rank.values, cr.values)[0, 1]
            if not np.isnan(corr):
                factor_ic_lists[fn].append(corr)

    return factor_ic_lists


def _calibrate_concepts_worker(concepts_chunk):
    """Worker: 并行计算一组概念的IC"""
    global _calib_factor_df, _calib_concept_to_codes, _calib_candidates
    factor_df = _calib_factor_df
    concept_to_codes = _calib_concept_to_codes
    candidate_factors = _calib_candidates

    regime_map = {1: 'bull', 0: 'neutral', -1: 'bear'}
    results = {}

    for concept in concepts_chunk:
        if concept_to_codes:
            codes = concept_to_codes.get(concept, [])
            if not codes:
                continue
            ind_df = factor_df[factor_df['code'].isin(codes)]
        else:
            ind_df = factor_df[factor_df['industry'] == concept]
        if len(ind_df) < 100:
            continue

        results[concept] = {}

        for regime_val, regime_name in regime_map.items():
            regime_df = ind_df[ind_df['regime'] == regime_val]
            if len(regime_df) < 50:
                continue

            factor_ic_lists = _cross_sectional_ic_batch(regime_df, candidate_factors)
            factor_metrics = {}

            for factor_name, ic_list in factor_ic_lists.items():
                if len(ic_list) < 10:
                    continue
                ic_mean = np.mean(ic_list)
                ic_std = np.std(ic_list) + 1e-10
                if ic_mean <= 0:
                    continue
                ic_signs = np.sign(ic_list)
                ic_stability = np.abs(np.sum(ic_signs)) / len(ic_signs)
                if ic_stability < 0.1:
                    continue
                t_stat = ic_mean / (ic_std / np.sqrt(len(ic_list)))
                if abs(t_stat) < 1.5:
                    continue
                factor_metrics[factor_name] = {
                    'ic_mean': float(ic_mean),
                    'ic_std': float(ic_std),
                    'ir': float(ic_mean / ic_std),
                    'ic_stability': float(ic_stability),
                    'combined_ir': float((ic_mean / ic_std) * (0.5 + 0.5 * ic_stability)),
                    'n_dates': int(len(ic_list)),
                }
            if factor_metrics:
                results[concept][regime_name] = factor_metrics
    return results


def calibrate_industry_regime(factor_df, candidate_factors, concept_map=None):
    """按概念板块×市场状态标定因子（并行：fork pool）

    关键：每个概念使用ALL属于它的股票（不限于主概念），
    与信号引擎的查找逻辑保持一致（一只股票→第一个有配置的概念）。

    Args:
        factor_df: 因子数据DataFrame（含code列）
        candidate_factors: 候选因子列表
        concept_map: {code: [concept_names]} 概念板块映射

    Returns:
        dict: {concept: {regime_name: {factor: ic_metrics}}}
    """
    if factor_df.empty:
        return {}

    if factor_df['date'].dtype != 'datetime64[ns]':
        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])

    if concept_map:
        codes_in_df = set(factor_df['code'].unique())
        concept_to_codes = defaultdict(list)
        for code, concepts in concept_map.items():
            if code in codes_in_df:
                for c in concepts:
                    concept_to_codes[c].append(code)
        min_codes = 20
        concepts = sorted(
            c for c, clist in concept_to_codes.items()
            if len(clist) >= min_codes
        )
        print(f"标定概念数: {len(concepts)} (min_codes={min_codes})")
    else:
        concepts = sorted(factor_df['industry'].dropna().unique())
        concept_to_codes = None
        print(f"标定行业数: {len(concepts)} (回退industry列)")

    # 并行: fork pool, factor_df通过COW共享
    n_workers = min(multiprocessing.cpu_count(), 8)
    concept_chunks = [list(c) for c in np.array_split(concepts, n_workers) if len(c) > 0]

    # 设置全局变量（fork后子进程继承）
    global _calib_factor_df, _calib_concept_to_codes, _calib_candidates
    _calib_factor_df = factor_df
    _calib_concept_to_codes = concept_to_codes
    _calib_candidates = candidate_factors

    calibration_results = {}
    with multiprocessing.get_context('fork').Pool(n_workers) as pool:
        for worker_result in tqdm(pool.imap(_calibrate_concepts_worker, concept_chunks),
                                   total=len(concept_chunks), desc="标定概念"):
            calibration_results.update(worker_result)

    return calibration_results


def _compute_combined_factor_ic(regime_df, factor_names, weights):
    """计算组合因子的截面IC（groupby方式，避免pivot内存爆炸）

    Args:
        regime_df: 行业×市场状态的因子DataFrame
        factor_names: 因子名列表
        weights: 因子权重列表（归一化后）

    Returns:
        dict: {ic_mean, ic_std, ir, ic_stability, n_dates} or None
    """
    available = [f for f in factor_names if f in regime_df.columns]
    if not available:
        return None

    w_map = {f: w for f, w in zip(factor_names, weights) if f in available}
    total_w = sum(abs(w) for w in w_map.values())
    if total_w == 0:
        return None

    # 向量化计算组合因子值 (dropna而非fillna(0): 缺失≠中性)
    valid_mask = np.ones(len(regime_df), dtype=bool)
    for f in w_map:
        valid_mask &= regime_df[f].notna().values
    combined = np.zeros(len(regime_df), dtype=np.float32)
    for f, w in w_map.items():
        combined[valid_mask] += regime_df[f].values[valid_mask] * w
    combined[valid_mask] /= total_w
    combined[~valid_mask] = np.nan  # 有缺失则不参与IC计算

    # 使用groupby计算截面IC（替代pivot）
    combined_series = pd.Series(combined, index=regime_df.index, dtype=np.float32)
    temp_df = pd.DataFrame({
        '_combined': combined_series,
        'future_ret': regime_df['future_ret'],
        'date': regime_df['date'],
    })
    ic_list = _cross_sectional_ic(temp_df, '_combined')
    del temp_df, combined_series

    if len(ic_list) < 10:
        return None

    ic_mean = np.mean(ic_list)
    if ic_mean <= 0:
        return None

    ic_std = np.std(ic_list) + 1e-10
    ir = ic_mean / ic_std
    ic_signs = np.sign(ic_list)
    ic_stability = np.abs(np.sum(ic_signs)) / len(ic_signs)

    return {
        'ic_mean': float(ic_mean),
        'ic_std': float(ic_std),
        'ir': float(ir),
        'ic_stability': float(ic_stability),
        'n_dates': len(ic_list),
    }


def _select_best_factors_worker(concepts_chunk):
    """Worker: 并行贪心前向选择"""
    global _calib_factor_df, _calib_concept_to_codes
    factor_df = _calib_factor_df
    concept_to_codes = _calib_concept_to_codes

    regime_map = {'neutral': 0, 'bull': 1, 'bear': -1}
    result_config = {}

    for concept, regimes in concepts_chunk:
        config = {}
        if concept_to_codes and concept in concept_to_codes:
            ind_df = factor_df[factor_df['code'].isin(concept_to_codes[concept])]
        elif 'industry' in factor_df.columns:
            ind_df = factor_df[factor_df['industry'] == concept]
        else:
            ind_df = factor_df

        for regime_name in ['neutral', 'bull', 'bear']:
            if regime_name not in regimes:
                continue

            sorted_factors = sorted(
                regimes[regime_name].items(),
                key=lambda x: x[1]['combined_ir'], reverse=True)

            candidates = [(fn, m) for fn, m in sorted_factors
                         if m['combined_ir'] >= 0.02]
            if not candidates:
                continue

            regime_val = regime_map[regime_name]
            regime_df = ind_df[ind_df['regime'] == regime_val] if 'regime' in ind_df.columns else ind_df
            if len(regime_df) < 50:
                selected = candidates[:3]
                factors = [f[0] for f in selected]
                total_ir = sum(f[1]['combined_ir'] for f in selected)
                weights = [f[1]['combined_ir'] / total_ir for f in selected]
                avg_ic = float(np.mean([f[1]['ic_mean'] for f in selected]))
                combined_ic = avg_ic
            else:
                best_result = None
                best_combined_ic = 0
                for k in range(1, min(4, len(candidates) + 1)):
                    top_k = candidates[:k]
                    factors_k = [f[0] for f in top_k]
                    total_ir = sum(f[1]['combined_ir'] for f in top_k)
                    weights_k = [f[1]['combined_ir'] / total_ir for f in top_k]
                    combo_ic = _compute_combined_factor_ic(regime_df, factors_k, weights_k)
                    if combo_ic and combo_ic['ic_mean'] > best_combined_ic:
                        best_combined_ic = combo_ic['ic_mean']
                        best_result = {
                            'factors': factors_k, 'weights': weights_k,
                            'avg_ic': float(np.mean([f[1]['ic_mean'] for f in top_k])),
                            'combined_ic': combo_ic['ic_mean'],
                            'combined_ir': combo_ic['ir'],
                            'combined_stability': combo_ic['ic_stability'],
                        }
                if best_result is None:
                    top1 = candidates[:1]
                    best_result = {
                        'factors': [top1[0][0]], 'weights': [1.0],
                        'avg_ic': top1[0][1]['ic_mean'],
                        'combined_ic': top1[0][1]['ic_mean'],
                        'combined_ir': top1[0][1]['ir'],
                        'combined_stability': top1[0][1]['ic_stability'],
                    }
                factors = best_result['factors']
                weights = best_result['weights']
                avg_ic = best_result['avg_ic']
                combined_ic = best_result['combined_ic']

            if regime_name == 'neutral':
                config['factors'] = factors
                config['weights'] = [round(w, 4) for w in weights]
                config['ic'] = round(avg_ic, 4)
                config['combined_ic'] = round(combined_ic, 4)
            elif regime_name == 'bull':
                config['bull_factors'] = factors
                config['bull_weights'] = [round(w, 4) for w in weights]
                config['bull_ic'] = round(avg_ic, 4)
                config['bull_combined_ic'] = round(combined_ic, 4)
            elif regime_name == 'bear':
                config['bear_factors'] = factors
                config['bear_weights'] = [round(w, 4) for w in weights]
                config['bear_ic'] = round(avg_ic, 4)
                config['bear_combined_ic'] = round(combined_ic, 4)

        if config:
            result_config[concept] = config
    return result_config


def select_best_factors(calibration_results, factor_df, concept_map=None, max_factors=3, min_combined_ir=0.02):
    """贪心前向选择最优因子组合（并行：fork pool）"""
    if concept_map:
        codes_in_df = set(factor_df['code'].unique())
        concept_to_codes = defaultdict(list)
        for code, concepts in concept_map.items():
            if code in codes_in_df:
                for c in concepts:
                    concept_to_codes[c].append(code)
    else:
        concept_to_codes = None

    # 并行
    items = list(calibration_results.items())
    n_workers = min(multiprocessing.cpu_count(), 8)
    chunks = [list(c) for c in np.array_split(items, n_workers) if len(c) > 0]

    global _calib_factor_df, _calib_concept_to_codes
    _calib_factor_df = factor_df
    _calib_concept_to_codes = concept_to_codes

    result_config = {}
    with multiprocessing.get_context('fork').Pool(n_workers) as pool:
        for worker_result in tqdm(pool.imap(_select_best_factors_worker, chunks),
                                   total=len(chunks), desc="因子选择"):
            result_config.update(worker_result)
    return result_config


def update_factor_config(industry_config, config_path):
    """更新factor_config.yaml中的industry_factors部分"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新industry_factors
    config['industry_factors'] = industry_config

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"已更新 {config_path}")


def generate_report(calibration_results, industry_config, output_path):
    """生成标定报告"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 离线标定报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # 汇总统计
        total_combos = 0
        good_ic_combos = 0  # IC > 5%

        for industry, regimes in calibration_results.items():
            for regime_name, factors in regimes.items():
                for factor_name, metrics in factors.items():
                    total_combos += 1
                    if metrics['ic_mean'] > 0.05:
                        good_ic_combos += 1

        f.write(f"## 汇总\n\n")
        f.write(f"- 行业数: {len(calibration_results)}\n")
        f.write(f"- 总标定组合: {total_combos}\n")
        f.write(f"- IC>5%的组合: {good_ic_combos} ({100*good_ic_combos/max(total_combos,1):.1f}%)\n\n")

        # 每个行业详细结果
        f.write(f"## 详细结果\n\n")

        for industry in sorted(calibration_results.keys()):
            f.write(f"### {industry}\n\n")

            # 输出选中的因子
            if industry in industry_config:
                cfg = industry_config[industry]
                if 'factors' in cfg:
                    c_ic = cfg.get('combined_ic', cfg.get('ic', 'N/A'))
                    f.write(f"- **Neutral**: {cfg['factors']} (单因子IC={cfg.get('ic', 'N/A')}, 组合IC={c_ic})\n")
                    f.write(f"  - weights: {cfg.get('weights', [])}\n")
                if 'bull_factors' in cfg:
                    c_ic = cfg.get('bull_combined_ic', cfg.get('bull_ic', 'N/A'))
                    f.write(f"- **Bull**: {cfg['bull_factors']} (单因子IC={cfg.get('bull_ic', 'N/A')}, 组合IC={c_ic})\n")
                    f.write(f"  - bull_weights: {cfg.get('bull_weights', [])}\n")
                if 'bear_factors' in cfg:
                    c_ic = cfg.get('bear_combined_ic', cfg.get('bear_ic', 'N/A'))
                    f.write(f"- **Bear**: {cfg['bear_factors']} (单因子IC={cfg.get('bear_ic', 'N/A')}, 组合IC={c_ic})\n")
                    f.write(f"  - bear_weights: {cfg.get('bear_weights', [])}\n")

            # 所有候选因子IC
            f.write(f"\n| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |\n")
            f.write(f"|------|------|---------|--------|-----|-----------|-------------|\n")

            for regime_name in ['neutral', 'bull', 'bear']:
                if regime_name in calibration_results[industry]:
                    factors = calibration_results[industry][regime_name]
                    for factor_name, metrics in sorted(factors.items(),
                                                        key=lambda x: x[1]['combined_ir'],
                                                        reverse=True):
                        f.write(f"| {factor_name} | {regime_name} | "
                                f"{metrics['ic_mean']:.4f} | {metrics['ic_std']:.4f} | "
                                f"{metrics['ir']:.4f} | {metrics['ic_stability']:.4f} | "
                                f"{metrics['combined_ir']:.4f} |\n")

            f.write(f"\n")

    print(f"标定报告已生成: {output_path}")


def main():
    config = load_config()

    # 候选因子列表（从backtest_factors读取）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    candidate_factors = raw_config.get('backtest_factors', [])
    if not candidate_factors:
        print("错误: factor_config.yaml中没有backtest_factors配置")
        return

    print(f"候选因子: {len(candidate_factors)} 个")
    print(f"因子列表: {candidate_factors}")

    # Step 1: 扫描元数据（不预加载全部股票数据）
    print("\n=== Step 1: 扫描数据 ===")
    stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates, concept_map = prepare_calibration_data()

    # Step 2: 流式计算因子数据（spawn模式，worker自行加载CSV和FundamentalData）
    print("\n=== Step 2: 计算因子数据 ===")
    factor_df = compute_factor_data(stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
                                     concept_map=concept_map)

    # 释放不再需要的中间数据
    del stock_file_map, fundamental_path, all_dates
    gc.collect()
    _log_memory("after cleanup")

    if factor_df.empty:
        print("错误: 无因子数据")
        return

    print(f"因子数据: {len(factor_df)} 条, "
          f"{factor_df['industry'].nunique()} 个行业, "
          f"{factor_df['code'].nunique()} 只股票")

    # Step 3: 标定（使用concept_map动态分组，所有属于该概念的股票参与IC计算）
    print("\n=== Step 3: 标定 ===")
    calibration_results = calibrate_industry_regime(factor_df, candidate_factors, concept_map=concept_map)

    # 统计
    n_industries = len(calibration_results)
    n_with_neutral = sum(1 for v in calibration_results.values() if 'neutral' in v)
    n_with_bull = sum(1 for v in calibration_results.values() if 'bull' in v)
    n_with_bear = sum(1 for v in calibration_results.values() if 'bear' in v)
    print(f"标定结果: {n_industries} 个行业, "
          f"neutral={n_with_neutral}, bull={n_with_bull}, bear={n_with_bear}")

    # Step 4: 贪心前向选择最优因子（验证组合IC）
    print("\n=== Step 4: 贪心前向选择最优因子 ===")
    industry_config = select_best_factors(calibration_results, factor_df, concept_map=concept_map)
    print(f"选中行业: {len(industry_config)} 个")
    for ind, cfg in industry_config.items():
        neutral = cfg.get('factors', [])
        bull = cfg.get('bull_factors', [])
        bear = cfg.get('bear_factors', [])
        n_cic = cfg.get('combined_ic', 'N/A')
        b_cic = cfg.get('bull_combined_ic', 'N/A')
        e_cic = cfg.get('bear_combined_ic', 'N/A')
        print(f"  {ind}: neutral={neutral}(组合IC={n_cic}), bull={bull}(组合IC={b_cic}), bear={bear}(组合IC={e_cic})")

    # Step 5: 更新配置
    print("\n=== Step 5: 更新配置 ===")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    update_factor_config(industry_config, config_path)

    # Step 6: 生成报告
    print("\n=== Step 6: 生成报告 ===")
    report_path = os.path.join(base_dir, 'analysis_results', 'calibration_report.md')
    generate_report(calibration_results, industry_config, report_path)

    # 汇总
    print("\n=== 标定完成 ===")
    ic_above_5 = 0
    ic_total = 0
    for ind, regimes in calibration_results.items():
        for regime_name, factors in regimes.items():
            for fn, metrics in factors.items():
                ic_total += 1
                if metrics['ic_mean'] > 0.05:
                    ic_above_5 += 1
    if ic_total > 0:
        print(f"IC>5%的因子-行业-状态组合: {ic_above_5}/{ic_total} ({100*ic_above_5/ic_total:.1f}%)")


if __name__ == '__main__':
    main()
