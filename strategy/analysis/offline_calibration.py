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
4. 按INDUSTRY_KEYWORDS分行业
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
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool
from tqdm import tqdm
import yaml
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.fundamental import FundamentalData
from core.factor_calculator import (
    calculate_indicators, compute_composite_factors,
    compress_fundamental_factor, get_default_params
)
from core.market_regime_detector import MarketRegimeDetector
from core.industry_mapping import INDUSTRY_KEYWORDS


# ========== 全局变量用于worker进程 ==========
_worker_fd = None
_worker_regime_data = None


def _init_calib_worker(fundamental_path, stock_codes):
    """Worker进程初始化"""
    global _worker_fd
    if fundamental_path and os.path.exists(fundamental_path):
        _worker_fd = FundamentalData(fundamental_path, stock_codes)
    else:
        _worker_fd = None


def _calibrate_stock_worker(args):
    """计算单只股票的因子数据（用于标定）

    与factor_preparer逻辑完全一致，使用统一的压缩函数
    """
    global _worker_fd, _worker_regime_data

    code, df, factor_dates, lookback, forward_period = args

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

    # 获取行业
    stock_industry = None
    if _worker_fd is not None:
        try:
            if len(factor_dates) > 0:
                raw_industry = _worker_fd.get_industry(code, factor_dates[0])
                if raw_industry:
                    for cat, keywords in INDUSTRY_KEYWORDS.items():
                        if any(kw in str(raw_industry) for kw in keywords):
                            stock_industry = cat
                            break
        except:
            pass

    # 批量获取基本面数据
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
                    'pg_improve': _worker_fd.get_profit_growth_improve(code, eval_date),
                    'rg_improve': _worker_fd.get_revenue_growth_improve(code, eval_date),
                    'cf_to_profit': None
                }
                operating_cf = _worker_fd.get_operating_cash_flow(code, eval_date)
                profit = _worker_fd.get_profit(code, eval_date)
                if operating_cf is not None and profit is not None and profit > 0:
                    fund_cache[eval_date]['cf_to_profit'] = operating_cf / profit
            except:
                fund_cache[eval_date] = {}

    results = []
    for sample_date in factor_dates:
        idx = date_to_idx.get(sample_date)
        if idx is None or idx < lookback:
            continue

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
        row.update(combo_factors)

        # 基本面因子 - 使用统一压缩
        if fund_data:
            raw_roe = fund_data.get('roe')
            raw_profit_growth = fund_data.get('profit_growth')
            raw_revenue_growth = fund_data.get('revenue_growth')
            raw_fund_score = fund_data.get('fund_score')
            raw_gross_margin = fund_data.get('gross_margin')
            raw_cf_to_profit = fund_data.get('cf_to_profit')
            raw_pg_improve = fund_data.get('pg_improve')
            raw_rg_improve = fund_data.get('rg_improve')

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
            if raw_pg_improve is not None and isinstance(raw_pg_improve, (int, float)):
                row['fund_pg_improve'] = compress_fundamental_factor(raw_pg_improve, 'fund_pg_improve')
            if raw_rg_improve is not None and isinstance(raw_rg_improve, (int, float)):
                row['fund_rg_improve'] = compress_fundamental_factor(raw_rg_improve, 'fund_rg_improve')

        # 计算未来收益
        if idx + forward_period < n:
            future_price = close_arr[idx + forward_period]
            current_price = close_arr[idx]
            if current_price > 0:
                row['future_ret'] = (future_price - current_price) / current_price
                results.append(row)

    return results


def load_data():
    """加载股票数据"""
    config = load_config()
    data_path = config.get('paths.data', '../data/stock_data/backtrader_data/')
    fundamental_path = config.get('paths.fundamental', '../data/stock_data/fundamental_data/')

    # 加载股票数据
    stock_data_dict = {}
    all_items = os.listdir(data_path)
    for item in tqdm(all_items, desc="loading data"):
        if item.endswith('_qfq.csv'):
            name = item[:-8]
        elif item.endswith('_hfq.csv'):
            name = item[:-8]
        else:
            continue
        data = pd.read_csv(os.path.join(data_path, item), parse_dates=['datetime'])
        stock_data_dict[name] = data

    # 加载基本面数据
    fundamental_data = None
    stock_codes = [name for name in stock_data_dict.keys() if name != "sh000001"]
    if fundamental_path and os.path.exists(fundamental_path):
        fundamental_data = FundamentalData(fundamental_path, stock_codes)

    # 生成市场状态
    regime_lookup = None
    if "sh000001" in stock_data_dict:
        detector = MarketRegimeDetector()
        index_df = stock_data_dict["sh000001"]
        # 使用与bt_execution一致的逻辑
        regime_result = detector.generate(index_df)
        # 构建日期->regime的查找表
        regime_lookup = {}
        if regime_result is not None and 'datetime' in regime_result.columns:
            for _, row in regime_result.iterrows():
                dt = pd.to_datetime(row['datetime'])
                regime_lookup[dt] = int(row.get('regime', 0))
        print(f"市场状态数据: {len(regime_lookup)} 天, "
              f"牛市={sum(1 for v in regime_lookup.values() if v == 1)}, "
              f"震荡={sum(1 for v in regime_lookup.values() if v == 0)}, "
              f"熊市={sum(1 for v in regime_lookup.values() if v == -1)}")

    return stock_data_dict, fundamental_data, regime_lookup, stock_codes


def compute_factor_data(stock_data_dict, fundamental_data, regime_lookup, stock_codes, num_workers=8):
    """计算所有股票的因子数据，然后合并市场状态"""
    config = load_config()
    lookback = config.get('industry_factor_config.lookback_days', 120)
    forward_period = config.get('dynamic_factor.forward_period', 20)

    # 收集所有日期
    all_dates = set()
    for df in stock_data_dict.values():
        if 'datetime' in df.columns:
            all_dates.update(df['datetime'].tolist())
        else:
            all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    factor_dates = all_dates[lookback:-forward_period]

    print(f"标定范围: {len(factor_dates)} 个时间点, {len(stock_codes)} 只股票")

    # 准备worker参数
    fundamental_path = fundamental_data.data_path if fundamental_data else None
    args_list = [
        (code, stock_data_dict[code], factor_dates, lookback, forward_period)
        for code in stock_codes
    ]

    # 并行计算
    all_factor_data = []
    with Pool(num_workers,
              initializer=_init_calib_worker,
              initargs=(fundamental_path, stock_codes)) as pool:
        for res in tqdm(pool.imap(_calibrate_stock_worker, args_list, chunksize=10),
                       total=len(args_list), desc="计算因子"):
            all_factor_data.extend(res)

    factor_df = pd.DataFrame(all_factor_data) if all_factor_data else pd.DataFrame()

    # 过滤极端未来收益
    if 'future_ret' in factor_df.columns:
        original_len = len(factor_df)
        factor_df = factor_df[
            (factor_df['future_ret'] > -0.5) & (factor_df['future_ret'] < 0.5)
        ]
        print(f"因子数据: {original_len} -> {len(factor_df)} (过滤极端值 {original_len - len(factor_df)})")

    # 合并市场状态（在主进程中，避免序列化大字典）
    if regime_lookup and not factor_df.empty and 'date' in factor_df.columns:
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        # 构建regime DataFrame用于merge
        regime_df = pd.DataFrame([
            {'date': dt, 'regime': regime}
            for dt, regime in regime_lookup.items()
        ])
        factor_df = factor_df.merge(regime_df, on='date', how='left')
        factor_df['regime'] = factor_df['regime'].fillna(0).astype(int)
        regime_counts = factor_df['regime'].value_counts().to_dict()
        print(f"市场状态分布: {regime_counts}")

    return factor_df


def calibrate_industry_regime(factor_df, candidate_factors):
    """按行业×市场状态标定因子

    Args:
        factor_df: 因子数据DataFrame
        candidate_factors: 候选因子列表

    Returns:
        dict: {industry: {regime_name: {factor: ic_metrics}}}
    """
    if factor_df.empty or 'industry' not in factor_df.columns:
        return {}

    # 确保日期类型正确
    if factor_df['date'].dtype != 'datetime64[ns]':
        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])

    industries = factor_df['industry'].dropna().unique()
    regime_map = {1: 'bull', 0: 'neutral', -1: 'bear'}

    calibration_results = {}

    for industry in tqdm(industries, desc="标定行业"):
        ind_df = factor_df[factor_df['industry'] == industry]
        if len(ind_df) < 100:
            continue

        calibration_results[industry] = {}

        for regime_val, regime_name in regime_map.items():
            regime_df = ind_df[ind_df['regime'] == regime_val]
            if len(regime_df) < 50:
                continue

            factor_metrics = {}

            for factor_name in candidate_factors:
                if factor_name not in regime_df.columns:
                    continue

                # 截面IC计算：每天对截面做Spearman rank corr
                try:
                    fn_pivot = regime_df.pivot(index='date', columns='code', values=factor_name)
                    ret_pivot = regime_df.pivot(index='date', columns='code', values='future_ret')

                    fn_rank = fn_pivot.rank(axis=1, na_option='keep')
                    ret_rank = ret_pivot.rank(axis=1, na_option='keep')

                    ic_series = fn_rank.corrwith(ret_rank, axis=1)
                    ic_list = ic_series.dropna().tolist()
                except:
                    ic_list = []

                if len(ic_list) < 10:
                    continue

                ic_mean = np.mean(ic_list)
                ic_std = np.std(ic_list) + 1e-10
                ir = ic_mean / ic_std
                ic_signs = np.sign(ic_list)
                ic_stability = np.abs(np.sum(ic_signs)) / len(ic_signs)
                n_dates = len(ic_list)

                # 只保留正向IC因子
                if ic_mean <= 0:
                    continue

                # 稳定性过滤（按市场状态分样本后stability普遍较低，适度放宽）
                if ic_stability < 0.1:
                    continue

                # t统计量过滤（替代高stability要求）
                t_stat = ic_mean / (ic_std / np.sqrt(n_dates)) if n_dates > 0 else 0
                if abs(t_stat) < 1.5:
                    continue

                combined_ir = ir * (0.5 + 0.5 * ic_stability)

                factor_metrics[factor_name] = {
                    'ic_mean': float(ic_mean),
                    'ic_std': float(ic_std),
                    'ir': float(ir),
                    'ic_stability': float(ic_stability),
                    'combined_ir': float(combined_ir),
                    'n_dates': int(n_dates),
                }

            if factor_metrics:
                calibration_results[industry][regime_name] = factor_metrics

    return calibration_results


def _compute_combined_factor_ic(regime_df, factor_names, weights):
    """计算组合因子的截面IC（向量化实现）

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

    # 向量化计算组合因子值
    combined = np.zeros(len(regime_df))
    for f, w in w_map.items():
        combined += regime_df[f].fillna(0).values * w
    combined /= total_w

    regime_df_copy = regime_df.copy()
    regime_df_copy['_combined'] = combined

    # 截面IC计算
    try:
        fn_pivot = regime_df_copy.pivot(index='date', columns='code', values='_combined')
        ret_pivot = regime_df_copy.pivot(index='date', columns='code', values='future_ret')
        fn_rank = fn_pivot.rank(axis=1, na_option='keep')
        ret_rank = ret_pivot.rank(axis=1, na_option='keep')
        ic_series = fn_rank.corrwith(ret_rank, axis=1)
        ic_list = ic_series.dropna().tolist()
    except:
        return None

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


def select_best_factors(calibration_results, factor_df, max_factors=3, min_combined_ir=0.02):
    """贪心前向选择最优因子组合，验证组合IC

    对每个行业×市场状态：
    1. 按combined_ir排序候选因子
    2. 贪心前向选择：逐步添加因子，选择使组合IC最高的因子
    3. 验证最终组合因子的IC
    4. 选择使组合IC最高的K值

    Args:
        calibration_results: calibrate_industry_regime的输出
        factor_df: 因子数据DataFrame（用于计算组合IC）
        max_factors: 每个行业每个状态最多选几个因子
        min_combined_ir: 最低combined_ir阈值

    Returns:
        dict: 适合写入factor_config.yaml的格式
    """
    industry_config = {}
    regime_map = {'neutral': 0, 'bull': 1, 'bear': -1}

    for industry, regimes in calibration_results.items():
        config = {}
        ind_df = factor_df[factor_df['industry'] == industry] if 'industry' in factor_df.columns else factor_df

        for regime_name in ['neutral', 'bull', 'bear']:
            if regime_name not in regimes:
                continue

            # 按combined_ir排序
            sorted_factors = sorted(
                regimes[regime_name].items(),
                key=lambda x: x[1]['combined_ir'],
                reverse=True
            )

            # 过滤低质量因子
            candidates = [
                (fn, m) for fn, m in sorted_factors
                if m['combined_ir'] >= min_combined_ir
            ]

            if not candidates:
                continue

            # 筛选该市场状态的因子数据
            regime_val = regime_map[regime_name]
            regime_df = ind_df[ind_df['regime'] == regime_val] if 'regime' in ind_df.columns else ind_df
            if len(regime_df) < 50:
                # 数据不足，退回简单选择
                selected = candidates[:max_factors]
                factors = [f[0] for f in selected]
                total_ir = sum(f[1]['combined_ir'] for f in selected)
                weights = [f[1]['combined_ir'] / total_ir for f in selected]
                avg_ic = float(np.mean([f[1]['ic_mean'] for f in selected]))
                combined_ic = avg_ic  # 无验证数据，用单因子均值IC估计
            else:
                # 贪心前向选择：top-k + 组合IC验证
                best_result = None
                best_combined_ic = 0

                for k in range(1, min(max_factors + 1, len(candidates) + 1)):
                    top_k = candidates[:k]
                    factors_k = [f[0] for f in top_k]
                    total_ir = sum(f[1]['combined_ir'] for f in top_k)
                    weights_k = [f[1]['combined_ir'] / total_ir for f in top_k]

                    combo_ic = _compute_combined_factor_ic(regime_df, factors_k, weights_k)

                    if combo_ic and combo_ic['ic_mean'] > best_combined_ic:
                        best_combined_ic = combo_ic['ic_mean']
                        best_result = {
                            'factors': factors_k,
                            'weights': weights_k,
                            'avg_ic': float(np.mean([f[1]['ic_mean'] for f in top_k])),
                            'combined_ic': combo_ic['ic_mean'],
                            'combined_ir': combo_ic['ir'],
                            'combined_stability': combo_ic['ic_stability'],
                        }

                if best_result is None:
                    top1 = candidates[:1]
                    best_result = {
                        'factors': [top1[0][0]],
                        'weights': [1.0],
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
            industry_config[industry] = config

    return industry_config


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

    # Step 1: 加载数据
    print("\n=== Step 1: 加载数据 ===")
    stock_data_dict, fundamental_data, regime_lookup, stock_codes = load_data()

    # Step 2: 计算因子数据
    print("\n=== Step 2: 计算因子数据 ===")
    factor_df = compute_factor_data(stock_data_dict, fundamental_data, regime_lookup, stock_codes)

    if factor_df.empty:
        print("错误: 无因子数据")
        return

    print(f"因子数据: {len(factor_df)} 条, "
          f"{factor_df['industry'].nunique()} 个行业, "
          f"{factor_df['code'].nunique()} 只股票")

    # Step 3: 标定
    print("\n=== Step 3: 标定 ===")
    calibration_results = calibrate_industry_regime(factor_df, candidate_factors)

    # 统计
    n_industries = len(calibration_results)
    n_with_neutral = sum(1 for v in calibration_results.values() if 'neutral' in v)
    n_with_bull = sum(1 for v in calibration_results.values() if 'bull' in v)
    n_with_bear = sum(1 for v in calibration_results.values() if 'bear' in v)
    print(f"标定结果: {n_industries} 个行业, "
          f"neutral={n_with_neutral}, bull={n_with_bull}, bear={n_with_bear}")

    # Step 4: 贪心前向选择最优因子（验证组合IC）
    print("\n=== Step 4: 贪心前向选择最优因子 ===")
    industry_config = select_best_factors(calibration_results, factor_df)
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
