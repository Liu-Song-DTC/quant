# core/dynamic_factor_selector.py
"""
动态因子选择器 — Walk-Forward IC验证 + 因子家族分散化

从 signal_engine.py 拆分，包含:
- DynamicFactorSelector 类
- 因子家族分类 (FACTOR_FAMILIES)
- IC计算辅助函数 (_compute_date_chunk, _compute_date_chunks_worker)
"""

import os
import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter, defaultdict
from datetime import date as date_type
from .cache_manager import load_ic_cache, save_ic_cache
from typing import Dict, List

from .config_loader import load_config


def _extract_factor_records(all_results: list, train_window_days: int = 250) -> list:
    """从 precompute 结果中提取所有因子评估记录, 用于 FactorStore 持久化.

    all_results: [(date, {factors, weights, directions, quality, _all_metrics})]
    """
    records = []
    for val_date, data in all_results:
        val_ts = pd.Timestamp(val_date)
        all_metrics = data.get('_all_metrics', [])
        for fm in all_metrics:
            records.append({
                'eval_date': val_ts,
                'factor_name': fm['factor'],
                'window_len': train_window_days,
                'industry': 'all',
                'ic_mean': fm['ic_mean'],
                'ir': fm['ir'],
                'ic_stability': fm['ic_stability'],
                'ret_spread': fm['ret_spread'],
                'direction': fm.get('direction', 1),
                'combined_ir': fm['combined_ir'],
                'coverage': 0.0,
                'n_dates': fm.get('n_dates', 0),
            })
    return records


# 因子家族分类（用于分散化约束，避免单一因子家族集体霸榜）
FACTOR_FAMILIES = {
    'momentum':  ['mom_diff_5_20', 'mom_diff_10_20', 'momentum_reversal',
                  'momentum_acceleration', 'mom_x_lowvol_20_20', 'mom_x_lowvol_10_10',
                  'mom_quality', 'trend_quality', 'trend_initiation', 'return_risk_ratio',
                  'momentum_reversal_chg20', 'momentum_acceleration_chg20',
                  'mom_x_lowvol_20_20_chg20', 'trend_lowvol', 'trend_lowvol_chg20'],
    'lowvol':    ['volatility', 'low_downside', 'inv_turnover', 'atr_ratio',
                  'downside_risk', 'bb_rsi_combo'],
    'value':     ['debt_ratio', 'log_operating_cf', 'log_revenue', 'log_total_assets',
                  'fund_cf_to_profit', 'fund_score', 'fund_score_chg20',
                  'fund_profit_growth', 'fund_revenue_growth', 'fund_pb', 'fund_pb_chg20',
                  'fund_pe_chg20', 'fund_roe_chg20', 'profit_growth_accel',
                  'revenue_growth_accel', 'quality_value', 'fund_gross_margin'],
    'quality':   ['turnover_stability', 'turnover_shrink', 'turnover_stability_chg20',
                  'volume_divergence', 'liquid_quality', 'consolidation_breakout'],
    'alpha':     ['kurtosis_20', 'overnight_ret', 'gap_ratio', 'amplitude_expansion',
                  'overnight_gap_raw', 'illiq_20', 'tail_risk'],
    'volume_price': ['wash_sale_score', 'gap_breakout_confirm', 'relative_volume_ratio',
                     'volume_dry_up', 'volume_price_resonance', 'turnover_burst',
                     'capital_flow_score', 'volume_surge', 'short_reversal'],
    'sentiment': ['northbound_signal', 'gate_g1', 'gate_g1_chg20', 'gate_g2',
                  'gate_g2_chg20', 'gate_g3', 'price_position_20'],
}


def get_factor_family(factor_name: str) -> str:
    """返回因子所属家族，未分类的返回 'other'"""
    for fam, members in FACTOR_FAMILIES.items():
        if factor_name in members:
            return fam
    return 'other'


# Chan/结构字段 — 具有绝对含义，不适合截面IC排名，需要从因子搜索中排除
_CHAN_EXCLUDE = {
    'pivot_position', 'pivot_present', 'pivot_zg', 'pivot_zd', 'pivot_zz',
    'pivot_level', 'pivot_count',
    'chan_pivot_present', 'chan_pivot_zg', 'chan_pivot_zd', 'chan_pivot_zz', 'chan_pivot_level',
    'breakout_above_pivot', 'breakout_below_pivot', 'consolidation_zone',
    'zhongyin', 'structure_complete', 'alignment_score',
    'trend_type', 'trend_strength',
    'top_fractals', 'bottom_fractals', 'fractal_type',
    'stroke_direction', 'stroke_id', 'stroke_count',
    'segment_direction', 'segment_id', 'segment_count',
    'buy_point', 'sell_point', 'buy_confidence', 'sell_confidence',
    'bi_buy_point', 'bi_sell_point', 'bi_buy_confidence', 'bi_sell_confidence', 'bi_td',
    'confirmed_buy', 'confirmed_sell', 'signal_level', 'buy_strength', 'sell_strength',
    'second_buy_point', 'second_buy_confidence', 'second_buy_b1_ref',
    'top_divergence', 'bottom_divergence', 'hidden_top_divergence',
    'hidden_bottom_divergence', 'divergence_active',
    'bottom_fractal_quality', 'bottom_fractal_strength',
    'bottom_fractal_vol_ratio', 'bottom_fractal_vol_spike', 'bottom_fractal_ema_dist',
    'chan_buy_score', 'chan_sell_score', 'structure_stop_price',
    'capital_flow_score', 'capital_flow_direction',
    'news_sentiment_score', 'news_sentiment_direction',
}


def _compute_date_chunk(args):
    """Compute global IC and factor selection for a date chunk.

    全局IC: 在所有股票上计算每个因子的截面IC, 不再按行业分组。
    因子家族分散化保证所选因子覆盖不同维度。

    Args:
        args: (date_chunk, factor_df, config)

    Returns:
        (result_dict, quality_stats) where result_dict is {val_date: {factors, weights, directions, quality, _all_metrics}}
    """
    date_chunk, factor_df, config = args
    _dyn_pass = 0
    _dyn_date_total = 0
    import pandas as pd
    from scipy import stats
    from tqdm import tqdm

    if factor_df['date'].dtype != 'datetime64[ns]':
        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])

    exclude_cols = {'code', 'date', 'future_ret', 'industry'} | _CHAN_EXCLUDE
    factor_names = [c for c in factor_df.columns if c not in exclude_cols and not c.endswith('_rank')]

    train_window_days = config['train_window_days']
    forward_period = config['forward_period']
    assert forward_period > 0, f"forward_period 必须>0，当前={forward_period}"

    top_n = config['top_n_factors']
    min_train_samples = config['min_train_samples']
    min_ic_dates = config.get('min_ic_dates', 5)
    ic_decay_factor = config.get('ic_decay_factor', 1.0)
    min_factor_count = config.get('min_factor_count', 2)
    min_factor_families = config.get('min_factor_families', 2)
    extra_candidate_factors = config.get('extra_candidate_factors', [])

    result = {}

    all_sorted_dates = sorted(factor_df['date'].unique())
    _date_idx = {pd.to_datetime(d): i for i, d in enumerate(all_sorted_dates)}

    def _trading_day_offset(date_val, offset):
        ts = pd.to_datetime(date_val)
        idx = _date_idx.get(ts)
        if idx is None:
            for i, d in enumerate(all_sorted_dates):
                if pd.to_datetime(d) > ts:
                    idx = i - 1
                    break
            if idx is None:
                idx = len(all_sorted_dates) - 1
        return all_sorted_dates[max(0, idx - offset)]

    chunk_end = date_chunk[-1]
    valid_end = _trading_day_offset(chunk_end, forward_period)
    valid_dates = [d for d in date_chunk if d <= valid_end]
    _dyn_date_total += len(valid_dates)

    # 预 pivot: 全局宽表 (避免每个日期重新 pivot)
    g_wide = factor_df.pivot(index='date', columns='code')
    g_ret = factor_df.pivot(index='date', columns='code', values='future_ret')
    all_codes = [c for c in g_ret.columns]

    if len(all_codes) < 3:
        quality_stats = {'pass': 0, 'fail': {'global': {'insufficient_codes': 1}}, 'date_total': _dyn_date_total}
        return result, quality_stats

    for val_date in tqdm(valid_dates, desc=f"IC计算({len(date_chunk)}天)", leave=False):
        val_date_ts = pd.to_datetime(val_date) if isinstance(val_date, str) else val_date
        train_end_date = _trading_day_offset(val_date_ts, forward_period)
        train_start_date = _trading_day_offset(train_end_date, train_window_days)

        date_rows = (factor_df['date'] >= train_start_date) & (factor_df['date'] < train_end_date)
        train_dates = factor_df.loc[date_rows, 'date'].unique()
        if len(train_dates) < min_ic_dates:
            continue

        g_ret_sliced = g_ret.loc[g_ret.index.isin(train_dates)]
        if len(g_ret_sliced) < min_train_samples:
            continue
        ret_rank = g_ret_sliced.rank(axis=1, na_option='keep')

        # 候选因子: 全局因子 + extra_candidate_factors
        if extra_candidate_factors:
            search_factors = [fn for fn in factor_names
                            if fn in extra_candidate_factors or fn not in set(extra_candidate_factors)]
        else:
            search_factors = list(factor_names)

        # 宽表层切片
        g_date_sliced = g_wide.loc[g_wide.index.isin(train_dates)]

        factor_metrics = []
        for fn in search_factors:
            if fn not in factor_df.columns:
                continue
            try:
                fn_wide = g_date_sliced[fn]
                fn_rank = fn_wide.rank(axis=1, na_option='keep')
                ic_series = fn_rank.corrwith(ret_rank, axis=1)
                ic_list = ic_series.dropna().tolist()
            except Exception:
                ic_list = []

            if len(ic_list) < min_ic_dates:
                continue

            n_dates = len(ic_list)
            if ic_decay_factor < 1.0:
                weights = np.array([ic_decay_factor ** (n_dates - i - 1) for i in range(n_dates)])
                weights = weights / weights.sum()
                ic_mean = np.sum(np.array(ic_list) * weights)
            else:
                ic_mean = np.mean(ic_list)

            ic_std = np.std(ic_list) + 1e-10
            ir = ic_mean / ic_std
            ic_signs = np.sign(ic_list)
            ic_stability = np.abs(np.sum(ic_signs)) / len(ic_signs)
            t_statistic = ic_mean / (ic_std / np.sqrt(n_dates))

            direction = 1 if ic_mean > 0 else -1
            abs_ic_mean = abs(ic_mean)
            abs_ir = abs(ir)

            # 收益差
            max_rank = fn_rank.max(axis=1).clip(lower=1)
            fv_rank = fn_rank.div(max_rank, axis=0)
            top_mask = fv_rank >= 0.80
            bot_mask = fv_rank <= 0.20
            top_ret = g_ret_sliced.where(top_mask).mean(axis=1, skipna=True)
            bot_ret = g_ret_sliced.where(bot_mask).mean(axis=1, skipna=True)
            spread_daily = (top_ret - bot_ret) * direction
            ret_spread = float(spread_daily.mean(skipna=True)) if spread_daily.notna().any() else 0.0

            ret_factor = min(1.0, max(0.0, ret_spread / 0.03))
            combined_ir = abs_ir * (0.4 + 0.3 * ic_stability + 0.3 * ret_factor)

            # IC质量过滤
            if ic_stability < 0.45:
                continue
            if abs_ic_mean < 0.015:
                continue
            if ic_std / (abs_ic_mean + 1e-10) > 6.0:
                continue
            if abs(t_statistic) < 1.2:
                continue
            if ret_spread < 0.004:
                continue
            if combined_ir < 0.01:
                continue
            if direction > 0:
                p_value = stats.norm.sf(t_statistic)
            else:
                p_value = stats.norm.cdf(t_statistic)
            if p_value > 0.15:
                continue
            if direction > 0:
                consistent_days = sum(1 for ic in ic_list if ic > 0)
            else:
                consistent_days = sum(1 for ic in ic_list if ic < 0)
            if consistent_days / n_dates < 0.50:
                continue

            factor_metrics.append({
                'factor': fn,
                'ic_mean': abs_ic_mean,
                'ir': abs_ir,
                'ic_stability': ic_stability,
                'combined_ir': combined_ir,
                'ret_spread': ret_spread,
                'direction': direction,
                'n_dates': n_dates,
            })

        if len(factor_metrics) >= min_factor_count:
            factor_metrics.sort(key=lambda x: x['combined_ir'], reverse=True)

            # 因子家族分散化
            family_best = {}
            for fm in factor_metrics:
                fam = get_factor_family(fm['factor'])
                if fam not in family_best:
                    family_best[fam] = fm

            families_ranked = sorted(family_best.values(),
                                     key=lambda x: x['combined_ir'], reverse=True)
            diversified = families_ranked[:top_n]
            used_families = {get_factor_family(f['factor']) for f in diversified}

            if len(used_families) < min_factor_families:
                for fm in factor_metrics:
                    fam = get_factor_family(fm['factor'])
                    if fam not in used_families:
                        if len(diversified) >= top_n:
                            for j in range(len(diversified) - 1, -1, -1):
                                to_replace_fam = get_factor_family(diversified[j]['factor'])
                                if sum(1 for d in diversified if get_factor_family(d['factor']) == to_replace_fam) > 1:
                                    diversified[j] = fm
                                    used_families = {get_factor_family(d['factor']) for d in diversified}
                                    break
                            else:
                                diversified[-1] = fm
                                used_families = {get_factor_family(d['factor']) for d in diversified}
                        else:
                            diversified.append(fm)
                            used_families.add(fam)
                        if len(used_families) >= min_factor_families:
                            break

            top_factors = diversified[:top_n]
            n_selected = len(top_factors)
            avg_quality = np.mean([f['combined_ir'] for f in top_factors])

            if n_selected >= min_factor_count and avg_quality >= 0.015:
                _dyn_pass += 1
                # R6: 平方加权 — 让IC最高的因子占主导, 避免次优因子稀释
                ir_sq = np.array([f['combined_ir'] ** 2 for f in top_factors])
                total_sq = ir_sq.sum() + 1e-10
                result[val_date] = {
                    'factors': [f['factor'] for f in top_factors],
                    'weights': (ir_sq / total_sq).tolist(),
                    'directions': [f.get('direction', 1) for f in top_factors],
                    'quality': avg_quality,
                    '_all_metrics': factor_metrics,
                }

    quality_stats = {
        'pass': _dyn_pass,
        'fail': {},
        'date_total': _dyn_date_total,
    }
    return result, quality_stats


def _compute_date_chunks_worker(args):
    """Worker: compute global IC for multiple overlapping chunks."""
    chunks, factor_df, config = args
    all_results = []
    for chunk in chunks:
        chunk_results, qs = _compute_date_chunk((chunk, factor_df, config))
        for date, factors in chunk_results.items():
            all_results.append((date, factors))
    return all_results, qs


class DynamicFactorSelector:
    """动态因子选择器 - 基于Walk-Forward验证的动态因子选择

    在每个验证时点，使用训练窗口内的历史数据计算各因子的IC/IR，
    动态选择IR最高的Top-N因子，避免静态配置的过拟合问题。
    """

    def __init__(self, config: dict = None, factor_library=None):
        self.config = config or {}
        self._load_config()

        self._factor_cache = {}
        self.factor_df = None
        self.industry_codes = {}
        self._all_dates_cache = None
        self.factor_library = factor_library  # FactorLibrary instance (optional)

    def set_factor_cache(self, factor_cache: dict, all_dates: list):
        """设置预计算的因子选择缓存（用于多进程共享）

        pandas 3.x修复: pd.Timestamp 继承自 datetime.date, isinstance 判断失效。
        统一用 .date() 方法归一化, datetime.date.date() 返回自身。
        """
        # 归一化 cache key → datetime.date (hash/eq 跨 pickle 一致)
        normalized = {}
        for k, v in factor_cache.items():
            key = k.date() if hasattr(k, 'date') else pd.to_datetime(k).date()
            normalized[key] = v
        self._factor_cache = normalized
        self._all_dates_cache = [
            d.date() if hasattr(d, 'date') else pd.to_datetime(d).date()
            for d in all_dates
        ]
        self._lookup_fail_count = 0
        self._lookup_fail_samples = []
        self._lookup_success_count = 0

    def _load_config(self):
        config_loader = load_config()
        dynamic_config = config_loader.get('dynamic_factor', {})

        factor_mode = config_loader.get('factor_mode', 'both')
        self.enabled = factor_mode in ['dynamic', 'both', 'reweight']

        self.train_window_days = dynamic_config.get('train_window_days', 250)
        self.forward_period = dynamic_config.get('forward_period', 20)
        self.top_n_factors = dynamic_config.get('top_n_factors', 3)
        self.min_train_samples = dynamic_config.get('min_train_samples', 50)
        self.min_ic_dates = dynamic_config.get('min_ic_dates', 5)
        self.ic_decay_factor = dynamic_config.get('ic_decay_factor', 1.0)
        self.min_factor_count = dynamic_config.get('min_factor_count', 2)
        self.use_static_candidates = dynamic_config.get('use_static_candidates', True)
        self.reweight_blend = dynamic_config.get('reweight_blend', 0.5)

    def set_factor_data(self, factor_df: pd.DataFrame):
        """设置因子数据"""
        if self.factor_df is not factor_df:
            self.factor_df = factor_df
            self._factor_cache.clear()
            if factor_df is not None and len(factor_df) > 0:
                self._all_dates_cache = [pd.to_datetime(d) for d in sorted(factor_df['date'].unique().tolist())]
            else:
                self._all_dates_cache = []

    def set_industry_mapping(self, industry_codes: Dict[str, List[str]]):
        """设置行业映射"""
        self.industry_codes = industry_codes

    def precompute_all_factor_selections(self, progress_callback=None, num_workers=4):
        """预计算因子选择（多进程并行）

        设计：使用重叠的日期chunk
        - Chunk [start:start+90] 只计算 [start:start+90-forward_period] 的IC
        - 相邻chunk重叠20天（forward_period），确保所有日期都被覆盖
        """
        if self.factor_df is None:
            return

        all_dates = self._all_dates_cache
        if not all_dates:
            return

        all_dates = [pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d for d in all_dates]

        total_dates = len(all_dates)
        chunk_size = 90
        overlap = self.train_window_days

        chunks = []
        start = 0
        while start < total_dates:
            end = min(start + chunk_size + overlap, total_dates)
            chunks.append(all_dates[start:end])
            start += chunk_size

        total = len(chunks)
        n_workers = min(num_workers, total, os.cpu_count() or 4)
        print(f"预计算因子选择(全局IC): {total} 个chunk, {n_workers} 个worker")

        if total == 0:
            return

        factor_df = self.factor_df
        if factor_df['date'].dtype != 'datetime64[ns]':
            factor_df['date'] = pd.to_datetime(factor_df['date'])
        config = {
            'train_window_days': self.train_window_days,
            'forward_period': self.forward_period,
            'top_n_factors': self.top_n_factors,
            'min_train_samples': self.min_train_samples,
            'min_ic_dates': self.min_ic_dates,
            'ic_decay_factor': self.ic_decay_factor,
            'min_factor_count': self.min_factor_count,
            'min_factor_families': load_config().get('dynamic_factor', {}).get('min_factor_families', 2),
            'reweight_blend': self.reweight_blend,
            'extra_candidate_factors': load_config().get('dynamic_factor', {}).get('extra_candidate_factors', []),
        }

        # === IC磁盘缓存: 避免每次回测重复计算 ===
        stock_count = factor_df['code'].nunique()
        date_count = factor_df['date'].nunique()
        cached = load_ic_cache(stock_count, date_count, config)
        if cached is not None:
            all_results = cached['cache']
            for date, factors in all_results:
                key = date.date() if hasattr(date, 'date') else pd.to_datetime(date).date()
                self._factor_cache[key] = factors
            print(f"IC缓存命中: 跳过{total}个chunk计算, 直接加载{len(all_results)}个日期结果")
            return

        # 每个worker只传chunk日期范围的factor_df子集
        worker_args = []
        for chunk in chunks:
            chunk_start = pd.to_datetime(chunk[0]) - pd.Timedelta(days=365)
            chunk_end = pd.to_datetime(chunk[-1])
            chunk_df = factor_df[(factor_df['date'] >= chunk_start) & (factor_df['date'] <= chunk_end)].copy()
            worker_args.append(([chunk], chunk_df, config))

        all_results = []
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            worker_results = pool.map(_compute_date_chunks_worker, worker_args)
            for wr in worker_results:
                chunk_results, _qs = wr
                all_results.extend(chunk_results)

        # 恢复factor_df引用
        self.factor_df = factor_df

        for date, factors in all_results:
            key = date.date() if hasattr(date, 'date') else pd.to_datetime(date).date()
            self._factor_cache[key] = factors

        # 保存IC缓存到磁盘, 下次回测直接加载
        save_ic_cache(all_results, all_dates, stock_count, date_count, config)

        # 保存所有因子评估指标到 FactorStore (供时变质量分析)
        if self.factor_library is not None:
            records = _extract_factor_records(all_results, self.train_window_days)
            if records:
                self.factor_library.store.save_batch(records)
                self.factor_library.register_batch(
                    list(set(r['factor_name'] for r in records)),
                    {r['factor_name']: get_factor_family(r['factor_name']) for r in records}
                )

        non_empty = sum(1 for v in self._factor_cache.values() if v)
        print(f"因子选择预计算完成(全局IC): {len(self._factor_cache)} 日期, "
              f"{non_empty} 非空, "
              f"日期范围={min(self._factor_cache.keys()) if self._factor_cache else 'EMPTY'}~{max(self._factor_cache.keys()) if self._factor_cache else 'EMPTY'}")

        if progress_callback:
            progress_callback(total, total)

    def select_factors_for_date(self, val_date: str, all_dates: List) -> Dict:
        """为指定日期选择全局最优因子.

        Returns:
            {'factors': [...], 'quality': ..., 'weights': [...], 'directions': [...]} or {}
        """
        val_ts = pd.Timestamp(val_date)
        val_key = val_ts.date()

        # 尝试因子库时变选择
        if self.factor_library is not None and self.factor_library.store.size > 0:
            try:
                result = self._select_via_library(val_date)
                if result:
                    return result
            except Exception:
                pass

        if val_key in self._factor_cache:
            return self._factor_cache[val_key]

        # Fallback: 最近的历史日期
        best_key = None
        for cache_key in self._factor_cache:
            if cache_key < val_key:
                if best_key is None or cache_key > best_key:
                    best_key = cache_key
        if best_key is not None:
            return self._factor_cache[best_key]

        self._lookup_fail_count = getattr(self, '_lookup_fail_count', 0) + 1
        if self._lookup_fail_count <= 5 or self._lookup_fail_count % 5000 == 0:
            print(f"[DYN_MISS] date={val_date} key={val_key} cache_size={len(self._factor_cache)}", flush=True)

        return {}

    def _select_via_library(self, val_date) -> Dict:
        """使用 FactorLibrary 全局时变评分选择因子."""
        library = self.factor_library
        if library is None:
            return {}

        selected = library.select(
            'all', val_date, top_n=self.top_n_factors,
            window_len=self.train_window_days,
            min_ic=0.015, exclude_decaying=True,
        )
        if len(selected) < 1:
            return {}

        factors = [s['factor_name'] for s in selected]
        scores = [s['score'] for s in selected]
        total_score = sum(scores) + 1e-10
        return {
            'factors': factors,
            'weights': [s / total_score for s in scores],
            'directions': [1] * len(factors),
            'quality': float(np.mean(scores)) if scores else 0.0,
        }

    def extend_to_date(self, target_date, new_factor_df: pd.DataFrame = None,
                       num_workers: int = 1):
        """增量更新因子缓存到目标日期 — 与回测使用相同的 _compute_date_chunk 逻辑。

        实盘每日调用, 只计算新日期的IC, 追加到现有缓存。
        无缓存时自动回退到 precompute_all_factor_selections。

        Args:
            target_date: 目标日期 (str/datetime/date)
            new_factor_df: 新增的因子数据行 (可选, 不传则使用 self.factor_df)
            num_workers: worker 数量 (默认1, 实盘单进程足够)
        """
        import pandas as pd
        target_ts = pd.Timestamp(target_date)

        # 更新 self.factor_df
        if new_factor_df is not None:
            if self.factor_df is not None:
                self.factor_df = pd.concat([self.factor_df, new_factor_df], ignore_index=True)
            else:
                self.factor_df = new_factor_df
            self._all_dates_cache = sorted(self.factor_df['date'].unique().tolist())

        if self.factor_df is None:
            return

        # 首次: 全量预计算
        if not self._factor_cache:
            print("[实盘] 首次初始化, 执行全量IC预计算...")
            self.precompute_all_factor_selections(num_workers=num_workers)
            return

        latest_cached = max(self._factor_cache.keys()) if self._factor_cache else None
        if latest_cached and target_ts.date() <= latest_cached:
            return  # 已是最新

        all_dates = self._all_dates_cache
        if not all_dates:
            return

        # 找新日期: 缓存最新之后 + 今天
        new_dates = [d for d in all_dates
                     if pd.Timestamp(d).date() > (latest_cached or pd.Timestamp('2000-01-01').date())]

        if not new_dates:
            return

        # 构建配置 (与 precompute_all_factor_selections 完全一致)
        cfg = load_config()
        config = {
            'train_window_days': self.train_window_days,
            'forward_period': self.forward_period,
            'top_n_factors': self.top_n_factors,
            'min_train_samples': self.min_train_samples,
            'min_ic_dates': self.min_ic_dates,
            'ic_decay_factor': self.ic_decay_factor,
            'min_factor_count': self.min_factor_count,
            'min_factor_families': cfg.get('dynamic_factor', {}).get('min_factor_families', 2),
            'reweight_blend': self.reweight_blend,
            'extra_candidate_factors': cfg.get('dynamic_factor', {}).get('extra_candidate_factors', []),
        }

        factor_df = self.factor_df
        if factor_df['date'].dtype != 'datetime64[ns]':
            factor_df = factor_df.copy()
            factor_df['date'] = pd.to_datetime(factor_df['date'])

        # 用最近一个 chunk 覆盖新日期 + 足够的训练窗口
        chunk_start = pd.Timestamp(new_dates[0]) - pd.Timedelta(days=self.train_window_days + 365)
        chunk = [d for d in all_dates if pd.Timestamp(d) >= chunk_start]
        chunk_df = factor_df[(factor_df['date'] >= chunk_start - pd.Timedelta(days=365))
                            & (factor_df['date'] <= pd.Timestamp(new_dates[-1]))].copy()

        result, _qs = _compute_date_chunk((chunk, chunk_df, config))
        if result:
            for date, factors in result.items():
                key = date.date() if hasattr(date, 'date') else pd.to_datetime(date).date()
                self._factor_cache[key] = factors
            print(f"[实盘] IC缓存已更新: +{len(result)} 日期, "
                  f"最新={max(self._factor_cache.keys())}, 总缓存={len(self._factor_cache)}")


def init_live_factor_cache(factor_df: pd.DataFrame, industry_codes: Dict[str, List[str]] = None,
                           num_workers: int = 4) -> DynamicFactorSelector:
    """实盘初始化: 用历史数据构建因子选择器并预计算全局IC。

    与回测使用完全相同的 _compute_date_chunk 逻辑, 保证等价性。

    Args:
        factor_df: 历史因子数据 DataFrame (columns: code, date, factor1, factor2, ..., future_ret)
        industry_codes: 行业映射 (可选, 用于 portfolio 行业分散)
        num_workers: 预计算并行度

    Returns:
        已初始化的 DynamicFactorSelector, 可直接用于 signal_engine

    Usage:
        from core.dynamic_factor_selector import init_live_factor_cache

        selector = init_live_factor_cache(factor_df, industry_codes)
        engine = SignalEngine()
        engine.dynamic_factor_selector = selector
    """
    selector = DynamicFactorSelector()
    selector.set_factor_data(factor_df)
    if industry_codes:
        selector.set_industry_mapping(industry_codes)
    selector.precompute_all_factor_selections(num_workers=num_workers)
    return selector
