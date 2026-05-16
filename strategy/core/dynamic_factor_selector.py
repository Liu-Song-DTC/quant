# core/dynamic_factor_selector.py
"""
动态因子选择器 — Walk-Forward IC验证 + 因子家族分散化

从 signal_engine.py 拆分，包含:
- DynamicFactorSelector 类
- 因子家族分类 (FACTOR_FAMILIES)
- IC计算辅助函数 (_compute_date_chunk, _compute_date_chunks_worker)
"""

import numpy as np
import pandas as pd
import multiprocessing
from typing import Dict, List

from .config_loader import load_config

# 因子家族分类（用于分散化约束，避免单一因子家族集体霸榜）
FACTOR_FAMILIES = {
    'momentum':  ['mom_10', 'mom_20', 'mom_diff_5_20', 'mom_diff_10_20',
                  'momentum_reversal', 'momentum_acceleration', 'max_ret_20',
                  'ret_vol_ratio_10', 'ret_vol_ratio_20', 'relative_strength'],
    'lowvol':    ['volatility', 'volatility_5', 'volatility_10', 'volatility_20',
                  'trend_lowvol', 'bb_width_20', 'atr_ratio_20', 'vol_confirm',
                  'low_downside', 'volume_contraction', 'consolidation_breakout'],
    'value':     ['fund_score', 'fund_roe', 'fund_profit_growth', 'fund_eps',
                  'fund_cf_to_profit', 'fund_gross_margin', 'fund_debt_ratio',
                  'fund_pg_improve', 'fund_rg_improve', 'inv_turnover'],
    'quality':   ['fund_revenue_growth', 'tech_fund_combo', 'turnover_stability',
                  'rsi_vol_combo', 'bb_rsi_combo', 'turnover_shrink'],
    'alpha':     ['skewness_20', 'kurtosis_20', 'tail_risk', 'volatility_skew',
                  'overnight_ret', 'intraday_ret', 'gap_ratio',
                  'price_volume_corr_20', 'illiq_20'],
    'volume_price': ['wash_sale_score', 'vol_price_breakout', 'smart_money_flow'],
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
    'smart_money_flow',
}


def _compute_date_chunk(args):
    """Compute IC and factor selection for a date chunk.

    设计：chunk [start:end] 只计算 [start:end-forward_period] 的IC
    即每个chunk的最后forward_period个日期无法计算IC，因为没有足够的未来收益数据
    这些无法计算的日期会由下一个chunk覆盖

    Args:
        args: (date_chunk, factor_df, industry_codes, config)

    Returns:
        {date: {industry: {'factors': [factors], 'quality': avg_quality}}}
    """
    import pandas as pd
    from scipy import stats
    from tqdm import tqdm

    date_chunk, factor_df, industry_codes, config = args

    if factor_df['date'].dtype != 'datetime64[ns]':
        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])

    exclude_cols = {'code', 'date', 'future_ret', 'industry'} | _CHAN_EXCLUDE
    factor_names = [c for c in factor_df.columns if c not in exclude_cols]

    train_window_days = config['train_window_days']
    forward_period = config['forward_period']
    top_n = config['top_n_factors']
    min_train_samples = config['min_train_samples']
    min_ic_dates = config.get('min_ic_dates', 5)
    ic_decay_factor = config.get('ic_decay_factor', 1.0)
    min_factor_count = config.get('min_factor_count', 2)
    min_factor_families = config.get('min_factor_families', 2)
    min_ic_1f = config.get('min_ic_1f', 0.05)
    use_static_candidates = config.get('use_static_candidates', True)
    industry_factor_config_static = config.get('industry_factor_config', {})
    extra_candidate_factors = config.get('extra_candidate_factors', [])

    result = {}

    chunk_start = date_chunk[0]
    chunk_end = date_chunk[-1]
    valid_end = chunk_end - pd.Timedelta(days=forward_period)

    valid_dates = [d for d in date_chunk if d <= valid_end]

    for val_date in tqdm(valid_dates, desc=f"IC计算({len(date_chunk)}天)", leave=False):
        val_date_ts = pd.to_datetime(val_date) if isinstance(val_date, str) else val_date

        train_start_date = val_date_ts - pd.Timedelta(days=train_window_days)
        train_end_date = val_date_ts - pd.Timedelta(days=forward_period)

        train_mask = (factor_df['date'] >= train_start_date) & (factor_df['date'] < train_end_date)
        train_df = factor_df[train_mask]

        if len(train_df) < min_train_samples:
            continue

        date_result = {}
        for industry, codes in industry_codes.items():
            if not codes:
                continue

            ind_df = train_df[train_df['code'].isin(codes)]
            if len(ind_df) < min_train_samples:
                continue

            if use_static_candidates and industry in industry_factor_config_static:
                static_cfg = industry_factor_config_static[industry]
                candidate_factors = set()
                for key in ['factors', 'bull_factors', 'bear_factors']:
                    candidate_factors.update(static_cfg.get(key, []))
                candidate_factors.update(extra_candidate_factors)
                search_factors = [fn for fn in factor_names if fn in candidate_factors]
            else:
                search_factors = factor_names

            factor_metrics = []
            for fn in search_factors:
                if fn not in ind_df.columns:
                    continue

                try:
                    fn_pivot = ind_df.pivot(index='date', columns='code', values=fn)
                    ret_pivot = ind_df.pivot(index='date', columns='code', values='future_ret')

                    fn_rank = fn_pivot.rank(axis=1, na_option='keep')
                    ret_rank = ret_pivot.rank(axis=1, na_option='keep')

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

                # IC质量过滤
                if ic_stability < 0.48:
                    continue
                ic_variance = ic_std / (abs(ic_mean) + 1e-10) if abs(ic_mean) > 1e-10 else 999
                if ic_variance > 10.0:
                    continue
                combined_ir = ir * (0.5 + 0.5 * ic_stability)
                if abs(t_statistic) < 0.4:
                    continue
                if ic_mean <= 0:
                    continue
                if combined_ir < 0.02:
                    continue
                p_value = stats.norm.sf(ic_mean / (ic_std / np.sqrt(n_dates)))
                if p_value > 0.35:
                    continue
                n_positive = sum(1 for ic in ic_list if ic > 0)
                if n_positive / n_dates < 0.45:
                    continue
                if n_dates >= 10:
                    split_idx = int(n_dates * 0.8)
                    oos_ic_list = ic_list[split_idx:]
                    if len(oos_ic_list) >= 2:
                        oos_ic = np.mean(oos_ic_list)
                        if oos_ic <= 0:
                            continue

                factor_metrics.append({
                    'factor': fn,
                    'ic_mean': ic_mean,
                    'ir': ir,
                    'ic_stability': ic_stability,
                    'combined_ir': combined_ir,
                })

            if len(factor_metrics) >= 1:
                factor_metrics.sort(key=lambda x: x['combined_ir'], reverse=True)

                # 因子家族分散化：两轮选择
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

                if n_selected < min_factor_count:
                    continue

                total_quality = sum(f['combined_ir'] for f in top_factors) + 1e-10
                date_result[industry] = {
                    'factors': [f['factor'] for f in top_factors],
                    'weights': [f['combined_ir'] / total_quality for f in top_factors],
                    'quality': avg_quality,
                }

        if date_result:
            result[val_date] = date_result

    return result


def _compute_date_chunks_worker(args):
    """Worker: compute IC for multiple overlapping chunks."""
    chunks, factor_df, industry_codes, config = args
    all_results = []
    for chunk in chunks:
        chunk_result = _compute_date_chunk((chunk, factor_df, industry_codes, config))
        for date, factors in chunk_result.items():
            all_results.append((date, factors))
    return all_results


class DynamicFactorSelector:
    """动态因子选择器 - 基于Walk-Forward验证的动态因子选择

    在每个验证时点，使用训练窗口内的历史数据计算各因子的IC/IR，
    动态选择IR最高的Top-N因子，避免静态配置的过拟合问题。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._load_config()

        self._factor_cache = {}
        self.factor_df = None
        self.industry_codes = {}
        self._all_dates_cache = None

    def set_factor_cache(self, factor_cache: dict, all_dates: list):
        """设置预计算的因子选择缓存（用于多进程共享）"""
        self._factor_cache = factor_cache
        self._all_dates_cache = all_dates

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
        self.min_ic_1f = dynamic_config.get('min_ic_1f', 0.05)
        self.use_static_candidates = dynamic_config.get('use_static_candidates', True)
        self.reweight_blend = dynamic_config.get('reweight_blend', 0.5)

    def set_factor_data(self, factor_df: pd.DataFrame):
        """设置因子数据"""
        if self.factor_df is not factor_df:
            self.factor_df = factor_df
            self._factor_cache.clear()
            if factor_df is not None and len(factor_df) > 0:
                self._all_dates_cache = sorted(factor_df['date'].unique().tolist())
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
        if self.factor_df is None or self.industry_codes is None:
            return

        all_dates = self._all_dates_cache
        if not all_dates:
            return

        all_dates = [pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d for d in all_dates]

        total_dates = len(all_dates)
        n_workers = min(num_workers, total_dates // 50)
        if n_workers < 1:
            n_workers = 1

        stride = total_dates // n_workers
        overlap = self.train_window_days

        chunks = []
        for i in range(n_workers):
            start = i * stride
            end = min((i + 1) * stride + overlap, total_dates)
            chunk_dates = all_dates[start:end]
            chunks.append(chunk_dates)

        total = len(chunks)
        print(f"预计算因子选择: {total} 个chunk, 每段约 {stride} 天, 重叠 {overlap} 天")

        if total == 0:
            return

        factor_df = self.factor_df
        if factor_df['date'].dtype != 'datetime64[ns]':
            factor_df['date'] = pd.to_datetime(factor_df['date'])
        industry_codes = self.industry_codes
        config = {
            'train_window_days': self.train_window_days,
            'forward_period': self.forward_period,
            'top_n_factors': self.top_n_factors,
            'min_train_samples': self.min_train_samples,
            'min_ic_dates': self.min_ic_dates,
            'ic_decay_factor': self.ic_decay_factor,
            'min_factor_count': self.min_factor_count,
            'min_factor_families': load_config().get('dynamic_factor', {}).get('min_factor_families', 2),
            'min_ic_1f': self.min_ic_1f,
            'use_static_candidates': self.use_static_candidates,
            'industry_factor_config': load_config().get('industry_factors', {}),
            'reweight_blend': self.reweight_blend,
            'extra_candidate_factors': load_config().get('dynamic_factor', {}).get('extra_candidate_factors', []),
        }

        worker_args = [
            ([chunk], factor_df, industry_codes, config)
            for chunk in chunks
        ]

        all_results = []
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(total) as pool:
            results = pool.map(_compute_date_chunks_worker, worker_args)
            for r in results:
                all_results.extend(r)

        for date, factors in all_results:
            self._factor_cache[date] = factors

        print(f"因子选择预计算完成: {len(self._factor_cache)} 个日期")

        if progress_callback:
            progress_callback(total, total)

    def select_factors_for_date(self, val_date: str, all_dates: List[str]) -> Dict[str, Dict]:
        """为指定日期选择各行业的最优因子

        Returns:
            {industry: {'factors': [factors], 'quality': avg_quality}}
        """
        val_date_ts = pd.to_datetime(val_date) if isinstance(val_date, str) else val_date

        if val_date_ts in self._factor_cache:
            return self._factor_cache[val_date_ts]

        for i in range(len(all_dates) - 1, -1, -1):
            if all_dates[i] < val_date_ts:
                nearest_date = all_dates[i]
                if nearest_date in self._factor_cache:
                    return self._factor_cache[nearest_date]

        return {}
