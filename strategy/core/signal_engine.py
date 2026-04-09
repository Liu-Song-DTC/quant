# core/signal_engine.py
"""
信号生成引擎

基于行业验证结果的因子配置，支持:
- 行业自适应因子选择
- 市场状态动态权重
- 风格因子调整
- 动态因子选择（Walk-Forward）
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats

from .signal import Signal
from .signal_store import SignalStore
from .config_loader import load_config
from .industry_mapping import INDUSTRY_KEYWORDS, get_industry_category
from .factor_calculator import calculate_indicators as calc_indicators, compute_composite_factors
import yaml
import os

import warnings
warnings.filterwarnings('ignore')


# 行业因子配置（从YAML加载）
def _load_industry_factors():
    """加载行业因子配置"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('industry_factors', {})
    return {}


INDUSTRY_FACTOR_CONFIG = _load_industry_factors()


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

    # Ensure date column is datetime
    if factor_df['date'].dtype != 'datetime64[ns]':
        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])

    # Factor columns
    exclude_cols = ['code', 'date', 'future_ret', 'industry']
    factor_names = [c for c in factor_df.columns if c not in exclude_cols]

    train_window_days = config['train_window_days']
    forward_period = config['forward_period']
    top_n = config['top_n_factors']
    min_train_samples = config['min_train_samples']
    min_ic_dates = config.get('min_ic_dates', 5)
    ic_decay_factor = config.get('ic_decay_factor', 1.0)  # 1.0=不衰减
    # === 阶段1优化参数 ===
    min_factor_count = config.get('min_factor_count', 2)  # 最少因子数量
    min_ic_1f = config.get('min_ic_1f', 0.05)  # 1因子最低IC门槛

    result = {}

    # Chunk [start:end] 只计算 [start:end-forward_period] 的IC
    # 即跳过最后forward_period个日期，这些日期由下一个chunk覆盖
    chunk_start = date_chunk[0]
    chunk_end = date_chunk[-1]
    valid_end = chunk_end - pd.Timedelta(days=forward_period)

    # 计算实际需要处理的日期数量（用于进度条）
    valid_dates = [d for d in date_chunk if d <= valid_end]
    total = len(valid_dates)

    for val_date in tqdm(valid_dates, desc=f"IC计算({len(date_chunk)}天)", leave=False):
        val_date_ts = pd.to_datetime(val_date) if isinstance(val_date, str) else val_date

        # Training window: [val_date - train_window_days, val_date - forward_period]
        train_start_date = val_date_ts - pd.Timedelta(days=train_window_days)
        train_end_date = val_date_ts - pd.Timedelta(days=forward_period)

        # Filter training data
        train_mask = (factor_df['date'] >= train_start_date) & (factor_df['date'] < train_end_date)
        train_df = factor_df[train_mask]

        if len(train_df) < min_train_samples:
            continue

        # Compute IC for each industry
        date_result = {}
        for industry, codes in industry_codes.items():
            if not codes:
                continue

            ind_df = train_df[train_df['code'].isin(codes)]
            if len(ind_df) < min_train_samples:
                continue

            # 一次性向量化计算所有日期的截面Spearman IC
            # 使用 pivot + rank + corr 实现真正的向量化
            factor_metrics = []
            for fn in factor_names:
                if fn not in ind_df.columns:
                    continue

                try:
                    # pivot: dates x stocks
                    fn_pivot = ind_df.pivot(index='date', columns='code', values=fn)
                    ret_pivot = ind_df.pivot(index='date', columns='code', values='future_ret')

                    # rank across stocks for each date (截面秩)
                    fn_rank = fn_pivot.rank(axis=1, na_option='keep')
                    ret_rank = ret_pivot.rank(axis=1, na_option='keep')

                    # row-wise Pearson correlation of ranks = Spearman IC
                    ic_series = fn_rank.corrwith(ret_rank, axis=1)
                    ic_list = ic_series.dropna().tolist()
                except:
                    ic_list = []

                if len(ic_list) < min_ic_dates:
                    continue

                # 计算IC质量指标
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

                # 稳定性过滤：至少30%的IC同号即可通过
                MIN_STABILITY = 0.3
                if ic_stability < MIN_STABILITY:
                    continue

                # combined_ir = ir * (0.5 + 0.5 * stability)
                combined_ir = ir * (0.5 + 0.5 * ic_stability)

                # t_statistic过滤（统计显著性）
                if abs(t_statistic) < 1.0:
                    continue

                # 因子方向过滤：只考虑正向因子（ic_mean > 0）
                if ic_mean <= 0:
                    continue

                # 最小质量阈值：过滤噪声因子
                MIN_COMBINED_IR = 0.03
                if combined_ir < MIN_COMBINED_IR:
                    continue

                factor_metrics.append({
                    'factor': fn,
                    'ic_mean': ic_mean,
                    'ir': ir,
                    'ic_stability': ic_stability,
                    'combined_ir': combined_ir,
                })

            if len(factor_metrics) >= 1:
                # 按质量排序，取所有通过阈值的因子（动态N）
                factor_metrics.sort(key=lambda x: x['combined_ir'], reverse=True)
                # 最多取top_n，但如果有超过top_n个因子通过质量阈值，也只取top_n
                top_factors = factor_metrics[:top_n]

                # === 阶段1优化：1F因子需要更高IC门槛 ===
                # 1F因子IC仅0.64%（接近随机），而3F因子IC=7.25%
                n_selected = len(top_factors)
                avg_quality = np.mean([f['combined_ir'] for f in top_factors])

                # 如果只有1个因子，检查是否达到1F专用门槛
                if n_selected < min_factor_count:
                    if n_selected == 1 and avg_quality < min_ic_1f:
                        # 1F质量不足，不记录（让调用方fallback到静态因子）
                        continue

                # 质量加权：combined_ir 反映因子质量，用于权重分配
                total_quality = sum(f['combined_ir'] for f in top_factors) + 1e-10
                date_result[industry] = {
                    'factors': [f['factor'] for f in top_factors],
                    'weights': [f['combined_ir'] / total_quality for f in top_factors],  # 质量加权
                    'quality': avg_quality
                }

        if date_result:
            result[val_date] = date_result

    return result


def _compute_date_chunks_worker(args):
    """Worker function: compute IC for multiple overlapping chunks.

    Args:
        args: (chunks, factor_df, industry_codes, config)
            - chunks: list of date lists (each chunk is a list of dates)
            - factor_df: DataFrame with factor data
            - industry_codes: {industry: [stock_codes]}
            - config: computation config

    Returns:
        list of (date, factors) tuples for all dates in all chunks
    """
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

        # 缓存: {date: {industry: [factors]}}
        self._factor_cache = {}

        # 因子数据: DataFrame with columns [code, date, factor1, factor2, ..., future_ret]
        self.factor_df = None

        # 行业映射: {category: [codes]}
        self.industry_codes = {}

        # 缓存: 已排序的唯一日期列表（避免每次都重新计算）
        self._all_dates_cache = None

    def set_factor_cache(self, factor_cache: dict, all_dates: list):
        """设置预计算的因子选择缓存（用于多进程共享）

        Args:
            factor_cache: {date: {industry: [factors]}}
            all_dates: 已排序的日期列表
        """
        self._factor_cache = factor_cache
        self._all_dates_cache = all_dates

    def _load_config(self):
        """加载配置"""
        config_loader = load_config()
        dynamic_config = config_loader.get('dynamic_factor', {})

        # enabled 由 factor_mode 决定：'dynamic' 或 'both' 时启用
        factor_mode = config_loader.get('factor_mode', 'both')
        self.enabled = factor_mode in ['dynamic', 'both']

        self.train_window_days = dynamic_config.get('train_window_days', 250)
        self.forward_period = dynamic_config.get('forward_period', 20)
        self.top_n_factors = dynamic_config.get('top_n_factors', 3)
        self.min_train_samples = dynamic_config.get('min_train_samples', 50)
        self.min_ic_dates = dynamic_config.get('min_ic_dates', 5)
        self.ic_decay_factor = dynamic_config.get('ic_decay_factor', 1.0)  # 1.0=不衰减
        # === 阶段1优化参数 ===
        self.min_factor_count = dynamic_config.get('min_factor_count', 2)  # 最少因子数量
        self.min_ic_1f = dynamic_config.get('min_ic_1f', 0.05)  # 1因子最低IC门槛

    def set_factor_data(self, factor_df: pd.DataFrame):
        """设置因子数据

        Args:
            factor_df: DataFrame with columns [code, date, factor1, factor2, ..., future_ret]
        """
        # 只在factor_df实际改变时才清除缓存（避免每次调用都清除缓存）
        if self.factor_df is not factor_df:
            self.factor_df = factor_df
            self._factor_cache.clear()
            # 缓存已排序的唯一日期列表
            if factor_df is not None and len(factor_df) > 0:
                self._all_dates_cache = sorted(factor_df['date'].unique().tolist())
            else:
                self._all_dates_cache = []

    def set_industry_mapping(self, industry_codes: Dict[str, List[str]]):
        """设置行业映射

        Args:
            industry_codes: {category: [stock_codes]}
        """
        self.industry_codes = industry_codes

    def precompute_all_factor_selections(self, progress_callback=None, num_workers=4):
        """预计算因子选择（多进程并行）

        设计：使用重叠的日期chunk
        - Chunk [start:start+90] 只计算 [start:start+90-forward_period] 的IC
        - 相邻chunk重叠20天（forward_period），确保所有日期都被覆盖
        - 例如：Chunk1 [0:90] -> 计算[0:70], Chunk2 [70:160] -> 计算[70:140]

        Args:
            progress_callback: 进度回调函数，接受 (current, total) 参数
            num_workers: 并行进程数
        """
        from multiprocessing import Pool
        import numpy as np

        if self.factor_df is None or self.industry_codes is None:
            return

        all_dates = self._all_dates_cache
        if not all_dates:
            return

        # Ensure all_dates are Timestamp type
        all_dates = [pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d for d in all_dates]

        # 按进程数划分日期范围，每个进程处理一段，保证边界有重叠
        total_dates = len(all_dates)
        n_workers = min(num_workers, total_dates // 50)  # 至少保证每段有50天
        if n_workers < 1:
            n_workers = 1

        # 每个worker处理一段日期，重叠窗口 = train_window_days
        # chunk i: [i*stride : i*stride + chunk_size + overlap]
        # 其中 stride = total_dates // n_workers
        stride = total_dates // n_workers
        overlap = self.train_window_days  # 重叠窗口保证边界IC计算完整

        chunks = []
        for i in range(n_workers):
            start = i * stride
            end = min((i + 1) * stride + overlap, total_dates)
            chunk_dates = all_dates[start:end]
            chunks.append(chunk_dates)

        total = len(chunks)
        print(f"预计算因子选择: {total} 个chunk (按进程数划分), 每段约 {stride} 天, 重叠 {overlap} 天")

        if total == 0:
            return

        # Prepare data
        factor_df = self.factor_df.copy()
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
            # === 阶段1优化参数 ===
            'min_factor_count': self.min_factor_count,
            'min_ic_1f': self.min_ic_1f,
        }

        # 每个worker处理一个chunk
        worker_args = [
            ([chunk], factor_df, industry_codes, config)
            for chunk in chunks
        ]

        # 并行计算
        print(f"开始并行计算 {total} 个chunk...")
        all_results = []
        with Pool(total) as pool:
            results = pool.map(_compute_date_chunks_worker, worker_args)
            for r in results:
                all_results.extend(r)

        # 汇总结果
        for date, factors in all_results:
            self._factor_cache[date] = factors

        print(f"因子选择预计算完成: {len(self._factor_cache)} 个日期")

        if progress_callback:
            progress_callback(total, total)

    def select_factors_for_date(self, val_date: str, all_dates: List[str]) -> Dict[str, Dict]:
        """为指定日期选择各行业的最优因子

        Args:
            val_date: 验证日期
            all_dates: 所有可用日期列表

        Returns:
            {industry: {'factors': [factors], 'quality': avg_quality}} 各行业选中的因子及质量分数
        """
        # 统一转换为 Timestamp（确保与缓存键类型一致）
        val_date_ts = pd.to_datetime(val_date) if isinstance(val_date, str) else val_date

        # 检查缓存（使用 Timestamp 类型的键）
        if val_date_ts in self._factor_cache:
            return self._factor_cache[val_date_ts]

        # 如果缓存未命中，尝试找最近的之前日期的缓存结果
        for i in range(len(all_dates) - 1, -1, -1):
            if all_dates[i] < val_date_ts:
                nearest_date = all_dates[i]
                if nearest_date in self._factor_cache:
                    return self._factor_cache[nearest_date]

        # 缓存完全未命中
        return {}


class SignalEngine:
    """信号生成引擎 - 使用行业验证后的高质量因子"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_score = 0.35

        # 加载配置
        self._load_config()

        # 市场状态信息
        self.market_regime_data = None
        self.current_idx = 0

        # 因子选择统计
        self._stats = {
            'dynamic_success': 0,
            'dynamic_fallback_fixed': 0,
            'dynamic_fallback_default': 0,
            'fixed_industry': 0,
            'fixed_default': 0,
            'ic_values': [],
        }

    def _load_config(self):
        """从配置文件加载参数"""
        config_loader = load_config()

        # 信号阈值（从配置文件加载）
        signal_config = config_loader.get('signal', {})
        self.buy_threshold = signal_config.get('buy_threshold', 0.18)  # 阶段2优化：默认0.18
        self.sell_threshold = signal_config.get('sell_threshold', -0.15)

        # 基本面因子配置
        self.fundamental_enabled = True
        self.fundamental_weight = config_loader.get('fundamental_weight', 0.3)

        # 市场状态乘数
        self.regime_multiplier = config_loader.get('regime_multiplier', {
            'bull': 1.0, 'neutral': 0.85, 'bear': 0.6
        })

        # 风格因子开关
        self.style_enabled = config_loader.get('style_factor_enabled', True)

        # 行业因子开关
        industry_config = config_loader.get_industry_factor_config()
        self.industry_factor_enabled = industry_config.get('enabled', True)

        # 技术指标参数
        self.indicator_params = config_loader.get_indicator_params()

        # 动态因子选择器
        self.dynamic_factor_selector = DynamicFactorSelector()

        # 动态因子模式配置：dynamic(仅动态) / fixed(仅固定) / both(动态优先+固定兜底)
        self.factor_mode = config_loader.get('factor_mode', 'both')
        self.factor_fallback_to_fixed = config_loader.get('dynamic_factor.fallback_to_fixed', True)

    def set_factor_data(self, factor_df: pd.DataFrame):
        """设置因子数据（用于动态因子选择）

        Args:
            factor_df: DataFrame with columns [code, date, factor1, factor2, ..., future_ret]
        """
        self.dynamic_factor_selector.set_factor_data(factor_df)

    def set_industry_mapping(self, industry_codes: Dict[str, List[str]]):
        """设置行业映射（用于动态因子选择）

        Args:
            industry_codes: {category: [stock_codes]}
        """
        self.dynamic_factor_selector.set_industry_mapping(industry_codes)
        # 同时保存到 SignalEngine 自身，方便 _select_factor_dynamic 访问
        self.industry_codes = industry_codes

    def set_fundamental_data(self, fundamental_data):
        """设置基本面数据"""
        self.fundamental_data = fundamental_data

    def set_market_regime(self, regime_df: pd.DataFrame):
        """设置市场状态数据"""
        if 'datetime' in regime_df.columns:
            regime_df = regime_df.copy()
            regime_df['datetime'] = pd.to_datetime(regime_df['datetime'])
            self.market_regime_data = regime_df.set_index('datetime')

    def print_factor_stats(self):
        """打印因子选择统计"""
        stats = self._stats
        # 排除 ic_values 列表，只计算整数统计
        total = sum(v for k, v in stats.items() if k != 'ic_values')
        if total == 0:
            print("\n因子选择统计: 无数据")
            return

        print("\n========== 因子选择统计 ==========")
        print(f"动态因子成功:     {stats['dynamic_success']:6d} ({100*stats['dynamic_success']/total:.1f}%)")
        print(f"动态->固定fallback: {stats['dynamic_fallback_fixed']:6d} ({100*stats['dynamic_fallback_fixed']/total:.1f}%)")
        print(f"动态->默认fallback: {stats['dynamic_fallback_default']:6d} ({100*stats['dynamic_fallback_default']/total:.1f}%)")
        print(f"固定行业因子:    {stats['fixed_industry']:6d} ({100*stats['fixed_industry']/total:.1f}%)")
        print(f"固定默认因子:    {stats['fixed_default']:6d} ({100*stats['fixed_default']/total:.1f}%)")
        print(f"总计:            {total}")
        print("==================================\n")

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        """生成信号"""
        dates = market_data["datetime"].values
        close = market_data['close'].values

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)

        last_sig = None
        for i in range(len(close)):
            sig = self._generate_signal(indicators, i, last_sig, dates[i], code)
            last_sig = sig
            date = pd.to_datetime(dates[i]).date()
            signal_store.set(code, date, sig)

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """计算技术指标（委托给factor_calculator）"""
        params = self.indicator_params
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        # 使用统一的因子计算器
        result = calc_indicators(close, high, low, volume, params)

        # 收益率
        result['ret_30'] = close / self._shift(close, 30) - 1

        # ATR比率
        result['atr_ratio_20'] = result['atr_20'] / (close + 1e-10)

        return result

    def _get_market_info(self, date) -> Dict[str, Any]:
        """获取指定日期的市场状态信息"""
        if self.market_regime_data is None:
            return {
                'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
                'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
                'style_regime': 'balanced', 'style_score': 0.0,
                'size_score': 0.0, 'style_confidence': 0.0,
            }

        dt = pd.to_datetime(date)
        if dt in self.market_regime_data.index:
            row = self.market_regime_data.loc[dt]
            return {
                'regime': int(row.get('regime', 0)),
                'confidence': float(row.get('confidence', 0.0)),
                'momentum_score': float(row.get('momentum_score', 0.0)),
                'trend_score': float(row.get('trend_score', 0.0)),
                'volatility': float(row.get('volatility', 0.15)),
                'is_extreme': bool(row.get('is_extreme', False)),
                'style_regime': str(row.get('style_regime', 'balanced')),
                'style_score': float(row.get('style_score', 0.0)),
                'size_score': float(row.get('size_score', 0.0)),
                'style_confidence': float(row.get('style_confidence', 0.0)),
            }
        return {
            'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
            'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
            'style_regime': 'balanced', 'style_score': 0.0,
            'size_score': 0.0, 'style_confidence': 0.0,
        }

    def _generate_signal(self, ind: dict, idx: int, last_sig, current_date=None, code=None) -> Signal:
        """生成信号"""
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0, factor_value=0.0,
                factor_name='V41', risk_vol=0.03, risk_regime=0,
                risk_confidence=0.0, risk_extreme=False, adjusted_score=0.0
            )

        # 市场状态
        market_info = self._get_market_info(current_date)
        risk_regime = market_info['regime']
        risk_extreme = market_info['is_extreme']
        style_regime = market_info.get('style_regime', 'balanced')
        style_score = market_info.get('style_score', 0.0)
        style_confidence = market_info.get('style_confidence', 0.0)

        # 获取行业类型
        industry_category = self._get_industry_category(code, current_date)

        # 因子选择和计算
        factor_name, factor_value, risk_info, is_industry = self._select_factor(
            ind, idx, risk_regime, industry_category, code=code, current_date=current_date
        )

        # 基本面因子
        fundamental_score = 0.0
        has_fundamental = False
        if self.fundamental_enabled and code:
            fundamental_score = self._get_fundamental_score(code, current_date)
            has_fundamental = fundamental_score > 0

        # 风格因子
        style_factor_score = self._get_style_score(ind, idx, market_info)

        # === 信号系统 v3 (优化版) ===
        # 核心思想: 信号分数反映相对强弱，组合层用排名排序

        # 1. 基础分数 = 因子值（已clip到合理范围）
        # clip极端值：factor_value通常在-10到10范围
        base_score = np.clip(factor_value, -10, 10)

        # 2. 基本面增强（仅对非行业因子生效）
        if not is_industry and fundamental_score > 0:
            base_score = base_score + fundamental_score * 0.1

        # === 优化：加入动量确认 ===
        # 获取短期动量（mom_5已经计算好在ind中）
        mom_5 = self._safe_get(ind, 'mom_5', idx, 0.0)
        mom_20 = self._safe_get(ind, 'mom_20', idx, 0.0)

        # 动量标准化：将动量缩放到与factor_value可比的范围
        # mom_5超过5%算强动量，10%算极强
        mom_5_norm = np.clip(mom_5 / 0.05, -2, 2)  # 标准化到-2到2
        mom_20_norm = np.clip(mom_20 / 0.10, -2, 2)

        # 组合分数：因子值为主(70%)，动量为辅(30%)
        score = base_score * 0.7 + mom_5_norm * 0.2 + mom_20_norm * 0.1

        # 4. 波动率风险指标
        risk_vol = self._safe_get(ind, 'volatility_10', idx, 0.02)

        # 5. 极端市场调整
        regime_weight = 0.9 if risk_extreme else 1.0
        adjusted_score = score * regime_weight

        # 添加标签（先生成标签再决定buy信号）
        factor_tags = []
        if has_fundamental:
            factor_tags.append('F')
        if style_confidence > 0.3:
            factor_tags.append(style_regime[:2].upper())
        factor_name = factor_name + ('_' + ''.join(factor_tags) if factor_tags else '_T')

        # 6. 交易信号
        # 买入条件：factor_value在合理范围内
        # === 核心优化：排除极端factor_value ===
        # 分析发现：factor_value > 1.5 的准确率只有44.84%（均值回归风险）
        # 方案：设置factor_value上限，过滤动量过热的股票
        FACTOR_VALUE_UPPER_LIMIT = 1.2  # factor_value上限（避免动量过热）

        # 下限阈值：市场择时调整
        market_momentum = market_info.get('momentum_score', 0.0)
        if market_momentum < -0.5:
            adjusted_buy_threshold = self.buy_threshold * 2.0
        elif market_momentum < -0.2:
            adjusted_buy_threshold = self.buy_threshold * 1.5
        elif market_momentum < 0:
            adjusted_buy_threshold = self.buy_threshold * 1.2
        else:
            adjusted_buy_threshold = self.buy_threshold

        # 买入条件：下限 < factor_value < 上限
        # === 优化2：过滤_T因子（负IC） ===
        # _T因子IC=-0.59%，买入准确率46.62%，拖累整体表现
        is_t_factor = factor_name.endswith('_T')

        buy = (factor_value > adjusted_buy_threshold and
               factor_value < FACTOR_VALUE_UPPER_LIMIT and
               abs(score) < 5.0 and
               not is_t_factor)  # 排除_T因子
        sell = factor_value < self.sell_threshold

        # 获取具体行业用于组合层减配
        specific_industry = self._get_specific_industry(code, current_date) if code else ''

        return Signal(
            buy=buy, sell=sell, score=score, factor_value=factor_value,
            factor_name=factor_name, industry=specific_industry or '',
            risk_vol=risk_vol, risk_regime=risk_regime,
            risk_confidence=market_info.get('confidence', 0.0),
            risk_extreme=risk_extreme or risk_info.get('is_high_vol', False),
            adjusted_score=adjusted_score
        )

    def _select_factor(self, ind: dict, idx: int, regime: int, industry_category: str = 'default',
                       code=None, current_date=None) -> tuple:
        """根据行业选择因子

        mode配置:
            - dynamic: 只用动态因子（不用固定因子）
            - fixed: 只用固定因子（跳过动态选择）
            - both: 动态优先，失败则用固定因子

        Returns:
            (factor_name, factor_value, risk_info, is_industry_factor)
        """
        # 动态因子优先
        if self.factor_mode in ['dynamic', 'both']:
            if self.dynamic_factor_selector.enabled and code and current_date:
                result = self._select_factor_dynamic(ind, idx, regime, code, current_date)
                if result:
                    self._stats['dynamic_success'] += 1
                    return result
                # 动态选择失败
                if self.factor_mode == 'dynamic' and not self.factor_fallback_to_fixed:
                    # mode=dynamic且不允许fallback，直接用默认因子
                    self._stats['dynamic_fallback_default'] += 1
                    factor_name, factor_value, risk_info = self._calculate_default_factor(ind, idx, regime, industry_category)
                    return factor_name, factor_value, risk_info, False
                else:
                    self._stats['dynamic_fallback_fixed'] += 1

        # 固定因子（行业特定或默认）
        if self.factor_mode in ['fixed', 'both']:
            if self.industry_factor_enabled and code and current_date:
                specific_industry = self._get_specific_industry(code, current_date)
                # === 阶段1优化：禁用负IC静态因子 ===
                # 金融行业的静态因子IC=-3.53%（负向），应跳过使用默认因子
                SKIP_STATIC_INDUSTRIES = ['金融']  # 这些行业跳过静态因子，使用默认因子
                if specific_industry and specific_industry in INDUSTRY_FACTOR_CONFIG:
                    if specific_industry not in SKIP_STATIC_INDUSTRIES:
                        result = self._calculate_industry_factor_score(ind, idx, specific_industry,
                                                                       code=code, current_date=current_date)
                        if result:
                            self._stats['fixed_industry'] += 1
                            factor_name, factor_value, risk_info = result
                            if regime == -1:  # 熊市
                                factor_value = factor_value * 0.7
                                factor_name = factor_name + '_bear'
                            factor_name = factor_name + f'_{specific_industry[:2]}'
                            return factor_name, factor_value, risk_info, True

        # 默认因子组合（固定因子的兜底）
        self._stats['fixed_default'] += 1
        factor_name, factor_value, risk_info = self._calculate_default_factor(ind, idx, regime, industry_category)
        return factor_name, factor_value, risk_info, False

    def _select_factor_dynamic(self, ind: dict, idx: int, regime: int,
                                code=None, current_date=None) -> Optional[tuple]:
        """动态因子选择

        使用DynamicFactorSelector在每个时点动态选择最优因子

        Returns:
            (factor_name, factor_value, risk_info, is_industry_factor) or None
        """
        if not code or not current_date:
            return None

        # DEBUG: 记录尝试
        # 获取当前日期的字符串形式
        if hasattr(current_date, 'date'):
            current_date_str = str(current_date.date())
        else:
            current_date_str = str(current_date)

        # 获取股票所属行业
        specific_industry = self._get_specific_industry(code, current_date)
        if not specific_industry:
            return None

        # 检查因子数据是否存在
        if self.dynamic_factor_selector.factor_df is None:
            return None

        # 获取动态选择的因子
        try:
            all_dates = self.dynamic_factor_selector._all_dates_cache
            if not all_dates:
                return None
            industry_factors = self.dynamic_factor_selector.select_factors_for_date(current_date_str, all_dates)
        except Exception as e:
            return None

        if not industry_factors or specific_industry not in industry_factors:
            return None

        # 提取因子列表和质量指标（新返回格式）
        selected_info = industry_factors[specific_industry]
        if not selected_info or 'factors' not in selected_info:
            return None
        selected_factors = selected_info['factors']
        factor_weights = selected_info.get('weights', None)  # IC权重列表
        dyn_quality = selected_info.get('quality', 0)

        # 条件fallback: DYN质量过低时返回None，触发fallback到FIXED
        # 阈值0.05: combined_ir < 0.05认为动态因子质量不足
        # combined_ir > 0 表示因子有正向预测能力
        # 阈值0.05: 要求动态因子质量达到较高水平才使用
        DYN_QUALITY_THRESHOLD = 0.05  # 阈值0.05: combined_ir < 0.05认为动态因子质量不足
        if dyn_quality < DYN_QUALITY_THRESHOLD:
            return None

        # === 阶段1优化：检查因子数量，1F fallback到静态 ===
        # 1F因子IC仅0.64%（接近随机），而3F因子IC=7.25%
        # 当只有1个因子时，需要更严格的质量门槛
        n_factors = len(selected_factors)
        min_factor_count = self.dynamic_factor_selector.min_factor_count
        min_ic_1f = self.dynamic_factor_selector.min_ic_1f

        if n_factors < min_factor_count:
            # 1F因子需要达到更高IC门槛才使用
            if n_factors == 1 and dyn_quality < min_ic_1f:
                return None  # 1F质量不足，fallback到静态因子

        # 计算动态因子得分
        factor_scores = []
        valid_weights = []
        valid_factors = []

        for i, factor_name in enumerate(selected_factors):
            # 基本面因子
            if factor_name.startswith('fund_'):
                factor_val = self._get_fundamental_factor_value(code, current_date, factor_name)
            else:
                # 技术因子
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 验证因子值的合理性
            if factor_val is not None and not np.isnan(factor_val) and not np.isinf(factor_val):
                # 检查是否在合理范围（技术因子通常在-10到10范围）
                if abs(factor_val) > 1000:
                    # 记录异常因子值用于调试
                    import warnings
                    warnings.warn(f'Extreme factor value: {factor_name}={factor_val:.2e} for {code} on {current_date}')
                    # 将极端值裁剪到合理范围
                    factor_val = np.sign(factor_val) * 1000
                factor_scores.append(factor_val)
                w = factor_weights[i] if factor_weights and i < len(factor_weights) else 1.0
                valid_weights.append(w)
                valid_factors.append(factor_name)

        if not factor_scores:
            return None

        # IC加权平均（使用带符号的权重，保留因子方向）
        if len(valid_weights) > 0 and sum(abs(w) for w in valid_weights) > 0:
            weights_arr = np.array(valid_weights)
            # 对于带符号的权重，用绝对值之和归一化（保留方向信息）
            weights_arr = weights_arr / sum(abs(w) for w in valid_weights)
            factor_value = np.sum(np.array(factor_scores) * weights_arr)
        else:
            factor_value = np.mean(factor_scores)

        # 最终安全检查：确保factor_value在合理范围
        # 技术因子通常在-10到10范围，IC加权平均后也应该在此范围
        if abs(factor_value) > 100:
            import warnings
            warnings.warn(f'Extreme factor_value after weighted sum: {factor_value:.2e}, clipping to ±100')
            factor_value = np.clip(factor_value, -100, 100)

        # 熊市调整
        if regime == -1:
            factor_value = factor_value * 0.7

        factor_name = f'DYN_{specific_industry[:4]}_{len(valid_factors)}F'
        risk_info = {'is_high_vol': False, 'dynamic_factor': True, 'n_factors': len(valid_factors),
                     'dyn_quality': dyn_quality}

        return factor_name, factor_value, risk_info, True

    def _get_fundamental_factor_value(self, code, current_date, factor_name: str) -> Optional[float]:
        """获取基本面因子值 - 使用clip+tanh压缩极端值"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
            return None

        try:
            if factor_name == 'fund_score':
                raw = self.fundamental_data.get_fundamental_score(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-100, min(100, raw))  # clip到合理范围
                    return np.tanh((raw_clipped - 50) / 50)  # 中心化后压缩
                return None
            elif factor_name == 'fund_profit_growth':
                raw = self.fundamental_data.get_profit_growth(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-100, min(100, raw))  # clip极端值
                    return np.tanh(raw_clipped)
                return None
            elif factor_name == 'fund_roe':
                raw = self.fundamental_data.get_roe(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-50, min(50, raw))  # clip极端值
                    return np.tanh((raw_clipped - 10) / 20)  # 中心化后压缩
                return None
            elif factor_name == 'fund_revenue_growth':
                raw = self.fundamental_data.get_revenue_growth(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-100, min(100, raw))  # clip极端值
                    return np.tanh(raw_clipped)
                return None
            elif factor_name == 'fund_eps':
                raw = self.fundamental_data.get_eps(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-10, min(10, raw))  # clip极端值
                    return np.tanh(raw_clipped)
                return None
            elif factor_name == 'fund_cf_to_profit':
                raw = self.fundamental_data.get_cf_to_profit(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-5, min(5, raw))  # clip极端值
                    return np.tanh(raw_clipped - 1)  # 偏移后压缩
                return None
            elif factor_name == 'fund_debt_ratio':
                raw = self.fundamental_data.get_debt_ratio(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(0, min(100, raw))  # clip到[0,100]
                    return np.tanh((50 - raw_clipped) / 50)  # 反向+压缩
                return None
            elif factor_name == 'fund_gross_margin':
                raw = self.fundamental_data.get_gross_margin(code, current_date)
                if raw is not None and not np.isnan(raw):
                    raw_clipped = max(-20, min(80, raw))  # clip极端值
                    return np.tanh((raw_clipped - 30) / 30)  # 中心化后压缩
                return None
        except:
            pass
        return None

    def _calculate_default_factor(self, ind: dict, idx: int, regime: int, industry_category: str) -> tuple:
        """计算默认因子组合

        使用稳定的基本面+动量因子组合，值域标准化到[-1, 1]左右
        """
        # 获取原始指标
        vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
        mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
        mom_20 = self._safe_get(ind, 'mom_20', idx, 0)

        # 动量×低波动因子（最稳定的技术面因子）
        # 使用 np.tanh 压缩替代 clip，保留相对大小信息
        # np.tanh(x) 将任意值平滑压缩到 (-1, 1)，比 clip 更好地保留极端值信息
        mom_lowvol = mom_20 * (1 - vol_10 * 10)  # 低波动加成
        mom_lowvol = np.tanh(mom_lowvol / 0.5) * 0.5  # 平滑压缩到 [-0.5, 0.5]

        # 组合因子值
        factor_value = mom_lowvol

        # 熊市折扣
        if regime == -1:
            factor_value = factor_value * 0.7

        factor_name = 'DEFAULT_MomLowVol'
        if regime == -1:
            factor_name = factor_name + '_bear'

        risk_info = {'is_high_vol': vol_10 > 0.04}
        return factor_name, factor_value, risk_info

    def _calculate_industry_factor_score(self, ind: dict, idx: int, industry: str,
                                           code=None, current_date=None) -> tuple:
        """计算行业特定因子得分

        直接使用原始因子值（与因子验证一致），然后等权平均合成
        """
        config = INDUSTRY_FACTOR_CONFIG.get(industry)
        if not config:
            return None

        factors = config.get('factors', [])
        direction = config.get('direction', {}) if 'direction' in config else {}

        factor_scores = []
        valid_factors = []

        for i, factor_name in enumerate(factors):
            factor_val = None

            # 基本面因子：从 fundamental_data 获取，并标准化到合理范围
            if factor_name.startswith('fund_'):
                if hasattr(self, 'fundamental_data') and self.fundamental_data and code and current_date:
                    try:
                        if factor_name == 'fund_score':
                            raw = self.fundamental_data.get_fundamental_score(code, current_date)
                            # fund_score 可能异常（负值或极大值），用tanh压缩到[-1,1]
                            # 先做合理范围clip，再tanh
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-100, min(100, raw))  # 先clip到合理范围
                                factor_val = np.tanh((raw_clipped - 50) / 50)  # 中心化后压缩
                            else:
                                factor_val = None
                        elif factor_name == 'fund_profit_growth':
                            raw = self.fundamental_data.get_profit_growth(code, current_date)
                            # 利润增长率可能极端，用tanh压缩到[-1,1]
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-100, min(100, raw))  # clip极端值
                                factor_val = np.tanh(raw_clipped)
                            else:
                                factor_val = None
                        elif factor_name == 'fund_roe':
                            raw = self.fundamental_data.get_roe(code, current_date)
                            # ROE 可能为负，用tanh压缩
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-50, min(50, raw))
                                factor_val = np.tanh((raw_clipped - 10) / 20)  # 中心化后压缩
                            else:
                                factor_val = None
                        elif factor_name == 'fund_revenue_growth':
                            raw = self.fundamental_data.get_revenue_growth(code, current_date)
                            # 营收增长率，用tanh压缩
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-100, min(100, raw))
                                factor_val = np.tanh(raw_clipped)
                            else:
                                factor_val = None
                        elif factor_name == 'fund_eps':
                            raw = self.fundamental_data.get_eps(code, current_date)
                            # EPS 可能为负或极大，用tanh压缩
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-10, min(10, raw))
                                factor_val = np.tanh(raw_clipped)
                            else:
                                factor_val = None
                        elif factor_name == 'fund_cf_to_profit':
                            raw = self.fundamental_data.get_cf_to_profit(code, current_date)
                            # 现金流/利润比可能极端，用tanh压缩
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-5, min(5, raw))
                                factor_val = np.tanh(raw_clipped - 1)
                            else:
                                factor_val = None
                        elif factor_name == 'fund_debt_ratio':
                            raw = self.fundamental_data.get_debt_ratio(code, current_date)
                            # 负债率，反向（低负债好），用tanh压缩
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(0, min(100, raw))
                                factor_val = np.tanh((50 - raw_clipped) / 50)
                            else:
                                factor_val = None
                        elif factor_name == 'fund_gross_margin':
                            raw = self.fundamental_data.get_gross_margin(code, current_date)
                            # 毛利率，用tanh压缩
                            if raw is not None and not np.isnan(raw):
                                raw_clipped = max(-20, min(80, raw))
                                factor_val = np.tanh((raw_clipped - 30) / 30)
                            else:
                                factor_val = None
                    except:
                        factor_val = None
            else:
                # 技术因子：从 ind 字典获取
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 直接使用原始因子值
            if factor_val is not None and not np.isnan(factor_val):
                factor_dir = direction.get(factor_name, 1)
                factor_scores.append(factor_val * factor_dir)
                valid_factors.append(factor_name)

        if not factor_scores:
            return None

        # 根据method选择合并方式 (2026-03-18优化: 默认使用equal)
        method = config.get('method', 'equal')
        weights = config.get('weights', None)

        if method == 'weighted' and weights and len(weights) >= len(factor_scores):
            # 使用指定权重
            w = weights[:len(factor_scores)]
            total_w = sum(w)
            factor_value = sum(s * w_i for s, w_i in zip(factor_scores, w)) / total_w
        elif method == 'equal':
            # 等权平均（推荐，更稳健）
            factor_value = np.mean(factor_scores)
        else:
            # Top1（保留兼容性，但不推荐）
            factor_value = factor_scores[0]

        return f'IND_{industry[:4]}', factor_value, {'is_high_vol': False, 'industry_factor': True, 'n_factors': len(factor_scores)}

    def _get_style_score(self, ind: dict, idx: int, market_info: dict) -> float:
        """获取风格因子分数"""
        style_regime = market_info.get('style_regime', 'balanced')
        style_confidence = market_info.get('style_confidence', 0.0)

        if style_confidence < 0.3 or style_regime == 'balanced':
            return 0.0

        if style_regime == 'small_cap':
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return -price_pos * 0.5 + 0.25
        elif style_regime == 'large_cap':
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return price_pos * 0.5 - 0.25
        elif style_regime == 'growth':
            mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
            # 使用 np.tanh 压缩替代 clip，保留相对大小
            return np.tanh(mom_10 * 2) * 0.3
        elif style_regime == 'value':
            vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
            # 使用 np.tanh 压缩替代 clip，保留相对大小
            return np.tanh((0.02 - vol_10) * 5) * 0.3
        return 0.0

    def _get_industry_category(self, code, current_date) -> str:
        """获取股票所属行业类型"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 'default'
        try:
            industry = self.fundamental_data.get_industry(code, current_date)
            if not industry:
                return 'default'
            # 简化：返回行业本身
            return industry
        except:
            return 'default'

    def _get_specific_industry(self, code, current_date) -> str:
        """获取具体行业名（使用INDUSTRY_KEYWORDS映射）"""
        # 动态因子模式下，允许没有 fundamental_data
        # 但如果没有 fundamental_data，尝试从 industry_codes 推断
        has_fd = hasattr(self, 'fundamental_data') and self.fundamental_data
        has_ic = hasattr(self, 'industry_codes') and self.industry_codes

        # 首先尝试从 industry_codes 查找（更可靠）
        if has_ic:
            for ind_name, codes in self.industry_codes.items():
                if code in codes:
                    return ind_name

        # 如果没找到，尝试从 fundamental_data 获取
        if has_fd:
            try:
                raw_industry = self.fundamental_data.get_industry(code, current_date)
                if not raw_industry:
                    return None

                # 清理行业名（去除Ⅱ、Ⅲ等特殊字符）
                cleaned_industry = raw_industry.replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()

                # 使用 INDUSTRY_KEYWORDS 将原始行业名转换为配置行业键
                for config_key, keywords in INDUSTRY_KEYWORDS.items():
                    # 精确匹配或包含匹配
                    if raw_industry in keywords or cleaned_industry in keywords:
                        # 检查该行业键是否在 INDUSTRY_FACTOR_CONFIG 中
                        if config_key in INDUSTRY_FACTOR_CONFIG:
                            return config_key
                    # 额外检查：关键词是否包含在原始行业中
                    for kw in keywords:
                        if kw in raw_industry or kw in cleaned_industry:
                            if config_key in INDUSTRY_FACTOR_CONFIG:
                                return config_key
            except:
                pass
        return None

    def _get_fundamental_score(self, code, current_date) -> float:
        """获取基本面因子评分"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 0.0

        score = 0.0
        roe = self.fundamental_data.get_roe(code, current_date)
        if roe is not None:
            if roe > 0.15:
                score += 0.35
            elif roe > 0.10:
                score += 0.25
            elif roe > 0.05:
                score += 0.15

        profit_growth = self.fundamental_data.get_profit_growth(code, current_date)
        if profit_growth is not None:
            if profit_growth > 0.50:
                score += 0.30
            elif profit_growth > 0.20:
                score += 0.20
            elif profit_growth > 0:
                score += 0.10

        revenue_growth = self.fundamental_data.get_revenue_growth(code, current_date)
        if revenue_growth is not None:
            if revenue_growth > 0.30:
                score += 0.20
            elif revenue_growth > 0.15:
                score += 0.12
            elif revenue_growth > 0:
                score += 0.05

        eps = self.fundamental_data.get_eps(code, current_date)
        if eps is not None and eps > 0:
            if eps > 1.0:
                score += 0.20
            elif eps > 0.5:
                score += 0.12

        return min(1.0, score)

    # === 辅助函数 ===
    def _safe_get(self, ind: dict, key: str, idx: int, default: float = 0.0) -> float:
        arr = ind.get(key)
        if arr is None:
            return default
        # 检查是否是真正的标量（不包括 numpy 标量）
        if np.isscalar(arr) and not hasattr(arr, '__len__'):
            if isinstance(arr, (int, float)) and (np.isnan(arr) or np.isinf(arr)):
                return default
            return arr
        if hasattr(arr, '__len__') and not isinstance(arr, str):
            if len(arr) <= idx:
                return default
            val = arr[idx]
            # 检查 NaN 和 Inf
            if isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val)):
                return default
            return val
        return default

    def _sma(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        result[window-1:] = np.convolve(arr, np.ones(window)/window, mode='valid')
        return result

    def _ema(self, arr, span):
        result = np.zeros_like(arr, dtype=float)
        result[0] = arr[0]
        alpha = 2 / (span + 1)
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result

    def _rsi(self, close, window):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = self._sma(gain, window)
        avg_loss = self._sma(loss, window)
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _bollinger(self, close, window, num_std):
        middle = self._sma(close, window)
        n = len(close)
        if n < window:
            std = np.zeros(n)
        else:
            from numpy.lib.stride_tricks import sliding_window_view
            sw = sliding_window_view(close, window)
            std = np.zeros(n)
            std[window:] = sw.std(axis=1)[:n-window]
        return middle + num_std * std, middle, middle - num_std * std

    def _atr(self, high, low, close, window):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        return self._sma(tr, window)

    def _rolling_max(self, arr, window):
        """滚动最大值 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        # sliding_window_view creates (n - window + 1, window) windows
        # But we need (n - window) results for indices [window, n-1]
        result = np.full(n, np.nan)
        result[window:] = sw.max(axis=1)[:n-window]
        return result

    def _rolling_min(self, arr, window):
        """滚动最小值 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        result = np.full(n, np.nan)
        result[window:] = sw.min(axis=1)[:n-window]
        return result

    def _shift(self, arr, periods):
        result = np.zeros_like(arr, dtype=float)
        result[periods:] = arr[:-periods]
        result[:periods] = np.nan
        return result

    def _rolling_std(self, arr, window):
        """滚动标准差 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        result = np.full(n, np.nan)
        result[window:] = sw.std(axis=1)[:n-window]
        return result

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
