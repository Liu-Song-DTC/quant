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
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper

from .signal import Signal
from .signal_store import SignalStore
from .config_loader import load_config
from .industry_mapping import INDUSTRY_KEYWORDS, get_industry_category
from .factor_calculator import calculate_indicators as calc_indicators, compute_composite_factors, compress_fundamental_factor
from .multi_timeframe import MultiTimeframeAnalyzer
import yaml
import os

import warnings


from .dynamic_factor_selector import (
    DynamicFactorSelector,
    FACTOR_FAMILIES,
    get_factor_family,
    _compute_date_chunk,
    _compute_date_chunks_worker,
)
# 向后兼容别名
_get_factor_family = get_factor_family

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


def _safe_get_arr(ind: dict, key: str, n: int, default):
    """向量化版 _safe_get：从ind字典取key对应的numpy数组，不存在时返回填充默认值的数组。"""
    arr = ind.get(key)
    if arr is not None:
        arr = np.asarray(arr)
        if len(arr) == n:
            return arr
    # key不存在或长度不匹配 → 返回填充默认值的数组
    if isinstance(default, (int, float)):
        return np.full(n, default, dtype=np.float64)
    elif isinstance(default, bool):
        return np.full(n, default, dtype=bool)
    return np.full(n, default)


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
            'dynamic_skip_low_ic': 0,  # 低IC行业跳过动态因子
            'dynamic_fallback_fixed': 0,
            'dynamic_fallback_default': 0,
            'dynamic_fallback_none': 0,  # 无高质量因子时不产生信号
            'fixed_industry': 0,
            'fixed_default': 0,
            'ic_values': [],
        }

        # 动态买入阈值：滚动缓冲区（最近800个score值，约半年数据，提高市场风格切换响应速度）
        self._factor_value_buffer = deque(maxlen=800)
        # 行业内因子值缓存：{industry: deque(maxlen=500)}
        self._industry_factor_cache = {}
        # 旧路径状态（向后兼容 _generate_signal 独立调用）
        self._pct_counter = 0
        self._cached_buy_threshold = self.buy_threshold
        self._cached_sell_threshold = self.sell_threshold

    def _load_config(self):
        """从配置文件加载参数"""
        config_loader = load_config()

        # 信号阈值（从配置文件加载）
        signal_config = config_loader.get('signal', {})
        self.buy_threshold = signal_config.get('buy_threshold', 0.12)
        self.sell_threshold = signal_config.get('sell_threshold', -0.2)

        # 卖出趋势保护阈值（防止趋势中途虚假卖出）
        self.trend_sell_threshold_strong = signal_config.get('trend_sell_threshold_strong', -0.15)
        self.trend_sell_threshold_weak = signal_config.get('trend_sell_threshold_weak', -0.10)
        self.trend_sell_ti_threshold = signal_config.get('trend_sell_ti_threshold', 0.05)
        self.trend_sell_ti_relax = signal_config.get('trend_sell_ti_relax', -0.20)

        # Fix#4: 动态阈值百分位 — 从YAML正确读取 (self.config可能为{})
        self._buy_threshold_pct_map = {
            1: signal_config.get('buy_threshold_pct_bull', 0.50),
            0: signal_config.get('buy_threshold_pct', 0.45),  # YAML默认0.45
            -1: signal_config.get('buy_threshold_pct_bear', 0.70),
        }
        self._buy_threshold_pct = self._buy_threshold_pct_map[0]

        # 基本面因子配置
        self.fundamental_enabled = True
        self.fundamental_weight = config_loader.get('fundamental_weight', 0.3)

        # 缠论融合权重: 因子分 vs 缠论分的比例, 0.35 = chan占35%
        self.chan_fusion_weight = config_loader.get('signal.chan_fusion_weight', 0.35)

        # 风格因子开关
        self.style_enabled = config_loader.get('style_factor_enabled', True)

        # 行业因子开关
        industry_config = config_loader.get_industry_factor_config()
        self.industry_factor_enabled = industry_config.get('enabled', True)

        # 技术指标参数
        self.indicator_params = config_loader.get_indicator_params()

        # 动态因子选择器
        self.dynamic_factor_selector = DynamicFactorSelector()

        # === ML预测层 ===
        ml_config = config_loader.get('ml', {})
        self.ml_enabled = ml_config.get('enabled', False)
        self.ml_blend_weight = ml_config.get('blend_weight', 0.30)
        self.ml_predictor = None
        self._ml_predictions = {}  # {(code, date): float}

        # === 缠论增强配置 ===
        chan_config = config_loader.get('chan_theory', {})
        div_config = chan_config.get('divergence', {})
        self.chan_bottom_div_threshold = div_config.get('bottom_div_threshold', 0.3)
        self.chan_top_div_threshold = div_config.get('top_div_threshold', 0.3)

        # 动态因子模式配置：dynamic(仅动态) / fixed(仅固定) / both(动态优先+固定兜底)
        self.factor_mode = config_loader.get('factor_mode', 'both')
        # 兼容两种配置key: fallback_to_fixed 和 fallback_to_static
        self.factor_fallback_to_fixed = config_loader.get('dynamic_factor.fallback_to_fixed',
                                                          config_loader.get('dynamic_factor.fallback_to_static', True))

        # === 统一多时间框架分析器 ===
        self.mtf_analyzer = MultiTimeframeAnalyzer(config_loader.config if config_loader.config else {})

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

    def set_ml_predictor(self, predictor):
        """设置ML预测器"""
        self.ml_predictor = predictor

    def set_ml_predictions(self, predictions: dict):
        """设置预计算的ML预测值 {(code, date): float}"""
        self._ml_predictions = predictions

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
        print(f"动态跳过(低IC):   {stats['dynamic_skip_low_ic']:6d} ({100*stats['dynamic_skip_low_ic']/total:.1f}%)")
        print(f"动态->固定fallback: {stats['dynamic_fallback_fixed']:6d} ({100*stats['dynamic_fallback_fixed']/total:.1f}%)")
        print(f"动态->默认fallback: {stats['dynamic_fallback_default']:6d} ({100*stats['dynamic_fallback_default']/total:.1f}%)")
        print(f"动态->无信号: {stats['dynamic_fallback_none']:6d} ({100*stats['dynamic_fallback_none']/total:.1f}%)")
        print(f"固定行业因子:    {stats['fixed_industry']:6d} ({100*stats['fixed_industry']/total:.1f}%)")
        print(f"固定默认因子:    {stats['fixed_default']:6d} ({100*stats['fixed_default']/total:.1f}%)")
        print(f"总计:            {total}")
        print("==================================\n")

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore,
                 latest_only: bool = False):
        """生成信号（向量化批处理：收集标量→数组装配→向量化阈值→买卖判定）

        latest_only=True 时采用混合路径：
        - 所有 bar 用快速默认因子计算分数（用于动态阈值 quantile 估计）
        - 仅最新 bar 走完整 _select_factor + _get_chan_boost 链
        - 信号构造也只处理最新 bar
        用于实盘选股，回测不可用（需要全部历史信号）。
        """
        dates = market_data["datetime"].values
        close = market_data['close'].values

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)
        self._preload_stock_fundamentals(code, dates)

        n = len(close)

        # 向量化预计算市场状态（避免逐 bar _get_market_info）
        regimes = self._precompute_regimes(dates, n)

        # ===== Phase 1: 逐bar收集复杂方法调用的标量结果 =====
        scalars = self._collect_bar_scalars(indicators, code, dates, n,
                                            regimes=regimes, latest_only=latest_only)

        # ===== Phase 2: 向量化分数装配 =====
        result = self._vectorized_score_assembly(scalars, indicators, n, code)
        del scalars

        # ===== Phase 3: 动态阈值（历史 bar 有快速分数，可用于 quantile 估计） =====
        buy_thresholds, sell_thresholds = self._compute_dynamic_thresholds(
            result['score'], result['risk_regime']
        )

        # ===== Phase 4: 买卖判定 + 构造Signal对象 =====
        valid = result['valid']
        close_arr = _safe_get_arr(indicators, 'close', n, 0.0)
        ma20_arr = _safe_get_arr(indicators, 'ma20', n, 0.0)
        bottom_div_arr = _safe_get_arr(indicators, 'bottom_divergence', n, 0.0)
        top_div_arr = _safe_get_arr(indicators, 'top_divergence', n, 0.0)

        bar_indices = [n - 1] if latest_only else range(n)
        for i in bar_indices:
            date = pd.to_datetime(dates[i]).date()

            if i < 60 or not valid[i]:
                # 无效bar返回空信号
                sig = Signal(
                    buy=False, sell=False, score=0.0, factor_value=0.0,
                    factor_name='V41' if i < 60 else 'NONE',
                    risk_vol=0.03, risk_regime=int(result['risk_regime'][i]),
                    risk_confidence=0.0, risk_extreme=bool(result['risk_extreme'][i]),
                    adjusted_score=0.0, pre_discount_score=0.0,
                    industry=result['industry'][i] if code else '',
                    exhaustion_risk=0.0, gap_breakout_confirm=0.0,
                    stroke_phase=0.0, top_fractal_volume=0.0,
                    ma_trend_up=False, profit_declining=False,
                    mom_60d=0.0, dist_ma60=0.0, max_dd_20d=0.0, vol_regime=1.0,
                    weekly_trend_up=False, monthly_trend_up=False,
                    weekly_trend_strength=0.0, monthly_trend_strength=0.0,
                    mtf_alignment_score=0.0, mtf_discount_factor=1.0,
                    weekly_pattern_signal=0.0, nearest_resistance_pct=0.0,
                    nearest_support_pct=0.0,
                    _chan_buy_signal=False, _chan_sell_signal=False, _dist_ma20=0.0,
                )
                signal_store.set(code, date, sig)
                continue

            # 买入判定
            score = float(result['score'][i])
            buy_th = float(buy_thresholds[i])
            sell_th = float(sell_thresholds[i])
            chan_buy_sig = bool(result['_chan_buy_signal'][i])
            chan_sell_sig = bool(result['_chan_sell_signal'][i])
            bp_buy = int(result['chan_buy_point'][i])
            sp_sell = int(result['chan_sell_point'][i])
            sl = int(result['signal_level'][i])
            dist_ma20 = float(result['_dist_ma20'][i])

            chan_force_buy = chan_buy_sig and (sl >= 2 or bp_buy == 1)
            close_p = float(close_arr[i])
            ma20_v = float(ma20_arr[i])
            price_above_ma20 = close_p > 0 and ma20_v > 0 and close_p > ma20_v

            # ti地板: trend_initiation强势时禁止lowvol因子把score扣成负数
            # ti正确识别趋势起点(0.3~0.7)，但trend_lowvol因高波动率而扣分
            # 导致大牛股系统性漏掉。ti>0.15 且站上MA20时设score≥ti×0.5
            trend_init = float(_safe_get_arr(indicators, 'trend_initiation', n, 0.0)[i])
            if trend_init > 0.15 and price_above_ma20 and not np.isnan(score):
                ti_floor = trend_init * 0.5
                if score < ti_floor:
                    score = ti_floor

            # MA60趋势过滤：非缠论买点必须站上MA60，防止下跌趋势中反复抄底
            ma60_v = float(_safe_get_arr(indicators, 'ma60', n, 0.0)[i])
            price_above_ma60 = close_p > 0 and ma60_v > 0 and close_p > ma60_v

            b1_strong = (bp_buy == 1 and chan_buy_sig and sl >= 2)
            if b1_strong:
                price_ok = dist_ma20 > -0.05
            elif chan_force_buy:
                price_ok = price_above_ma20
            else:
                # 非缠论买入: 必须多头排列(MA20>MA60) + price站上MA20/MA60，确认中期趋势向上
                # 避免下跌趋势中的反弹被误认为反转(82%套牢率根因)
                ma20_above_ma60 = ma20_v > 0 and ma60_v > 0 and ma20_v > ma60_v
                price_ok = price_above_ma20 and price_above_ma60 and ma20_above_ma60

            is_b3 = (bp_buy == 3 and chan_buy_sig)
            if is_b3 and sl >= 2:
                max_dist = 0.40
            elif is_b3:
                max_dist = 0.35
            elif b1_strong:
                max_dist = 0.25
            else:
                max_dist = 0.30
            price_not_extended = dist_ma20 < max_dist

            # 趋势突破辅助入场: 技术面强势时用技术score兜底因子score
            # 解决行业因子对部分股票天然看空导致大牛股全部漏掉的问题
            vol_spike = float(result['volume_ratio'][i]) > 1.5 if 'volume_ratio' in result else False
            price_above_ma20 = close_p > 0 and ma20_v > 0 and close_p > ma20_v

            # 计算纯技术面score（0~1），用于兜底因子score为负的情况
            tech_score = 0.0
            if vol_spike and price_above_ma20:
                # MA20斜率: 5日MA20变化率
                if i >= 5 and ma20_v > 0:
                    ma20_slope = (ma20_arr[i] - ma20_arr[i-5]) / ma20_arr[i-5] if ma20_arr[i-5] > 0 else 0
                else:
                    ma20_slope = 0
                # 成交量确认
                vol_score = min(float(result['volume_ratio'][i]) / 3.0, 1.0) if 'volume_ratio' in result else 0
                # 距MA20距离（越近越好，突破时最好在MA20附近）
                dist_score = max(0, 1.0 - abs(dist_ma20) / 0.15) if not np.isnan(dist_ma20) else 0.5
                # 5日动量
                mom5d = (close_p / float(close_arr[max(0,i-5)]) - 1) if i >= 5 and float(close_arr[max(0,i-5)]) > 0 else 0
                mom_score = min(max(mom5d / 0.10, 0), 1.0)  # 5%+ momentum = full score
                # 综合技术score
                tech_score = (ma20_slope * 0.35 + vol_score * 0.25 + dist_score * 0.20 + mom_score * 0.20)
                tech_score = max(0.0, min(1.0, tech_score))

            # 当技术面强势(score>0.4)时，用max(因子score, 技术score)作为有效score
            effective_score = score
            if tech_score > 0.4 and not np.isnan(score):
                effective_score = max(score, tech_score * 0.6)  # 技术score打6折，避免完全覆盖因子

            trend_breakout = (vol_spike and price_above_ma20
                            and tech_score > 0.4
                            and not np.isnan(effective_score) and effective_score > buy_th * 0.7)

            buy = (not np.isnan(score) and
                   (effective_score > buy_th or chan_force_buy or trend_breakout) and
                   price_ok and price_not_extended)

            # 趋势确认: 非缠论买入时分級检查trend_initiation
            # 均线多头排列时容忍ti略负（趋势结构已健康），否则仍需ti>0
            # 趋势突破买入golden_cross已确认趋势方向，免除ti检查
            if buy and not chan_force_buy and not trend_breakout:
                trend_init = float(_safe_get_arr(indicators, 'trend_initiation', n, 0.0)[i])
                ma20_above_ma60 = ma20_v > 0 and ma60_v > 0 and ma20_v > ma60_v
                if ma20_above_ma60 and price_above_ma20:
                    # 均线多头排列: 趋势结构已确认，容忍ti略负
                    if trend_init <= -0.03:
                        buy = False
                elif ma20_above_ma60:
                    if trend_init <= -0.01:
                        buy = False
                else:
                    if trend_init <= 0:
                        buy = False

            chan_force_sell = chan_sell_sig and (sl <= -2 or sp_sell >= 1)
            # MA60止损: 跌破MA60且score转负 → 强制卖出，截断下跌趋势中的持仓
            ma60_stop = (not price_above_ma60) and score < 0 and close_p > 0 and ma60_v > 0

            # 趋势保护: 三级卖出阈值，防止趋势中途虚假卖出
            # 旧动量因子在趋势起点天然滞后(20日窗口仍包含下跌段)，score短暂为负不应卖出
            trend_init = float(_safe_get_arr(indicators, 'trend_initiation', n, 0.0)[i])
            ma20_above_ma60 = ma20_v > 0 and ma60_v > 0 and ma20_v > ma60_v
            if ma20_above_ma60 and price_above_ma20:
                # 强趋势: MA20>MA60多头排列，大幅提高卖出容忍度
                if trend_init > self.trend_sell_ti_threshold:
                    sell_tolerance = self.trend_sell_ti_relax
                else:
                    sell_tolerance = self.trend_sell_threshold_strong
            elif price_above_ma20 and ma20_v > 0:
                sell_tolerance = self.trend_sell_threshold_weak
            else:
                sell_tolerance = sell_th
            trend_sell = score < sell_tolerance or ma60_stop or chan_force_sell

            sell = not np.isnan(score) and trend_sell

            # 因子标签
            factor_tags = []
            if result['has_fundamental'][i]:
                factor_tags.append('F')
            if float(result['style_confidence'][i]) > 0.3:
                factor_tags.append(str(result['style_regime'][i])[:2].upper())
            if float(result['smart_money'][i]) > 0.15:
                factor_tags.append('V')
            if float(bottom_div_arr[i]) > 0.3 or float(top_div_arr[i]) > 0.3:
                factor_tags.append('D')
            n_buy = int(result['n_buy_systems'][i])
            if n_buy >= 2:
                factor_tags.append(f'R{n_buy}')
            fn = result['factor_name'][i]
            fn = fn + ('_' + ''.join(factor_tags) if factor_tags else '_T')

            sig = Signal(
                buy=buy, sell=sell, score=score,
                factor_value=float(result['factor_value'][i]),
                factor_name=fn,
                industry=str(result['industry'][i]) if result['industry'][i] else '',
                risk_vol=float(result['risk_vol'][i]),
                risk_regime=int(result['risk_regime'][i]),
                risk_confidence=float(result['risk_confidence'][i]),
                risk_extreme=bool(result['risk_extreme'][i]),
                adjusted_score=float(result['adjusted_score'][i]),
                pre_discount_score=float(result['pre_discount_score'][i]),
                factor_quality=float(result['factor_quality'][i]),
                signal_confidence=float(result['signal_confidence'][i]),
                chan_divergence_type=str(result['chan_divergence_type'][i]),
                chan_divergence_strength=float(result['chan_divergence_strength'][i]),
                chan_structure_score=float(result['chan_structure_score'][i]),
                chan_buy_point=int(result['chan_buy_point'][i]),
                chan_sell_point=int(result['chan_sell_point'][i]),
                signal_level=int(result['signal_level'][i]),
                resonance_systems=int(result['resonance_systems'][i]),
                capital_flow_score=float(result['capital_flow_score'][i]),
                news_sentiment_score=float(result['news_sentiment_score'][i]),
                trend_type=int(result['trend_type'][i]),
                chan_pivot_zg=float(result['chan_pivot_zg'][i]),
                chan_pivot_zd=float(result['chan_pivot_zd'][i]),
                chan_pivot_zz=float(result['chan_pivot_zz'][i]),
                daily_return=float(result['daily_return'][i]),
                volume_ratio=float(result['volume_ratio'][i]),
                volume_ratio_raw=float(result['volume_ratio_raw'][i]),
                exhaustion_risk=float(result['exhaustion_risk'][i]),
                gap_breakout_confirm=float(result['gap_breakout_confirm'][i]),
                stroke_phase=float(result['stroke_phase'][i]),
                top_fractal_volume=float(result['top_fractal_volume'][i]),
                ma_trend_up=bool(result['ma_trend_up'][i]),
                profit_declining=bool(result['profit_declining'][i]),
                mom_60d=float(result['mom_60d'][i]),
                dist_ma60=float(result['dist_ma60'][i]),
                max_dd_20d=float(result['max_dd_20d'][i]),
                vol_regime=float(result['vol_regime'][i]),
                weekly_trend_up=bool(result['weekly_trend_up'][i]),
                monthly_trend_up=bool(result['monthly_trend_up'][i]),
                weekly_trend_strength=float(result['weekly_trend_strength'][i]),
                monthly_trend_strength=float(result['monthly_trend_strength'][i]),
                mtf_alignment_score=float(result['mtf_alignment_score'][i]),
                mtf_discount_factor=float(result['mtf_discount_factor'][i]),
                weekly_pattern_signal=float(result['weekly_pattern_signal'][i]),
                nearest_resistance_pct=float(result['nearest_resistance_pct'][i]),
                nearest_support_pct=float(result['nearest_support_pct'][i]),
                _chan_buy_signal=chan_buy_sig,
                _chan_sell_signal=chan_sell_sig,
                _dist_ma20=dist_ma20,
            )
            signal_store.set(code, date, sig)

        # 显式释放中间数组，避免大规模循环中GC延迟导致内存堆积
        del result, close_arr, ma20_arr, bottom_div_arr, top_div_arr
        del buy_thresholds, sell_thresholds, valid

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """计算技术指标（委托给factor_calculator）"""
        params = self.indicator_params
        dates = data['datetime'].values
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        turnover_rate = data['turnover_rate'].values if 'turnover_rate' in data.columns else None
        open_price = data['open'].values if 'open' in data.columns else None
        amplitude = data['amplitude'].values if 'amplitude' in data.columns else None

        # 使用统一的因子计算器
        result = calc_indicators(close, high, low, volume, params,
                                 turnover_rate=turnover_rate, open_arr=open_price,
                                 amplitude=amplitude)

        # 收益率
        result['ret_30'] = close / self._shift(close, 30) - 1

        # ATR比率
        result['atr_ratio_20'] = result['atr_20'] / (close + 1e-10)

        # ── 预信号价格特征（均值回归过滤） ──
        # 1. 60日动量
        result['mom_60d'] = close / self._shift(close, 60) - 1

        # 2. 距MA60偏离
        ma60 = self._rolling_mean(close, 60)
        result['dist_ma60'] = (close - ma60) / (ma60 + 1e-10)

        # 3. 20日最大回撤
        result['max_dd_20d'] = self._compute_max_dd(close, 20)

        # 4. 波动率区间 (10日vol / 60日vol)
        ret_1d = close / self._shift(close, 1) - 1
        vol_10d = self._rolling_std(ret_1d, 10)
        vol_60d = self._rolling_std(ret_1d, 60)
        result['vol_regime'] = vol_10d / (vol_60d + 1e-10)

        # 5. 统一多时间框架分析（替代旧版固定步长降采样）
        open_arr = open_price if open_price is not None else close
        mtf_result = self.mtf_analyzer.analyze(
            close, high, low, open_arr, volume, dates
        )
        result['weekly_trend_up'] = mtf_result.weekly_trend_up
        result['monthly_trend_up'] = mtf_result.monthly_trend_up
        result['weekly_trend_strength'] = mtf_result.weekly_trend_strength
        result['monthly_trend_strength'] = mtf_result.monthly_trend_strength
        result['mtf_alignment_score'] = mtf_result.alignment_score
        result['mtf_discount_factor'] = mtf_result.discount_factor
        result['weekly_pattern_signal'] = mtf_result.weekly_pattern_signal
        result['nearest_resistance_pct'] = mtf_result.nearest_resistance_pct
        result['nearest_support_pct'] = mtf_result.nearest_support_pct

        # 保存close数组（供信号生成中的均线位置过滤使用）
        result['close'] = close

        return result

    def _precompute_regimes(self, dates: np.ndarray, n: int) -> np.ndarray:
        """向量化批量获取市场状态，替代逐 bar _get_market_info"""
        if self.market_regime_data is None or len(self.market_regime_data) == 0:
            return np.zeros(n, dtype=int)
        dt_index = pd.DatetimeIndex(dates)
        aligned = self.market_regime_data['regime'].reindex(dt_index, method='ffill')
        return aligned.fillna(0).astype(int).values

    def _get_market_info(self, date) -> Dict[str, Any]:
        """获取指定日期的市场状态信息"""
        if self.market_regime_data is None:
            return {
                'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
                'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
                'style_regime': 'balanced', 'style_score': 0.0,
                'size_score': 0.0, 'style_confidence': 0.0,
                'bear_risk': False,
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
                'bear_risk': bool(row.get('bear_risk', False)),
            }
        return {
            'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
            'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
            'style_regime': 'balanced', 'style_score': 0.0,
            'size_score': 0.0, 'style_confidence': 0.0,
            'bear_risk': False,
        }

    def _get_chan_boost(self, ind: dict, idx: int, market_info: dict = None, industry: str = '') -> dict:
        """
        缠论增强 (czsc风格 v2): 使用分型→笔→线段→中枢→买卖点的完整层级。

        核心规则（对齐czsc三类买卖点）:
        - 一买(B1): 下跌趋势结束+底背离 → 最强买入信号
        - 二买(B2): 一买后回调不破前低 → 二次确认
        - 三买(B3): 突破中枢上沿后回调不进中枢 → 趋势加速
        - 一卖(S1): 上涨趋势结束+顶背离 → 最强卖出信号
        - 二卖(S2): 一卖后反弹不破前高 → 二次确认
        - 三卖(S3): 跌破中枢下沿后反弹不回中枢 → 加速下跌
        """
        mult = 1.0
        is_buy_boost = False
        is_sell_boost = False
        div_quality = 0.0

        # === 新数据: 买卖点 (来自 chan_theory + chanlun-pro增强) ===
        buy_point = int(self._safe_get(ind, 'buy_point', idx, 0))
        sell_point = int(self._safe_get(ind, 'sell_point', idx, 0))
        buy_conf = self._safe_get(ind, 'buy_confidence', idx, 0.0)
        sell_conf = self._safe_get(ind, 'sell_confidence', idx, 0.0)
        chan_buy = self._safe_get(ind, 'chan_buy_score', idx, 0.0)
        chan_sell = self._safe_get(ind, 'chan_sell_score', idx, 0.0)
        # chanlun-pro 增强字段
        signal_level = int(self._safe_get(ind, 'signal_level', idx, 0))
        confirmed_buy = bool(self._safe_get(ind, 'confirmed_buy', idx, 0))
        confirmed_sell = bool(self._safe_get(ind, 'confirmed_sell', idx, 0))
        buy_strength = self._safe_get(ind, 'buy_strength', idx, 0.0)
        sell_strength = self._safe_get(ind, 'sell_strength', idx, 0.0)
        bi_td = bool(self._safe_get(ind, 'bi_td', idx, 0))
        bi_buy = int(self._safe_get(ind, 'bi_buy_point', idx, 0))
        bi_sell = int(self._safe_get(ind, 'bi_sell_point', idx, 0))

        trend_type = int(self._safe_get(ind, 'trend_type', idx, 0))
        trend_strength = self._safe_get(ind, 'trend_strength', idx, 0.0)
        pivot_pos = int(self._safe_get(ind, 'pivot_position', idx, 0))
        consolidation = self._safe_get(ind, 'consolidation_zone', idx, 0.0)
        stroke_dir = int(self._safe_get(ind, 'stroke_direction', idx, 0))
        alignment = self._safe_get(ind, 'alignment_score', idx, 0.0)
        structure_complete = self._safe_get(ind, 'structure_complete', idx, 0.0)
        zhongyin = self._safe_get(ind, 'zhongyin', idx, 0.0)

        # === 旧数据 (divergence_detector) 作为补充 ===
        bottom_div = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        top_div = self._safe_get(ind, 'top_divergence', idx, 0.0)
        hidden_bottom = self._safe_get(ind, 'hidden_bottom_divergence', idx, 0.0)

        # === Layer 1: 多级别确认 (chanlun-pro最高优先级) ===

        # 双级别确认 (笔+线段同时) → 最强信号
        # v8: IC分析显示chan_buy_point IC≈0, chan_sell_point IC=+0.027(反号)
        # 买卖点预测力弱, 降低boost幅度; 卖点惩罚大幅缩小
        if signal_level == 3 and confirmed_buy:
            mult *= 1.18  # 双级别确认买点 → 降幅(原1.35)
            div_quality = max(div_quality, buy_strength)
            is_buy_boost = True
        elif signal_level == -3 and confirmed_sell:
            mult *= 0.70  # 双级别确认卖点 → 减罚(原0.55)
            div_quality = max(div_quality, sell_strength)
            is_sell_boost = True

        # 线段级别信号
        elif signal_level == 2 and confirmed_buy:
            mult *= 1.12
            div_quality = max(div_quality, buy_strength)
            is_buy_boost = True
        elif signal_level == -2 and confirmed_sell:
            mult *= 0.75
            div_quality = max(div_quality, sell_strength)
            is_sell_boost = True

        # 笔级别信号
        elif signal_level == 1 and confirmed_buy:
            mult *= 1.08
            div_quality = max(div_quality, buy_strength * 0.7)
            is_buy_boost = True
        elif signal_level == -1 and confirmed_sell:
            mult *= 0.82
            div_quality = max(div_quality, sell_strength * 0.7)
            is_sell_boost = True

        # === Layer 1.5: 底部分型观察 ===
        # 底部分型是缠论最基础的转折信号，比笔/线段更早发现底部
        bottom_fx_quality = self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0)
        bottom_fx_strength = self._safe_get(ind, 'bottom_fractal_strength', idx, 0.0)
        bottom_fx_vol = self._safe_get(ind, 'bottom_fractal_vol_ratio', idx, 1.0)
        bottom_fx_spike = self._safe_get(ind, 'bottom_fractal_vol_spike', idx, 1.0)
        # 量在价先: 相对前日放量3倍以上 = 极度恐慌抛售 = 强反转
        is_volume_spike_3x = bottom_fx_spike >= 3.0

        if bottom_fx_quality > 0.25:  # 降低最低门槛（原来0.35），让量在价先有机会进入
            if is_buy_boost:
                # 已有买点信号 + 强底分型 → 增强确认
                base_boost = 0.08 * bottom_fx_quality
                if is_volume_spike_3x:
                    base_boost += 0.08  # 量在价先额外增强
                mult *= 1.0 + base_boost
                div_quality = max(div_quality, bottom_fx_quality * 0.65)
            elif not is_sell_boost:
                # 量在价先: 3x放量+底分型 → 降低独立买信号门槛到0.30
                standalone_threshold = 0.30 if is_volume_spike_3x else 0.55
                mild_threshold = 0.25 if is_volume_spike_3x else 0.35

                if bottom_fx_quality > standalone_threshold:
                    # 高质量底分型 → 独立买信号
                    boost = 0.10 * bottom_fx_quality
                    if is_volume_spike_3x:
                        boost += 0.10  # 量在价先显著增强
                    mult *= 1.0 + boost
                    is_buy_boost = True
                    div_quality = max(div_quality, bottom_fx_quality * 0.55)
                elif bottom_fx_quality > mild_threshold:
                    # 温和买倾向
                    mult *= 1.0 + 0.05 * bottom_fx_quality
                    if is_volume_spike_3x:
                        mult *= 1.06  # 3x放量翻倍温和信号
                    elif bottom_fx_vol > 1.3:
                        mult *= 1.03  # 放量底分型额外加分
                # 量在价先 + 笔趋势耗尽 → 强确认
                if bottom_fx_quality > 0.45 and bi_td:
                    mult *= 1.06
                if is_volume_spike_3x and bi_td:
                    mult *= 1.04  # 3x放量+趋势耗尽双重确认

        # === Layer 1.6: 二买 (B2) — 缠论最安全的买点 ===
        # B2 = 一买后回调不破前低 + 底分型 + MACD背离
        second_buy = bool(self._safe_get(ind, 'second_buy_point', idx, 0))
        second_buy_conf = self._safe_get(ind, 'second_buy_confidence', idx, 0.0)
        b1_ref = int(self._safe_get(ind, 'second_buy_b1_ref', idx, -1))

        if second_buy and second_buy_conf >= 0.45:
            # B2是缠论中最可靠的买点 — 趋势已反转 + 回调确认
            b2_boost = 0.12 * second_buy_conf  # 最高约 +12% (原0.18, IC分析显示买点预测力弱)
            if not is_sell_boost:
                mult *= 1.0 + b2_boost
                is_buy_boost = True
                div_quality = max(div_quality, second_buy_conf * 0.75)
            elif is_buy_boost:
                # 已有买信号 + B2确认 → 强力共振
                mult *= 1.0 + b2_boost * 0.7
                div_quality = max(div_quality, second_buy_conf * 0.75)

        # 回退到旧的三类买卖点 (无多级别确认时)
        # B3/B2/B1 互斥优先级：一个时间点只能处于一种买点状态
        # 优先级：B3 > B2 > B1（趋势加速 > 回调确认 > 底部反转）
        if not is_buy_boost and not is_sell_boost:
            # === 买点互斥选择 ===
            if buy_point == 3 and buy_conf > 0.2:
                # B3: 趋势加速买点 — 质量门控
                # v8.1 fix: 移除chan_pivot_zg ZG止损 (IC=-0.8% 噪音)
                b3_mult = self._calc_b3_multiplier(ind, idx, market_info, industry)
                if b3_mult is not None:
                    mult *= b3_mult
                    div_quality = max(div_quality, buy_conf * 0.90)
                    is_buy_boost = True
                else:
                    b3_mult = self._calc_b3_multiplier(ind, idx, market_info, industry)
                    if b3_mult is not None:
                        mult *= b3_mult
                        div_quality = max(div_quality, buy_conf * 0.90)
                        is_buy_boost = True

            elif buy_point == 2 and buy_conf > 0.2:
                # B2: 回调确认买点 — 最安全
                b2_mult = self._calc_b2_multiplier(ind, idx, market_info)
                if b2_mult is not None:
                    mult *= b2_mult
                    div_quality = max(div_quality, buy_conf * 0.85)
                    is_buy_boost = True

            elif buy_point == 1 and buy_conf > 0.3:
                # B1: 底部反转买点 — 熊市最佳
                b1_mult = self._calc_b1_multiplier(ind, idx, market_info)
                if b1_mult is not None:
                    mult *= b1_mult
                    div_quality = max(div_quality, buy_conf * 0.80)
                    is_buy_boost = True

            # === 卖点互斥选择（买点与卖点也互斥，买点优先） ===
            if not is_buy_boost:
                if sell_point == 1 and sell_conf > 0.3:
                    mult *= 0.78  # IC分析: 卖点预测正向收益, 大幅减罚(原0.65)
                    div_quality = max(div_quality, sell_conf)
                    is_sell_boost = True
                elif sell_point == 2 and sell_conf > 0.2:
                    mult *= 0.82  # 减罚(原0.70)
                    div_quality = max(div_quality, sell_conf * 0.8)
                    is_sell_boost = True
                elif sell_point == 3 and sell_conf > 0.2:
                    mult *= 0.85  # 减罚(原0.75)
                    div_quality = max(div_quality, sell_conf * 0.85)
                    is_sell_boost = True

        # === Layer 2: 统一Chan信号 (当买卖点不明显时使用) ===
        if buy_point == 0 and sell_point == 0:
            if chan_buy > 0.5:
                mult *= 1.0 + 0.10 * chan_buy  # 降幅(原0.15)
                is_buy_boost = True
                div_quality = max(div_quality, chan_buy * 0.6)
            if chan_sell > 0.5:
                mult *= 1.0 - 0.15 * chan_sell  # 减罚(原0.25)
                is_sell_boost = True
                div_quality = max(div_quality, chan_sell * 0.6)

        # === Layer 3: 背离检测 (兜底，但必须有趋势 — "没有趋势，没有背驰") ===
        if not is_buy_boost and not is_sell_boost:
            if bottom_div > top_div and bottom_div > self.chan_bottom_div_threshold:
                # 底背离仅在下跌趋势中有意义（盘整中的背离无效）
                if trend_type != 0:  # 有趋势才启用背离
                    mult *= 1.14  # 降幅(原1.20), divergence_strength IC仅+0.009
                    div_quality = bottom_div
                    is_buy_boost = True
            elif top_div > bottom_div and top_div > self.chan_top_div_threshold:
                # 顶背离仅在上涨趋势中有意义
                if trend_type != 0:
                    mult *= 0.78  # 减罚(原0.70), 卖点预测力弱
                    div_quality = top_div
                    is_sell_boost = True

        # === Layer 4: 趋势类型调整 ===
        if trend_type == 2:  # 上涨趋势
            if is_buy_boost:
                mult *= 1.08  # 顺势做多
            elif is_sell_boost:
                mult *= 0.92  # 逆势做空打折扣
        elif trend_type == -2:  # 下跌趋势
            if is_sell_boost:
                mult *= 0.92  # 顺势做空（mult已是sell信号）
            elif is_buy_boost:
                mult *= 0.92  # 逆势抄底需谨慎
        elif trend_type == 1:  # 盘整
            if abs(mult - 1.0) > 0.1:
                mult = 1.0 + (mult - 1.0) * 0.80  # 盘整时信号打8折(原7折), 更宽松

        # === Layer 5: 中枢位置调整 ===
        if pivot_pos == -1 and trend_type >= 1:  # 中枢下方+非下跌趋势
            if is_buy_boost:
                mult *= 1.10  # 中枢下方买入 = 低成本
        elif pivot_pos == 1 and trend_type <= -1:  # 中枢上方+非上涨趋势
            if is_sell_boost:
                mult *= 0.90  # 中枢上方卖出 = 好价位

        # === Layer 6: 状态调整 ===
        if hidden_bottom > 0.15 and not is_sell_boost:
            mult *= 1.08
        if abs(alignment) > 0.5:
            mult *= (1.0 + 0.10 * alignment)
        # 中阴阶段: 方向不明，大幅降分（缠论第99课：中阴阶段不操作）
        if zhongyin > 0 and abs(alignment) < 0.3:
            mult *= 0.50
        if structure_complete < 0.5 and abs(alignment) < 0.3:
            mult *= 0.85

        # === Layer 7: 中枢附近 + 底背离 → 2买可能位置 ===
        chan_pivot_present = self._safe_get(ind, 'chan_pivot_present', idx, 0.0)
        if chan_pivot_present > 0 and bottom_div > 0.15:
            mult *= 1.06

        # Layer 8 (大级别趋势保护) 已合并至 multi_timeframe.py 统一折扣因子,
        # 不再在此处重复应用。均线乖离检查已移至 portfolio.py 均值回归调整。

        # 均线乖离检查已移至 portfolio.py 均值回归调整（统一处理）

        return {
            'boost_multiplier': float(np.clip(mult, 0.5, 1.3)),
            'is_chan_buy_boost': is_buy_boost,
            'is_chan_sell_boost': is_sell_boost,
            'divergence_quality': div_quality,
        }

    def _calc_b3_multiplier(self, ind: dict, idx: int,
                            market_info: dict = None, industry: str = '') -> float:
        """B3 动态乘数: 大盘环境 × 趋势结构 × 背离共振 × 量能确认 × 回调质量

        返回 None 表示不产生B3信号 (大盘下跌时禁用)
        返回 float 为最终乘数 (1.0 基准)

        5层过滤体系 (由粗到细):
          Layer 1 - 大盘环境: 熊市禁用, 震荡仅强板块可做
          Layer 2 - 趋势结构: 首中枢最优, 第三中枢后大幅折扣
          Layer 3 - 背离共振: 底背离+放量底分型确认
          Layer 4 - 量能确认: 突破放量+回调缩量 (量在价先)
          Layer 5 - 回调质量: 浅回踩=强控盘, 深回踩=弱势
        """
        if market_info is None:
            market_info = {}
        regime = market_info.get('regime', 0)  # -1=bear, 0=neutral, 1=bull

        # ── Layer 0: B3 硬门控 (四条件至少满足两个，否则拒收) ──
        # 条件a: 突破放量 (≥1.3x 均量)
        b3_bo_vol = self._safe_get(ind, 'b3_breakout_vol_ratio', idx, 0.0)
        breakout_pass = (b3_bo_vol >= 1.3)

        # 条件b: 回调缩量 (≤0.85x 突破量)
        b3_pb_vol = self._safe_get(ind, 'b3_pullback_vol_ratio', idx, 1.0)
        contraction_pass = (b3_pb_vol <= 0.85)

        # 条件c: 结构确认 (趋势确认 或 背离 或 底分型)
        b3_tc = bool(self._safe_get(ind, 'b3_trend_confirmed', idx, False))
        bd = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        fx_q = self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0)
        confirm_pass = (b3_tc or bd > 0.3 or fx_q > 0.25)

        # 条件d: 回调质量 (浅回踩 ≥ 0.5)
        pb_shallow = self._safe_get(ind, 'b3_pullback_shallowness', idx, 0.5)
        shallow_pass = (pb_shallow >= 0.5)

        n_pass = sum([breakout_pass, contraction_pass, confirm_pass, shallow_pass])
        if n_pass < 2:
            return None

        # ── Layer 1: 大盘环境 ──
        if regime == -1:
            # 大盘下跌: 检测逆势抗跌基因后才允许
            anti_market_strength = self._safe_get(ind, 'relative_strength', idx, 0.0)
            pivot_count = int(self._safe_get(ind, 'pivot_count', idx, 0))
            # 仅当个股强于大盘(RS>0.3)且正在构筑第一个上涨中枢时才允许B3
            if anti_market_strength < 0.3 or pivot_count > 1:
                return None
            # 熊市B3仅给70%基准，需极强证据才能盈利
            regime_mult = 0.65
        elif regime == 0:
            # 大盘震荡: 仅强势板块允许B3
            if not self._is_industry_strong(industry):
                return None
            regime_mult = 0.85
        else:
            # 大盘上涨: B3最适合的环境
            regime_mult = 1.0

        # ── Layer 2: 趋势结构 (中枢"段位"和"地基") ──
        b3_trend_confirmed = bool(self._safe_get(ind, 'b3_trend_confirmed', idx, False))
        b3_trend_rank = int(self._safe_get(ind, 'b3_trend_rank', idx, 0))
        if not b3_trend_confirmed:
            # 趋势未确认 (<2个非重叠上涨中枢): 大幅折扣
            trend_mult = 0.7
        elif b3_trend_rank <= 1:
            # 底部第一个上涨中枢的B3: "首中枢最佳" — 最优位置
            trend_mult = 1.0
        elif b3_trend_rank == 2:
            # 第二个中枢B3: 尚可
            trend_mult = 0.85
        else:
            # 第三中枢及以上B3: "中枢末段易背驰见顶"
            trend_mult = 0.6

        # ── Layer 3: 背离共振 ──
        bottom_div = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        has_div_resonance = bottom_div > 0.3
        # 底分型+放量确认
        bottom_fx_qual = self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0)
        bottom_fx_vol = self._safe_get(ind, 'bottom_fractal_vol_ratio', idx, 1.0)
        has_fx_vol = bottom_fx_qual > 0.25 and bottom_fx_vol > 1.2

        if has_div_resonance and has_fx_vol:
            # 背离 + 放量底分型 共振 → B3入场确认
            div_mult = 1.0
        elif has_div_resonance or has_fx_vol:
            # 仅一项满足
            div_mult = 0.8
        else:
            # 无背离无放量底分型 → B3可能回调未结束
            div_mult = 0.6

        # ── Layer 4: 量能确认 (过滤器二: 突破放量+回调缩量) ──
        b3_breakout_vol = self._safe_get(ind, 'b3_breakout_vol_ratio', idx, 0.0)
        b3_pullback_vol = self._safe_get(ind, 'b3_pullback_vol_ratio', idx, 1.0)

        if b3_breakout_vol > 1.5 and b3_pullback_vol < 0.7:
            # 突破放量(>1.5x均量) + 回调缩量(<0.7x突破量) = 标准量价配合
            vol_mult = 1.0
        elif b3_breakout_vol > 2.0 and b3_pullback_vol < 0.5:
            # 强突破(>2x均量) + 强缩量(<0.5x) = "一脚踹开"+"抛压衰竭"
            vol_mult = 1.15
        elif b3_breakout_vol > 1.0 and b3_pullback_vol < 0.8:
            # 及格线
            vol_mult = 0.85
        elif b3_breakout_vol > 0:
            # 有量数据但不达标 → 需打折
            vol_mult = 0.65
        else:
            # 无量数据(老股/无volume字段) → 中性，不影响
            vol_mult = 0.9

        # ── Layer 5: 回调质量 (过滤器四: 极限浅回踩) ──
        b3_pullback_shallow = self._safe_get(ind, 'b3_pullback_shallowness', idx, 0.5)

        if b3_pullback_shallow > 0.8:
            # 极浅回踩: 不破5日线, 主力控盘度高 → 爆发力强
            pullback_mult = 1.15
        elif b3_pullback_shallow > 0.6:
            # 标准浅回踩: 正常回调到位
            pullback_mult = 1.0
        elif b3_pullback_shallow > 0.35:
            # 较深回踩: 多头防守偏弱
            pullback_mult = 0.8
        elif b3_pullback_shallow > 0:
            # 深回踩: 几乎跌回中枢, 空头势能大
            pullback_mult = 0.6
        else:
            # 无数据 → 中性
            pullback_mult = 0.9

        # ── 合成 ──
        # 5层乘数相乘: 每层独立判断，任一层弱则整体大幅折扣
        # 修改: 允许 <1.0 的结果，使B3质量真正具有区分度
        # 质量差的B3(combined<0.33)会产生折扣(<1.0)，质量好的B3才有加成
        combined = regime_mult * trend_mult * div_mult * vol_mult * pullback_mult
        final_mult = 0.55 + 0.75 * combined
        return max(0.70, min(final_mult, 1.35))

    def _calc_b2_multiplier(self, ind: dict, idx: int, market_info: dict = None) -> float:
        """B2 质量乘数: 大盘环境 × B1质量 × 回调深度 × MACD改善 × 量能

        B2 = 一买后回调不破前低 → 缠论最安全的买点
        支持增强版B2(second_buy_confidence>0)和简易版B2(buy_point=2)
        返回 None 表示B2质量不足，应递补到B1
        """
        if market_info is None:
            market_info = {}
        regime = market_info.get('regime', 0)

        # ── Layer 1: 大盘环境 ──
        if regime == -1:
            regime_mult = 1.10   # 熊市B2性价比高
        elif regime == 0:
            regime_mult = 1.05   # 震荡市B2最佳
        else:
            regime_mult = 0.95   # 牛市B2稍弱于B3

        b2_confidence = self._safe_get(ind, 'second_buy_confidence', idx, 0.0)

        # ── 增强版B2 (有second_buy数据) ──
        if b2_confidence > 0.01:
            b1_ref_idx = int(self._safe_get(ind, 'second_buy_b1_ref', idx, -1))

            # Layer 2: 引用B1质量
            if b2_confidence > 0.7:
                b1_qual_mult = 1.10
            elif b2_confidence > 0.5:
                b1_qual_mult = 1.0
            elif b2_confidence > 0.3:
                b1_qual_mult = 0.85
            else:
                b1_qual_mult = 0.70

            # Layer 3: B1后反弹幅度
            close_arr = ind.get('close', None)
            if close_arr is not None and b1_ref_idx >= 0 and b1_ref_idx < idx:
                b1_price = close_arr[b1_ref_idx]
                post_high = np.max(close_arr[b1_ref_idx:idx + 1])
                bounce_pct = (post_high - b1_price) / (b1_price + 1e-10)
                if bounce_pct > 0.05:
                    bounce_mult = 1.10
                elif bounce_pct > 0.02:
                    bounce_mult = 1.0
                else:
                    bounce_mult = 0.8
            else:
                bounce_mult = 0.95

            # Layer 4: MACD改善
            macd_improved = False
            macd = ind.get('macd', None)
            if macd is not None and b1_ref_idx >= 0:
                macd_b2 = float(macd[idx]) if idx < len(macd) and not np.isnan(macd[idx]) else 0
                macd_b1 = float(macd[b1_ref_idx]) if b1_ref_idx < len(macd) and not np.isnan(macd[b1_ref_idx]) else 0
                macd_improved = macd_b2 > macd_b1
            macd_mult = 1.08 if macd_improved else 0.90

            # Layer 5: 回调缩量
            b3_pb_vol = self._safe_get(ind, 'b3_pullback_vol_ratio', idx, 1.0)
            if 0 < b3_pb_vol < 0.6:
                vol_mult = 1.10
            elif 0 < b3_pb_vol < 0.8:
                vol_mult = 1.0
            elif b3_pb_vol > 0:
                vol_mult = 0.85
            else:
                vol_mult = 0.95

            # Layer 6: MA20支撑回踩（B2回调到MA20附近获得支撑=最强确认）
            close_arr_ma = ind.get('close', None)
            ma20_arr = ind.get('ma20', None)
            if close_arr_ma is not None and ma20_arr is not None:
                cur_price = float(close_arr_ma[idx])
                ma20_val = float(ma20_arr[idx])
                if ma20_val > 0:
                    dist_ma20_pct = (cur_price - ma20_val) / ma20_val
                    if -0.02 <= dist_ma20_pct <= 0.02:
                        ma20_mult = 1.15   # 精准回踩MA20 → 最强支撑确认
                    elif -0.03 <= dist_ma20_pct <= 0.05:
                        ma20_mult = 1.08   # 靠近MA20 → 中等支撑
                    elif dist_ma20_pct > 0.10:
                        ma20_mult = 0.92   # 远离MA20 → 不是典型B2回调
                    else:
                        ma20_mult = 1.0
                else:
                    ma20_mult = 1.0
            else:
                ma20_mult = 1.0

            # 硬门控: B2需要至少b2_confidence>0.3
            if b2_confidence < 0.30:
                return None

            combined = regime_mult * b1_qual_mult * bounce_mult * macd_mult * vol_mult * ma20_mult
            final_mult = (1.0 + (1.25 - 1.0) * combined)
            return max(1.0, min(final_mult, 1.40))

        # ── 简易版B2 (buy_point=2但无增强数据) ──
        else:
            buy_conf_simple = self._safe_get(ind, 'buy_confidence', idx, 0.0)
            if buy_conf_simple < 0.2:
                return None

            # 简易评估: buy_confidence + MACD + 量能 + 底分型
            conf_mult = 0.7 + buy_conf_simple * 0.6

            macd = ind.get('macd', None)
            macd_ok = False
            if macd is not None and idx >= 10:
                macd_now = float(macd[idx]) if not np.isnan(macd[idx]) else 0
                macd_prev = float(np.mean(macd[max(0,idx-10):idx]))
                macd_ok = macd_now > macd_prev
            macd_mult_s = 1.05 if macd_ok else 0.92

            pullback_vol = self._safe_get(ind, 'b3_pullback_vol_ratio', idx, 1.0)
            if 0 < pullback_vol < 0.7:
                vol_mult_s = 1.08
            elif pullback_vol > 0:
                vol_mult_s = 0.90
            else:
                vol_mult_s = 0.95

            bottom_fx = self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0)
            fx_mult_s = 1.06 if bottom_fx > 0.25 else 1.0

            # MA20支撑: 简易B2回调到MA20附近获得支撑
            close_arr_s = ind.get('close', None)
            ma20_arr_s = ind.get('ma20', None)
            if close_arr_s is not None and ma20_arr_s is not None:
                cur_p = float(close_arr_s[idx])
                ma20_v = float(ma20_arr_s[idx])
                if ma20_v > 0:
                    d_ma20 = (cur_p - ma20_v) / ma20_v
                    if -0.02 <= d_ma20 <= 0.02:
                        ma20_mult_s = 1.12
                    elif -0.03 <= d_ma20 <= 0.05:
                        ma20_mult_s = 1.06
                    else:
                        ma20_mult_s = 1.0
                else:
                    ma20_mult_s = 1.0
            else:
                ma20_mult_s = 1.0

            combined = regime_mult * conf_mult * macd_mult_s * vol_mult_s * fx_mult_s * ma20_mult_s
            final_mult = (1.0 + (1.30 - 1.0) * combined)
            return max(1.0, min(final_mult, 1.30))

    def _calc_b1_multiplier(self, ind: dict, idx: int, market_info: dict = None) -> float:
        """B1 质量乘数: 大盘环境 × 背离强度 × 恐慌量能 × 超卖深度 × 趋势耗尽

        B1 = 下跌趋势末端底背离 → 反转买点，熊市最佳
        返回 None 表示B1质量不足
        """
        if market_info is None:
            market_info = {}
        regime = market_info.get('regime', 0)

        # ── Layer 1: 大盘环境 ──
        if regime == -1:
            regime_mult = 1.10   # 熊市B1: 抄底正当时
        elif regime == 0:
            regime_mult = 1.0    # 震荡: B1中性
        else:
            regime_mult = 0.85   # 牛市B1: 逆势抄底需谨慎

        # ── Layer 2: 背离强度 ──
        bottom_div = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        if bottom_div > 0.5:
            div_mult = 1.15      # 强背离 → 高概率反转
        elif bottom_div > 0.3:
            div_mult = 1.0
        elif bottom_div > 0.15:
            div_mult = 0.80
        else:
            div_mult = 0.60

        # ── Layer 3: 恐慌抛售量 ──
        fx_vol_spike = self._safe_get(ind, 'bottom_fractal_vol_spike', idx, 1.0)
        if fx_vol_spike >= 3.0:
            vol_mult = 1.20      # 3x放量恐慌 = 强反转信号(量在价先)
        elif fx_vol_spike >= 2.0:
            vol_mult = 1.10
        elif fx_vol_spike >= 1.5:
            vol_mult = 1.0
        elif fx_vol_spike >= 1.0:
            vol_mult = 0.85
        else:
            vol_mult = 0.90

        # ── Layer 4: 超卖深度 (价格低于MA60的幅度) ──
        ema60 = ind.get('ema60', None)
        close_arr = ind.get('close', None)
        if ema60 is not None and close_arr is not None:
            dist_from_ma60 = (ema60[idx] - close_arr[idx]) / (ema60[idx] + 1e-10)
            if dist_from_ma60 > 0.10:
                oversold_mult = 1.15    # 深度超卖 → 反弹空间大
            elif dist_from_ma60 > 0.05:
                oversold_mult = 1.05
            elif dist_from_ma60 > 0:
                oversold_mult = 1.0
            else:
                oversold_mult = 0.85    # 价格在MA60之上 → 不是真正B1
        else:
            oversold_mult = 0.95

        # ── Layer 5: 笔趋势耗尽 ──
        bi_td = bool(self._safe_get(ind, 'bi_td', idx, 0))
        td_mult = 1.10 if bi_td else 0.90

        # ── 硬门控: B1需要至少背离+恐慌量能之一达标 ──
        if bottom_div < 0.15 and fx_vol_spike < 1.5:
            return None

        combined = regime_mult * div_mult * vol_mult * oversold_mult * td_mult
        final_mult = (1.0 + (1.30 - 1.0) * combined)
        return max(1.0, min(final_mult, 1.30))

    def _is_industry_strong(self, industry: str) -> bool:
        """判断行业是否强势 (IC > 0.06 或 配置中标注为主线)"""
        if not industry:
            return False
        cfg = INDUSTRY_FACTOR_CONFIG.get(industry, {})
        if not cfg:
            return False
        combined_ic = cfg.get('combined_ic', 0)
        base_ic = cfg.get('ic', 0)
        return combined_ic > 0.06 or base_ic > 0.06

    def _generate_signal(self, ind: dict, idx: int, last_sig, current_date=None, code=None,
                         effective_buy_threshold=None, effective_sell_threshold=None) -> Signal:
        """生成信号

        Args:
            effective_buy_threshold: 外部注入的买入阈值，None则使用内部状态
            effective_sell_threshold: 外部注入的卖出阈值，None则使用内部状态
        """
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0, factor_value=0.0,
                factor_name='V41', risk_vol=0.03, risk_regime=0,
                risk_confidence=0.0, risk_extreme=False, adjusted_score=0.0,
                pre_discount_score=0.0,
                industry=self._get_specific_industry(code, current_date) if code else '',
                exhaustion_risk=0.0, gap_breakout_confirm=0.0,
                stroke_phase=0.0, top_fractal_volume=0.0,
                ma_trend_up=False, profit_declining=False,
                mom_60d=0.0, dist_ma60=0.0, max_dd_20d=0.0, vol_regime=1.0,
                weekly_trend_up=False, monthly_trend_up=False,
                weekly_trend_strength=0.0, monthly_trend_strength=0.0,
                mtf_alignment_score=0.0, mtf_discount_factor=1.0,
                weekly_pattern_signal=0.0, nearest_resistance_pct=0.0,
                nearest_support_pct=0.0,
                _chan_buy_signal=False, _chan_sell_signal=False, _dist_ma20=0.0,
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
        factor_result = self._select_factor(
            ind, idx, risk_regime, industry_category, code=code, current_date=current_date
        )

        # 如果_select_factor返回None（如负IC行业），返回空信号
        if factor_result is None:
            return Signal(
                buy=False, sell=False, score=0.0, factor_value=0.0,
                factor_name='NONE', risk_vol=0.03, risk_regime=risk_regime,
                risk_confidence=0.0, risk_extreme=risk_extreme, adjusted_score=0.0,
                pre_discount_score=0.0,
                industry=self._get_specific_industry(code, current_date) if code else '',
                exhaustion_risk=0.0, gap_breakout_confirm=0.0,
                stroke_phase=0.0, top_fractal_volume=0.0,
                ma_trend_up=False, profit_declining=False,
                mom_60d=0.0, dist_ma60=0.0, max_dd_20d=0.0, vol_regime=1.0,
                weekly_trend_up=False, monthly_trend_up=False,
                weekly_trend_strength=0.0, monthly_trend_strength=0.0,
                mtf_alignment_score=0.0, mtf_discount_factor=1.0,
                weekly_pattern_signal=0.0, nearest_resistance_pct=0.0,
                nearest_support_pct=0.0,
                _chan_buy_signal=False, _chan_sell_signal=False, _dist_ma20=0.0,
            )

        factor_name, factor_value, risk_info, is_industry = factor_result

        # 基本面因子
        fundamental_score = 0.0
        has_fundamental = False
        if self.fundamental_enabled and code:
            fundamental_score = self._get_fundamental_score(code, current_date)
            has_fundamental = fundamental_score > 0

        # 基本面排雷: 近两季度利润同比持续下滑
        profit_declining = False
        if self.fundamental_enabled and code:
            profit_declining = self._check_profit_decline(code, current_date)

        # 风格因子
        style_factor_score = self._get_style_score(ind, idx, market_info)

        # === 信号系统 v4 ===
        # 核心思想: score = factor_value，与标定验证的IC完全对齐
        # 组合层通过截面rank_pct排序选股，不再依赖信号层的额外增强

        # 1. 基础分数 = 因子值（不做行业归一化，保留行业轮动alpha）
        base_score = np.clip(factor_value, -10, 10)

        # 获取行业（用于标签）
        specific_industry = self._get_specific_industry(code, current_date) if code else ''

        # 2. 基本面增强（仅对非行业因子生效）
        if not is_industry and fundamental_score > 0:
            base_score = base_score + fundamental_score * self.fundamental_weight

        # 纯动量：直接使用因子值，不做二次加工
        score = base_score

        # === 缠论增强（czsc风格 7层融合）===
        chan_boost = self._get_chan_boost(ind, idx, market_info,
                                          specific_industry if code else '')
        chan_sl = int(self._safe_get(ind, 'signal_level', idx, 0))

        # === 缠论+因子 加法融合 (替代乘法叠加，提高可解释性) ===
        # 将缠论信号转为独立加法项，每层贡献独立可追溯
        # score = α × factor_score + β × chan_score
        chan_fusion_weight = getattr(self, 'chan_fusion_weight', 0.35)
        factor_fusion_weight = 1.0 - chan_fusion_weight
        chan_mult = chan_boost['boost_multiplier']
        chan_div_quality = chan_boost.get('divergence_quality', 0.0)
        is_chan_buy = chan_boost.get('is_chan_buy_boost', False)
        is_chan_sell = chan_boost.get('is_chan_sell_boost', False)

        # 计算缠论得分：基于信号层级和背离质量
        if chan_sl >= 2 and is_chan_buy:
            # 线段级/双级别买入 → 缠论强信号
            chan_score = (0.3 + 0.4 * chan_div_quality) * max(1.0, chan_mult)
        elif chan_sl <= -2 and is_chan_sell:
            # 线段级/双级别卖出 → 缠论强卖出
            sell_intensity = max(0.3, (1.0 - chan_mult) * 3.0)
            chan_score = -(0.3 + 0.4 * chan_div_quality) * sell_intensity
        elif is_chan_buy:
            # 笔级买入 → 缠论辅助信号
            chan_score = 0.15 * chan_div_quality * max(1.0, chan_mult)
        elif is_chan_sell:
            # 笔级卖出 → 缠论辅助信号
            chan_score = -0.15 * chan_div_quality * max(0.5, chan_mult)
        else:
            # 无缠论信号 → 缠论调整仍以温和乘法作用于因子分
            chan_score = 0.0
            score = score * max(0.7, min(chan_mult, 1.3))  # 无信号时保留温和调整

        if chan_score != 0.0:
            # Fix#1: 标准化加法融合 — 两分量纲不同，必须归一化到[0,1]后再融合
            # base_score ∈ [-10, 10] → [0, 1]
            base_norm = (base_score + 10) / 20.0
            base_norm = float(np.clip(base_norm, 0.0, 1.0))
            # chan_score ∈ [-0.65, 0.65] → [0, 1] (中性点=0.5)
            # 正值信号映射到(0.5, 1.0]，负值映射到[0.0, 0.5)
            if chan_score >= 0:
                chan_norm = 0.5 + 0.5 * float(np.clip(chan_score / 0.65, 0.0, 1.0))
            else:
                chan_norm = 0.5 - 0.5 * float(np.clip(abs(chan_score) / 0.65, 0.0, 1.0))
            chan_norm = float(np.clip(chan_norm, 0.0, 1.0))
            # 归一化后加权融合，再还原到合理分数范围
            fused_norm = factor_fusion_weight * base_norm + chan_fusion_weight * chan_norm  # ∈ [0, 1]
            score = fused_norm * 2.0 - 1.0  # 还原到 [-1, 1]，贴近因子分分布

        smart_money = self._safe_get(ind, 'smart_money_flow', idx, 0)
        bottom_div = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        top_div = self._safe_get(ind, 'top_divergence', idx, 0.0)

        # === P0-P2: 因子-缠论冲突解决 (兜底检查，仅对无质量过滤的信号生效) ===
        buy_point_raw = int(self._safe_get(ind, 'buy_point', idx, 0))
        trend_type_raw = int(self._safe_get(ind, 'trend_type', idx, 0))
        bottom_fx_q = self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0)

        # B3/B2/B1质量系统是否已介入 (chan_mult > 1.0 表示通过了某级质量门控)
        chan_quality_applied = chan_boost.get('is_chan_buy_boost', False)

        # P0: 纯因子信号 — 无Chan结构确认 → 温和降分
        # v8.1 fix: chan_buy_point IC≈0, 改用强Chan字段判断结构存在
        has_chan_structure = (chan_sl >= 1 or
                              bottom_div > 0.3 or bottom_fx_q > 0.25 or
                              chan_div_quality > 0.2)
        if not has_chan_structure and not chan_quality_applied and score > 0:
            score = score * 0.70  # 无缠论结构 → 温和降分(原0.55)

        # P1: Chan内部冲突 — 已移除 (v8.1 fix)
        # chan_sell_point IC=0.0005 p=0.656 纯幻觉, 基于幻觉的惩罚无意义
        # if buy_point_raw > 0 and sell_point_raw > 0: score *= 0.65

        # P2: 无趋势的Chan结构 — 仅当没有其他质量确认时才处罚
        # v8.1 fix: 改用chan_sl替代buy_point_raw(IC≈0)作为触发条件
        has_chan_hint = (chan_sl >= 1 or bottom_div > 0.3 or bottom_fx_q > 0.25)
        if has_chan_hint and trend_type_raw == 0 and not chan_quality_applied:
            if bottom_div > 0.3 or bottom_fx_q > 0.35:
                pass
            else:
                score = score * 0.72  # 减罚(原0.65)

        # === ML预测混合（非线性因子交互） ===
        if self.ml_enabled and self.ml_predictor is not None and \
           self.ml_predictor.is_trained() and code and current_date:
            ml_pred = self._ml_predictions.get((code, current_date))
            if ml_pred is not None and not np.isnan(ml_pred):
                ml_pred_scaled = np.tanh(ml_pred * 3)  # 压缩到(-1, 1)
                score = (1 - self.ml_blend_weight) * score + \
                        self.ml_blend_weight * ml_pred_scaled

        # 4. 波动率风险指标
        risk_vol = self._safe_get(ind, 'volatility_10', idx, 0.02)

        # 5. 市场状态: 信号层不做折扣 (Fix#7: 组合层通过敞口控制统一处理)
        # 信号层专注截面排序, 组合层专注仓位规模 — 各司其职
        regime_weight = 0.85 if risk_extreme else 1.0
        adjusted_score = score * regime_weight

        # === 缠论三系统共振 ===
        # 系统1(技术指标): score (已计算)
        # 系统2(资金流向): capital_flow_score
        # 系统3(资讯热点): news_sentiment_score
        # 三个系统独立运行，互不影响。多系统共振才产生强信号。
        cf_score = self._safe_get(ind, 'capital_flow_score', idx, 0.0)
        ns_score = self._safe_get(ind, 'news_sentiment_score', idx, 0.0)
        cf_dir = int(self._safe_get(ind, 'capital_flow_direction', idx, 0))
        ns_dir = int(self._safe_get(ind, 'news_sentiment_direction', idx, 0))

        # 判定各系统是否发出买入信号
        sys1_buy = score > self.buy_threshold * 0.7  # 放宽阈值，查共振而非绝对强度
        sys2_buy = cf_score > 0.5 and cf_dir == 1    # 明确资金流入(排除中性)
        sys3_buy = ns_score > 0.3 and ns_dir == 1    # 明确利好冲击(排除中性)
        sys1_sell = score < self.sell_threshold
        sys2_sell = cf_score > 0.5 and cf_dir == -1  # 资金流出
        sys3_sell = ns_score > 0.3 and ns_dir == -1  # 利空冲击

        n_buy_systems = sum([sys1_buy, sys2_buy, sys3_buy])
        n_sell_systems = sum([sys1_sell, sys2_sell, sys3_sell])

        # === 统一多时间框架折扣（月线→周线→日线） ===
        # 替代旧版 tf_agree 三档折扣 + Layer 8 EMA60/120 大级别保护
        # 使用日历感知重采样 + K线形态 + 支撑/阻力综合计算
        pre_discount_score = score  # 保存折扣前原始分数（用于参数网格扫描）
        mtf_discount = float(self._safe_get(ind, 'mtf_discount_factor', idx, 1.0))
        weekly_trend_up = bool(self._safe_get(ind, 'weekly_trend_up', idx, False))
        monthly_trend_up = bool(self._safe_get(ind, 'monthly_trend_up', idx, False))
        if score > 0:
            # 买入信号: MTF反转折扣 (v8.1 fix)
            # 数据: MTF对齐时买入准确率48.9% vs 不对齐时59.2%
            # 高级别趋势"好"=短期已透支→折扣; "差"=潜在反转→溢价
            mtf_buy_factor = 2.0 - mtf_discount
            mtf_buy_factor = float(np.clip(mtf_buy_factor, 0.5, 1.5))
            score *= mtf_buy_factor
            adjusted_score *= mtf_buy_factor
        elif score < 0:
            # 卖出信号: MTF对称折扣 — 顺势卖出(周月线偏空)折扣小, 逆势卖出折扣大
            sell_mtf_discount = 0.5 + (1.0 - mtf_discount) * 0.7
            sell_mtf_discount = max(0.35, min(1.0, sell_mtf_discount))
            score *= sell_mtf_discount
            adjusted_score *= sell_mtf_discount

        # 共振逻辑: 只有多系统同向时才放大信号
        if n_buy_systems >= 3:
            # 三系统共振 → 最强买入
            score = score * 1.25
            adjusted_score = adjusted_score * 1.25
        elif n_buy_systems == 2:
            # 双系统共振 → 较强买入
            score = score * 1.12
            adjusted_score = adjusted_score * 1.12
        elif n_buy_systems == 1 and not sys1_buy:
            # 仅资金或资讯系统有信号，技术系统无信号 → 观望
            score = score * 0.7
            adjusted_score = adjusted_score * 0.7
        elif n_sell_systems >= 2:
            # 多系统看空 → 强化卖出
            score = score * 0.7
            adjusted_score = adjusted_score * 0.7
        elif n_buy_systems == 0 and n_sell_systems == 0:
            # 三系统全无信号 → 不交易
            score = score * 0.8
            adjusted_score = adjusted_score * 0.8

        # 添加标签（先生成标签再决定buy信号）
        factor_tags = []
        if has_fundamental:
            factor_tags.append('F')
        if style_confidence > 0.3:
            factor_tags.append(style_regime[:2].upper())
        if smart_money > 0.15:
            factor_tags.append('V')
        if bottom_div > 0.3 or top_div > 0.3:
            factor_tags.append('D')
        # 三系统共振标签
        if n_buy_systems >= 2:
            factor_tags.append(f'R{n_buy_systems}')  # R2=双系统, R3=三系统
        factor_name = factor_name + ('_' + ''.join(factor_tags) if factor_tags else '_T')

        # 6. 信号信心度评分
        # 先计算MA20位置（供置信度评分和买入决策共用）
        close_price = self._safe_get(ind, 'close', idx, 0.0)
        ma20_val = self._safe_get(ind, 'ma20', idx, 0.0)
        price_above_ma20 = (close_price > 0 and ma20_val > 0 and close_price > ma20_val)
        dist_ma20 = (close_price - ma20_val) / ma20_val if ma20_val > 0 else 1.0

        confidence_factors = []
        if risk_info and risk_info.get('dyn_quality', 0) > 0.02:
            confidence_factors.append(min(risk_info['dyn_quality'] * 5, 0.3))
        if abs(smart_money) > 0.15:
            confidence_factors.append(0.10)
        if abs(bottom_div) > 0.3 or abs(top_div) > 0.3:
            confidence_factors.append(0.15)
        bottom_fx_qual = self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0)
        if bottom_fx_qual > 0.45:
            confidence_factors.append(0.12)
        elif bottom_fx_qual > 0.35:
            confidence_factors.append(0.06)
        # 放量突破MA20确认: 成交量>1.5x均量 + 价格在MA20上方 = 真突破
        vol_ratio_raw = self._safe_get(ind, 'volume_ratio_raw', idx, 1.0)
        if vol_ratio_raw > 1.5 and price_above_ma20:
            confidence_factors.append(0.12)
        elif vol_ratio_raw > 1.2 and price_above_ma20:
            confidence_factors.append(0.06)
        # 三系统共振置信度
        if n_buy_systems >= 3:
            confidence_factors.append(0.25)
        elif n_buy_systems == 2:
            confidence_factors.append(0.15)
        elif n_buy_systems == 1:
            confidence_factors.append(0.05)
        # 缠论结构信号置信度：结构层级越高越可信
        chan_sl = int(self._safe_get(ind, 'signal_level', idx, 0))
        chan_div_q = chan_boost.get('divergence_quality', 0.0)
        if chan_sl >= 3 and chan_div_q > 0.3:
            confidence_factors.append(0.20)   # 双级别确认 → 高置信
        elif chan_sl >= 2 and chan_div_q > 0.2:
            confidence_factors.append(0.12)   # 线段级 → 中高置信
        elif chan_sl == 1 and chan_div_q > 0.15:
            confidence_factors.append(0.06)   # 笔级 → 基础置信
        signal_confidence = min(sum(confidence_factors), 0.8) + 0.2

        # 7. 交易信号：基于score（含所有增强），而非裸factor_value
        if effective_buy_threshold is None:
            # 原有状态路径：滚动缓冲区 + 周期性百分位重算
            if score is not None and not np.isnan(score):
                self._factor_value_buffer.append(float(score))
            self._pct_counter += 1
            if len(self._factor_value_buffer) > 100 and self._pct_counter % 20 == 0:
                buf = list(self._factor_value_buffer)
                current_pct = self._buy_threshold_pct_map.get(risk_regime, self._buy_threshold_pct_map[0])
                pct = current_pct * 100
                self._cached_buy_threshold = max(
                    float(np.percentile(buf, pct)),
                    self.buy_threshold * 0.6
                )
                self._cached_sell_threshold = min(
                    float(np.percentile(buf, 100 - pct)),
                    self.sell_threshold
                )
            effective_buy_threshold = self._cached_buy_threshold
            effective_sell_threshold = self._cached_sell_threshold

        # 缠论买点信号: 强结构买点直接触发（与卖出逻辑对称）
        # Fix#5: 成交量门控已移除 — 2.0x/1.5x双重确认在当前市场中几乎无信号通过
        # 量能枯竭由 portfolio.py 短期趋势过滤#1 (vol_ratio<-0.30) 处理
        chan_buy_signal = chan_boost.get('is_chan_buy_boost', False)
        sl_buy = int(self._safe_get(ind, 'signal_level', idx, 0))
        bp_buy = int(self._safe_get(ind, 'buy_point', idx, 0))
        chan_force_buy = chan_buy_signal and (sl_buy >= 2 or bp_buy == 1)

        # MA20趋势+乖离过滤（前向计算已在置信度评分段完成）
        # B1底部反转: 价格天然在MA20下方，强Chan确认时允许最多-5%偏离
        b1_strong = (bp_buy == 1 and chan_buy_signal and sl_buy >= 2)
        if b1_strong:
            price_ok = dist_ma20 > -0.05  # B1允许在MA20下方5%以内
        else:
            price_ok = price_above_ma20   # 其他买点必须在MA20上方

        # 乖离上限: B3趋势加速可容忍更高偏离
        is_b3 = (bp_buy == 3 and chan_buy_signal)
        if is_b3 and sl_buy >= 2:
            max_dist = 0.40   # B3段级确认 → 40%
        elif is_b3:
            max_dist = 0.35   # B3通过质量过滤 → 35% (Fix#2)
        elif b1_strong:
            max_dist = 0.25   # B1底部反转 → 25%上限
        else:
            max_dist = 0.30   # 默认

        price_not_extended = dist_ma20 < max_dist

        buy = (score is not None and
               not np.isnan(score) and
               (score > effective_buy_threshold or chan_force_buy) and
               price_ok and
               price_not_extended)

        # 缠论卖点: 强卖点直接触发sell（与买入逻辑对称）
        chan_sell_signal = chan_boost.get('is_chan_sell_boost', False)
        sl_for_sell = int(self._safe_get(ind, 'signal_level', idx, 0))
        sp_for_sell = int(self._safe_get(ind, 'sell_point', idx, 0))
        chan_force_sell = chan_sell_signal and (sl_for_sell <= -2 or sp_for_sell >= 1)
        sell = (score is not None and
                not np.isnan(score) and
                (score < effective_sell_threshold or chan_force_sell))

        # 提取因子质量（用于组合层权重调整）
        factor_quality = risk_info.get('dyn_quality', 0.0) if risk_info else 0.0

        # 缠论背离类型判定（chanlun-pro增强版：多级别确认优先）
        buy_point = int(self._safe_get(ind, 'buy_point', idx, 0))
        sell_point = int(self._safe_get(ind, 'sell_point', idx, 0))
        bi_buy = int(self._safe_get(ind, 'bi_buy_point', idx, 0))
        bi_sell = int(self._safe_get(ind, 'bi_sell_point', idx, 0))
        signal_level = int(self._safe_get(ind, 'signal_level', idx, 0))
        confirmed_buy = bool(self._safe_get(ind, 'confirmed_buy', idx, 0))
        confirmed_sell = bool(self._safe_get(ind, 'confirmed_sell', idx, 0))
        bottom_div = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        top_div = self._safe_get(ind, 'top_divergence', idx, 0.0)

        # 双级别确认 (chanlun-pro核心): 笔+线段同时确认 → 最强信号
        if signal_level == 3 and confirmed_buy:
            chan_div_type = f'bi{bi_buy}_seg{buy_point}_buy'
        elif signal_level == -3 and confirmed_sell:
            chan_div_type = f'bi{bi_sell}_seg{sell_point}_sell'
        elif signal_level == 2 and confirmed_buy:
            chan_div_type = f'buy{buy_point}'
        elif signal_level == -2 and confirmed_sell:
            chan_div_type = f'sell{sell_point}'
        elif signal_level == 1 and confirmed_buy:
            chan_div_type = f'bi{bi_buy}'
        elif signal_level == -1 and confirmed_sell:
            chan_div_type = f'bi{bi_sell}'
        elif bottom_div > top_div and bottom_div > 0.3:
            chan_div_type = 'bottom'
        elif top_div > bottom_div and top_div > 0.3:
            chan_div_type = 'top'
        elif self._safe_get(ind, 'hidden_bottom_divergence', idx, 0.0) > 0.15:
            chan_div_type = 'hidden_bottom'
        elif self._safe_get(ind, 'hidden_top_divergence', idx, 0.0) > 0.15:
            chan_div_type = 'hidden_top'
        elif self._safe_get(ind, 'second_buy_point', idx, 0) > 0:
            chan_div_type = 'B2'  # 二买 — 最安全的缠论买点
        elif self._safe_get(ind, 'bottom_fractal_vol_spike', idx, 0.0) >= 3.0:
            chan_div_type = 'bottom_fx_3x'  # 量在价先
        elif self._safe_get(ind, 'bottom_fractal_quality', idx, 0.0) > 0.35:
            chan_div_type = 'bottom_fx'
        else:
            chan_div_type = 'none'

        return Signal(
            buy=buy, sell=sell, score=score, factor_value=factor_value,
            factor_name=factor_name, industry=specific_industry or '',
            risk_vol=risk_vol, risk_regime=risk_regime,
            risk_confidence=market_info.get('confidence', 0.0),
            risk_extreme=risk_extreme or risk_info.get('is_high_vol', False),
            adjusted_score=adjusted_score,
            pre_discount_score=pre_discount_score,
            factor_quality=factor_quality,
            signal_confidence=signal_confidence,
            chan_divergence_type=chan_div_type,
            chan_divergence_strength=max(bottom_div, top_div, self._safe_get(ind, 'buy_confidence', idx, 0.0),
                                         self._safe_get(ind, 'sell_confidence', idx, 0.0)),
            chan_structure_score=float(self._safe_get(ind, 'alignment_score', idx, 0.0)),
            chan_buy_point=buy_point,
            chan_sell_point=sell_point,
            signal_level=int(self._safe_get(ind, 'signal_level', idx, 0)),
            resonance_systems=max(n_buy_systems, n_sell_systems),
            capital_flow_score=cf_score,
            news_sentiment_score=ns_score,
            trend_type=int(self._safe_get(ind, 'trend_type', idx, 0)),
            chan_pivot_zg=float(self._safe_get(ind, 'chan_pivot_zg', idx, np.nan)),
            chan_pivot_zd=float(self._safe_get(ind, 'chan_pivot_zd', idx, np.nan)),
            chan_pivot_zz=float(self._safe_get(ind, 'chan_pivot_zz', idx, np.nan)),
            daily_return=float(self._safe_get(ind, 'ret', idx, 0.0)),
            volume_ratio=float(self._safe_get(ind, 'volume_ratio', idx, 0.0)),
            volume_ratio_raw=float(self._safe_get(ind, 'volume_ratio_raw', idx, 1.0)),
            exhaustion_risk=float(self._safe_get(ind, 'exhaustion_risk', idx, 0.0)),
            gap_breakout_confirm=float(self._safe_get(ind, 'gap_breakout_confirm', idx, 0.0)),
            stroke_phase=float(self._safe_get(ind, 'stroke_phase', idx, 0.0)),
            top_fractal_volume=float(self._safe_get(ind, 'top_fractal_volume', idx, 0.0)),
            ma_trend_up=bool(self._safe_get(ind, 'ema20_above_60', idx, False)),
            profit_declining=profit_declining,
            mom_60d=float(self._safe_get(ind, 'mom_60d', idx, 0.0)),
            dist_ma60=float(self._safe_get(ind, 'dist_ma60', idx, 0.0)),
            max_dd_20d=float(self._safe_get(ind, 'max_dd_20d', idx, 0.0)),
            vol_regime=float(self._safe_get(ind, 'vol_regime', idx, 1.0)),
            weekly_trend_up=bool(self._safe_get(ind, 'weekly_trend_up', idx, False)),
            monthly_trend_up=bool(self._safe_get(ind, 'monthly_trend_up', idx, False)),
            weekly_trend_strength=float(self._safe_get(ind, 'weekly_trend_strength', idx, 0.0)),
            monthly_trend_strength=float(self._safe_get(ind, 'monthly_trend_strength', idx, 0.0)),
            mtf_alignment_score=float(self._safe_get(ind, 'mtf_alignment_score', idx, 0.0)),
            mtf_discount_factor=float(self._safe_get(ind, 'mtf_discount_factor', idx, 1.0)),
            weekly_pattern_signal=float(self._safe_get(ind, 'weekly_pattern_signal', idx, 0.0)),
            nearest_resistance_pct=float(self._safe_get(ind, 'nearest_resistance_pct', idx, 0.0)),
            nearest_support_pct=float(self._safe_get(ind, 'nearest_support_pct', idx, 0.0)),
            _chan_buy_signal=is_chan_buy, _chan_sell_signal=is_chan_sell, _dist_ma20=dist_ma20,
        )

    def _reevaluate_buy_sell(self, sig: Signal, buy_threshold: float, sell_threshold: float,
                              ind: dict, idx: int) -> Tuple[bool, bool]:
        """基于已有Signal字段 + 新阈值重新判定买卖，不重复昂贵计算。

        使用Signal中的内部字段 _chan_buy_signal, _chan_sell_signal, _dist_ma20
        以及公开字段 score, chan_buy_point, chan_sell_point, signal_level 完成判定。
        """
        score = sig.score
        if score is None or np.isnan(score):
            return False, False

        dist_ma20 = sig._dist_ma20
        chan_buy_signal = sig._chan_buy_signal
        chan_sell_signal = sig._chan_sell_signal
        bp_buy = sig.chan_buy_point
        sp_for_sell = sig.chan_sell_point
        sl = sig.signal_level

        # chan_force_buy（与_generate_signal内逻辑完全一致）
        chan_force_buy = chan_buy_signal and (sl >= 2 or bp_buy == 1)

        # MA20价格位置检查
        close_price = float(self._safe_get(ind, 'close', idx, 0.0))
        ma20_val = float(self._safe_get(ind, 'ma20', idx, 0.0))
        price_above_ma20 = close_price > 0 and ma20_val > 0 and close_price > ma20_val

        b1_strong = (bp_buy == 1 and chan_buy_signal and sl >= 2)
        if b1_strong:
            price_ok = dist_ma20 > -0.05
        else:
            price_ok = price_above_ma20

        # 乖离上限
        is_b3 = (bp_buy == 3 and chan_buy_signal)
        if is_b3 and sl >= 2:
            max_dist = 0.40
        elif is_b3:
            max_dist = 0.35
        elif b1_strong:
            max_dist = 0.25
        else:
            max_dist = 0.30

        price_not_extended = dist_ma20 < max_dist

        buy = (score > buy_threshold or chan_force_buy) and price_ok and price_not_extended

        # chan_force_sell（与_generate_signal内逻辑完全一致）
        chan_force_sell = chan_sell_signal and (sl <= -2 or sp_for_sell >= 1)
        sell = score < sell_threshold or chan_force_sell

        return buy, sell

    # ========================================================================
    # 向量化批处理：标量收集 + 数组级联装配
    # ========================================================================

    def _collect_bar_scalars(self, ind: dict, code: str, dates: np.ndarray, n: int,
                              regimes: np.ndarray = None, latest_only: bool = False) -> dict:
        """逐bar收集复杂方法调用的标量结果（仅"硬"部分，不含算术）。

        latest_only=True 时采用混合路径：
        - 所有 i>=60 的 bar 用 _calculate_default_factor 快速生成分数（用于动态阈值）
        - 仅 i=n-1 走完整 _select_factor + _get_chan_boost 链（用于最终信号）
        - 预期提速 >95%，同时保留动态阈值的自适应能力

        Returns:
            dict of arrays, 每个array长度为n。valid_mask标记有效bar。
        """
        # 预分配数组
        valid = np.zeros(n, dtype=bool)
        mkt_regime = np.zeros(n, dtype=int)
        mkt_extreme = np.zeros(n, dtype=bool)
        mkt_style_regime = np.full(n, 'balanced', dtype=object)
        mkt_style_score = np.zeros(n)
        mkt_style_conf = np.zeros(n)
        mkt_conf = np.zeros(n)
        ind_cat = np.full(n, 'default', dtype=object)
        fname = np.full(n, 'V41', dtype=object)
        fval = np.zeros(n)
        risk_qual = np.zeros(n)
        risk_hvol = np.zeros(n, dtype=bool)
        is_ind = np.zeros(n, dtype=bool)
        fund_score = np.zeros(n)
        has_fund = np.zeros(n, dtype=bool)
        profit_dec = np.zeros(n, dtype=bool)
        style_score = np.zeros(n)
        chan_mult = np.ones(n)
        chan_div_q = np.zeros(n)
        chan_buy = np.zeros(n, dtype=bool)
        chan_sell = np.zeros(n, dtype=bool)
        spec_ind = np.full(n, '', dtype=object)

        if latest_only and regimes is not None:
            # ── 混合路径：所有 bar 快速分数 + 最新 bar 完整链 ──
            for i in range(60, n):
                # 快速分数：纯指标计算，无外部查询
                fn, fv, risk_info = self._calculate_default_factor(ind, i, int(regimes[i]), 'default')
                valid[i] = True
                mkt_regime[i] = int(regimes[i])
                fname[i] = fn
                fval[i] = fv
                risk_qual[i] = risk_info.get('dyn_quality', 0.0) if risk_info else 0.0

            # 最新 bar：完整链覆盖
            last = n - 1
            current_date = dates[last]
            market_info = self._get_market_info(current_date)
            industry_category = self._get_industry_category(code, current_date)

            factor_result = self._select_factor(
                ind, last, market_info['regime'], industry_category,
                code=code, current_date=current_date
            )
            if factor_result is not None:
                fn, fv, risk_info, is_ind_f = factor_result
                # 覆盖快速路径的值
                mkt_regime[last] = market_info['regime']
                mkt_extreme[last] = market_info['is_extreme']
                mkt_style_regime[last] = market_info.get('style_regime', 'balanced')
                mkt_style_score[last] = market_info.get('style_score', 0.0)
                mkt_style_conf[last] = market_info.get('style_confidence', 0.0)
                mkt_conf[last] = market_info.get('confidence', 0.0)
                ind_cat[last] = industry_category
                fname[last] = fn
                fval[last] = fv
                risk_qual[last] = risk_info.get('dyn_quality', 0.0) if risk_info else 0.0
                risk_hvol[last] = risk_info.get('is_high_vol', False) if risk_info else False
                is_ind[last] = is_ind_f

                if self.fundamental_enabled and code:
                    fs = self._get_fundamental_score(code, current_date)
                    fund_score[last] = fs
                    has_fund[last] = fs > 0
                    profit_dec[last] = self._check_profit_decline(code, current_date)

                style_score[last] = self._get_style_score(ind, last, market_info)
                spec_ind[last] = self._get_specific_industry(code, current_date) if code else ''
                cb = self._get_chan_boost(ind, last, market_info, spec_ind[last] if code else '')
                chan_mult[last] = cb['boost_multiplier']
                chan_div_q[last] = cb.get('divergence_quality', 0.0)
                chan_buy[last] = cb.get('is_chan_buy_boost', False)
                chan_sell[last] = cb.get('is_chan_sell_boost', False)
        else:
            # ── 原路径：所有 bar 完整链（回测用） ──
            for i in range(n):
                if i < 60:
                    continue

                current_date = dates[i]
                market_info = self._get_market_info(current_date)
                industry_category = self._get_industry_category(code, current_date)

                factor_result = self._select_factor(
                    ind, i, market_info['regime'], industry_category,
                    code=code, current_date=current_date
                )
                if factor_result is None:
                    continue

                fn, fv, risk_info, is_ind_f = factor_result
                valid[i] = True
                mkt_regime[i] = market_info['regime']
                mkt_extreme[i] = market_info['is_extreme']
                mkt_style_regime[i] = market_info.get('style_regime', 'balanced')
                mkt_style_score[i] = market_info.get('style_score', 0.0)
                mkt_style_conf[i] = market_info.get('style_confidence', 0.0)
                mkt_conf[i] = market_info.get('confidence', 0.0)
                ind_cat[i] = industry_category
                fname[i] = fn
                fval[i] = fv
                risk_qual[i] = risk_info.get('dyn_quality', 0.0) if risk_info else 0.0
                risk_hvol[i] = risk_info.get('is_high_vol', False) if risk_info else False
                is_ind[i] = is_ind_f

                if self.fundamental_enabled and code:
                    fs = self._get_fundamental_score(code, current_date)
                    fund_score[i] = fs
                    has_fund[i] = fs > 0
                    profit_dec[i] = self._check_profit_decline(code, current_date)

                style_score[i] = self._get_style_score(ind, i, market_info)

                spec_ind[i] = self._get_specific_industry(code, current_date) if code else ''
                cb = self._get_chan_boost(ind, i, market_info, spec_ind[i] if code else '')
                chan_mult[i] = cb['boost_multiplier']
                chan_div_q[i] = cb.get('divergence_quality', 0.0)
                chan_buy[i] = cb.get('is_chan_buy_boost', False)
                chan_sell[i] = cb.get('is_chan_sell_boost', False)

        return {
            'valid': valid,
            'market_regime': mkt_regime, 'market_extreme': mkt_extreme,
            'market_style_regime': mkt_style_regime, 'market_style_score': mkt_style_score,
            'market_style_confidence': mkt_style_conf, 'market_confidence': mkt_conf,
            'industry_category': ind_cat,
            'factor_name': fname, 'factor_value': fval,
            'risk_dyn_quality': risk_qual, 'risk_is_high_vol': risk_hvol,
            'is_industry': is_ind,
            'fundamental_score': fund_score, 'has_fundamental': has_fund,
            'profit_declining': profit_dec, 'style_factor_score': style_score,
            'chan_boost_multiplier': chan_mult, 'chan_divergence_quality': chan_div_q,
            'chan_is_buy_boost': chan_buy, 'chan_is_sell_boost': chan_sell,
            'specific_industry': spec_ind,
        }

    def _vectorized_score_assembly(self, s: dict, ind: dict, n: int, code: str = '') -> dict:
        """向量化分数装配（低内存版）：原位运算 + 按段释放中间数组。

        输入 s = _collect_bar_scalars 的输出 + ind指标字典。
        输出 r = 所有Signal构造所需字段的数组（尽量引用 s/ind 已有数组，不复制）。
        """
        valid = s['valid']
        cfw = getattr(self, 'chan_fusion_weight', 0.35)
        ffw = 1.0 - cfw

        # === 1. 基础分数（原位运算） ===
        score = np.clip(s['factor_value'], -10.0, 10.0)
        # 基本面增强
        fund_mask = ~s['is_industry'] & (s['fundamental_score'] > 0)
        score[fund_mask] += s['fundamental_score'][fund_mask] * self.fundamental_weight

        # === 1.5. trend_initiation 增强: ti>0时温和提升score，帮助克服旧动量因子滞后 ===
        ti_arr = _safe_get_arr(ind, 'trend_initiation', n, 0.0)
        ti_boost = np.tanh(np.maximum(0, ti_arr) * 2.0) * 0.12
        score += ti_boost

        # 保存 pre_discount_score 在 MTF 折扣前（引用，后面会 copy）
        pre_discount_score = None  # 延迟到 MTF 前赋值

        # === 2. 缠论+因子 加法融合 ===
        chan_sl = _safe_get_arr(ind, 'signal_level', n, 0).astype(int)
        chan_mult = s['chan_boost_multiplier']
        chan_div_q = s['chan_divergence_quality']
        is_chan_buy = s['chan_is_buy_boost']
        is_chan_sell = s['chan_is_sell_boost']

        # 条件掩码
        cond_buy_strong = (chan_sl >= 2) & is_chan_buy
        cond_sell_strong = (chan_sl <= -2) & is_chan_sell
        cond_buy_weak = is_chan_buy & ~cond_buy_strong
        cond_sell_weak = is_chan_sell & ~cond_sell_strong
        cond_no_chan = ~(is_chan_buy | is_chan_sell)

        chan_score = np.zeros(n)
        chan_score[cond_buy_strong] = (0.3 + 0.4 * chan_div_q[cond_buy_strong]) * np.maximum(1.0, chan_mult[cond_buy_strong])
        sell_intensity = np.maximum(0.3, (1.0 - chan_mult) * 3.0)
        chan_score[cond_sell_strong] = -(0.3 + 0.4 * chan_div_q[cond_sell_strong]) * sell_intensity[cond_sell_strong]
        chan_score[cond_buy_weak] = 0.15 * chan_div_q[cond_buy_weak] * np.maximum(1.0, chan_mult[cond_buy_weak])
        chan_score[cond_sell_weak] = -0.15 * chan_div_q[cond_sell_weak] * np.maximum(0.5, chan_mult[cond_sell_weak])

        # 无缠论信号时温和乘法调整（原位）
        score[cond_no_chan] *= np.clip(chan_mult[cond_no_chan], 0.7, 1.3)

        # 标准化加法融合（仅对有缠论信号的bar）
        has_chan = chan_score != 0.0
        if has_chan.any():
            base_norm = np.clip((score + 10.0) / 20.0, 0.0, 1.0)
            ch_pos = chan_score >= 0
            ch_neg = ~ch_pos
            chan_norm = np.empty(n)
            chan_norm[ch_pos] = 0.5 + 0.5 * np.clip(chan_score[ch_pos] / 0.65, 0.0, 1.0)
            chan_norm[ch_neg] = 0.5 - 0.5 * np.clip(np.abs(chan_score[ch_neg]) / 0.65, 0.0, 1.0)
            np.clip(chan_norm, 0.0, 1.0, out=chan_norm)
            # 自适应融合权重: ti强劲时趋势端权重大，缓解新旧因子分歧
            ti_mean = float(np.mean(ti_arr[has_chan]))
            adaptive_cfw = cfw + np.clip(ti_mean * 0.3, 0.0, 0.12)
            adaptive_ffw = 1.0 - adaptive_cfw
            fused_norm = adaptive_ffw * base_norm + adaptive_cfw * chan_norm
            score[has_chan] = fused_norm[has_chan] * 2.0 - 1.0
            del base_norm, chan_norm, fused_norm, ch_pos, ch_neg
        del chan_score, cond_buy_strong, cond_sell_strong, cond_buy_weak, cond_sell_weak, sell_intensity

        # === 3. P0-P2 冲突解决（原位乘法） ===
        buy_point_raw = _safe_get_arr(ind, 'buy_point', n, 0).astype(int)
        sell_point_raw = _safe_get_arr(ind, 'sell_point', n, 0).astype(int)
        trend_type_raw = _safe_get_arr(ind, 'trend_type', n, 0).astype(int)
        bottom_fx_q = _safe_get_arr(ind, 'bottom_fractal_quality', n, 0.0)
        bottom_div = _safe_get_arr(ind, 'bottom_divergence', n, 0.0)
        top_div = _safe_get_arr(ind, 'top_divergence', n, 0.0)

        # v8.1 fix: 移除chan_buy_point(IC≈0), 改用强Chan字段
        has_chan_struct = (chan_sl >= 1) | (bottom_div > 0.3) | (bottom_fx_q > 0.25) | (chan_div_q > 0.2)

        # P0
        mask = ~has_chan_struct & ~is_chan_buy & (score > 0)
        score[mask] *= 0.70
        # P1: 已移除 (chan_sell_point IC=0.0005 纯幻觉)
        # P2
        mask = has_chan_struct & (trend_type_raw == 0) & ~is_chan_buy & \
               ~((bottom_div > 0.3) | (bottom_fx_q > 0.35))
        score[mask] *= 0.72
        del has_chan_struct

        # === 4. 波动率风险 + 市场状态调整 ===
        risk_vol = _safe_get_arr(ind, 'volatility_10', n, 0.02)
        regime_weight = np.where(s['market_extreme'], 0.85, 1.0)
        adjusted_score = score * regime_weight

        # === 5. 三系统共振 ===
        cf_score = _safe_get_arr(ind, 'capital_flow_score', n, 0.0)
        ns_score = _safe_get_arr(ind, 'news_sentiment_score', n, 0.0)
        cf_dir = _safe_get_arr(ind, 'capital_flow_direction', n, 0).astype(int)
        ns_dir = _safe_get_arr(ind, 'news_sentiment_direction', n, 0).astype(int)

        s1b = score > self.buy_threshold * 0.7
        s2b = (cf_score > 0.5) & (cf_dir == 1)
        s3b = (ns_score > 0.3) & (ns_dir == 1)
        s1s = score < self.sell_threshold
        s2s = (cf_score > 0.5) & (cf_dir == -1)
        s3s = (ns_score > 0.3) & (ns_dir == -1)
        n_buy = s1b.astype(np.int8) + s2b.astype(np.int8) + s3b.astype(np.int8)
        n_sell = s1s.astype(np.int8) + s2s.astype(np.int8) + s3s.astype(np.int8)

        # === 6. MTF 折扣（原位） ===
        pre_discount_score = score.copy()  # 此时才 copy，避免之前中间状态的 score
        mtf_discount = _safe_get_arr(ind, 'mtf_discount_factor', n, 1.0)

        pos_mask = score > 0
        # v8.1 fix: MTF反转 — MTF对齐时折扣, 不对齐时溢价
        mtf_buy = np.clip(2.0 - mtf_discount, 0.5, 1.5)
        score[pos_mask] *= mtf_buy[pos_mask]
        adjusted_score[pos_mask] *= mtf_buy[pos_mask]

        neg_mask = score < 0
        sell_mtf = np.clip(0.5 + (1.0 - mtf_discount) * 0.7, 0.35, 1.0)
        score[neg_mask] *= sell_mtf[neg_mask]
        adjusted_score[neg_mask] *= sell_mtf[neg_mask]
        del pos_mask, neg_mask, sell_mtf

        # === 7. 共振逻辑（原位乘法） ===
        mask = n_buy >= 3
        score[mask] *= 1.25; adjusted_score[mask] *= 1.25
        mask = n_buy == 2
        score[mask] *= 1.12; adjusted_score[mask] *= 1.12
        mask = (n_buy == 1) & ~s1b
        score[mask] *= 0.7; adjusted_score[mask] *= 0.7
        mask = n_sell >= 2
        score[mask] *= 0.7; adjusted_score[mask] *= 0.7
        mask = (n_buy == 0) & (n_sell == 0)
        score[mask] *= 0.8; adjusted_score[mask] *= 0.8
        del s1b, s2b, s3b, s1s, s2s, s3s

        # === 8. 置信度评分（按贡献项逐次叠加，避免中间数组膨胀） ===
        close_price = _safe_get_arr(ind, 'close', n, 0.0)
        ma20_val = _safe_get_arr(ind, 'ma20', n, 0.0)
        price_above_ma20 = (close_price > 0) & (ma20_val > 0) & (close_price > ma20_val)
        dist_ma20 = np.where(ma20_val > 0, (close_price - ma20_val) / ma20_val, 1.0)

        smart_money = _safe_get_arr(ind, 'smart_money_flow', n, 0.0)
        bottom_fx_qual = _safe_get_arr(ind, 'bottom_fractal_quality', n, 0.0)
        vol_ratio_raw = _safe_get_arr(ind, 'volume_ratio_raw', n, 1.0)

        conf = np.full(n, 0.2)
        conf = np.where(s['risk_dyn_quality'] > 0.02,
                        conf + np.clip(s['risk_dyn_quality'] * 5, 0, 0.3), conf)
        conf = np.where(np.abs(smart_money) > 0.15, conf + 0.10, conf)
        conf = np.where((np.abs(bottom_div) > 0.3) | (np.abs(top_div) > 0.3), conf + 0.15, conf)
        conf = np.where(bottom_fx_qual > 0.45, conf + 0.12,
                        np.where(bottom_fx_qual > 0.35, conf + 0.06, conf))
        conf = np.where((vol_ratio_raw > 1.5) & price_above_ma20, conf + 0.12,
                        np.where((vol_ratio_raw > 1.2) & price_above_ma20, conf + 0.06, conf))
        conf = np.where(n_buy >= 3, conf + 0.25,
                        np.where(n_buy == 2, conf + 0.15,
                                 np.where(n_buy == 1, conf + 0.05, conf)))
        conf = np.where((chan_sl >= 3) & (chan_div_q > 0.3), conf + 0.20,
                        np.where((chan_sl >= 2) & (chan_div_q > 0.2), conf + 0.12,
                                 np.where((chan_sl == 1) & (chan_div_q > 0.15), conf + 0.06, conf)))
        signal_confidence = np.clip(conf, 0.0, 1.0)
        del conf

        # === 9. 缠论背离类型判定 ===
        confirmed_buy = _safe_get_arr(ind, 'confirmed_buy', n, 0).astype(bool)
        confirmed_sell = _safe_get_arr(ind, 'confirmed_sell', n, 0).astype(bool)
        bi_buy = _safe_get_arr(ind, 'bi_buy_point', n, 0).astype(int)
        bi_sell = _safe_get_arr(ind, 'bi_sell_point', n, 0).astype(int)
        hidden_bottom = _safe_get_arr(ind, 'hidden_bottom_divergence', n, 0.0)
        hidden_top = _safe_get_arr(ind, 'hidden_top_divergence', n, 0.0)
        second_buy = _safe_get_arr(ind, 'second_buy_point', n, 0).astype(int)
        bottom_fx_spike = _safe_get_arr(ind, 'bottom_fractal_vol_spike', n, 0.0)

        div_type = np.full(n, 'none', dtype=object)
        for i in range(n):
            if not valid[i]:
                continue
            sl_i = chan_sl[i]
            if sl_i == 3 and confirmed_buy[i]:
                div_type[i] = f'bi{bi_buy[i]}_seg{buy_point_raw[i]}_buy'
            elif sl_i == -3 and confirmed_sell[i]:
                div_type[i] = f'bi{bi_sell[i]}_seg{sell_point_raw[i]}_sell'
            elif sl_i == 2 and confirmed_buy[i]:
                div_type[i] = f'buy{buy_point_raw[i]}'
            elif sl_i == -2 and confirmed_sell[i]:
                div_type[i] = f'sell{sell_point_raw[i]}'
            elif sl_i == 1 and confirmed_buy[i]:
                div_type[i] = f'bi{bi_buy[i]}'
            elif sl_i == -1 and confirmed_sell[i]:
                div_type[i] = f'bi{bi_sell[i]}'
            elif bottom_div[i] > top_div[i] and bottom_div[i] > 0.3:
                div_type[i] = 'bottom'
            elif top_div[i] > bottom_div[i] and top_div[i] > 0.3:
                div_type[i] = 'top'
            elif hidden_bottom[i] > 0.15:
                div_type[i] = 'hidden_bottom'
            elif hidden_top[i] > 0.15:
                div_type[i] = 'hidden_top'
            elif second_buy[i] > 0:
                div_type[i] = 'B2'
            elif bottom_fx_spike[i] >= 3.0:
                div_type[i] = 'bottom_fx_3x'
            elif bottom_fx_qual[i] > 0.35:
                div_type[i] = 'bottom_fx'

        # === 10. 组装返回（尽量引用 s/ind 已有数组，不复制） ===
        return {
            'valid': valid,
            'score': score,
            'adjusted_score': adjusted_score,
            'pre_discount_score': pre_discount_score,
            'factor_value': s['factor_value'],
            'factor_name': s['factor_name'],
            'industry': s['specific_industry'],
            'risk_vol': risk_vol,
            'risk_regime': s['market_regime'],
            'risk_confidence': s['market_confidence'],
            'risk_extreme': s['market_extreme'] | s['risk_is_high_vol'],
            'factor_quality': s['risk_dyn_quality'],
            'signal_confidence': signal_confidence,
            'chan_divergence_type': div_type,
            'chan_divergence_strength': np.maximum(
                np.maximum(bottom_div, top_div),
                np.maximum(_safe_get_arr(ind, 'buy_confidence', n, 0.0),
                           _safe_get_arr(ind, 'sell_confidence', n, 0.0))
            ),
            'chan_structure_score': _safe_get_arr(ind, 'alignment_score', n, 0.0),
            'chan_buy_point': buy_point_raw,
            'chan_sell_point': sell_point_raw,
            'signal_level': chan_sl,
            'resonance_systems': np.maximum(n_buy, n_sell),
            'capital_flow_score': cf_score,
            'news_sentiment_score': ns_score,
            'trend_type': trend_type_raw,
            'chan_pivot_zg': _safe_get_arr(ind, 'chan_pivot_zg', n, np.nan),
            'chan_pivot_zd': _safe_get_arr(ind, 'chan_pivot_zd', n, np.nan),
            'chan_pivot_zz': _safe_get_arr(ind, 'chan_pivot_zz', n, np.nan),
            'daily_return': _safe_get_arr(ind, 'ret', n, 0.0),
            'volume_ratio': _safe_get_arr(ind, 'volume_ratio', n, 0.0),
            'volume_ratio_raw': vol_ratio_raw,
            'exhaustion_risk': _safe_get_arr(ind, 'exhaustion_risk', n, 0.0),
            'gap_breakout_confirm': _safe_get_arr(ind, 'gap_breakout_confirm', n, 0.0),
            'stroke_phase': _safe_get_arr(ind, 'stroke_phase', n, 0.0),
            'top_fractal_volume': _safe_get_arr(ind, 'top_fractal_volume', n, 0.0),
            'ma_trend_up': _safe_get_arr(ind, 'ema20_above_60', n, 0).astype(bool),
            'profit_declining': s['profit_declining'],
            'mom_60d': _safe_get_arr(ind, 'mom_60d', n, 0.0),
            'dist_ma60': _safe_get_arr(ind, 'dist_ma60', n, 0.0),
            'max_dd_20d': _safe_get_arr(ind, 'max_dd_20d', n, 0.0),
            'vol_regime': _safe_get_arr(ind, 'vol_regime', n, 1.0),
            # MTF fields
            'weekly_trend_up': _safe_get_arr(ind, 'weekly_trend_up', n, 0).astype(bool),
            'monthly_trend_up': _safe_get_arr(ind, 'monthly_trend_up', n, 0).astype(bool),
            'weekly_trend_strength': _safe_get_arr(ind, 'weekly_trend_strength', n, 0.0),
            'monthly_trend_strength': _safe_get_arr(ind, 'monthly_trend_strength', n, 0.0),
            'mtf_alignment_score': _safe_get_arr(ind, 'mtf_alignment_score', n, 0.0),
            'mtf_discount_factor': mtf_discount,
            'weekly_pattern_signal': _safe_get_arr(ind, 'weekly_pattern_signal', n, 0.0),
            'nearest_resistance_pct': _safe_get_arr(ind, 'nearest_resistance_pct', n, 0.0),
            'nearest_support_pct': _safe_get_arr(ind, 'nearest_support_pct', n, 0.0),
            # 内部字段（用于第二遍阈值重评估）
            '_chan_buy_signal': is_chan_buy,
            '_chan_sell_signal': is_chan_sell,
            '_dist_ma20': dist_ma20,
            # 辅助
            'has_fundamental': s['has_fundamental'],
            'style_regime': s['market_style_regime'],
            'style_confidence': s['market_style_confidence'],
            'smart_money': smart_money,
            'n_buy_systems': n_buy,
        }

    def _compute_dynamic_thresholds(self, scores: np.ndarray, regimes: np.ndarray) -> tuple:
        """向量化动态阈值：rolling quantile替代逐bar buffer/counter。

        原逻辑等价转换：
        - deque(maxlen=800) → rolling(window=800, min_periods=100)
        - 每20根K线重算 → 每根K线直接重算（原为性能折衷，向量化后无此必要）
        - 按市场状态选择百分位 → 按regime分组计算不同分位数

        Args:
            scores: shape (n,), 含NaN表示无有效分数
            regimes: shape (n,), 市场状态编码 {1:bull, 0:neutral, -1:bear}

        Returns:
            buy_thresholds: shape (n,), 各bar的买入阈值
            sell_thresholds: shape (n,), 各bar的卖出阈值
        """
        n = len(scores)
        # 默认阈值（rolling quantile不足100根时使用）
        buy_thresholds = np.full(n, self.buy_threshold)
        sell_thresholds = np.full(n, self.sell_threshold)

        # 逐regime计算rolling quantile（regime种类少，循环开销可忽略）
        for regime, pct_raw in self._buy_threshold_pct_map.items():
            mask = regimes == regime
            if not mask.any():
                continue

            pct = pct_raw  # e.g. 0.45
            # NaN-mask non-matching bars so rolling quantile only sees this regime
            s_regime = pd.Series(np.where(mask, scores, np.nan))
            # rolling quantile: 包含当前bar，等价于deque append后算percentile
            # min_periods=100 等价于 len(buffer) > 100
            roll_buy = s_regime.rolling(window=800, min_periods=100).quantile(pct)
            roll_sell = s_regime.rolling(window=800, min_periods=100).quantile(1.0 - pct)

            # 只覆写rolling quantile有效的bar（>=100根），其余保留默认阈值
            buy_vals = roll_buy.values
            sell_vals = roll_sell.values
            valid = mask & ~np.isnan(buy_vals)
            buy_thresholds[valid] = np.maximum(buy_vals[valid], self.buy_threshold * 0.6)
            sell_thresholds[valid] = np.minimum(sell_vals[valid], self.sell_threshold)

        return buy_thresholds, sell_thresholds

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
        specific_industry = self._get_specific_industry(code, current_date) if code else ''

        # fixed模式：直接使用默认因子（跳过行业配置）
        if self.factor_mode == 'fixed':
            self._stats['fixed_default'] += 1
            factor_name, factor_value, risk_info = self._calculate_default_factor(ind, idx, regime, industry_category)
            return factor_name, factor_value, risk_info, False

        # 动态因子优先 (仅dynamic/both模式)
        if self.factor_mode in ['dynamic', 'both']:
            if self.dynamic_factor_selector.enabled and code and current_date:
                result = self._select_factor_dynamic(ind, idx, regime, code, current_date)
                if result:
                    self._stats['dynamic_success'] += 1
                    return result
                # 动态选择失败
                if self.factor_mode == 'dynamic' and not self.factor_fallback_to_fixed:
                    self._stats['dynamic_fallback_none'] += 1
                    return None

        # reweight模式: 走静态因子选择路径, 但使用walk-forward IC调整权重
        # fixed/both/reweight模式: 走标准静态路径

        # reweight模式: 获取walk-forward IC权重
        dyn_ic_weights = None
        if self.factor_mode == 'reweight' and self.dynamic_factor_selector.enabled and code and current_date:
            dyn_ic_weights = self._get_dynamic_ic_weights(specific_industry, current_date)

        # 固定因子（行业特定或默认）
        # 注意：当factor_mode='dynamic'时，只有fallback允许时才会到达这里
        if self.factor_mode in ['fixed', 'both', 'reweight'] or (self.factor_mode == 'dynamic' and self.factor_fallback_to_fixed):
            if self.industry_factor_enabled and code and current_date:
                # 使用行业特定因子（已按市场状态优化）
                if specific_industry and specific_industry in INDUSTRY_FACTOR_CONFIG:
                    result = self._calculate_industry_factor_score(ind, idx, specific_industry,
                                                                   code=code, current_date=current_date,
                                                                   regime=regime,
                                                                   dynamic_ic_weights=dyn_ic_weights)
                    if result:
                        self._stats['fixed_industry'] += 1
                        factor_name, factor_value, risk_info = result
                        # 不做熊市折扣：组合层用截面rank_pct排序，均匀缩放不改变排名
                        factor_name = factor_name + f'_{specific_industry[:2]}'
                        return factor_name, factor_value, risk_info, True

        # Fix#13: 最终兜底 — 使用行业category的配置（优先）
        # 如果specific_industry不在配置中,尝试用industry_category匹配
        if self.industry_factor_enabled:
            fallback_ind = specific_industry if specific_industry in INDUSTRY_FACTOR_CONFIG else industry_category
            if fallback_ind in INDUSTRY_FACTOR_CONFIG:
                result = self._calculate_industry_factor_score(
                    ind, idx, fallback_ind, code=code, current_date=current_date,
                    regime=0, dynamic_ic_weights=None
                )
                if result:
                    self._stats['fixed_default'] += 1
                    factor_name, factor_value, risk_info = result
                    return factor_name, factor_value, risk_info, True

        # 绝对兜底: 原始市场状态信号(MOM/REV/SHARPE)
        self._stats['fixed_default'] += 1
        factor_name, factor_value, risk_info = self._calculate_default_factor(ind, idx, regime, industry_category)
        return factor_name, factor_value, risk_info, False

    def _get_dynamic_ic_weights(self, industry: str, current_date) -> Optional[dict]:
        """获取walk-forward IC权重（reweight模式）

        Returns:
            {factor_name: ic_weight} or None
        """
        if not industry or self.dynamic_factor_selector.factor_df is None:
            return None

        if hasattr(current_date, 'date'):
            current_date_str = str(current_date.date())
        else:
            current_date_str = str(current_date)

        all_dates = self.dynamic_factor_selector._all_dates_cache
        if not all_dates:
            return None

        try:
            industry_factors = self.dynamic_factor_selector.select_factors_for_date(current_date_str, all_dates)
        except Exception:
            logger.warning(f"动态因子选择失败 date={current_date_str}", exc_info=True)
            return None

        if not industry_factors or industry not in industry_factors:
            return None

        selected_info = industry_factors[industry]
        if not selected_info or 'factors' not in selected_info:
            return None

        factors = selected_info['factors']
        weights = selected_info.get('weights', None)
        if not weights or len(weights) != len(factors):
            return None

        return {f: w for f, w in zip(factors, weights)}

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

        # 检查因子数据是否存在（factor_df 或预计算的 factor_cache 至少有一个）
        if self.dynamic_factor_selector.factor_df is None and not self.dynamic_factor_selector._factor_cache:
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
        # dyn_quality = avg combined_IR, <0.08 ≈ 因子几乎无预测力
        DYN_QUALITY_THRESHOLD = 0.04
        if dyn_quality < DYN_QUALITY_THRESHOLD:
            return None

        # 只要有因子通过IC验证就使用
        n_factors = len(selected_factors)
        if n_factors < 1:
            return None

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
                # 因子已压缩到(-1, 1)范围，检查是否有超出范围的异常
                if abs(factor_val) > 5:
                    # 记录异常因子值用于调试
                    import warnings
                    warnings.warn(f'Extreme factor value after compression: {factor_name}={factor_val:.2e} for {code} on {current_date}')
                    # 将极端值裁剪到合理范围
                    factor_val = np.sign(factor_val) * np.tanh(abs(factor_val))
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

        # 最终安全检查：因子值应该在(-1, 1)范围
        # 多因子加权平均后可能略微超出，使用tanh再次压缩
        if abs(factor_value) > 1.5:
            factor_value = np.tanh(factor_value)

        # 不做熊市折扣：组合层用截面rank_pct排序，均匀缩放不改变排名

        factor_name = f'DYN_{specific_industry[:4]}_{len(valid_factors)}F'
        risk_info = {'is_high_vol': False, 'dynamic_factor': True, 'n_factors': len(valid_factors),
                     'dyn_quality': dyn_quality, 'selected_factors': valid_factors}

        return factor_name, factor_value, risk_info, True

    def _get_fundamental_factor_value(self, code, current_date, factor_name: str) -> Optional[float]:
        """获取基本面因子值 - 使用统一压缩函数"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
            return None

        raw_value = self._get_raw_fundamental_value(code, current_date, factor_name)
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            return None
        return compress_fundamental_factor(raw_value, factor_name)

    def _check_profit_decline(self, code, current_date) -> bool:
        """基本面排雷: 利润下滑且降幅在扩大(恶化), 而非收窄(拐点)

        国茂股份案例: 利润同比 -33%→-27%→-11%, 降幅收窄+营收转正=拐点, 不应过滤
        真正该过滤的是: -10%→-20%→-35%, 降幅扩大=基本面加速恶化
        """
        try:
            if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
                return False
            fd = self.fundamental_data
            if code not in fd.stock_data:
                return False

            df = fd.stock_data[code]
            if '数据可用日期' not in df.columns or len(df) == 0:
                return False

            # 找到当前日期之前的最新报告，然后取最近3份报告
            current_date_str = str(current_date)[:10].replace('-', '')
            df_sorted = df.sort_values('数据可用日期').reset_index(drop=True)
            df_sorted['数据可用日期_str'] = df_sorted['数据可用日期'].astype(str)

            # 找到 <= current_date 的最新报告
            mask = df_sorted['数据可用日期_str'] <= current_date_str
            if not mask.any():
                return False

            # 取最近的3份报告
            recent = df_sorted[mask].tail(3)
            if len(recent) < 3:
                return False

            pg_values = []
            rg_values = []
            for _, row in recent.iterrows():
                pg = row.get('净利润-同比增长')
                rg = row.get('营业总收入-同比增长')
                if pg is not None:
                    try:
                        pg_values.append(float(str(pg).replace('%', '')))
                    except (ValueError, TypeError):
                        pass
                if rg is not None:
                    try:
                        rg_values.append(float(str(rg).replace('%', '')))
                    except (ValueError, TypeError):
                        pass

            if len(pg_values) >= 3:
                # 三个季度利润同比均为负 → 持续下滑
                if all(v < 0 for v in pg_values):
                    # 检查趋势: 降幅扩大(恶化) or 收窄(拐点)
                    # pg_values[0]=最早, pg_values[-1]=最新
                    trend = pg_values[-1] - pg_values[0]
                    if trend < -5:
                        # 降幅扩大超过5个百分点 → 加速恶化 → 过滤
                        return True
                    # 降幅收窄或持平 → 可能是拐点 → 再看看营收
                    if trend > 5 and len(rg_values) >= 3 and all(v < 0 for v in rg_values):
                        # 利润改善但营收仍在下滑 → 利润改善不可持续 → 过滤
                        return True
                    # 利润+营收都在改善 → 拐点确认, 不过滤
                    return False
            return False
        except Exception:
            logger.debug(f"基本面过滤异常 code={code}", exc_info=True)
            return False

    def _get_raw_fundamental_value(self, code, current_date, factor_name: str) -> Optional[float]:
        """获取基本面因子原始值（使用预加载缓存）"""
        # 优先使用预加载缓存
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                row = entry.get('row')
                if row is not None:
                    try:
                        return self._extract_fund_value_from_row(row, factor_name)
                    except Exception:
                        pass

        # Fallback
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
            return None
        try:
            if factor_name == 'fund_score':
                return self.fundamental_data.get_fundamental_score(code, current_date)
            elif factor_name == 'fund_profit_growth':
                return self.fundamental_data.get_profit_growth(code, current_date)
            elif factor_name == 'fund_roe':
                return self.fundamental_data.get_roe(code, current_date)
            elif factor_name == 'fund_revenue_growth':
                return self.fundamental_data.get_revenue_growth(code, current_date)
            elif factor_name == 'fund_eps':
                return self.fundamental_data.get_eps(code, current_date)
            elif factor_name == 'fund_cf_to_profit':
                return self.fundamental_data.get_cf_to_profit(code, current_date)
            elif factor_name == 'fund_debt_ratio':
                return self.fundamental_data.get_debt_ratio(code, current_date)
            elif factor_name == 'fund_gross_margin':
                return self.fundamental_data.get_gross_margin(code, current_date)
            elif factor_name == 'fund_pg_improve':
                return self.fundamental_data.get_profit_growth_improve(code, current_date)
            elif factor_name == 'fund_rg_improve':
                return self.fundamental_data.get_revenue_growth_improve(code, current_date)
        except Exception:
            logger.debug(f"基本面因子值获取失败 code={code} factor={factor_name}", exc_info=True)
            pass
        return None

    @staticmethod
    def _extract_fund_value_from_row(row, factor_name: str) -> Optional[float]:
        """从缓存row提取基本面原始值"""
        col_map = {
            'fund_score': None,  # 单独计算
            'fund_profit_growth': '净利润-同比增长',
            'fund_roe': '净资产收益率',
            'fund_revenue_growth': '营业总收入-同比增长',
            'fund_eps': '每股收益',
            'fund_cf_to_profit': None,
            'fund_debt_ratio': 'zcfz_资产负债率',
            'fund_gross_margin': '销售毛利率',
            'fund_pg_improve': None,
            'fund_rg_improve': None,
        }
        col = col_map.get(factor_name)
        if col is None:
            return None
        val = row.get(col)
        if val is None:
            return None
        try:
            if isinstance(val, str):
                return float(val.strip('%')) / 100
            return float(val)
        except (ValueError, TypeError):
            return None

    def _calculate_default_factor(self, ind: dict, idx: int, regime: int, industry_category: str) -> tuple:
        """指数条件信号：指数涨=追涨，指数跌=抄底

        所有输出均tanh压缩到(-1,1)，与动态因子和行业因子的量纲一致。
        """
        mom_20 = self._safe_get(ind, 'mom_20', idx, 0)
        mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
        mom_5 = self._safe_get(ind, 'mom_5', idx, 0)
        vol_20 = self._safe_get(ind, 'volatility', idx, 0)
        rel_str = self._safe_get(ind, 'relative_strength', idx, 0)

        momentum = mom_20 * 0.5 + mom_10 * 0.3 + rel_str * 0.2
        reversal = -(mom_20 * 0.4 + mom_10 * 0.3 + mom_5 * 0.3)

        # 用市场状态决定方向：牛追涨，熊抄底
        # tanh压缩统一量纲，与动态/行业因子对齐
        if regime == 1:
            factor_value = np.tanh(momentum * 3)
            factor_name = 'MOM'
        elif regime == -1:
            factor_value = np.tanh(reversal * 3)
            factor_name = 'REV'
        else:
            # 中性：用风险调整动量
            vol = abs(vol_20) + 0.01
            sharpe_raw = momentum / vol
            factor_value = np.tanh(sharpe_raw)
            factor_name = 'SHARPE'

        risk_info = {'is_high_vol': False}
        return factor_name, factor_value, risk_info

    def _calculate_industry_factor_score(self, ind: dict, idx: int, industry: str,
                                           code=None, current_date=None, regime=0,
                                           dynamic_ic_weights=None) -> tuple:
        """计算行业特定因子得分

        支持按市场状态选择不同的因子组合，使用IC权重加权
        支持tech_fund_combo等复合因子
        dynamic_ic_weights: walk-forward IC权重 (reweight模式), {factor_name: ic_weight}
        """
        config = INDUSTRY_FACTOR_CONFIG.get(industry)
        if not config:
            return None

        # 根据市场状态选择因子和权重
        # regime: 1=bull, 0=neutral, -1=bear
        if regime == 1:
            factors = config.get('bull_factors', config.get('factors', []))
            weights = config.get('bull_weights', None)
        elif regime == -1:
            factors = config.get('bear_factors', config.get('factors', []))
            weights = config.get('bear_weights', None)
        else:
            factors = config.get('factors', [])
            weights = config.get('weights', None)

        if not factors:
            return None

        # 获取基本面压缩评分（用于tech_fund_combo等复合因子）
        compressed_fund_score = 0.0
        if code and current_date and hasattr(self, 'fundamental_data') and self.fundamental_data:
            raw_fund_score = self._get_raw_fundamental_value(code, current_date, 'fund_score')
            if raw_fund_score is not None and isinstance(raw_fund_score, (int, float)):
                compressed_fund_score = compress_fundamental_factor(raw_fund_score, 'fund_score')

        direction = config.get('direction', {}) if 'direction' in config else {}

        factor_scores = []
        valid_factors = []
        valid_weights = []

        for i, factor_name in enumerate(factors):
            factor_val = None

            if factor_name == 'tech_fund_combo':
                # tech_fund_combo 需要基本面数据，通过 compute_composite_factors 计算
                combo = compute_composite_factors(ind, idx, fund_score=compressed_fund_score)
                factor_val = combo.get('tech_fund_combo')
            elif factor_name.startswith('fund_'):
                # 基本面因子：使用统一压缩函数
                factor_val = self._get_fundamental_factor_value(code, current_date, factor_name)
            else:
                # 技术因子：从 ind 字典获取
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 直接使用因子值
            if factor_val is not None and not np.isnan(factor_val):
                factor_dir = direction.get(factor_name, 1)
                factor_scores.append(factor_val * factor_dir)
                valid_factors.append(factor_name)
                # 获取IC权重
                if weights and i < len(weights):
                    valid_weights.append(weights[i])

        if not factor_scores:
            return None

        # 使用IC权重加权平均
        if valid_weights and len(valid_weights) == len(factor_scores):
            # reweight模式: 混合静态权重和walk-forward IC权重
            if dynamic_ic_weights and self.factor_mode == 'reweight':
                blend = self.dynamic_factor_selector.reweight_blend
                blended_weights = []
                for i, (w_s, fn) in enumerate(zip(valid_weights, valid_factors)):
                    w_d = dynamic_ic_weights.get(fn, None)
                    if w_d is not None and w_d > 0:
                        # 混合: blend * static + (1-blend) * dynamic
                        blended_weights.append(blend * abs(w_s) + (1 - blend) * w_d)
                    else:
                        # 无动态IC数据时保留静态权重
                        blended_weights.append(abs(w_s))
                total_w = sum(blended_weights)
                if total_w > 0:
                    factor_value = sum(s * w for s, w in zip(factor_scores, blended_weights)) / total_w
                else:
                    factor_value = np.mean(factor_scores)
            else:
                # 标准模式: 使用标定产出的权重
                total_w = sum(abs(w) for w in valid_weights)
                if total_w > 0:
                    factor_value = sum(s * abs(w) for s, w in zip(factor_scores, valid_weights)) / total_w
                else:
                    factor_value = np.mean(factor_scores)
        else:
            # 无权重时等权平均
            factor_value = np.mean(factor_scores)

        # 添加市场状态标记到因子名称
        regime_suffix = {1: '_B', -1: '_E', 0: ''}.get(regime, '')
        return f'IND_{industry[:4]}{regime_suffix}', factor_value, {'is_high_vol': False, 'industry_factor': True, 'n_factors': len(factor_scores)}

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

    def _preload_stock_fundamentals(self, code, dates):
        """预加载股票基本面数据缓存，避免每根K线查询DataFrame

        核心优化：基本面数据按季度发布，对2000根K线只需计算~16次而非2000次。
        通过指针推进找到每根K线适用的最新财报，预计算score和industry。
        """
        self._fund_cache = {}  # date_str -> {'score': float, 'industry': str, ...}
        self._has_fund_data_cache = False

        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return

        fd = self.fundamental_data
        if code not in fd.stock_data:
            return

        df = fd.stock_data[code]
        if '数据可用日期' not in df.columns or len(df) == 0:
            return

        df = df.copy()
        df['数据可用日期_str'] = df['数据可用日期'].astype(str)
        df = df.sort_values('数据可用日期_str').reset_index(drop=True)

        # 对每个K线日期，用指针推进找到适用财报
        sorted_dates = sorted(dates)
        report_idx = -1
        n_reports = len(df)

        for d in sorted_dates:
            d_ts = pd.Timestamp(d)
            d_str = d_ts.strftime('%Y%m%d')

            # 推进指针：下一份财报已可用
            while report_idx + 1 < n_reports and df.iloc[report_idx + 1]['数据可用日期_str'] <= d_str:
                report_idx += 1

            if report_idx >= 0:
                row = df.iloc[report_idx]
                cache_key = str(d)[:10]
                if cache_key not in self._fund_cache:
                    score = self._compute_fund_score_from_row(row)
                    industry = row.get('所处行业', None)
                    # 存原始row数据供 _get_fundamental_factor_value 使用
                    self._fund_cache[cache_key] = {
                        'score': score,
                        'industry': industry,
                        'row': row,
                    }

        self._has_fund_data_cache = True

    @staticmethod
    @staticmethod
    def _compute_fund_score_from_row(row) -> float:
        """从基本面数据行计算评分 — 委托给 factor_calculator.compute_fundamental_score"""
        from .factor_calculator import compute_fundamental_score

        def _parse_pct(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    return float(val.strip('%')) / 100
                return float(val)
            except (ValueError, TypeError):
                return None

        return compute_fundamental_score(
            roe=_parse_pct(row.get('净资产收益率')),
            profit_growth=_parse_pct(row.get('净利润-同比增长')),
            revenue_growth=_parse_pct(row.get('营业总收入-同比增长')),
            eps=_parse_pct(row.get('每股收益')),
        )

    def _get_industry_category(self, code, current_date) -> str:
        """获取股票所属行业类型"""
        # 使用预加载缓存
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                industry = entry.get('industry')
                if industry:
                    return industry
            return 'default'

        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 'default'
        try:
            industry = self.fundamental_data.get_industry(code, current_date)
            if not industry:
                return 'default'
            return industry
        except Exception:
            return 'default'

    def _get_specific_industry(self, code, current_date) -> str:
        """获取具体行业名（使用INDUSTRY_KEYWORDS映射）"""
        has_ic = hasattr(self, 'industry_codes') and self.industry_codes

        # 首先尝试从 industry_codes 查找（使用预建反向索引，O(1)）
        if has_ic:
            if not hasattr(self, '_industry_code_reverse_map'):
                self._industry_code_reverse_map = {
                    c: ind_name for ind_name, codes in self.industry_codes.items() for c in codes
                }
            result = self._industry_code_reverse_map.get(code)
            if result:
                return result

        # 使用预加载缓存
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                raw_industry = entry.get('industry')
                if raw_industry:
                    cleaned = raw_industry.replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()
                    for config_key, keywords in INDUSTRY_KEYWORDS.items():
                        if raw_industry in keywords or cleaned in keywords:
                            if config_key in INDUSTRY_FACTOR_CONFIG:
                                return config_key
                        for kw in keywords:
                            if kw in raw_industry or kw in cleaned:
                                if config_key in INDUSTRY_FACTOR_CONFIG:
                                    return config_key
                    return raw_industry

        # Fallback: 直接查询 fundamental_data
        has_fd = hasattr(self, 'fundamental_data') and self.fundamental_data
        if has_fd:
            try:
                raw_industry = self.fundamental_data.get_industry(code, current_date)
                if not raw_industry:
                    return None
                cleaned_industry = raw_industry.replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()
                for config_key, keywords in INDUSTRY_KEYWORDS.items():
                    if raw_industry in keywords or cleaned_industry in keywords:
                        if config_key in INDUSTRY_FACTOR_CONFIG:
                            return config_key
                    for kw in keywords:
                        if kw in raw_industry or kw in cleaned_industry:
                            if config_key in INDUSTRY_FACTOR_CONFIG:
                                return config_key
            except Exception:
                pass
        return None

    def _get_fundamental_score(self, code, current_date) -> float:
        """获取基本面因子评分（使用预加载缓存）"""
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                return entry.get('score', 0.0)

        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 0.0

        from .factor_calculator import compute_fundamental_score
        return compute_fundamental_score(
            roe=self.fundamental_data.get_roe(code, current_date),
            profit_growth=self.fundamental_data.get_profit_growth(code, current_date),
            revenue_growth=self.fundamental_data.get_revenue_growth(code, current_date),
            eps=self.fundamental_data.get_eps(code, current_date),
        )

    # === 辅助函数 ===
    @staticmethod
    def _safe_get(ind: dict, key: str, idx: int, default: float = 0.0) -> float:
        """快速安全获取指标值（假设ind值为numpy数组，跳过昂贵类型检查）"""
        arr = ind.get(key)
        if arr is None:
            return default
        try:
            if idx >= len(arr):
                return default
            val = arr[idx]
            # NaN/Inf检查: val!=val 只在NaN时为True, 比np.isnan快
            if val != val or val == float('inf') or val == float('-inf'):
                return default
            return float(val)
        except (TypeError, IndexError):
            return default

    def _sma(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        result[window-1:] = np.convolve(arr, np.ones(window)/window, mode='valid')
        return result

    @staticmethod
    @njit
    def _ema(arr, span):
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

    def _rolling_mean(self, arr, window):
        """滚动均值 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        result = np.full(n, np.nan)
        result[window:] = sw.mean(axis=1)[:n-window]
        return result

    def _compute_max_dd(self, close, window):
        """计算滚动窗口内的最大回撤（负值，如-0.15表示-15%）

        向量化实现：对每个 i，计算 [i-window, i) 内的最大回撤
        max_dd = min over j in [i-window, i) of (price[j] - peak[j]) / peak[j]
        """
        n = len(close)
        result = np.full(n, np.nan)
        if n < window + 1:
            return result

        # 向量化：对每个起点，计算窗口内的回撤
        from numpy.lib.stride_tricks import sliding_window_view
        # 取 [window-1, n-1) 的窗口视图，每个窗口有 window 个元素
        sw = sliding_window_view(close, window)
        # sw.shape = (n - window + 1, window)
        # sw[i] = close[i : i+window]

        # 对每个窗口计算：当前价格/累计最大值 - 1
        peak = np.maximum.accumulate(sw, axis=1)
        dd = (sw - peak) / np.where(peak > 0, peak, 1)
        min_dd = dd.min(axis=1)  # shape = (n - window + 1,)

        result[window:] = min_dd[:n - window]
        return result

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
