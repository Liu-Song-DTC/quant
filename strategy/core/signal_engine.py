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
from collections import deque

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper

from .signal import Signal
from .signal_store import SignalStore
from .config_loader import load_config
from .industry_mapping import INDUSTRY_KEYWORDS, get_industry_category
from .factor_calculator import calculate_indicators as calc_indicators, compute_composite_factors, compress_fundamental_factor
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

        # 动态买入阈值：滚动缓冲区（最近1000个因子值）
        self._factor_value_buffer = deque(maxlen=2000)
        # 行业内因子值缓存：{industry: deque(maxlen=500)}
        self._industry_factor_cache = {}
        # 动态阈值基准百分位
        self._buy_threshold_pct = self.config.get('signal', {}).get('buy_threshold_pct', 0.60)

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

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        """生成信号"""
        dates = market_data["datetime"].values
        close = market_data['close'].values

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)

        # 预加载基本面缓存（避免每根K线查询DataFrame，提速100x+）
        self._preload_stock_fundamentals(code, dates)

        # 预分配百分位节流计数器
        self._pct_counter = 0
        self._cached_buy_threshold = self.buy_threshold
        self._cached_sell_threshold = self.sell_threshold

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

        return result

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

    def _get_chan_boost(self, ind: dict, idx: int) -> dict:
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
        if signal_level == 3 and confirmed_buy:
            mult *= 1.35  # 双级别确认买点 → 最强买入
            div_quality = max(div_quality, buy_strength)
            is_buy_boost = True
        elif signal_level == -3 and confirmed_sell:
            mult *= 0.55  # 双级别确认卖点 → 最强卖出
            div_quality = max(div_quality, sell_strength)
            is_sell_boost = True

        # 线段级别信号
        elif signal_level == 2 and confirmed_buy:
            mult *= 1.25
            div_quality = max(div_quality, buy_strength)
            is_buy_boost = True
        elif signal_level == -2 and confirmed_sell:
            mult *= 0.65
            div_quality = max(div_quality, sell_strength)
            is_sell_boost = True

        # 笔级别信号
        elif signal_level == 1 and confirmed_buy:
            mult *= 1.15
            div_quality = max(div_quality, buy_strength * 0.7)
            is_buy_boost = True
        elif signal_level == -1 and confirmed_sell:
            mult *= 0.75
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
                base_boost = 0.12 * bottom_fx_quality
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
                    boost = 0.14 * bottom_fx_quality
                    if is_volume_spike_3x:
                        boost += 0.10  # 量在价先显著增强
                    mult *= 1.0 + boost
                    is_buy_boost = True
                    div_quality = max(div_quality, bottom_fx_quality * 0.55)
                elif bottom_fx_quality > mild_threshold:
                    # 温和买倾向
                    mult *= 1.0 + 0.07 * bottom_fx_quality
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
            b2_boost = 0.18 * second_buy_conf  # 最高约 +17%
            if not is_sell_boost:
                mult *= 1.0 + b2_boost
                is_buy_boost = True
                div_quality = max(div_quality, second_buy_conf * 0.75)
            elif is_buy_boost:
                # 已有买信号 + B2确认 → 强力共振
                mult *= 1.0 + b2_boost * 0.7
                div_quality = max(div_quality, second_buy_conf * 0.75)

        # 回退到旧的三类买卖点 (无多级别确认时)
        if not is_buy_boost and not is_sell_boost:
            if buy_point == 1 and buy_conf > 0.3:
                mult *= 1.25
                div_quality = max(div_quality, buy_conf)
                is_buy_boost = True
            elif buy_point == 2 and buy_conf > 0.2:
                mult *= 1.15
                div_quality = max(div_quality, buy_conf * 0.8)
                is_buy_boost = True
            elif buy_point == 3 and buy_conf > 0.2:
                # B3=突破中枢后回调确认，缠论最可靠的趋势加速买点
                mult *= 1.30
                div_quality = max(div_quality, buy_conf * 0.90)
                is_buy_boost = True

            if sell_point == 1 and sell_conf > 0.3:
                mult *= 0.65
                div_quality = max(div_quality, sell_conf)
                is_sell_boost = True
            elif sell_point == 2 and sell_conf > 0.2:
                mult *= 0.70
                div_quality = max(div_quality, sell_conf * 0.8)
                is_sell_boost = True
            elif sell_point == 3 and sell_conf > 0.2:
                mult *= 0.75
                div_quality = max(div_quality, sell_conf * 0.85)
                is_sell_boost = True

        # === Layer 2: 统一Chan信号 (当买卖点不明显时使用) ===
        if buy_point == 0 and sell_point == 0:
            if chan_buy > 0.5:
                mult *= 1.0 + 0.15 * chan_buy
                is_buy_boost = True
                div_quality = max(div_quality, chan_buy * 0.6)
            if chan_sell > 0.5:
                mult *= 1.0 - 0.25 * chan_sell
                is_sell_boost = True
                div_quality = max(div_quality, chan_sell * 0.6)

        # === Layer 3: 背离检测 (兜底，但必须有趋势 — "没有趋势，没有背驰") ===
        if not is_buy_boost and not is_sell_boost:
            if bottom_div > top_div and bottom_div > self.chan_bottom_div_threshold:
                # 底背离仅在下跌趋势中有意义（盘整中的背离无效）
                if trend_type != 0:  # 有趋势才启用背离
                    mult *= 1.20
                    div_quality = bottom_div
                    is_buy_boost = True
            elif top_div > bottom_div and top_div > self.chan_top_div_threshold:
                # 顶背离仅在上涨趋势中有意义
                if trend_type != 0:
                    mult *= 0.70
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
                mult = 1.0 + (mult - 1.0) * 0.7  # 盘整时信号打7折

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

        return {
            'boost_multiplier': float(np.clip(mult, 0.5, 1.5)),
            'is_chan_buy_boost': is_buy_boost,
            'is_chan_sell_boost': is_sell_boost,
            'divergence_quality': div_quality,
        }

    def _generate_signal(self, ind: dict, idx: int, last_sig, current_date=None, code=None) -> Signal:
        """生成信号"""
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0, factor_value=0.0,
                factor_name='V41', risk_vol=0.03, risk_regime=0,
                risk_confidence=0.0, risk_extreme=False, adjusted_score=0.0,
                industry=self._get_specific_industry(code, current_date) if code else ''
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
                industry=self._get_specific_industry(code, current_date) if code else ''
            )

        factor_name, factor_value, risk_info, is_industry = factor_result

        # 基本面因子
        fundamental_score = 0.0
        has_fundamental = False
        if self.fundamental_enabled and code:
            fundamental_score = self._get_fundamental_score(code, current_date)
            has_fundamental = fundamental_score > 0

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
            base_score = base_score + fundamental_score * 0.15

        # 纯动量：直接使用因子值，不做二次加工
        score = base_score

        # === 缠论增强（czsc风格 7层融合）===
        chan_boost = self._get_chan_boost(ind, idx)
        chan_sl = int(self._safe_get(ind, 'signal_level', idx, 0))

        # === 缠论门控：强结构信号时缠论主导，弱信号时因子主导 ===
        # 门控公式保留 chan_boost['boost_multiplier'] 的7层调整 (趋势/中枢/中阴等)
        chan_mult = chan_boost['boost_multiplier']
        if chan_sl >= 2 and chan_boost.get('is_chan_buy_boost'):
            # 线段级/双级别买入 → 缠论主导，因子辅助，7层调整保留
            chan_score = 0.5 + 0.5 * chan_boost.get('divergence_quality', 0.5)
            score = chan_score * 1.5 * max(1.0, chan_mult) + base_score * 0.2
        elif chan_sl <= -2 and chan_boost.get('is_chan_sell_boost'):
            # 线段级/双级别卖出 → 缠论主导，7层调整保留 (sell_mult<1转为负分强度)
            chan_score = 0.5 + 0.5 * chan_boost.get('divergence_quality', 0.5)
            sell_intensity = max(0.3, (1.0 - chan_mult) * 3.0)
            score = -chan_score * sell_intensity + base_score * 0.2
        else:
            # 无强结构信号或仅笔级 → 因子主导，缠论乘法增强（原行为）
            score = score * chan_mult

        smart_money = self._safe_get(ind, 'smart_money_flow', idx, 0)
        bottom_div = self._safe_get(ind, 'bottom_divergence', idx, 0.0)
        top_div = self._safe_get(ind, 'top_divergence', idx, 0.0)

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

        # 5. 市场状态调整（portfolio层管理敞口，信号层仅温和折扣）
        if risk_regime == -1:
            score = score * 0.85  # 熊市温和降分（给portfolio层留空间）
        elif risk_regime == 0:
            score = score * 0.95  # 中性市场轻微折扣
        # 牛市不调整

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
        sys2_buy = cf_score > 0.5 and cf_dir >= 0    # 资金流入
        sys3_buy = ns_score > 0.3 and ns_dir >= 0    # 利好冲击
        sys1_sell = score < self.sell_threshold
        sys2_sell = cf_score > 0.5 and cf_dir == -1  # 资金流出
        sys3_sell = ns_score > 0.3 and ns_dir == -1  # 利空冲击

        n_buy_systems = sum([sys1_buy, sys2_buy, sys3_buy])
        n_sell_systems = sum([sys1_sell, sys2_sell, sys3_sell])

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
        # 更新score滚动缓冲区（替代factor_value buffer）
        if score is not None and not np.isnan(score):
            self._factor_value_buffer.append(float(score))

        # 动态阈值：每20根K线重算一次（百分位计算很贵，阈值变化缓慢）
        self._pct_counter += 1
        if len(self._factor_value_buffer) > 100 and self._pct_counter % 20 == 0:
            buf = list(self._factor_value_buffer)
            pct = self._buy_threshold_pct * 100
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

        buy = (score is not None and
               not np.isnan(score) and
               score > effective_buy_threshold and
               abs(score) < 5.0)

        # 缠论止盈: 强卖点信号直接触发sell=True，不依赖分数阈值
        chan_sell_signal = chan_boost.get('is_chan_sell_boost', False)
        sl_for_sell = int(self._safe_get(ind, 'signal_level', idx, 0))
        sp_for_sell = int(self._safe_get(ind, 'sell_point', idx, 0))
        sell = (score is not None and
                not np.isnan(score) and
                (score < effective_sell_threshold or
                 (chan_sell_signal and (sl_for_sell <= -2 or sp_for_sell >= 1))))

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

        # 默认因子组合（固定因子的兜底）
        # 注意：只有当允许使用固定因子时才执行DEFAULT
        # factor_mode='dynamic'且fallback=False时，应该已经返回None了
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
        DYN_QUALITY_THRESHOLD = 0.01
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

        # 首先尝试从 industry_codes 查找（更可靠，O(1)）
        if has_ic:
            for ind_name, codes in self.industry_codes.items():
                if code in codes:
                    return ind_name

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

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
