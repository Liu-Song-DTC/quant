# core/signal_engine.py
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .signal import Signal
from .signal_store import SignalStore
from .factor_library import calc_factor_volatility_10, calc_factor_rsi_8, calc_factor_bb_width_20
from .config_loader import load_config

import warnings
warnings.filterwarnings('ignore')

# 加载配置
_config = load_config()

# 从配置文件加载行业配置
INDUSTRY_CATEGORY = _config.get_industry_category()
INDUSTRY_FACTOR_WEIGHTS = _config.get_industry_factor_weights()

# 行业最优因子配置（基于滚动验证结果）
# 结构: {具体行业名: {factors: [因子列表], direction: {因子名: 方向}}}
# direction: 1=正向(因子值高买), -1=反向(因子值低买)
INDUSTRY_FACTORS = {
    '银行Ⅱ': {
        'factors': ['trend_30', 'price_ma_30', 'ret_30', 'rsi_14', 'vol_10'],
        'direction': {'trend_30': -1, 'price_ma_30': -1, 'ret_30': -1, 'rsi_14': -1, 'vol_10': -1},
    },
    '贵金属': {
        'factors': ['ma_cross_10_60', 'ret_30', 'mom_30', 'price_ma_60', 'vol_20'],
        'direction': {'ma_cross_10_60': 1, 'ret_30': 1, 'mom_30': 1, 'price_ma_60': 1, 'vol_20': 1},
    },
    '消费电子': {
        'factors': ['bb_pos_30', 'price_ma_60', 'price_pos_20', 'rsi_20', 'vol_10'],
        'direction': {'bb_pos_30': 1, 'price_ma_60': -1, 'price_pos_20': -1, 'rsi_20': 1, 'vol_10': -1},
    },
    '电池': {
        'factors': ['ma_cross_10_60', 'price_ma_60', 'price_ma_120', 'vol_20', 'atr_ratio_20'],
        'direction': {'ma_cross_10_60': -1, 'price_ma_60': -1, 'price_ma_120': -1, 'vol_20': -1, 'atr_ratio_20': -1},
    },
    '工业金属': {
        'factors': ['price_ma_60', 'ma_cross_10_60', 'ma_golden_20_60', 'rsi_14', 'bb_width_20'],
        'direction': {'price_ma_60': -1, 'ma_cross_10_60': -1, 'ma_golden_20_60': -1, 'rsi_14': -1, 'bb_width_20': -1},
    },
    '基础建设': {
        'factors': ['rsi_14', 'rsi_20', 'bb_width_20', 'vol_20', 'atr_20'],
        'direction': {'rsi_14': -1, 'rsi_20': -1, 'bb_width_20': -1, 'vol_20': -1, 'atr_20': -1},
    },
    '电网设备': {
        'factors': ['price_ma_60', 'ma_all', 'vol_30', 'atr_10', 'rsi_20'],
        'direction': {'price_ma_60': 1, 'ma_all': 1, 'vol_30': 1, 'atr_10': 1, 'rsi_20': 1},
    },
    '光伏设备': {
        'factors': ['price_ma_60', 'price_ma_120', 'ma_cross_10_60', 'vol_10', 'bb_width_30'],
        'direction': {'price_ma_60': 1, 'price_ma_120': 1, 'ma_cross_10_60': 1, 'vol_10': 1, 'bb_width_30': 1},
    },
    '通信设备': {
        'factors': ['bb_width_30', 'vol_20', 'vol_30', 'rsi_20', 'ma_cross_10_20'],
        'direction': {'bb_width_30': 1, 'vol_20': 1, 'vol_30': 1, 'rsi_20': 1, 'ma_cross_10_20': 1},
    },
    '电力': {
        'factors': ['trend_30', 'price_ma_30', 'rsi_14', 'rsi_20', 'bb_pos_30'],
        'direction': {'trend_30': -1, 'price_ma_30': -1, 'rsi_14': -1, 'rsi_20': -1, 'bb_pos_30': -1},
    },
    '元件': {
        'factors': ['price_ma_60', 'ma_cross_10_60', 'price_ma_30', 'rsi_20', 'bb_pos_30'],
        'direction': {'price_ma_60': -1, 'ma_cross_10_60': -1, 'price_ma_30': 1, 'rsi_20': 1, 'bb_pos_30': 1},
    },
    '半导体': {
        'factors': ['ret_30', 'mom_30', 'price_ma_60', 'vol_10', 'rsi_14'],
        'direction': {'ret_30': 1, 'mom_30': 1, 'price_ma_60': 1, 'vol_10': 1, 'rsi_14': 1},
    },
}


class SignalEngine:
    """信号生成引擎 - 使用高质量因子库"""

    def __init__(self, config=None):
        self.config = config or {}
        self.min_score = 0.35

        # 加载因子配置
        self._load_config()

        # 市场状态信息
        self.market_regime_data = None
        self.current_idx = 0

    def _load_config(self):
        """从配置文件加载参数（使用默认值，与原有逻辑一致）"""
        try:
            from .config_loader import load_config
            config_loader = load_config()
            tech_weights = config_loader.get_technical_weights()
            thresholds = config_loader.get_signal_thresholds()

            # 技术面因子权重
            self.vol_weight = tech_weights.get('volatility_10', 0.30)
            self.rsi_weight = tech_weights.get('rsi_average', 0.25)
            self.bb_weight = tech_weights.get('bb_width', 0.15)
            self.mom_weight = tech_weights.get('momentum', 0.30)

            # 信号阈值
            self.buy_threshold = thresholds.get('buy', 0.15)
            self.adjusted_buy_threshold = thresholds.get('adjusted_buy', 0.05)
            self.sell_threshold = thresholds.get('sell', -0.05)

            # 极端状态权重
            self.extreme_weight = config_loader.get('extreme_adjustments.regime_weight', 0.85)

            # 基本面因子配置
            fundamental_config = config_loader.get('fundamental_weights', {})
            self.fundamental_enabled = fundamental_config.get('enabled', True)
            self.fundamental_weight = fundamental_config.get('weight', 0.30)
            self.technical_weight = fundamental_config.get('technical_weight', 0.70)

            # 市场状态动态权重
            self.regime_weights = config_loader.get('regime_weights', {})

            # 风格因子配置 (新增)
            style_config = config_loader.get('style_weights', {})
            self.style_enabled = style_config.get('enabled', True)
            self.style_weight = style_config.get('weight', 0.25)

        except Exception:
            # 使用默认值（平衡版）
            self.vol_weight = 0.20
            self.rsi_weight = 0.30  # 提高RSI权重
            self.bb_weight = 0.20
            self.mom_weight = 0.30
            self.buy_threshold = 0.15
            self.adjusted_buy_threshold = 0.05
            self.sell_threshold = -0.05
            self.extreme_weight = 0.85
            # 基本面默认值
            self.fundamental_enabled = True
            self.fundamental_weight = 0.30
            self.technical_weight = 0.70
            # 市场状态权重默认值（平衡版）
            self.regime_weights = {
                'bull': {'volatility_10': 0.15, 'rsi_average': 0.30, 'bb_width': 0.15, 'momentum': 0.40},
                'bear': {'volatility_10': 0.40, 'rsi_average': 0.30, 'bb_width': 0.10, 'momentum': 0.20},
                'neutral': {'volatility_10': 0.20, 'rsi_average': 0.30, 'bb_width': 0.20, 'momentum': 0.30},
            }
            # 风格因子默认值
            self.style_enabled = True
            self.style_weight = 0.25

    def set_fundamental_data(self, fundamental_data):
        """设置基本面数据"""
        self.fundamental_data = fundamental_data

    def set_market_regime(self, regime_df: pd.DataFrame):
        """
        设置市场状态数据
        DataFrame应包含列: datetime, regime, confidence, momentum_score, trend_score, volatility, is_extreme
        """
        if 'datetime' in regime_df.columns:
            regime_df = regime_df.copy()
            regime_df['datetime'] = pd.to_datetime(regime_df['datetime'])
            self.market_regime_data = regime_df.set_index('datetime')

    def _get_market_info(self, date) -> Dict[str, Any]:
        """获取指定日期的市场状态信息"""
        if self.market_regime_data is None:
            return {
                'regime': 0,
                'confidence': 0.0,
                'momentum_score': 0.0,
                'trend_score': 0.0,
                'volatility': 0.15,
                'is_extreme': False,
                'style_regime': 'balanced',
                'style_score': 0.0,
                'size_score': 0.0,
                'style_confidence': 0.0,
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
            'regime': 0,
            'confidence': 0.0,
            'momentum_score': 0.0,
            'trend_score': 0.0,
            'volatility': 0.15,
            'is_extreme': False,
            'style_regime': 'balanced',
            'style_score': 0.0,
            'size_score': 0.0,
            'style_confidence': 0.0,
        }

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        dates = market_data["datetime"].values
        close = market_data['close']

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)

        last_sig = None
        for idx in close.index:
            sig = self._generate_signal(indicators, idx, last_sig, dates[idx], code)
            last_sig = sig
            date = pd.to_datetime(dates[idx]).date()
            signal_store.set(code, date, sig)

    def _calculate_indicators(self, data: pd.DataFrame):
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        result = {'close': close, 'high': high, 'low': low, 'volume': volume}

        # EMA
        for span in [5, 10, 20, 60]:
            result[f'ema{span}'] = self._ema(close, span)

        # MA
        for span in [5, 10, 20, 30, 60]:
            result[f'ma{span}'] = self._sma(close, span)

        # RSI - 更多周期
        result['rsi'] = self._rsi(close, 14)
        result['rsi_14'] = self._rsi(close, 14)
        result['rsi_8'] = self._rsi(close, 8)
        result['rsi_6'] = self._rsi(close, 6)
        result['rsi_10'] = self._rsi(close, 10)

        # 布林带
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = self._bollinger(close, 20, 2)
        # 布林带宽度
        bb_std = np.zeros_like(close)
        bb_std[20:] = np.array([np.std(close[max(0,i-20):i]) for i in range(20, len(close))])
        bb_middle = result['bb_middle']
        result['bb_width_20'] = 4 * bb_std / (bb_middle + 1e-10)

        # 成交量
        result['volume_ma20'] = self._sma(volume, 20)
        result['volume_ratio'] = volume / (result['volume_ma20'] + 1e-10)

        # ATR
        result['atr'] = self._atr(high, low, close, 14)
        result['atr_ratio'] = result['atr'] / close

        # 动量 - 更多周期
        for period in [3, 5, 10, 20, 30]:
            result[f'mom_{period}'] = close / self._shift(close, period) - 1

        # 价格位置
        result['high_20'] = self._rolling_max(high, 20)
        result['low_20'] = self._rolling_min(low, 20)
        result['price_position_20'] = (close - result['low_20']) / (result['high_20'] - result['low_20'] + 1e-10)

        # 波动率
        returns = np.diff(close, prepend=close[0]) / close
        result['ret'] = returns
        result['volatility_10'] = self._rolling_std(returns, 10)
        result['volatility_5'] = self._rolling_std(returns, 5)
        result['volatility_20'] = self._rolling_std(returns, 20)

        # 均线关系
        result['ema5_above_20'] = result['ema5'] > result['ema20']
        result['ema20_above_60'] = result['ema20'] > result['ema60']
        result['full_golden'] = (result['ema5'] > result['ema20']) & (result['ema20'] > result['ema60'])
        result['full_death'] = (result['ema5'] < result['ema20']) & (result['ema20'] < result['ema60'])

        # 趋势强度
        result['trend_strength'] = (result['ema20'] - result['ema60']) / (result['ema60'] + 1e-10)

        # 斜率
        result['ema20_slope'] = result['ema20'] / self._shift(result['ema20'], 10) - 1

        # MACD
        result['macd'], result['macd_signal'], result['macd_hist'] = self._macd(close)

        # ==== 行业因子补充计算 ====
        # RSI 20周期
        result['rsi_20'] = self._rsi(close, 20)

        # 布林带位置和宽度 (30周期)
        bb_upper_30, bb_middle_30, bb_lower_30 = self._bollinger(close, 30, 2)
        bb_std_30 = np.zeros_like(close)
        bb_std_30[30:] = np.array([np.std(close[max(0,i-30):i]) for i in range(30, len(close))])
        result['bb_width_30'] = 4 * bb_std_30 / (bb_middle_30 + 1e-10)
        result['bb_pos_30'] = (close - bb_lower_30) / (bb_upper_30 - bb_lower_30 + 1e-10)

        # 价格相对均线 (30, 60, 120周期)
        result['price_ma_30'] = close / (result['ma30'] + 1e-10) - 1
        result['price_ma_60'] = close / (result['ma60'] + 1e-10) - 1
        # 计算ma120
        ma120 = self._sma(close, 120)
        result['price_ma_120'] = close / (ma120 + 1e-10) - 1

        # 均线交叉
        ma10 = result['ma10']
        ma20 = result['ma20']
        ma60 = result['ma60']
        result['ma_cross_10_60'] = ma10 / (ma60 + 1e-10) - 1
        result['ma_cross_10_20'] = ma10 / (ma20 + 1e-10) - 1
        result['ma_golden_20_60'] = (ma20 > ma60).astype(float)
        result['ma_all'] = ((result['ma5'] > ma20) & (ma20 > ma60)).astype(float)

        # 趋势因子
        ma30 = result['ma30']
        result['trend_30'] = (close - ma30) / (ma30 + 1e-10)

        # 价格位置 (20周期)
        result['price_pos_20'] = (close - result['low_20']) / (result['high_20'] - result['low_20'] + 1e-10)

        # 成交量相关
        result['vol_20'] = self._rolling_std(returns, 20)
        result['vol_30'] = self._rolling_std(returns, 30)

        # ATR相关
        result['atr_10'] = self._atr(high, low, close, 10)
        result['atr_20'] = self._atr(high, low, close, 20)
        result['atr_ratio_20'] = result['atr_20'] / close

        # 收益率
        result['ret_30'] = close / self._shift(close, 30) - 1

        return result

    def _rolling_std(self, arr, window):
        """计算滚动标准差"""
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window, len(arr)):
            result[i] = np.std(arr[i-window:i])
        return result

    def _generate_signal(self, ind, idx, last_sig, current_date=None, code=None):
        """
        信号生成 - 因子组合版（技术面 + 基本面 + 风格因子）
        不依赖市场状态判断，依赖因子本身的自适应性
        """
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0,
                factor_value=0.0, factor_name='V41',
                risk_vol=0.03, risk_regime=0, risk_confidence=0.0,
                risk_extreme=False, adjusted_score=0.0
            )

        # 获取市场状态（仅用于记录，不影响因子计算）
        market_info = self._get_market_info(current_date)
        risk_regime = market_info['regime']
        risk_confidence = market_info['confidence']
        risk_extreme = market_info['is_extreme']
        style_regime = market_info.get('style_regime', 'balanced')
        style_score = market_info.get('style_score', 0.0)
        style_confidence = market_info.get('style_confidence', 0.0)

        # === 获取行业类型用于因子权重调整 ===
        industry_category = self._get_industry_category(code, current_date)

        # === 使用纯因子组合，不依赖市场状态 ===
        factor_name, factor_value, risk_info = self._select_factor(
            ind, idx, risk_regime, industry_category, code=code, current_date=current_date
        )

        # === 获取基本面因子 ===
        fundamental_score = 0.0
        has_fundamental = False
        if self.fundamental_enabled and code:
            fundamental_score = self._get_fundamental_score(code, current_date)
            has_fundamental = fundamental_score > 0

        # === 获取风格因子 (新增) ===
        style_factor_score = self._get_style_score(ind, idx, market_info)

        # === 技术面分数 (已经是 0+ 的值) ===
        tech_score = max(0, factor_value)

        # === 组合分数：技术面分数 + 基本面加分 + 风格因子 ===
        # 基本面因子作为加成：当基本面良好时，在技术面分数基础上加分
        # 风格因子根据市场风格调整选股偏好
        if fundamental_score > 0:
            # 基本面分数归一化 (0-1范围) 作为加成系数
            # 例如: tech_score=0.2, fundamental_score=0.5 => combined = 0.2 * (1 + 0.5*0.3) = 0.23
            bonus = 1.0 + fundamental_score * self.fundamental_weight
            combined_score = tech_score * bonus
        else:
            combined_score = tech_score

        # 加上风格因子调整
        if style_confidence > 0.3 and abs(style_factor_score) > 0.05:
            # 风格因子权重
            style_weight = getattr(self, 'style_weight', 0.25)
            # 风格调整: 直接加成而不是乘法，这样更明显
            combined_score = combined_score + style_factor_score * style_weight

        score = max(0, combined_score)
        risk_vol = ind['atr_ratio'][idx] * 2

        # === 极端波动时降低仓位 ===
        regime_weight = 1.0
        if risk_extreme:
            regime_weight = 0.85

        adjusted_score = score * regime_weight

        # === 交易信号 ===
        buy = factor_value > 0.15 and adjusted_score > 0.05
        sell = factor_value <= -0.05

        # 添加基本面和风格标记到因子名称
        factor_tags = []
        if has_fundamental:
            factor_tags.append('F')
        if style_confidence > 0.3:
            factor_tags.append(style_regime[:2].upper())
        if factor_tags:
            factor_name = factor_name + '_' + ''.join(factor_tags)
        else:
            factor_name = factor_name + '_T'

        return Signal(
            buy=buy,
            sell=sell,
            score=score,
            factor_value=factor_value,
            factor_name=factor_name,
            risk_vol=risk_vol,
            risk_regime=risk_regime,
            risk_confidence=risk_confidence,
            risk_extreme=risk_extreme or risk_info.get('is_high_vol', False),
            adjusted_score=adjusted_score
        )

    def _get_style_score(self, ind, idx, market_info) -> float:
        """
        获取风格因子分数

        根据市场风格状态调整:
        - 小盘风格: 偏好小市值、高成长
        - 大盘风格: 偏好大市值、稳定收益
        - 成长风格: 偏好高动量、高增长
        - 价值风格: 偏好低估值、高股息
        """
        style_regime = market_info.get('style_regime', 'balanced')
        style_score = market_info.get('style_score', 0.0)
        style_confidence = market_info.get('style_confidence', 0.0)

        if style_confidence < 0.3 or style_regime == 'balanced':
            return 0.0

        # 根据风格状态返回调整分数
        if style_regime == 'small_cap':
            # 小盘风格: 偏好小市值股票 (使用价格位置作为代理)
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            # 价格位置偏低说明可能是小盘股（便宜）
            return -price_pos * 0.5 + 0.25  # 调整为 -0.25 到 0.25

        elif style_regime == 'large_cap':
            # 大盘风格: 偏好稳定大盘股
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return price_pos * 0.5 - 0.25  # 调整为 -0.25 到 0.25

        elif style_regime == 'growth':
            # 成长风格: 偏好高动量
            mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
            return np.clip(mom_10 * 2, -0.3, 0.3)

        elif style_regime == 'value':
            # 价值风格: 偏好低波动、稳定收益
            vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
            return np.clip((0.02 - vol_10) * 5, -0.3, 0.3)

        return 0.0

    def _select_factor(self, ind, idx, regime: int, industry_category: str = 'default', code=None, current_date=None):
        """
        根据市场状态和行业选择因子组合

        优先级:
        1. 如果行业有特定因子配置（INDUSTRY_FACTORS），使用行业特定因子
        2. 否则使用默认的4大类因子组合

        使用高质量因子库:
        - volatility_10: IC=4.02%, IR=1.07 (A股最佳)
        - volatility_5: IC=3.21%
        - rsi_8: IC=2.37%
        - bb_width_20: IC=2.33%

        根据市场状态动态调整权重:
        - 牛市: 提高动量权重
        - 熊市: 提高波动率/防御权重
        - 震荡: 均衡配置

        根据行业调整权重:
        - 科技/成长: 提高RSI和动量权重
        - 周期/资源: 提高波动率权重
        - 消费/稳定: 提高布林带权重
        - 金融/大盘: 提高动量和波动率权重
        """
        # === 优先使用行业特定因子 ===
        if code and current_date:
            specific_industry = self._get_specific_industry(code, current_date)
            if specific_industry:
                result = self._calculate_industry_factor_score(ind, idx, specific_industry)
                if result:
                    factor_name, factor_value, risk_info = result

                    # 熊市: 降低权重
                    if regime == -1:
                        factor_value = factor_value * 0.7
                        factor_name = factor_name + '_bear'

                    # 添加行业标记
                    factor_name = factor_name + f'_{specific_industry[:2]}'
                    return factor_name, factor_value, risk_info

        # === 默认因子组合逻辑 ===
        # 获取各因子值
        vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
        vol_5 = self._safe_get(ind, 'volatility_5', idx, 0.02)
        vol_20 = self._safe_get(ind, 'volatility_20', idx, 0.02)
        rsi = self._safe_get(ind, 'rsi_14', idx, 50)
        rsi_8 = self._safe_get(ind, 'rsi_8', idx, 50)
        rsi_10 = self._safe_get(ind, 'rsi_10', idx, 50)
        rsi_6 = self._safe_get(ind, 'rsi_6', idx, 50)
        bb_width = self._safe_get(ind, 'bb_width_20', idx, 0.05)
        mom_20 = self._safe_get(ind, 'mom_20', idx, 0)
        mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
        mom_3 = self._safe_get(ind, 'mom_3', idx, 0)
        price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)

        # 计算各因子
        # 波动率因子 - 保持中性，略微偏向低波动
        # 原: vol_factor = vol_10 * 10 (高波动=高分)
        # 修改为: 保持原有方向，但降低权重
        vol_factor = vol_10 * 10

        # RSI因子 (-1 到 1)
        rsi_val = (rsi_8 - 50) / 50

        # 多周期RSI平均
        rsi_avg = (rsi_6 + rsi_8 + rsi_10) / 3
        rsi_avg_val = (rsi_avg - 50) / 50

        # 布林带因子
        bb_val = bb_width

        # 动量因子
        mom_val = mom_10 * 2

        # 风险信息
        risk_info = {
            'is_high_vol': vol_20 > 0.05,  # 5%以上才算高波动
            'vol_factor': vol_factor,
        }

        # === 根据市场状态动态调整权重 ===
        # 默认权重
        vol_w = self.vol_weight
        rsi_w = self.rsi_weight
        bb_w = self.bb_weight
        mom_w = self.mom_weight

        # 从配置中获取动态权重
        if self.regime_weights:
            if regime == 1:  # 牛市
                regime_cfg = self.regime_weights.get('bull', {})
                vol_w = regime_cfg.get('volatility_10', vol_w)
                rsi_w = regime_cfg.get('rsi_average', rsi_w)
                bb_w = regime_cfg.get('bb_width', bb_w)
                mom_w = regime_cfg.get('momentum', mom_w)
            elif regime == -1:  # 熊市
                regime_cfg = self.regime_weights.get('bear', {})
                vol_w = regime_cfg.get('volatility_10', vol_w)
                rsi_w = regime_cfg.get('rsi_average', rsi_w)
                bb_w = regime_cfg.get('bb_width', bb_w)
                mom_w = regime_cfg.get('momentum', mom_w)
            else:  # 震荡市
                regime_cfg = self.regime_weights.get('neutral', {})
                vol_w = regime_cfg.get('volatility_10', vol_w)
                rsi_w = regime_cfg.get('rsi_average', rsi_w)
                bb_w = regime_cfg.get('bb_width', bb_w)
                mom_w = regime_cfg.get('momentum', mom_w)

        # === 根据行业类型调整权重 ===
        if industry_category != 'default':
            industry_weights = INDUSTRY_FACTOR_WEIGHTS.get(industry_category, INDUSTRY_FACTOR_WEIGHTS['default'])
            # 行业权重与市场状态权重混合 (50%行业 + 50%市场状态)
            vol_w = vol_w * 0.5 + industry_weights.get('volatility_10', vol_w) * 0.5
            rsi_w = rsi_w * 0.5 + industry_weights.get('rsi_average', rsi_w) * 0.5
            bb_w = bb_w * 0.5 + industry_weights.get('bb_width', bb_w) * 0.5
            mom_w = mom_w * 0.5 + industry_weights.get('momentum', mom_w) * 0.5

        # === 核心组合: 使用动态权重 ===
        factor_value = (vol_factor * vol_w +
                       rsi_avg_val * rsi_w +
                       bb_val * bb_w +
                       mom_val * mom_w)
        factor_name = f'V{int(vol_w*100)}_RSIavg{int(rsi_w*100)}_BB{int(bb_w*100)}_Mom{int(mom_w*100)}'

        # === 熊市: 降低权重 ===
        if regime == -1:
            factor_value = factor_value * 0.7
            factor_name = factor_name + '_bear'

        # === 添加行业标记 ===
        if industry_category != 'default':
            # 简写行业名称
            industry_short = {'科技/成长': 'TC', '周期/资源': 'CY', '消费/稳定': 'XF', '金融/大盘': 'JR'}
            ind_tag = industry_short.get(industry_category, industry_category[:2])
            factor_name = factor_name + f'_{ind_tag}'

        return factor_name, factor_value, risk_info

    def _safe_get(self, ind: dict, key: str, idx: int, default: float = 0.0) -> float:
        """安全获取数组元素"""
        arr = ind.get(key)
        if arr is None:
            return default
        if isinstance(arr, (int, float)):
            return default
        if hasattr(arr, '__len__'):
            if len(arr) <= idx:
                return default
            val = arr[idx]
            if isinstance(val, (int, float)) and not np.isnan(val):
                return val
        return default

    def _get_industry_category(self, code, current_date) -> str:
        """获取股票所属行业类型

        Returns:
            行业类型字符串: '科技/成长', '周期/资源', '消费/稳定', '金融/大盘', 'default'
        """
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 'default'

        try:
            industry = self.fundamental_data.get_industry(code, current_date)
            if not industry:
                return 'default'

            # 查找行业类型
            for category, industries in INDUSTRY_CATEGORY.items():
                if any(ind in str(industry) for ind in industries):
                    return category
        except:
            pass
        return 'default'

    def _get_specific_industry(self, code, current_date) -> str:
        """获取股票所属具体行业名

        Returns:
            具体行业名（如'银行Ⅱ'、'半导体'）或None
        """
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return None

        try:
            industry = self.fundamental_data.get_industry(code, current_date)
            if industry and industry in INDUSTRY_FACTORS:
                return industry
        except:
            pass
        return None

    def _calculate_industry_factor_score(self, ind, idx, industry: str) -> tuple:
        """计算行业特定因子的得分

        Args:
            ind: 因子数据字典
            idx: 当前索引
            industry: 具体行业名

        Returns:
            (factor_name, factor_value, risk_info)
        """
        if industry not in INDUSTRY_FACTORS:
            return None

        config = INDUSTRY_FACTORS[industry]
        factors = config.get('factors', [])
        direction = config.get('direction', {})

        factor_scores = []
        factor_names = []

        for factor_name in factors:
            # 获取因子值
            factor_val = self._safe_get(ind, factor_name, idx, 0.0)

            # 标准化因子值到 -1 到 1 范围
            if factor_name.startswith('rsi_'):
                # RSI: 50为中心，0-100范围
                normalized = (factor_val - 50) / 50
            elif factor_name.startswith('vol_') or factor_name.startswith('volatility_'):
                # 波动率: 0.02为中性
                normalized = (factor_val - 0.02) / 0.02
            elif factor_name.startswith('bb_pos_'):
                # 布林带位置: 0.5为中性
                normalized = (factor_val - 0.5) * 2
            elif factor_name.startswith('bb_width_'):
                # 布林带宽度: 0.05为中性
                normalized = (factor_val - 0.05) / 0.05
            elif factor_name.startswith('price_pos_') or factor_name.startswith('price_position_'):
                # 价格位置: 0.5为中性
                normalized = (factor_val - 0.5) * 2
            elif factor_name.startswith('price_ma_'):
                # 价格相对均线: 0为中性
                normalized = factor_val * 5  # 放大
            elif factor_name.startswith('ma_cross_'):
                # 均线交叉: 0为中性
                normalized = factor_val * 3
            elif factor_name.startswith('ma_golden_') or factor_name == 'ma_all':
                # 黄金交叉信号: 0或1
                normalized = factor_val * 2 - 1
            elif factor_name.startswith('trend_'):
                # 趋势: 0为中性
                normalized = factor_val * 3
            elif factor_name.startswith('mom_') or factor_name.startswith('ret_'):
                # 动量/收益率: 0为中性
                normalized = factor_val * 3
            elif factor_name.startswith('atr_'):
                # ATR: 归一化处理
                normalized = np.clip(factor_val / (ind['close'][idx] + 1e-10) * 10, -2, 2)
            elif factor_name == 'vol_20' or factor_name == 'vol_30':
                normalized = (factor_val - 0.02) / 0.02
            else:
                normalized = factor_val

            # 应用方向
            factor_dir = direction.get(factor_name, 1)
            adjusted_score = normalized * factor_dir

            factor_scores.append(adjusted_score)
            factor_names.append(factor_name)

        # 计算综合得分
        if factor_scores:
            # 使用平均得分，乘以1.5放大信号
            factor_value = np.mean(factor_scores) * 1.5
            # 限制范围
            factor_value = np.clip(factor_value, -1, 1)
        else:
            factor_value = 0

        risk_info = {'is_high_vol': False, 'industry_factor': True}

        return f'IND_{industry[:4]}', factor_value, risk_info

    def _get_fundamental_score(self, code, current_date):
        """获取基本面因子评分

        数据已经是小数形式（如ROE=0.12表示12%）
        """
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 0.0

        score = 0.0

        # ROE评分 - 最重要的因子（小数形式，如0.15表示15%）
        roe = self.fundamental_data.get_roe(code, current_date)
        if roe is not None:
            if roe > 0.15:  # > 15%
                score += 0.35
            elif roe > 0.10:  # > 10%
                score += 0.25
            elif roe > 0.05:  # > 5%
                score += 0.15

        # 净利润增长（小数形式）
        profit_growth = self.fundamental_data.get_profit_growth(code, current_date)
        if profit_growth is not None:
            if profit_growth > 0.50:  # > 50%
                score += 0.30
            elif profit_growth > 0.20:  # > 20%
                score += 0.20
            elif profit_growth > 0:  # > 0%
                score += 0.10

        # 营业收入增长（小数形式）
        revenue_growth = self.fundamental_data.get_revenue_growth(code, current_date)
        if revenue_growth is not None:
            if revenue_growth > 0.30:  # > 30%
                score += 0.20
            elif revenue_growth > 0.15:  # > 15%
                score += 0.12
            elif revenue_growth > 0:  # > 0%
                score += 0.05

        # 每股收益
        eps = self.fundamental_data.get_eps(code, current_date)
        if eps is not None and eps > 0:
            if eps > 1.0:
                score += 0.20
            elif eps > 0.5:
                score += 0.12

        # 归一化到 0-1 范围
        return min(1.0, score)

    # 辅助函数
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
        std = np.array([np.std(close[i-window:i]) if i >= window else 0 for i in range(len(close))])
        return middle + num_std * std, middle, middle - num_std * std

    def _atr(self, high, low, close, window):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        return self._sma(tr, window)

    def _rolling_max(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result

    def _rolling_min(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result

    def _shift(self, arr, periods):
        result = np.zeros_like(arr, dtype=float)
        result[periods:] = arr[:-periods]
        result[:periods] = np.nan
        return result

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
