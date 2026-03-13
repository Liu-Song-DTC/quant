# core/signal_engine.py
import numpy as np
import pandas as pd

from .signal import Signal
from .signal_store import SignalStore

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class SignalEngine:
    """信号生成器，支持多种策略模式"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化信号生成器

        参数:
            config: 配置字典，包含策略参数
        """
        self.config = config or {}

        # 默认配置
        self.default_config = {
            # 动量策略参数
            'momentum_window': 20,
            'rsi_window': 14,
            'ma_short': 10,
            'ma_mid': 20,
            'ma_long': 60,
            'boll_window': 20,
            'boll_std': 2,
            'volume_window': 10,
            'volatility_window': 20,

            # 阈值参数
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'min_volume_ratio': 1.8,

            # 风险参数
            'base_risk_vol': 0.02,
            'max_position_score': 10,
            'score_smoothing': 0.6,  # 增加平滑，减少信号震荡

            # 新增：持仓相关参数
            'min_hold_days': 3,      # 最小持仓天数
            'profit_target': 0.10,   # 止盈目标 10%
            'trailing_stop': 0.07,   # 移动止盈回撤 7%

            # 新增：信号质量参数
            'signal_threshold_buy': 0.25,  # 提高买入门槛
            'signal_threshold_sell': -0.15, # 提高卖出门槛
            'signal_confidence': 0.8,      # 提高信号置信度
        }

        # 更新配置
        self.default_config.update(self.config)
        self.config = self.default_config

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        """
        生成交易信号

        参数:
            code: 证券代码
            market_data: 市场数据DataFrame
            signal_store: 信号存储字典

        返回:
            包含buy, sell, score, risk_vol的信号字典
        """
        dates = market_data["datetime"].values
        close = market_data["close"]
        if len(market_data) < self.config['ma_long']:
            # 数据不足，返回中性信号
            return

        # 计算技术指标
        indicators = self._calculate_indicators(market_data)

        last_sig = None
        for idx in close.index:
            regime = self._detect_regime(indicators, idx)

            # 组合多个策略信号
            signals = self._combine_signals(indicators, market_data, idx)

            # 计算最终信号
            sig = self._generate_final_signal(signals, indicators, market_data, last_sig, idx, regime)
            last_sig = sig
            date = pd.to_datetime(dates[idx]).date()
            signal_store.set(code, date, sig)


    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算技术指标"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        open_price = data['open'].values

        # 移动平均线
        self.ma_short = self._rolling_mean(close, self.config['ma_short'])
        self.ma_mid = self._rolling_mean(close, self.config['ma_mid'])
        self.ma_long = self._rolling_mean(close, self.config['ma_long'])

        # 动量指标
        self.momentum = self._calculate_momentum(close, self.config['momentum_window'])

        # RSI指标
        self.rsi = self._calculate_rsi(close, self.config['rsi_window'])

        # 布林带
        self.boll_upper, self.boll_middle, self.boll_lower = self._calculate_bollinger_bands(
            close, self.config['boll_window'], self.config['boll_std']
        )

        # 成交量指标
        volume_ma = self._rolling_mean(volume, self.config['volume_window'])
        self.volume_ratio = volume / (volume_ma + 1e-10)

        # 价格波动率（ATR近似）
        self.volatility = self._calculate_volatility(high, low, close, self.config['volatility_window'])

        # 价格位置（相对于近期高低点）
        self.price_position = self._calculate_price_position(close, self.config['momentum_window'])

        # 量价关系
        self.price_volume_corr = self._calculate_price_volume_correlation(close, volume, self.config['volume_window'])

        # =============== 新增指标 ===============

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self._calculate_macd(close)

        # KDJ
        self.kdj_k, self.kdj_d, self.kdj_j = self._calculate_kdj(high, low, close)

        # OBV（能量潮）
        self.obv = self._calculate_obv(close, volume)
        self.obv_ma = self._rolling_mean(self.obv, 20)

        # 价格通道突破（Donchian Channel）
        self.donchian_upper, self.donchian_lower = self._calculate_donchian(high, low, 20)

        # 相对强度（需要后续与指数比较）
        self.roc = self._calculate_roc(close, 10)  # 10日变化率

        # 成交量加权价格（VWAP近似）
        self.vwap = self._calculate_vwap(close, high, low, volume, 20)

        # 价格动量背离检测
        self.price_divergence = self._detect_divergence(close, self.rsi, 14)

        # 大阳线/大阴线检测
        self.candle_pattern = self._detect_candle_pattern(open_price, close, high, low)

        # =============== 新增核心指标 ===============

        # 趋势强度指标 (ADX简化版)
        self.trend_strength = self._calculate_trend_strength(high, low, close, 14)

        # 回调深度检测（用于回调买入）
        self.pullback_depth = self._calculate_pullback_depth(close, high, 20)

        # 价格相对位置（相对于N日高低点）
        self.price_channel_pos = self._calculate_channel_position(close, high, low, 60)

        # 成交量趋势
        self.volume_trend = self._calculate_volume_trend(volume, 20)

        # 波动率收缩检测（突破前兆）
        self.volatility_squeeze = self._detect_volatility_squeeze(close, 20)

        return {
            'ma_short': self.ma_short,
            'ma_mid': self.ma_mid,
            'ma_long': self.ma_long,
            'momentum': self.momentum,
            'rsi': self.rsi,
            'boll_upper': self.boll_upper,
            'boll_middle': self.boll_middle,
            'boll_lower': self.boll_lower,
            'volume_ratio': self.volume_ratio,
            'volatility': self.volatility,
            'price_position': self.price_position,
            'price_volume_corr': self.price_volume_corr,
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            # 新增
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_hist': self.macd_hist,
            'kdj_k': self.kdj_k,
            'kdj_d': self.kdj_d,
            'kdj_j': self.kdj_j,
            'obv': self.obv,
            'obv_ma': self.obv_ma,
            'donchian_upper': self.donchian_upper,
            'donchian_lower': self.donchian_lower,
            'roc': self.roc,
            'vwap': self.vwap,
            'price_divergence': self.price_divergence,
            'candle_pattern': self.candle_pattern,
            # 新增
            'trend_strength': self.trend_strength,
            'pullback_depth': self.pullback_depth,
            'price_channel_pos': self.price_channel_pos,
            'volume_trend': self.volume_trend,
            'volatility_squeeze': self.volatility_squeeze,
        }

    def _combine_signals(self, indicators: Dict[str, np.ndarray], data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """组合多个策略信号"""
        # 1. 趋势策略信号
        trend_score = self._trend_strategy(indicators, idx)

        # 2. 均值回归策略信号
        mean_reversion_score = self._mean_reversion_strategy(indicators, idx)

        # 3. 动量策略信号
        momentum_score = self._momentum_strategy(indicators, idx)

        # 4. 量价策略信号
        volume_price_score = self._volume_price_strategy(indicators, data, idx)

        # 5. 波动率策略信号
        volatility_score = self._volatility_strategy(indicators, idx)

        # 6. MACD策略
        macd_score = self._macd_strategy(indicators, idx)

        # 7. KDJ策略
        kdj_score = self._kdj_strategy(indicators, idx)

        # 8. 突破策略
        breakout_score = self._breakout_strategy(indicators, idx)

        # 9. OBV策略
        obv_score = self._obv_strategy(indicators, idx)

        # 10. 背离策略
        divergence_score = self._divergence_strategy(indicators, idx)

        # 11. K线形态
        candle_score = self._candle_strategy(indicators, idx)

        # =============== 新增策略 ===============

        # 12. 回调买入策略
        pullback_score = self._pullback_strategy(indicators, idx)

        # 13. 趋势强度过滤
        trend_strength_score = self._trend_strength_strategy(indicators, idx)

        # 14. 波动率收缩突破
        squeeze_score = self._squeeze_breakout_strategy(indicators, idx)

        # 15. 市场情绪指标
        sentiment_score = self._market_sentiment_strategy(indicators, data, idx)

        return {
            'trend': trend_score,
            'mean_reversion': mean_reversion_score,
            'momentum': momentum_score,
            'volume_price': volume_price_score,
            'volatility': volatility_score,
            'macd': macd_score,
            'kdj': kdj_score,
            'breakout': breakout_score,
            'obv': obv_score,
            'divergence': divergence_score,
            'candle': candle_score,
            'pullback': pullback_score,
            'trend_strength': trend_strength_score,
            'squeeze': squeeze_score,
            'sentiment': sentiment_score,
        }

    def _trend_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """趋势跟踪策略 - 强化版"""
        # 多重均线排列
        if (
            np.isnan(indicators['ma_short'][idx]) or
            np.isnan(indicators['ma_mid'][idx]) or
            np.isnan(indicators['ma_long'][idx])
        ):
            return 0.0

        ma_short = indicators['ma_short'][idx]
        ma_mid = indicators['ma_mid'][idx]
        ma_long = indicators['ma_long'][idx]
        close = indicators['close'][idx]

        # 计算均线斜率（趋势强度）
        if idx >= 10:
            ma_short_slope = (ma_short - indicators['ma_short'][idx-10]) / ma_short
            ma_mid_slope = (ma_mid - indicators['ma_mid'][idx-10]) / ma_mid
            ma_long_slope = (ma_long - indicators['ma_long'][idx-10]) / ma_long
        else:
            ma_short_slope = 0
            ma_mid_slope = 0
            ma_long_slope = 0

        # 短期>中期>长期为强势多头趋势
        if ma_short > ma_mid > ma_long:
            # 趋势强度加成
            strength = 0.75
            if ma_short_slope > 0.015 and ma_mid_slope > 0.008:
                strength = 1.0
            elif ma_short_slope > 0.005:
                strength = 0.85
            # 长期均线也向上时更强
            if ma_long_slope > 0:
                strength = min(1.0, strength * 1.1)
            return strength

        # 短期<中期<长期为强势空头趋势
        elif ma_short < ma_mid < ma_long:
            strength = -0.75
            if ma_short_slope < -0.015 and ma_mid_slope < -0.008:
                strength = -1.0
            elif ma_short_slope < -0.005:
                strength = -0.85
            if ma_long_slope < 0:
                strength = max(-1.0, strength * 1.1)
            return strength

        # 价格在长期均线上方，均线开始多头排列
        elif close > ma_long and ma_short > ma_mid:
            return 0.4

        # 价格在长期均线上方，但均线纠缠
        elif close > ma_long and ma_short > ma_long:
            return 0.25

        # 价格在长期均线下方，均线开始空头排列
        elif close < ma_long and ma_short < ma_mid:
            return -0.4

        # 价格在长期均线下方
        elif close < ma_long:
            return -0.25

        # 均线纠缠，趋势不明
        else:
            return 0.0

    def _mean_reversion_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """均值回归策略"""
        if (
            np.isnan(indicators['ma_short'][idx]) or
            np.isnan(indicators['ma_long'][idx]) or
            np.isnan(indicators['boll_middle'][idx])
        ):
            return 0.0

        trend = indicators['ma_short'][idx] - indicators['ma_long'][idx]
        trend_strength = abs(trend) / indicators['close'][idx]
        if trend_strength > 0.03:
            return 0.0
        score = 0.0

        # RSI超买超卖
        rsi = indicators['rsi'][idx]
        if rsi > self.config['rsi_overbought']:
            score -= 1.0  # 超买，看空
        elif rsi < self.config['rsi_oversold']:
            score += 1.0  # 超卖，看多

        # 布林带回归
        close = indicators['close'][idx]
        boll_middle = indicators['boll_middle'][idx]
        boll_upper = indicators['boll_upper'][idx]
        boll_lower = indicators['boll_lower'][idx]
        band_width = boll_upper - boll_lower
        if band_width <= 0 or np.isnan(band_width):
            return 0.0

        # 价格偏离中轨程度
        deviation = (close - boll_middle) / band_width * 2

        if deviation > 0.8:  # 严重偏离上轨
            score -= 0.8
        elif deviation < -0.8:  # 严重偏离下轨
            score += 0.8

        return np.clip(score, -1, 1)

    def _momentum_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """动量策略"""
        momentum = indicators['momentum'][idx]
        hist = indicators['momentum'][idx-60:idx]

        if idx < 60:
            return 0.0
        z = momentum / np.nanstd(hist)
        return np.clip(z / 2, -1, 1)


    def _volume_price_strategy(self, indicators: Dict[str, np.ndarray], data: pd.DataFrame, idx: int) -> float:
        """量价策略"""
        score = 0.0

        # 成交量放大确认
        volume_ratio = indicators['volume_ratio'][idx]
        change_percent = data['change_percent'].values[idx]
        if np.isnan(volume_ratio) or volume_ratio <= 0:
            return 0.0

        # 放量上涨
        if volume_ratio > self.config['min_volume_ratio']:
            score += np.sign(change_percent) * 0.8

        # 量价背离检测（简化版）
        price_volume_corr = indicators['price_volume_corr'][idx]
        if price_volume_corr > 0.3:  # 正相关较强
            score += 0.3
        elif price_volume_corr < -0.3:  # 负相关较强
            score -= 0.3

        return np.clip(score, -1, 1)

    def _volatility_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """波动率策略"""
        if (
            np.isnan(indicators['boll_upper'][idx]) or
            np.isnan(indicators['boll_lower'][idx])
        ):
            return 0.0

        range_width = (
            indicators['boll_upper'][idx] -
            indicators['boll_lower'][idx]
        ) / indicators['close'][idx]

        if range_width < 0.04:
            return 0.0
        volatility = indicators['volatility'][idx]
        price_position = indicators['price_position'][idx]

        # 低波动率突破策略
        if volatility < 0.02:  # 低波动
            if price_position > 0.8:  # 价格在高位
                return 0.6
            elif price_position < 0.2:  # 价格在低位
                return -0.6

        # 高波动率均值回归
        elif volatility > 0.05:  # 高波动
            return 0.0  # 观望

        return 0.0

    def _detect_regime(self, indicators: Dict[str, np.ndarray], idx: int) -> int:
        """
        市场结构识别
        返回:
            1  -> 上行结构
            0  -> 震荡
           -1  -> 下行结构
        """
        ma_mid = indicators['ma_mid'][idx]
        ma_long = indicators['ma_long'][idx]
        close = indicators['close'][idx]

        if np.isnan(ma_mid) or np.isnan(ma_long):
            return 0

        slope = (ma_mid - ma_long) / close

        # 阈值可以后续调，但不要太小
        if slope > 0.01:
            return 1
        elif slope < -0.01:
            return -1
        else:
            return 0

    def _generate_final_signal(self, signals: Dict[str, float],
                              indicators: Dict[str, np.ndarray],
                               data: pd.DataFrame, last_sig: Signal,
                               idx: int, regime: int) -> Signal:
        """生成最终交易信号 - 简化核心因子版"""
        # 核心趋势信号
        trend = signals['trend']
        momentum = signals['momentum']
        macd = signals['macd']

        # 辅助信号
        obv = signals['obv']
        kdj = signals['kdj']
        breakout = signals['breakout']
        sentiment = signals['sentiment']  # 新增市场情绪信号

        # =============== 市场环境调整 ===============
        if regime == -1:
            # 熊市：不做多，信号打折
            trend = min(trend, 0)
            macd = min(macd, 0)
            momentum = min(momentum, 0)
        elif regime == 0:
            # 震荡市：略微降低
            trend *= 0.85

        # =============== 核心评分（回归简洁） ===============

        # 趋势60% + MACD30% + 动量10%
        core_score = 0.60 * trend + 0.30 * macd + 0.10 * momentum

        # =============== 入场条件判断 ===============

        # 多头入场条件
        buy_conditions = 0
        if trend > 0.4:  # 趋势要求
            buy_conditions += 1
        if macd > 0.3:   # MACD要求
            buy_conditions += 1
        if momentum > 0:  # 动量为正
            buy_conditions += 1

        # 空头/卖出条件
        sell_conditions = 0
        if trend < -0.2:  # 降低卖出要求
            sell_conditions += 1
        if macd < -0.15:
            sell_conditions += 1
        if momentum < -0.2:
            sell_conditions += 1
        if obv < -0.1:
            sell_conditions += 1

        # =============== 综合分数调整 ===============

        composite_score = core_score

        # 买入信号加强（满足多个条件时）
        if buy_conditions >= 3 and core_score > 0.15:
            composite_score = max(core_score, 0.35)
        elif buy_conditions >= 4:
            composite_score *= 1.15

        # 卖出信号加强
        if sell_conditions >= 3 and core_score < -0.1:
            composite_score = min(core_score, -0.25)

        # =============== 信号平滑（降低震荡） ===============
        if last_sig and abs(last_sig.score) > 0.1:
            if np.sign(composite_score) == np.sign(last_sig.score):
                # 同向时平滑
                composite_score = 0.6 * composite_score + 0.4 * last_sig.score
            elif abs(composite_score) < 0.15:
                # 弱反向信号时延续上一期
                composite_score = last_sig.score * 0.3

        # =============== 生成买卖信号 ===============

        # 提高买入门槛，只接受高质量信号
        buy = False
        if regime >= 0:  # 非熊市
            if composite_score > self.config['signal_threshold_buy'] and buy_conditions >= 3:
                buy = True
            elif composite_score > 0.35:  # 强信号直接买
                buy = True

        # 卖出条件
        sell = composite_score < self.config['signal_threshold_sell'] or (sell_conditions >= 2 and composite_score < -0.05)

        # 熊市强制卖出
        if regime == -1 and (macd < -0.2 or momentum < -0.15):
            sell = True

        risk_vol = self._calculate_risk_volatility(indicators, data, idx)

        return Signal(
            buy=bool(buy),
            sell=bool(sell),
            score=float(np.clip(composite_score, -1, 1)),
            risk_vol=float(risk_vol),
        )

    def _calculate_risk_volatility(self, indicators: Dict[str, np.ndarray], data: pd.DataFrame, idx: int) -> float:
        """计算风险波动率"""
        volatility = indicators['volatility'][idx]

        # 基础风险波动率 + 当前波动率调整
        risk_vol = self.config['base_risk_vol'] * (1 + volatility * 10)

        # 限制风险波动率范围
        return max(0.005, min(0.1, risk_vol))

    # =============== 技术指标计算工具函数 ===============

    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """滚动均值"""
        result = np.full_like(data, np.nan)
        for i in range(window-1, len(data)):
            result[i] = np.mean(data[i-window+1:i+1])
        return result

    @staticmethod
    def _calculate_momentum(close: np.ndarray, window: int) -> np.ndarray:
        """计算动量"""
        momentum = np.zeros_like(close)
        for i in range(window, len(close)):
            momentum[i] = (close[i] / close[i-window]) - 1
        return momentum

    @staticmethod
    def _calculate_rsi(close: np.ndarray, window: int) -> np.ndarray:
        """计算RSI"""
        deltas = np.diff(close)
        seed = deltas[:window+1]

        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window

        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(close)
        rsi[:window] = 100 - 100 / (1 + rs)

        for i in range(window, len(close)-1):
            delta = deltas[i-1]

            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta

            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window

            rs = up / down if down != 0 else 0
            rsi[i] = 100 - 100 / (1 + rs)

        return rsi

    @staticmethod
    def _calculate_bollinger_bands(close: np.ndarray, window: int, num_std: float) -> tuple:
        """计算布林带"""
        middle = np.full_like(close, np.nan)
        upper = np.full_like(close, np.nan)
        lower = np.full_like(close, np.nan)

        for i in range(window-1, len(close)):
            slice_data = close[i-window+1:i+1]
            middle[i] = np.mean(slice_data)
            std = np.std(slice_data)
            upper[i] = middle[i] + num_std * std
            lower[i] = middle[i] - num_std * std

        return upper, middle, lower

    @staticmethod
    def _calculate_volatility(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
        """计算波动率（ATR近似）"""
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = np.zeros_like(close)
        atr[window] = np.mean(tr[:window])

        for i in range(window+1, len(close)):
            atr[i] = (atr[i-1] * (window - 1) + tr[i-1]) / window

        return atr / close  # 标准化波动率

    @staticmethod
    def _calculate_price_position(close: np.ndarray, window: int) -> np.ndarray:
        """计算价格位置（0-1之间，表示在近期高低点中的位置）"""
        position = np.zeros_like(close)

        for i in range(window, len(close)):
            window_data = close[i-window:i+1]
            high = np.max(window_data)
            low = np.min(window_data)

            if high != low:
                position[i] = (close[i] - low) / (high - low)
            else:
                position[i] = 0.5

        return position

    @staticmethod
    def _calculate_price_volume_correlation(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
        """计算价格与成交量的滚动相关系数"""
        corr = np.zeros_like(close)

        for i in range(window, len(close)):
            price_slice = close[i-window:i+1]
            volume_slice = volume[i-window:i+1]

            if len(price_slice) > 1 and len(volume_slice) > 1:
                corr_matrix = np.corrcoef(price_slice, volume_slice)
                corr[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0

        return corr

    # =============== 新增指标计算函数 ===============

    @staticmethod
    def _calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """计算MACD"""
        def ema(data, span):
            result = np.zeros_like(data)
            result[0] = data[0]
            alpha = 2 / (span + 1)
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def _calculate_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9, m1: int = 3, m2: int = 3) -> tuple:
        """计算KDJ指标"""
        length = len(close)
        rsv = np.zeros(length)
        k = np.zeros(length)
        d = np.zeros(length)
        j = np.zeros(length)

        for i in range(n-1, length):
            highest = np.max(high[i-n+1:i+1])
            lowest = np.min(low[i-n+1:i+1])
            if highest != lowest:
                rsv[i] = (close[i] - lowest) / (highest - lowest) * 100
            else:
                rsv[i] = 50

        k[n-1] = 50
        d[n-1] = 50

        for i in range(n, length):
            k[i] = (m1 - 1) / m1 * k[i-1] + 1 / m1 * rsv[i]
            d[i] = (m2 - 1) / m2 * d[i-1] + 1 / m2 * k[i]
            j[i] = 3 * k[i] - 2 * d[i]

        return k, d, j

    @staticmethod
    def _calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """计算OBV（能量潮）"""
        obv = np.zeros_like(close)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    @staticmethod
    def _calculate_donchian(high: np.ndarray, low: np.ndarray, window: int) -> tuple:
        """计算Donchian通道"""
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)

        for i in range(window, len(high)):
            upper[i] = np.max(high[i-window:i])
            lower[i] = np.min(low[i-window:i])

        return upper, lower

    @staticmethod
    def _calculate_roc(close: np.ndarray, window: int) -> np.ndarray:
        """计算变化率ROC"""
        roc = np.zeros_like(close)
        for i in range(window, len(close)):
            if close[i-window] != 0:
                roc[i] = (close[i] - close[i-window]) / close[i-window] * 100
        return roc

    @staticmethod
    def _calculate_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
        """计算滚动VWAP"""
        typical_price = (high + low + close) / 3
        vwap = np.zeros_like(close)

        for i in range(window, len(close)):
            tp_slice = typical_price[i-window+1:i+1]
            vol_slice = volume[i-window+1:i+1]
            total_vol = np.sum(vol_slice)
            if total_vol > 0:
                vwap[i] = np.sum(tp_slice * vol_slice) / total_vol
            else:
                vwap[i] = close[i]

        return vwap

    @staticmethod
    def _detect_divergence(close: np.ndarray, indicator: np.ndarray, window: int) -> np.ndarray:
        """检测价格与指标的背离"""
        divergence = np.zeros_like(close)

        for i in range(window * 2, len(close)):
            # 价格创新高但指标未创新高 -> 顶背离 (-1)
            # 价格创新低但指标未创新低 -> 底背离 (+1)

            price_window = close[i-window:i+1]
            ind_window = indicator[i-window:i+1]

            price_max_idx = np.argmax(price_window)
            price_min_idx = np.argmin(price_window)
            ind_max_idx = np.argmax(ind_window)
            ind_min_idx = np.argmin(ind_window)

            # 顶背离：价格在窗口末端创新高，但指标高点在前面
            if price_max_idx >= window - 2 and ind_max_idx < window - 3:
                if price_window[-1] > price_window[ind_max_idx]:
                    divergence[i] = -1

            # 底背离：价格在窗口末端创新低，但指标低点在前面
            if price_min_idx >= window - 2 and ind_min_idx < window - 3:
                if price_window[-1] < price_window[ind_min_idx]:
                    divergence[i] = 1

        return divergence

    @staticmethod
    def _detect_candle_pattern(open_price: np.ndarray, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """检测K线形态"""
        pattern = np.zeros_like(close)

        for i in range(1, len(close)):
            body = close[i] - open_price[i]
            body_size = abs(body)
            upper_shadow = high[i] - max(open_price[i], close[i])
            lower_shadow = min(open_price[i], close[i]) - low[i]
            total_range = high[i] - low[i]

            if total_range == 0:
                continue

            body_ratio = body_size / total_range

            # 大阳线：实体占比>70%，收盘>开盘
            if body > 0 and body_ratio > 0.7:
                pattern[i] = 1

            # 大阴线：实体占比>70%，收盘<开盘
            elif body < 0 and body_ratio > 0.7:
                pattern[i] = -1

            # 锤子线：下影线长，实体小，上影线短
            elif lower_shadow > 2 * body_size and upper_shadow < body_size * 0.5:
                pattern[i] = 0.5  # 看涨

            # 倒锤子：上影线长，实体小，下影线短
            elif upper_shadow > 2 * body_size and lower_shadow < body_size * 0.5:
                pattern[i] = -0.5  # 看跌

        return pattern

    # =============== 新增核心指标计算 ===============

    @staticmethod
    def _calculate_trend_strength(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
        """计算趋势强度 (简化版ADX)"""
        strength = np.zeros_like(close)

        for i in range(window + 1, len(close)):
            # 计算方向移动
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]

            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0

            # 简单平均
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

            if tr > 0:
                plus_di = plus_dm / tr
                minus_di = minus_dm / tr
                dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                strength[i] = dx

        # 平滑
        result = np.zeros_like(strength)
        for i in range(window, len(strength)):
            result[i] = np.mean(strength[i-window+1:i+1])

        return result

    @staticmethod
    def _calculate_pullback_depth(close: np.ndarray, high: np.ndarray, window: int) -> np.ndarray:
        """计算回调深度 - 用于回调买入判断"""
        depth = np.zeros_like(close)

        for i in range(window, len(close)):
            # 找到窗口内的最高点
            recent_high = np.max(high[i-window:i+1])
            recent_high_idx = np.argmax(high[i-window:i+1])

            # 计算从最高点的回调深度
            if recent_high > 0:
                pullback = (recent_high - close[i]) / recent_high

                # 如果最高点在最近，说明还在创新高，回调深度为0
                if recent_high_idx >= window - 3:
                    depth[i] = 0
                else:
                    depth[i] = pullback

        return depth

    @staticmethod
    def _calculate_channel_position(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
        """计算价格在通道中的位置"""
        position = np.zeros_like(close)

        for i in range(window, len(close)):
            highest = np.max(high[i-window:i])
            lowest = np.min(low[i-window:i])

            if highest > lowest:
                position[i] = (close[i] - lowest) / (highest - lowest)
            else:
                position[i] = 0.5

        return position

    @staticmethod
    def _calculate_volume_trend(volume: np.ndarray, window: int) -> np.ndarray:
        """计算成交量趋势"""
        trend = np.zeros_like(volume, dtype=float)

        for i in range(window * 2, len(volume)):
            # 比较近期均量和远期均量
            recent_avg = np.mean(volume[i-window//2:i+1])
            past_avg = np.mean(volume[i-window:i-window//2])

            if past_avg > 0:
                trend[i] = (recent_avg - past_avg) / past_avg

        return trend

    @staticmethod
    def _detect_volatility_squeeze(close: np.ndarray, window: int) -> np.ndarray:
        """检测波动率收缩（突破前兆）"""
        squeeze = np.zeros_like(close)

        for i in range(window * 2, len(close)):
            # 计算当前波动率
            current_std = np.std(close[i-window:i+1])
            current_mean = np.mean(close[i-window:i+1])

            # 计算历史波动率
            hist_std = np.std(close[i-window*2:i-window])
            hist_mean = np.mean(close[i-window*2:i-window])

            if hist_mean > 0 and current_mean > 0:
                current_vol = current_std / current_mean
                hist_vol = hist_std / hist_mean

                if hist_vol > 0:
                    # 波动率收缩比例
                    squeeze[i] = 1 - current_vol / hist_vol
                    squeeze[i] = max(0, squeeze[i])  # 只关注收缩

        return squeeze

    # =============== 新增策略函数 ===============

    def _macd_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """MACD策略 - 优化版"""
        if idx < 35:
            return 0.0

        macd = indicators['macd'][idx]
        signal = indicators['macd_signal'][idx]
        hist = indicators['macd_hist'][idx]
        hist_prev = indicators['macd_hist'][idx-1] if idx > 0 else 0
        hist_prev2 = indicators['macd_hist'][idx-2] if idx > 1 else 0

        # 计算MACD柱状图的趋势
        hist_trend = 0
        if idx >= 5:
            hist_5_ago = indicators['macd_hist'][idx-5]
            hist_trend = hist - hist_5_ago

        score = 0.0

        # MACD金叉/死叉
        if hist > 0 and hist_prev <= 0:  # 金叉
            # 零轴上方金叉更强
            if macd > 0:
                score += 0.75
            else:
                score += 0.5
        elif hist < 0 and hist_prev >= 0:  # 死叉
            if macd < 0:
                score -= 0.75
            else:
                score -= 0.5

        # MACD柱状图连续放大（趋势强化）
        if hist > 0 and hist > hist_prev > hist_prev2:
            score += 0.3
        elif hist < 0 and hist < hist_prev < hist_prev2:
            score -= 0.3

        # 零轴位置和趋势
        if macd > 0 and hist_trend > 0:
            score += 0.2
        elif macd < 0 and hist_trend < 0:
            score -= 0.2

        return np.clip(score, -1, 1)

    def _kdj_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """KDJ策略 - 更稳健"""
        if idx < 15:
            return 0.0

        k = indicators['kdj_k'][idx]
        d = indicators['kdj_d'][idx]
        j = indicators['kdj_j'][idx]
        k_prev = indicators['kdj_k'][idx-1]
        d_prev = indicators['kdj_d'][idx-1]

        score = 0.0

        # KDJ金叉/死叉（在合适位置）
        if k > d and k_prev <= d_prev:  # 金叉
            if k < 50:  # 低位金叉更有效
                score += 0.5
            else:
                score += 0.2
        elif k < d and k_prev >= d_prev:  # 死叉
            if k > 50:  # 高位死叉更有效
                score -= 0.5
            else:
                score -= 0.2

        # 超卖区反弹
        if j < 10 and k > k_prev:
            score += 0.4
        # 超买区回落
        elif j > 90 and k < k_prev:
            score -= 0.4

        return np.clip(score, -1, 1)

    def _breakout_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """突破策略（Donchian通道）"""
        if idx < 25:
            return 0.0

        close = indicators['close'][idx]
        upper = indicators['donchian_upper'][idx]
        lower = indicators['donchian_lower'][idx]
        volume_ratio = indicators['volume_ratio'][idx]

        if upper == 0 or lower == 0:
            return 0.0

        score = 0.0

        # 向上突破
        if close >= upper:
            score += 0.7
            # 放量突破加分
            if volume_ratio > 1.5:
                score += 0.3

        # 向下突破
        elif close <= lower:
            score -= 0.7
            if volume_ratio > 1.5:
                score -= 0.3

        # 在通道中间
        else:
            channel_width = upper - lower
            if channel_width > 0:
                position = (close - lower) / channel_width
                if position > 0.8:
                    score += 0.2
                elif position < 0.2:
                    score -= 0.2

        return np.clip(score, -1, 1)

    def _obv_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """OBV能量潮策略"""
        if idx < 25:
            return 0.0

        obv = indicators['obv'][idx]
        obv_ma = indicators['obv_ma'][idx]
        obv_prev = indicators['obv'][idx-5] if idx >= 5 else obv
        close = indicators['close'][idx]
        close_prev = indicators['close'][idx-5] if idx >= 5 else close

        if np.isnan(obv_ma) or obv_ma == 0:
            return 0.0

        score = 0.0

        # OBV在均线上方
        if obv > obv_ma:
            score += 0.3
        else:
            score -= 0.3

        # OBV趋势与价格趋势一致
        obv_trend = obv - obv_prev
        price_trend = close - close_prev

        if obv_trend > 0 and price_trend > 0:
            score += 0.4  # 量价齐升
        elif obv_trend < 0 and price_trend < 0:
            score -= 0.2  # 量价齐跌
        elif obv_trend > 0 and price_trend < 0:
            score += 0.3  # 底部吸筹
        elif obv_trend < 0 and price_trend > 0:
            score -= 0.5  # 顶部出货

        return np.clip(score, -1, 1)

    def _divergence_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """背离策略"""
        if idx < 30:
            return 0.0

        divergence = indicators['price_divergence'][idx]

        # 底背离看涨，顶背离看跌
        return float(divergence) * 0.8

    def _candle_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """K线形态策略"""
        if idx < 5:
            return 0.0

        pattern = indicators['candle_pattern'][idx]
        volume_ratio = indicators['volume_ratio'][idx]

        score = float(pattern)

        # 放量的K线形态更有意义
        if abs(pattern) > 0.5 and volume_ratio > 1.3:
            score *= 1.3

        return np.clip(score, -1, 1)

    def _pullback_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """回调买入策略 - 在上升趋势中等待回调"""
        if idx < 30:
            return 0.0

        pullback = indicators['pullback_depth'][idx]
        trend = indicators['ma_short'][idx]
        ma_long = indicators['ma_long'][idx]
        rsi = indicators['rsi'][idx]
        kdj_k = indicators['kdj_k'][idx]

        if np.isnan(trend) or np.isnan(ma_long):
            return 0.0

        score = 0.0

        # 只在上升趋势中寻找回调买入机会
        if trend > ma_long:
            # 理想回调深度: 3%-8%
            if 0.03 <= pullback <= 0.08:
                score += 0.6
                # RSI超卖加分
                if rsi < 40:
                    score += 0.3
                # KDJ低位加分
                if kdj_k < 30:
                    score += 0.2
            # 轻微回调
            elif 0.01 <= pullback < 0.03:
                score += 0.2
            # 回调过深，可能趋势反转
            elif pullback > 0.12:
                score -= 0.3

        # 下降趋势中的反弹做空
        elif trend < ma_long:
            if pullback < 0.02:  # 反弹到高点附近
                score -= 0.4

        return np.clip(score, -1, 1)

    def _trend_strength_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """趋势强度策略 - 过滤弱趋势"""
        if idx < 20:
            return 0.0

        strength = indicators['trend_strength'][idx]
        ma_short = indicators['ma_short'][idx]
        ma_long = indicators['ma_long'][idx]

        if np.isnan(ma_short) or np.isnan(ma_long):
            return 0.0

        score = 0.0

        # 趋势方向
        trend_up = ma_short > ma_long

        # 强趋势 (strength > 0.3) 时跟随趋势
        if strength > 0.3:
            score = 0.6 if trend_up else -0.6
        # 中等趋势
        elif strength > 0.2:
            score = 0.3 if trend_up else -0.3
        # 弱趋势/震荡，减少交易
        else:
            score = 0.0

        return score

    def _squeeze_breakout_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """波动率收缩突破策略"""
        if idx < 45:
            return 0.0

        squeeze = indicators['volatility_squeeze'][idx]
        close = indicators['close'][idx]
        upper = indicators['boll_upper'][idx]
        lower = indicators['boll_lower'][idx]
        volume_ratio = indicators['volume_ratio'][idx]

        if np.isnan(upper) or np.isnan(lower):
            return 0.0

        score = 0.0

        # 波动率收缩后的突破更可靠
        if squeeze > 0.3:  # 波动率收缩超过30%
            # 向上突破布林带
            if close > upper and volume_ratio > 1.3:
                score = 0.8
            # 向下突破布林带
            elif close < lower and volume_ratio > 1.3:
                score = -0.8
            # 即将突破
            elif close > upper * 0.98:
                score = 0.3
            elif close < lower * 1.02:
                score = -0.3

        return score

    def _market_sentiment_strategy(self, indicators: Dict[str, np.ndarray], data: pd.DataFrame, idx: int) -> float:
        """市场情绪策略 - 基于多个情绪指标综合判断"""
        if idx < 20:
            return 0.0

        close = indicators['close'][idx]
        volume = indicators['volume'][idx]
        volume_ratio = indicators['volume_ratio'][idx]
        rsi = indicators['rsi'][idx]
        momentum = indicators['momentum'][idx]

        score = 0.0

        # 1. 成交量情绪 - 放量上涨看多，缩量下跌看空
        if volume_ratio > 1.5 and data['change_percent'].values[idx] > 0:
            score += 0.3
        elif volume_ratio < 0.7 and data['change_percent'].values[idx] < 0:
            score -= 0.3

        # 2. 动量情绪 - 连续上涨看多，连续下跌看空
        if momentum > 0 and idx >= 3:
            recent_momentum = sum(indicators['momentum'][idx-3:idx+1])
            if recent_momentum > 0:
                score += 0.2

        # 3. RSI情绪 - 极端值后的反转机会
        if rsi < 30:  # 超卖，反弹机会
            score += 0.3
        elif rsi > 70:  # 超买，回调风险
            score -= 0.3

        # 4. 价格位置情绪 - 连创新高看多，连创新低看空
        if idx >= 5:
            recent_highs = sum(1 for i in range(idx-5, idx) if indicators['close'][i] < close)
            recent_lows = sum(1 for i in range(idx-5, idx) if indicators['close'][i] > close)

            if recent_highs >= 4:  # 连续创新高
                score += 0.2
            elif recent_lows >= 4:  # 连续创新低
                score -= 0.2

        # 5. 量价配合情绪 - 价升量增，价跌量缩
        price_change = data['change_percent'].values[idx]
        if price_change > 0 and volume_ratio > 1.2:
            score += 0.2  # 价升量增，强势
        elif price_change < 0 and volume_ratio < 0.8:
            score += 0.1  # 价跌量缩，可能企稳

        return np.clip(score, -1, 1)

