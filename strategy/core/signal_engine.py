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
            'momentum_window': 20,      # 动量计算窗口
            'rsi_window': 14,           # RSI窗口
            'ma_short': 5,              # 短期均线
            'ma_mid': 10,               # 中期均线
            'ma_long': 20,              # 长期均线
            'boll_window': 20,          # 布林带窗口
            'boll_std': 2,              # 布林带标准差倍数
            'volume_window': 10,        # 成交量均线窗口
            'volatility_window': 20,    # 波动率计算窗口

            # 阈值参数
            'rsi_overbought': 70,       # RSI超买阈值
            'rsi_oversold': 30,         # RSI超卖阈值
            'min_volume_ratio': 1.5,    # 最小成交量比率

            # 风险参数
            'base_risk_vol': 0.02,      # 基础风险波动
            'max_position_score': 10,   # 最大持仓分数
            'score_smoothing': 0.3,     # 分数平滑系数
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
            # 组合多个策略信号
            signals = self._combine_signals(indicators, market_data, idx)

            # 计算最终信号
            sig = self._generate_final_signal(signals, indicators, market_data, last_sig, idx)
            last_sig = sig
            date = pd.to_datetime(dates[idx]).date()
            signal_store.set(code, date, sig)


    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算技术指标"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

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
        self.volume_ratio = volume / volume_ma

        # 价格波动率（ATR近似）
        self.volatility = self._calculate_volatility(high, low, close, self.config['volatility_window'])

        # 价格位置（相对于近期高低点）
        self.price_position = self._calculate_price_position(close, self.config['momentum_window'])

        # 量价关系
        self.price_volume_corr = self._calculate_price_volume_correlation(close, volume, self.config['volume_window'])

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
            'close': close
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

        return {
            'trend': trend_score,
            'mean_reversion': mean_reversion_score,
            'momentum': momentum_score,
            'volume_price': volume_price_score,
            'volatility': volatility_score
        }

    def _trend_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """趋势跟踪策略"""
        # 多重均线排列
        ma_score = 0.0

        # 短期>中期>长期为强势多头趋势
        if (indicators['ma_short'][idx] > indicators['ma_mid'][idx] and
            indicators['ma_mid'][idx] > indicators['ma_long'][idx]):
            ma_score = 1.0
        # 短期<中期<长期为强势空头趋势
        elif (indicators['ma_short'][idx] < indicators['ma_mid'][idx] and
              indicators['ma_mid'][idx] < indicators['ma_long'][idx]):
            ma_score = -1.0
        # 均线纠缠，趋势不明
        else:
            # 检查短期均线方向
            if indicators['ma_short'][idx] > indicators['ma_short'][idx-5]:
                ma_score = 0.3
            else:
                ma_score = -0.3

        # 价格在布林带中的位置
        close = indicators['close'][idx]
        boll_upper = indicators['boll_upper'][idx]
        boll_lower = indicators['boll_lower'][idx]

        if close > boll_upper * 0.95:  # 接近上轨，强势
            boll_score = 0.5
        elif close < boll_lower * 1.05:  # 接近下轨，弱势
            boll_score = -0.5
        else:
            boll_score = 0.0

        return (ma_score + boll_score) / 2

    def _mean_reversion_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """均值回归策略"""
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

        # 价格偏离中轨程度
        deviation = (close - boll_middle) / (boll_upper - boll_lower) * 2

        if deviation > 0.8:  # 严重偏离上轨
            score -= 0.8
        elif deviation < -0.8:  # 严重偏离下轨
            score += 0.8

        return np.clip(score, -1, 1)

    def _momentum_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """动量策略"""
        momentum = indicators['momentum'][idx]

        # 标准化动量值
        if momentum > 0.05:  # 强正动量
            return 1.0
        elif momentum < -0.05:  # 强负动量
            return -1.0
        elif momentum > 0.02:  # 弱正动量
            return 0.5
        elif momentum < -0.02:  # 弱负动量
            return -0.5
        else:
            return 0.0

    def _volume_price_strategy(self, indicators: Dict[str, np.ndarray], data: pd.DataFrame, idx: int) -> float:
        """量价策略"""
        score = 0.0

        # 成交量放大确认
        volume_ratio = indicators['volume_ratio'][idx]
        change_percent = data['change_percent'].values[idx]

        # 放量上涨
        if volume_ratio > self.config['min_volume_ratio'] and change_percent > 0:
            score += 0.8
        # 放量下跌
        elif volume_ratio > self.config['min_volume_ratio'] and change_percent < -1:
            score -= 0.8

        # 量价背离检测（简化版）
        price_volume_corr = indicators['price_volume_corr'][idx]
        if price_volume_corr > 0.3:  # 正相关较强
            score += 0.3
        elif price_volume_corr < -0.3:  # 负相关较强
            score -= 0.3

        return np.clip(score, -1, 1)

    def _volatility_strategy(self, indicators: Dict[str, np.ndarray], idx: int) -> float:
        """波动率策略"""
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

    def _generate_final_signal(self, signals: Dict[str, float],
                              indicators: Dict[str, np.ndarray],
                               data: pd.DataFrame, last_sig: Signal, idx: int) -> Signal:
        """生成最终交易信号"""
        # 加权综合分数（可根据策略表现调整权重）
        weights = {
            'trend': 0.3,
            'mean_reversion': 0.25,
            'momentum': 0.25,
            'volume_price': 0.1,
            'volatility': 0.1
        }

        # 计算综合分数
        composite_score = sum(signals[key] * weights[key] for key in weights)

        # 平滑处理
        if last_sig:
            w = self.config['score_smoothing']
            composite_score =  w * composite_score + (1 - w) * last_sig.score


        # 生成买卖信号
        buy = composite_score > 0.2  # 分数阈值可调整
        sell = composite_score < -0.2

        # 计算风险波动率
        risk_vol = self._calculate_risk_volatility(indicators, data, idx)

        sig = Signal(
            buy=bool(buy),
            sell=bool(sell),
            score=float(composite_score),
            risk_vol=float(risk_vol),
        )
        return sig

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

