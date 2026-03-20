# core/factor_library.py
"""因子库 - 各种技术因子的计算函数"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable


class FactorRegistry:
    """因子注册表 - 管理所有可用因子"""

    _factors = {}
    _factor_groups = {}

    @classmethod
    def register(cls, name: str, func: Callable, group: str = 'technical'):
        """注册因子"""
        cls._factors[name] = func
        if group not in cls._factor_groups:
            cls._factor_groups[group] = []
        cls._factor_groups[group].append(name)

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """获取因子函数"""
        return cls._factors.get(name)

    @classmethod
    def list_factors(cls, group: str = None) -> list:
        """列出因子"""
        if group:
            return cls._factor_groups.get(group, [])
        return list(cls._factors.keys())


# ==================== 辅助函数 ====================

def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """计算滚动标准差"""
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    for i in range(window, len(arr)):
        result[i] = np.std(arr[i-window:i])
    return result


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均"""
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    result[window-1:] = np.convolve(arr, np.ones(window)/window, mode='valid')
    return result


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    result = np.zeros_like(arr, dtype=float)
    result[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    """计算RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = _sma(gain, window)
    avg_loss = _sma(loss, window)
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _bollinger(close: np.ndarray, window: int, num_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带"""
    middle = _sma(close, window)
    std = np.array([np.std(close[i-window:i]) if i >= window else 0 for i in range(len(close))])
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """计算ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    return _sma(tr, window)


def _shift(arr: np.ndarray, periods: int) -> np.ndarray:
    """数组位移"""
    result = np.zeros_like(arr, dtype=float)
    result[periods:] = arr[:-periods]
    result[:periods] = np.nan
    # 标记无效位移（价格为0或负数）
    shifted_vals = arr[:-periods]
    result[periods:][shifted_vals <= 0] = np.nan
    return result


# ==================== 波动率因子 ====================

def calc_factor_volatility_10(close: np.ndarray) -> np.ndarray:
    """计算10日波动率因子"""
    close_safe = np.where(close <= 0, np.nan, close)
    returns = np.diff(close_safe, prepend=close_safe[0])
    returns = returns / np.where(close_safe == 0, np.nan, close_safe)
    result = _rolling_std(returns, 10)
    result[returns == 0] = 0  # 价格为0或不变时波动率为0
    return result


def calc_factor_volatility_5(close: np.ndarray) -> np.ndarray:
    """计算5日波动率因子"""
    close_safe = np.where(close <= 0, np.nan, close)
    returns = np.diff(close_safe, prepend=close_safe[0])
    returns = returns / np.where(close_safe == 0, np.nan, close_safe)
    result = _rolling_std(returns, 5)
    result[returns == 0] = 0
    return result


def calc_factor_volatility_20(close: np.ndarray) -> np.ndarray:
    """计算20日波动率因子"""
    close_safe = np.where(close <= 0, np.nan, close)
    returns = np.diff(close_safe, prepend=close_safe[0])
    returns = returns / np.where(close_safe == 0, np.nan, close_safe)
    result = _rolling_std(returns, 20)
    result[returns == 0] = 0
    return result


# ==================== RSI因子 ====================

def calc_factor_rsi_8(close: np.ndarray) -> np.ndarray:
    """计算RSI-8因子"""
    return _rsi(close, 8)


def calc_factor_rsi_14(close: np.ndarray) -> np.ndarray:
    """计算RSI-14因子"""
    return _rsi(close, 14)


def calc_factor_rsi_6(close: np.ndarray) -> np.ndarray:
    """计算RSI-6因子"""
    return _rsi(close, 6)


def calc_factor_rsi_10(close: np.ndarray) -> np.ndarray:
    """计算RSI-10因子"""
    return _rsi(close, 10)


# ==================== 布林带因子 ====================

def calc_factor_bb_width_20(close: np.ndarray) -> np.ndarray:
    """计算布林带宽度因子"""
    bb_upper, bb_middle, bb_lower = _bollinger(close, 20, 2)
    bb_std = np.zeros_like(close)
    bb_std[20:] = np.array([np.std(close[max(0, i-20):i]) for i in range(20, len(close))])
    bb_width = 4 * bb_std / (bb_middle + 1e-10)
    return bb_width


# ==================== 动量因子 ====================

def calc_factor_momentum_10(close: np.ndarray) -> np.ndarray:
    """计算10日动量"""
    return close / _shift(close, 10) - 1


def calc_factor_momentum_20(close: np.ndarray) -> np.ndarray:
    """计算20日动量"""
    return close / _shift(close, 20) - 1


# ==================== ATR因子 ====================

def calc_factor_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """计算ATR因子"""
    return _atr(high, low, close, period)


# ==================== 组合因子 ====================

def calc_factor_composite(
    close: np.ndarray,
    weights: Dict[str, float] = None
) -> np.ndarray:
    """计算组合因子 - 与原代码逻辑一致

    使用配置文件的权重计算组合因子值
    """
    if weights is None:
        weights = {
            'volatility_10': 0.30,
            'rsi_average': 0.25,
            'bb_width': 0.15,
            'momentum': 0.30,
        }

    # 计算各基础因子
    vol_10 = calc_factor_volatility_10(close)
    rsi_6 = calc_factor_rsi_6(close)
    rsi_8 = calc_factor_rsi_8(close)
    rsi_10 = calc_factor_rsi_10(close)
    bb_width = calc_factor_bb_width_20(close)
    mom_10 = calc_factor_momentum_10(close)

    # 计算组合因子
    result = np.zeros_like(close, dtype=float)
    for i in range(len(close)):
        if np.isnan(vol_10[i]):
            result[i] = 0
            continue

        # 波动率因子 (放大到类似动量)
        vol_factor = vol_10[i] * 10

        # 多周期RSI平均
        rsi_avg = (rsi_6[i] + rsi_8[i] + rsi_10[i]) / 3
        rsi_avg_val = (rsi_avg - 50) / 50

        # 布林带因子
        bb_val = bb_width[i]

        # 动量因子
        mom_val = mom_10[i] * 2

        # 组合
        factor_value = (
            vol_factor * weights.get('volatility_10', 0.30) +
            rsi_avg_val * weights.get('rsi_average', 0.25) +
            bb_val * weights.get('bb_width', 0.15) +
            mom_val * weights.get('momentum', 0.30)
        )
        result[i] = factor_value

    return result


# 注册因子
FactorRegistry.register('volatility_10', calc_factor_volatility_10, 'volatility')
FactorRegistry.register('volatility_5', calc_factor_volatility_5, 'volatility')
FactorRegistry.register('volatility_20', calc_factor_volatility_20, 'volatility')
FactorRegistry.register('rsi_8', calc_factor_rsi_8, 'rsi')
FactorRegistry.register('rsi_14', calc_factor_rsi_14, 'rsi')
FactorRegistry.register('rsi_6', calc_factor_rsi_6, 'rsi')
FactorRegistry.register('rsi_10', calc_factor_rsi_10, 'rsi')
FactorRegistry.register('bb_width_20', calc_factor_bb_width_20, 'bollinger')
FactorRegistry.register('momentum_10', calc_factor_momentum_10, 'momentum')
FactorRegistry.register('momentum_20', calc_factor_momentum_20, 'momentum')
FactorRegistry.register('atr', calc_factor_atr, 'volatility')
