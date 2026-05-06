"""
MACD 背离检测器 — 缠中说禅核心理论实现

核心概念：
- 顶背离 (Top Divergence): 价格创新高，MACD柱状图未创新高 → 卖出信号
- 底背离 (Bottom Divergence): 价格创新低，MACD柱状图未创新低 → 买入信号
- 隐藏背离: 趋势延续中的反向确认
- "没有趋势，没有背驰" — 背离成立前验证趋势结构

纯 NumPy 实现，与现有 factor_calculator 风格一致。
"""
import numpy as np
from typing import Dict, Tuple, Optional


def detect_peak_trough(
    arr: np.ndarray,
    lookback: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测一维数组中的局部峰值和谷值。

    Args:
        arr: 输入数组（如价格或MACD柱状图）
        lookback: 每侧检查的点数（窗口 = 2*lookback + 1）

    Returns:
        peaks: bool数组，峰值位置为 True
        troughs: bool数组，谷值位置为 True
    """
    n = len(arr)
    peaks = np.zeros(n, dtype=bool)
    troughs = np.zeros(n, dtype=bool)

    if n < 2 * lookback + 1:
        return peaks, troughs

    for i in range(lookback, n - lookback):
        if np.isnan(arr[i]):
            continue
        window = arr[i - lookback : i + lookback + 1]
        if arr[i] == np.nanmax(window) and np.sum(~np.isnan(window)) >= 3:
            peaks[i] = True
        if arr[i] == np.nanmin(window) and np.sum(~np.isnan(window)) >= 3:
            troughs[i] = True

    return peaks, troughs


def _find_previous_peak(arr: np.ndarray, peaks: np.ndarray, idx: int) -> Optional[int]:
    """在 idx 之前查找最近的峰值索引"""
    for i in range(idx - 1, -1, -1):
        if peaks[i] and not np.isnan(arr[i]):
            return i
    return None


def _find_previous_trough(arr: np.ndarray, troughs: np.ndarray, idx: int) -> Optional[int]:
    """在 idx 之前查找最近的谷值索引"""
    for i in range(idx - 1, -1, -1):
        if troughs[i] and not np.isnan(arr[i]):
            return i
    return None


def _has_trend_structure(
    price: np.ndarray,
    ema20: np.ndarray,
    ema60: np.ndarray,
    idx: int,
    divergence_type: str,
    min_bars: int = 10,
) -> bool:
    """
    验证趋势结构是否存在（"没有趋势，没有背驰"）。

    顶背离前提：上升趋势（ema20 > ema60，且近期有一段上涨）
    底背离前提：下降趋势（ema20 < ema60，且近期有一段下跌）
    """
    if idx < min_bars:
        return False

    if divergence_type == 'top':
        # 需要上升趋势
        trend_up = ema20[idx] > ema60[idx]
        if not trend_up:
            return False
        # 还需要近期有上涨段（价格高于20天前）
        recent_up = price[idx] > np.nanmean(price[max(0, idx - 20):idx])
        return recent_up
    elif divergence_type == 'bottom':
        # 需要下降趋势
        trend_down = ema20[idx] < ema60[idx]
        if not trend_down:
            return False
        recent_down = price[idx] < np.nanmean(price[max(0, idx - 20):idx])
        return recent_down

    return True


def compute_divergence(
    price: np.ndarray,
    macd_hist: np.ndarray,
    ema20: Optional[np.ndarray] = None,
    ema60: Optional[np.ndarray] = None,
    lookback: int = 20,
    peak_trough_lookback: int = 5,
    strength_threshold: float = 0.3,
    verify_trend: bool = True,
) -> Dict[str, np.ndarray]:
    """
    检测 MACD 背离。

    Args:
        price: 收盘价数组
        macd_hist: MACD 柱状图 (macd_line - signal_line)
        ema20: 20日EMA（趋势验证用）
        ema60: 60日EMA（趋势验证用）
        lookback: 背离检测的回看窗口
        peak_trough_lookback: 极值检测的单侧点数
        strength_threshold: 最小背离强度
        verify_trend: 是否验证趋势结构

    Returns:
        字典包含:
        - top_divergence: float数组，顶背离强度 [0, 1]
        - bottom_divergence: float数组，底背离强度 [0, 1]
        - hidden_top: float数组，隐藏顶背离
        - hidden_bottom: float数组，隐藏底背离
        - divergence_active: bool数组
    """
    n = len(price)
    top_div = np.zeros(n)
    bottom_div = np.zeros(n)
    hidden_top = np.zeros(n)
    hidden_bottom = np.zeros(n)
    active = np.zeros(n, dtype=bool)

    if n < lookback * 2:
        return {
            'top_divergence': top_div,
            'bottom_divergence': bottom_div,
            'hidden_top': hidden_top,
            'hidden_bottom': hidden_bottom,
            'divergence_active': active,
        }

    price_peaks, price_troughs = detect_peak_trough(price, peak_trough_lookback)
    macd_peaks, macd_troughs = detect_peak_trough(macd_hist, peak_trough_lookback)

    # 安全检查：如果没有ema数组，创建默认值
    if ema20 is None:
        ema20 = np.full(n, price[0])
    if ema60 is None:
        ema60 = np.full(n, price[0])

    for i in range(lookback, n):
        if np.isnan(price[i]) or np.isnan(macd_hist[i]):
            continue

        # 顶背离检测：价格创新高，MACD柱状图未创新高
        if price_peaks[i]:
            prev_price_peak = _find_previous_peak(price, price_peaks, i)
            prev_macd_peak = _find_previous_peak(macd_hist, macd_peaks, i)

            if prev_price_peak is not None and prev_macd_peak is not None:
                price_higher = price[i] > price[prev_price_peak]
                macd_lower = macd_hist[i] < macd_hist[prev_macd_peak]

                if price_higher and macd_lower:
                    if not verify_trend or _has_trend_structure(price, ema20, ema60, i, 'top'):
                        strength = min(1.0, (price[i] / (price[prev_price_peak] + 1e-10) - 1) * 10)
                        strength *= (macd_hist[prev_macd_peak] - macd_hist[i]) / (abs(macd_hist[prev_macd_peak]) + 1e-10)
                        strength = max(0.0, min(1.0, strength))
                        if strength >= strength_threshold:
                            top_div[i] = strength
                            active[i] = True

        # 顶背离持续信号：峰值之后的K线也标记为背离活跃
        if i > 0 and top_div[i] == 0 and top_div[i - 1] > 0:
            # 背离信号持续 lookback//2 根K线
            for j in range(1, min(lookback // 2, i)):
                if top_div[i - j] > 0:
                    top_div[i] = top_div[i - j] * 0.5  # 衰减
                    active[i] = True
                    break

        # 底背离检测：价格创新低，MACD柱状图未创新低
        if price_troughs[i]:
            prev_price_trough = _find_previous_trough(price, price_troughs, i)
            prev_macd_trough = _find_previous_trough(macd_hist, macd_troughs, i)

            if prev_price_trough is not None and prev_macd_trough is not None:
                price_lower = price[i] < price[prev_price_trough]
                macd_higher = macd_hist[i] > macd_hist[prev_macd_trough]

                if price_lower and macd_higher:
                    if not verify_trend or _has_trend_structure(price, ema20, ema60, i, 'bottom'):
                        strength = min(1.0, (price[prev_price_trough] / (price[i] + 1e-10) - 1) * 10)
                        strength *= (macd_hist[i] - macd_hist[prev_macd_trough]) / (abs(macd_hist[prev_macd_trough]) + 1e-10)
                        strength = max(0.0, min(1.0, strength))
                        if strength >= strength_threshold:
                            bottom_div[i] = strength
                            active[i] = True

        # 底背离持续信号
        if i > 0 and bottom_div[i] == 0 and bottom_div[i - 1] > 0:
            for j in range(1, min(lookback // 2, i)):
                if bottom_div[i - j] > 0:
                    bottom_div[i] = bottom_div[i - j] * 0.5
                    active[i] = True
                    break

        # 隐藏顶背离：价格未创新高，MACD创新高（趋势减弱）
        if price_peaks[i]:
            prev_price_peak = _find_previous_peak(price, price_peaks, i)
            prev_macd_peak = _find_previous_peak(macd_hist, macd_peaks, i)
            if prev_price_peak is not None and prev_macd_peak is not None:
                if price[i] <= price[prev_price_peak] and macd_hist[i] > macd_hist[prev_macd_peak]:
                    strength = min(0.8, (macd_hist[i] - macd_hist[prev_macd_peak]) / (abs(macd_hist[prev_macd_peak]) + 1e-10))
                    if strength >= strength_threshold * 0.5:
                        hidden_top[i] = strength

        # 隐藏底背离：价格未创新低，MACD创新低（趋势增强）
        if price_troughs[i]:
            prev_price_trough = _find_previous_trough(price, price_troughs, i)
            prev_macd_trough = _find_previous_trough(macd_hist, macd_troughs, i)
            if prev_price_trough is not None and prev_macd_trough is not None:
                if price[i] >= price[prev_price_trough] and macd_hist[i] < macd_hist[prev_macd_trough]:
                    strength = min(0.8, (macd_hist[prev_macd_trough] - macd_hist[i]) / (abs(macd_hist[prev_macd_trough]) + 1e-10))
                    if strength >= strength_threshold * 0.5:
                        hidden_bottom[i] = strength

    return {
        'top_divergence': top_div,
        'bottom_divergence': bottom_div,
        'hidden_top': hidden_top,
        'hidden_bottom': hidden_bottom,
        'divergence_active': active,
    }


def compute_macd_divergence_signal(
    divergence: Dict[str, np.ndarray],
    idx: int,
    min_strength: float = 0.3,
) -> Dict[str, float]:
    """
    在时间点 idx 提取背离信号。

    Args:
        divergence: compute_divergence 的返回字典
        idx: 时间索引
        min_strength: 最小信号强度阈值

    Returns:
        字典包含:
        - divergence_type: 'none'/'bottom'/'top'/'hidden_bottom'/'hidden_top'
        - divergence_strength: float，信号强度
        - is_bullish: bool，是否看涨
        - is_bearish: bool，是否看跌
    """
    top_s = float(divergence['top_divergence'][idx]) if idx < len(divergence['top_divergence']) else 0.0
    bottom_s = float(divergence['bottom_divergence'][idx]) if idx < len(divergence['bottom_divergence']) else 0.0
    htop_s = float(divergence['hidden_top'][idx]) if idx < len(divergence['hidden_top']) else 0.0
    hbottom_s = float(divergence['hidden_bottom'][idx]) if idx < len(divergence['hidden_bottom']) else 0.0

    # 优先检测标准背离
    if bottom_s >= min_strength:
        return {
            'divergence_type': 'bottom',
            'divergence_strength': bottom_s,
            'is_bullish': True,
            'is_bearish': False,
        }
    elif top_s >= min_strength:
        return {
            'divergence_type': 'top',
            'divergence_strength': top_s,
            'is_bullish': False,
            'is_bearish': True,
        }
    elif hbottom_s >= min_strength * 0.5:
        return {
            'divergence_type': 'hidden_bottom',
            'divergence_strength': hbottom_s,
            'is_bullish': True,
            'is_bearish': False,
        }
    elif htop_s >= min_strength * 0.5:
        return {
            'divergence_type': 'hidden_top',
            'divergence_strength': htop_s,
            'is_bullish': False,
            'is_bearish': True,
        }

    return {
        'divergence_type': 'none',
        'divergence_strength': 0.0,
        'is_bullish': False,
        'is_bearish': False,
    }
