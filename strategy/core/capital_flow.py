"""
缠论系统2: 资金流向 (Capital Flow) — 独立选股系统

与系统1(技术指标)完全独立，只基于价量关系推断资金动向。

核心指标:
1. 主力资金流向 (smart_money_flow) — 量价关系推断主力意图
2. 量能积累 (volume_accumulation) — 放量滞涨/缩量下跌 = 吸筹
3. 大单代理 (large_order_proxy) — 振幅×量/换手率
4. 净流向 (net_flow_direction) — 5日累计 (close-open)*volume 方向
5. 量价突破 (vol_price_breakout) — 缩量盘整后放量突破
6. 洗盘检测 (wash_sale_score) — 下跌缩量+快速回升

输出: capital_flow_score ∈ [0, 1]，独立于系统1的技术指标评分。
"""

import numpy as np
from typing import Dict, Optional

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动均值 (cumsum实现)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cs = np.cumsum(np.insert(arr.astype(float), 0, 0))
    result[window - 1:] = (cs[window:] - cs[:-window]) / window
    return result


@njit
def _ema_smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
    """EMA平滑 (Numba JIT)"""
    result = arr.copy()
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动求和"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cs = np.cumsum(np.insert(arr.astype(float), 0, 0))
    result[window - 1:] = cs[window:] - cs[:-window]
    return result


def compute_capital_flow_signal(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    open_arr: np.ndarray,
    turnover_rate: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    计算资金流向独立信号 (向量化版本)。
    """
    n = len(close)

    capital_flow_score = np.zeros(n)
    smart_money = np.zeros(n)
    accumulation = np.zeros(n)
    net_flow = np.zeros(n)
    breakout = np.zeros(n)
    direction = np.zeros(n, dtype=int)

    if n < 20:
        return {
            'capital_flow_score': capital_flow_score,
            'smart_money_score': smart_money,
            'accumulation_score': accumulation,
            'net_flow_score': net_flow,
            'breakout_score': breakout,
            'capital_flow_direction': direction,
        }

    # 日收益率
    ret_1d = np.zeros(n)
    ret_1d[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)

    # 20日均量
    vol_ma20 = _rolling_mean(volume, 20)
    vr = np.ones(n)
    valid_mask = ~np.isnan(vol_ma20) & (vol_ma20 > 0)
    vr[valid_mask] = volume[valid_mask] / vol_ma20[valid_mask]

    # === 1. 主力资金流向 (向量化) ===
    # 每日贡献分
    daily_sm = np.zeros(n)
    # 放量上涨: r>0, vr>1.2
    mask1 = (ret_1d > 0) & (vr > 1.2)
    daily_sm[mask1] = ret_1d[mask1] * np.clip(vr[mask1], 0, 3.0) * 2
    # 缩量上涨: r>0, vr<0.7
    mask2 = (ret_1d > 0) & (vr < 0.7) & ~mask1
    daily_sm[mask2] = ret_1d[mask2] * 0.5
    # 放量下跌: r<0, vr>1.5
    mask3 = (ret_1d < 0) & (vr > 1.5)
    daily_sm[mask3] = ret_1d[mask3] * 0.3
    # 缩量下跌: r<0, vr<0.7
    mask4 = (ret_1d < 0) & (vr < 0.7) & ~mask3
    daily_sm[mask4] = ret_1d[mask4] * 1.0
    # 其他
    mask_other = ~(mask1 | mask2 | mask3 | mask4)
    daily_sm[mask_other] = ret_1d[mask_other]

    smart_money_5d = _rolling_sum(daily_sm, 5)
    smart_money[5:] = smart_money_5d[5:] / 5.0

    # === 2. 量能积累检测 (半向量化) ===
    for i in range(20, n):
        acc_score = 0.0
        price_change_10 = (close[i] - close[i-10]) / (close[i-10] + 1e-10)
        vol_trend = np.mean(volume[i-9:i+1]) / (np.mean(volume[i-19:i-9]) + 1e-10)

        # 模式A: 放量滞涨
        if abs(price_change_10) < 0.05 and vol_trend > 1.3:
            acc_score += 0.6
        elif abs(price_change_10) < 0.08 and vol_trend > 1.1:
            acc_score += 0.35

        # 模式B: 缩量下跌后放量回升
        if price_change_10 < -0.03 and vol_trend < 0.8:
            vol_recent = np.mean(volume[i-4:i+1])
            if vol_recent > np.mean(volume[i-9:i-4]) * 1.2 and ret_1d[i] > 0:
                acc_score += 0.4

        # 模式C: 低换手率缩量横盘
        if turnover_rate is not None:
            avg_turnover = np.mean(turnover_rate[i-9:i+1])
            if avg_turnover < 0.01 and abs(price_change_10) < 0.03:
                acc_score += 0.25

        accumulation[i] = np.clip(acc_score, 0, 1)

    # === 3. 净资金流向 (向量化) ===
    day_flow = (close - open_arr) * volume
    avg_vol_20 = _rolling_mean(volume, 20) + 1e-10
    daily_nf = day_flow / avg_vol_20
    nf_5d = _rolling_sum(daily_nf, 5)
    net_flow[5:] = nf_5d[5:] / 5.0

    # === 4. 量价突破 (向量化) ===
    avg_vol_20_arr = _rolling_mean(volume, 20)
    recent_high_20 = np.full(n, np.nan)
    if n >= 20:
        from numpy.lib.stride_tricks import sliding_window_view
        sw_high = sliding_window_view(high, 20)
        recent_high_20[19:] = sw_high.max(axis=1)

    # 放量突破
    bk_mask1 = (volume > avg_vol_20_arr * 1.5) & (close >= recent_high_20 * 0.98) & (close > open_arr)
    breakout[bk_mask1] = np.clip((volume[bk_mask1] / (avg_vol_20_arr[bk_mask1] + 1e-10) - 1.5) * 0.5 + 0.6, 0, 1)
    # 温和突破
    bk_mask2 = (volume > avg_vol_20_arr * 1.2) & (close >= recent_high_20 * 0.95) & ~bk_mask1
    breakout[bk_mask2] = 0.35

    # === 综合资金流向评分 (向量化) ===
    sm_norm = np.clip(smart_money * 4 + 0.5, 0, 1)
    # 向量化替代 per-bar np.std loop
    nf_std = np.full(n, 1.0)
    if n >= 61:
        from numpy.lib.stride_tricks import sliding_window_view
        sw_nf = sliding_window_view(net_flow, 61)
        nf_std[60:] = np.std(sw_nf, axis=1) + 1e-10
    nf_norm = np.clip(net_flow / nf_std * 0.15 + 0.5, 0, 1)
    acc_norm = accumulation
    bk_norm = breakout

    flow_sum = sm_norm * 0.3 + nf_norm * 0.3 + acc_norm * 0.2 + bk_norm * 0.2
    capital_flow_score = np.clip(flow_sum, 0, 1)

    # 方向判定
    direction[flow_sum > 0.55] = 1
    direction[flow_sum < 0.35] = -1

    # 3日EMA平滑 (JIT)
    capital_flow_score = _ema_smooth(capital_flow_score, 2.0 / 4.0)

    return {
        'capital_flow_score': capital_flow_score,
        'smart_money_score': smart_money,
        'accumulation_score': accumulation,
        'net_flow_score': net_flow,
        'breakout_score': breakout,
        'capital_flow_direction': direction,
    }
