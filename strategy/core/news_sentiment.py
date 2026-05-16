"""
缠论系统3: 资讯热点 (News Sentiment) — 独立选股系统 (价量代理)

由于无外部新闻数据源，使用价量异常作为资讯冲击的代理指标:

核心代理指标:
1. 跳空缺口 (gap_impact) — 隔夜信息冲击的直接体现
2. 量异动 (volume_shock) — 异常放量 = 信息驱动交易
3. 异常收益 (abnormal_return) — 超过3σ的收益 = 资讯冲击
4. 涨跌停距离 (limit_proximity) — A股特色，涨停板附近的信息聚集
5. 缺口持续性 (gap_persistence) — 缺口是否回补 = 信息持续影响力

输出: news_sentiment_score ∈ [0, 1]，独立于系统1和系统2。
"""

import numpy as np
from typing import Dict, Optional


def compute_news_sentiment_signal(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    open_arr: np.ndarray,
    amplitude: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """计算资讯热点独立信号（价量代理, 向量化版本）"""
    n = len(close)
    news_score = np.zeros(n)
    gap_signal = np.zeros(n)
    vol_shock = np.zeros(n)
    abnormal_ret = np.zeros(n)
    direction = np.zeros(n, dtype=int)

    if n < 20:
        return {
            'news_sentiment_score': news_score,
            'gap_signal': gap_signal,
            'volume_shock_signal': vol_shock,
            'abnormal_ret_signal': abnormal_ret,
            'news_sentiment_direction': direction,
        }

    # 日收益率
    ret_1d = np.zeros(n)
    ret_1d[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)

    # 隔夜跳空
    overnight_gap = np.zeros(n)
    overnight_gap[1:] = (open_arr[1:] - close[:-1]) / (close[:-1] + 1e-10)

    # === 1. 跳空缺口检测 (向量化) ===
    abs_gap = np.abs(overnight_gap)
    gap_signal = np.select(
        [abs_gap > 0.05, abs_gap > 0.03, abs_gap > 0.015, abs_gap > 0.005],
        [0.9, 0.65 + (abs_gap - 0.03) * 12.5, 0.3 + (abs_gap - 0.015) * 23.3, abs_gap * 30],
        default=0.0
    )

    # === 2. 量异动检测 (向量化) ===
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        cs = np.cumsum(np.insert(volume.astype(float), 0, 0))
        vol_ma20[19:] = (cs[20:] - cs[:-20]) / 20
    vol_ratio = volume / (vol_ma20 + 1e-10)
    vol_shock = np.select(
        [vol_ratio >= 4.0, vol_ratio >= 3.0, vol_ratio >= 2.0, vol_ratio >= 1.5],
        [0.9, 0.6 + (vol_ratio - 3.0) * 0.3, 0.3 + (vol_ratio - 2.0) * 0.3, 0.1 + (vol_ratio - 1.5) * 0.4],
        default=0.0
    )
    vol_shock[:20] = 0

    # === 3. 异常收益 (向量化) ===
    ret_std = np.full(n, np.nan)
    if n >= 20:
        # rolling std using cumsum
        ret_sma = np.full(n, np.nan)
        ret_sq_sma = np.full(n, np.nan)
        cs = np.cumsum(np.insert(ret_1d.astype(float), 0, 0))
        cs_sq = np.cumsum(np.insert(ret_1d ** 2, 0, 0))
        ret_sma[19:] = (cs[20:] - cs[:-20]) / 20
        ret_sq_sma[19:] = (cs_sq[20:] - cs_sq[:-20]) / 20
        ret_std[19:] = np.sqrt(np.maximum(0, ret_sq_sma[19:] - ret_sma[19:] ** 2))

    z_score = np.abs(ret_1d) / (ret_std + 1e-10)
    abnormal_ret = np.select(
        [z_score >= 3.0, z_score >= 2.5, z_score >= 2.0, z_score >= 1.5],
        [0.85, 0.55 + (z_score - 2.5) * 0.6, 0.25 + (z_score - 2.0) * 0.6, z_score * 0.17],
        default=0.0
    )
    abnormal_ret[:20] = 0

    # === 4. 涨跌停距离 (向量化) ===
    daily_ret_abs = np.abs(ret_1d)
    limit_prox = np.select(
        [daily_ret_abs > 0.07, daily_ret_abs > 0.05],
        [0.5 + (daily_ret_abs - 0.07) * 16.7, 0.2 + (daily_ret_abs - 0.05) * 15],
        default=0.0
    )

    # === 5. 缺口持续性 (半向量化) ===
    gap_persistence = np.zeros(n)
    for i in range(3, n):
        gap = overnight_gap[i-3]
        if abs(gap) < 0.01:
            continue
        pre_close = close[i-4] if i >= 4 else close[i-3]
        gap_filled = False
        for j in range(i-2, i+1):
            if gap > 0 and low[j] <= pre_close * 1.005:
                gap_filled = True
                break
            elif gap < 0 and high[j] >= pre_close * 0.995:
                gap_filled = True
                break
        if not gap_filled:
            gap_persistence[i] = min(0.7, abs(gap) * 10 + 0.2)

    # === 综合资讯冲击评分 (向量化) ===
    news_score = (
        gap_signal * 0.25 +
        vol_shock * 0.25 +
        abnormal_ret * 0.20 +
        limit_prox * 0.15 +
        gap_persistence * 0.15
    )
    news_score = np.clip(news_score, 0, 1)

    # 方向判定
    direction[(news_score > 0.3) & (ret_1d > 0) & (overnight_gap > 0)] = 1
    direction[(news_score > 0.3) & (ret_1d < 0) & (overnight_gap < 0)] = -1

    return {
        'news_sentiment_score': news_score,
        'gap_signal': gap_signal,
        'volume_shock_signal': vol_shock,
        'abnormal_ret_signal': abnormal_ret,
        'news_sentiment_direction': direction,
    }
