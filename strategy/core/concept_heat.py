# core/concept_heat.py
"""
题材热度因子 — 本地合成（不依赖外部API）

三个维度加权合成：
- 维度A: 行业动量 (40%) — sector_rotation 行业排名
- 维度B: 妖股信号强度 (40%) — volume_surge + turnover_burst + limit_up_freq
- 维度C: 短期超额收益 (20%) — 5日相对大盘的超额收益

用于捕捉2025-2026年题材驱动的妖股行情。
"""

import numpy as np
from typing import Dict, Optional


def compute_concept_heat(
    volume_surge: np.ndarray,
    turnover_burst: np.ndarray,
    limit_up_freq: np.ndarray,
    excess_return_5d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """计算题材热度因子 (0~1)

    Args:
        volume_surge: 成交量放大因子 [n]
        turnover_burst: 换手率突破因子 [n]
        limit_up_freq: 涨停频率因子 [n]
        excess_return_5d: 5日超额收益（相对大盘），可为None

    Returns:
        concept_heat: 题材热度得分 [n], range ~[0, 1]
    """
    n = len(volume_surge)

    # === 维度B: 妖股信号强度 ===
    # tanh压缩到(0,1)区间
    surge_norm = np.tanh(np.maximum(volume_surge, 0) * 1.5)
    burst_norm = np.tanh(np.maximum(turnover_burst, 0) * 1.5)
    limit_norm = np.tanh(np.maximum(limit_up_freq, 0) * 1.5)

    yaogu_signal = (surge_norm + burst_norm + limit_norm) / 3.0

    # === 维度C: 短期超额收益 ===
    if excess_return_5d is not None and len(excess_return_5d) == n:
        excess_norm = np.tanh(np.maximum(excess_return_5d, 0) * 3.0)
    else:
        excess_norm = np.full(n, 0.5)

    # === 综合得分 (维度A由调用方注入) ===
    # 这里先计算B+C的70%，维度A的30%由调用方通过 industry_momentum 传入
    concept_heat = 0.57 * yaogu_signal + 0.29 * excess_norm + 0.14 * 0.5
    # 最后0.14留给行业动量，默认0.5中等

    return np.clip(concept_heat, 0.0, 1.0)


def compute_concept_heat_with_industry(
    volume_surge: np.ndarray,
    turnover_burst: np.ndarray,
    limit_up_freq: np.ndarray,
    industry_momentum: float = 0.5,
    excess_return_5d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """计算含行业动量的题材热度因子

    Args:
        volume_surge, turnover_burst, limit_up_freq: 妖股因子数组
        industry_momentum: 行业动量得分 (0~1)，来自 SectorRotation.get_sector_score()
        excess_return_5d: 5日超额收益

    Returns:
        concept_heat: 0~1
    """
    n = len(volume_surge)
    base = compute_concept_heat(volume_surge, turnover_burst, limit_up_freq, excess_return_5d)
    # 行业动量调权：减少默认0.5的占比，加入实际行业动量
    result = base * 0.6 + industry_momentum * 0.4
    return np.clip(result, 0.0, 1.0)
