"""
多级别结构分析器 — 缠中说禅中枢与级别理论实现

核心概念：
- K线重叠 → 中枢级别判定（3K=1F, 4-9K=5F, 20K+=30F在日线图）
- 中枢区域检测（pivot zone）
- 趋势结构分类（完成/未完成/中阴）
- 多级别EMA对齐分数
- 三类买卖点识别

纯 NumPy 实现，与现有 factor_calculator 风格一致。
"""
import numpy as np
from typing import Dict, Optional


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])
    return result


def detect_kline_overlap(
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """
    计算每个位置的连续K线重叠数（缠论级别判定基础）。

    缠论规则（日线图）:
    - 3K重叠 ≈ 1F中枢
    - 4-9K重叠 ≈ 5F中枢
    - 10-19K重叠 ≈ 15F中枢
    - 20K+重叠 ≈ 30F中枢

    重叠定义: min(high[i], high[i-1]) - max(low[i], low[i-1]) > 0

    Returns:
        overlap_count: 每个位置的重叠K线数（0=无重叠，N=连续N根K线重叠）
    """
    n = len(high)
    overlap_count = np.zeros(n, dtype=int)

    if n < 2:
        return overlap_count

    for i in range(1, n):
        # 检查与前一K线是否重叠
        overlap_high = min(high[i], high[i - 1])
        overlap_low = max(low[i], low[i - 1])
        if overlap_high > overlap_low:
            overlap_count[i] = overlap_count[i - 1] + 1 if overlap_count[i - 1] > 0 else 2
        else:
            overlap_count[i] = 0

    return overlap_count


def _kline_level(count: int) -> int:
    """根据重叠K线数判定中枢级别"""
    if count >= 20:
        return 30
    elif count >= 10:
        return 15
    elif count >= 4:
        return 5
    elif count >= 3:
        return 1
    return 0


def detect_pivot_zone(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_overlap: int = 3,
    zone_buffer: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    检测中枢（pivot/consolidation）区域。

    中枢定义: 至少三个连续次级别走势重叠的区域。
    在日线图上，用K线重叠延伸区域近似中枢。

    Returns:
        pivot_present: bool数组，是否处于中枢中
        pivot_top: float数组，中枢上沿
        pivot_bottom: float数组，中枢下沿
        pivot_mid: float数组，中枢中轴
        pivot_level: int数组，中枢级别 (0/1/5/15/30)
        pivot_count: int数组，构成中枢的连续重叠K线数
    """
    n = len(high)
    pivot_present = np.zeros(n, dtype=bool)
    pivot_top = np.full(n, np.nan)
    pivot_bottom = np.full(n, np.nan)
    pivot_mid = np.full(n, np.nan)
    pivot_level = np.zeros(n, dtype=int)
    pivot_count = np.zeros(n, dtype=int)

    if n < min_overlap:
        return {
            'pivot_present': pivot_present, 'pivot_top': pivot_top,
            'pivot_bottom': pivot_bottom, 'pivot_mid': pivot_mid,
            'pivot_level': pivot_level, 'pivot_count': pivot_count,
        }

    overlap = detect_kline_overlap(high, low)

    # 向前填充中枢信息到所有重叠K线
    i = min_overlap
    while i < n:
        if overlap[i] >= min_overlap:
            # 找到重叠段的起点
            start = i
            while start > 0 and overlap[start] >= min_overlap:
                start -= 1
            start += 1

            # 找到重叠段的终点
            end = i
            while end < n and (end == start or overlap[end] > 0):
                end += 1

            # 计算中枢边界
            seg_high = high[start:end]
            seg_low = low[start:end]
            seg_close = close[start:end]

            if len(seg_high) >= min_overlap:
                pivot_h = np.percentile(seg_high, 90)  # 使用90分位减少噪声
                pivot_l = np.percentile(seg_low, 10)
                pivot_c = np.mean(seg_close)
                lvl = _kline_level(end - start)

                for j in range(start, end):
                    pivot_present[j] = True
                    pivot_top[j] = pivot_h
                    pivot_bottom[j] = pivot_l
                    pivot_mid[j] = pivot_c
                    pivot_level[j] = lvl
                    pivot_count[j] = end - start

            i = end
        else:
            i += 1

    return {
        'pivot_present': pivot_present,
        'pivot_top': pivot_top,
        'pivot_bottom': pivot_bottom,
        'pivot_mid': pivot_mid,
        'pivot_level': pivot_level,
        'pivot_count': pivot_count,
    }


def classify_trend_structure(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema20: np.ndarray,
    ema60: np.ndarray,
    pivot_info: Dict[str, np.ndarray],
    min_trend_bars: int = 8,
    zhongyin_threshold: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    分类趋势结构（走势终完美 + 中阴识别）。

    检查:
    1. 趋势方向 (ema20 vs ema60)
    2. 是否至少形成了一个中枢 (走势终完美)
    3. 结构是否完成 (足够K线超越中枢)
    4. 中阴状态 (EMA交织，方向不明)

    Returns:
        trend_up: bool数组，当前趋势向上
        structure_complete: bool数组，趋势结构基本完成
        zhongyin: bool数组，处于中阴/过渡状态
        breakout_above_pivot: bool数组，突破中枢上沿
        breakout_below_pivot: bool数组，跌破中枢下沿
        pivot_distance: float数组，到中枢边界的百分比距离
    """
    n = len(close)
    trend_up = np.zeros(n, dtype=bool)
    structure_complete = np.zeros(n, dtype=bool)
    zhongyin = np.zeros(n, dtype=bool)
    breakout_above = np.zeros(n, dtype=bool)
    breakout_below = np.zeros(n, dtype=bool)
    pivot_distance = np.zeros(n)

    pivot_present = pivot_info['pivot_present']
    pivot_top = pivot_info['pivot_top']
    pivot_bottom = pivot_info['pivot_bottom']
    pivot_level = pivot_info['pivot_level']

    for i in range(max(20, min_trend_bars), n):
        # 趋势方向
        trend_up[i] = ema20[i] > ema60[i]

        # 中阴判断: EMA间距很小 + 中枢存在
        ema_spread = abs(ema20[i] - ema60[i]) / (close[i] + 1e-10)
        if ema_spread < zhongyin_threshold and pivot_present[i]:
            zhongyin[i] = True
        elif ema_spread < zhongyin_threshold * 1.5:
            zhongyin[i] = True

        # 结构完成: 中枢形成 + 价格已离开中枢 + 有足够趋势K线
        if pivot_present[i]:
            pivot_top_i = pivot_top[i]
            pivot_bottom_i = pivot_bottom[i]

            # 中枢之上有趋势延续段 = 向上结构完整
            if trend_up[i] and not np.isnan(pivot_top_i):
                above_pivot = close[i] > pivot_top_i * (1 + 0.01)
                breakout_above[i] = above_pivot
                # 检查中枢前有上升，中枢后有离开 = 至少形成了a+A+b结构
                if above_pivot and pivot_level[i] >= 5:
                    structure_complete[i] = True
                # 到中枢边界距离
                pivot_distance[i] = (close[i] - pivot_top_i) / (close[i] + 1e-10)

            # 中枢之下 = 向下结构完整
            elif not trend_up[i] and not np.isnan(pivot_bottom_i):
                below_pivot = close[i] < pivot_bottom_i * (1 - 0.01)
                breakout_below[i] = below_pivot
                if below_pivot and pivot_level[i] >= 5:
                    structure_complete[i] = True
                pivot_distance[i] = (pivot_bottom_i - close[i]) / (close[i] + 1e-10)

    return {
        'trend_up': trend_up,
        'structure_complete': structure_complete,
        'zhongyin': zhongyin,
        'breakout_above_pivot': breakout_above,
        'breakout_below_pivot': breakout_below,
        'pivot_distance': pivot_distance,
    }


def compute_multi_level_alignment(
    close: np.ndarray,
    ema20: np.ndarray,
    ema60: np.ndarray,
    ema120: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算多级别EMA对齐分数。

    缠论核心: 大级别定方向，小级别定买卖点。
    - ema20 > ema60 > ema120: 多头排列，强趋势向上 → 正分
    - ema20 < ema60 < ema120: 空头排列，强趋势向下 → 负分
    - 交叉缠绕: 方向不明确 → 0附近

    Returns:
        alignment_score: [-1, 1]范围
            > 0.5: 强烈看涨排列
            < -0.5: 强烈看跌排列
            0附近: 无明显趋势（中阴）
    """
    n = len(close)
    alignment = np.zeros(n)

    # 如果没有ema120，用MA60近似代替（大级别方向）
    if ema120 is None:
        # 用120日简单移动平均近似
        ema120 = _sma(close, min(120, n))

    for i in range(20, n):
        if np.isnan(ema20[i]) or np.isnan(ema60[i]):
            continue

        score = 0.0

        # ema20 vs ema60: 短期方向
        mid_slope = (ema20[i] - ema20[max(0, i - 5)]) / (ema20[max(0, i - 5)] + 1e-10)
        if ema20[i] > ema60[i]:
            score += 0.3 * min(1.0, abs(mid_slope) * 100)
        else:
            score -= 0.3 * min(1.0, abs(mid_slope) * 100)

        # ema60 vs ema120: 中期方向
        if ema120 is not None and not np.isnan(ema120[i]):
            long_slope = (ema60[i] - ema60[max(0, i - 10)]) / (ema60[max(0, i - 10)] + 1e-10)
            if ema60[i] > ema120[i]:
                score += 0.3 * min(1.0, abs(long_slope) * 50)
            else:
                score -= 0.3 * min(1.0, abs(long_slope) * 50)

        # 全排列检查 (ema20 > ema60 > ema120)
        if ema120 is not None and not np.isnan(ema120[i]):
            if ema20[i] > ema60[i] > ema120[i]:
                score += 0.4
            elif ema20[i] < ema60[i] < ema120[i]:
                score -= 0.4

        # EMA间距: 间距越大，趋势越强
        spread = (ema20[i] - ema60[i]) / (close[i] + 1e-10)
        score += np.clip(spread * 20, -0.2, 0.2)

        alignment[i] = np.clip(score, -1.0, 1.0)

    return alignment


def compute_structure_signal(
    structure_info: Dict[str, np.ndarray],
    alignment: np.ndarray,
    pivot_info: Dict[str, np.ndarray],
    idx: int,
) -> Dict[str, float]:
    """
    在时间点 idx 提取结构信号。

    Returns:
        pivot_present: 0/1
        pivot_level: 0/1/5/15/30
        structure_complete: 0/1
        zhongyin: 0/1
        alignment_score: [-1, 1]
        breakout_strength: [0, 1]
        buy_point_type: 0/1/2/3
        sell_point_type: 0/1/2/3
    """
    result = {
        'pivot_present': 0.0,
        'pivot_level': 0.0,
        'structure_complete': 0.0,
        'zhongyin': 0.0,
        'alignment_score': 0.0,
        'breakout_strength': 0.0,
        'buy_point_type': 0,
        'sell_point_type': 0,
    }

    if idx < 20:
        return result

    pp = pivot_info.get('pivot_present', np.zeros(1))
    result['pivot_present'] = float(pp[idx]) if idx < len(pp) else 0.0

    pl = pivot_info.get('pivot_level', np.zeros(1))
    result['pivot_level'] = float(pl[idx]) if idx < len(pl) else 0.0

    sc = structure_info.get('structure_complete', np.zeros(1))
    result['structure_complete'] = float(sc[idx]) if idx < len(sc) else 0.0

    zy = structure_info.get('zhongyin', np.zeros(1))
    result['zhongyin'] = float(zy[idx]) if idx < len(zy) else 0.0

    result['alignment_score'] = float(alignment[idx]) if idx < len(alignment) else 0.0

    ba = structure_info.get('breakout_above_pivot', np.zeros(1))
    result['breakout_strength'] = float(ba[idx]) if idx < len(ba) else 0.0

    # 三类买卖点识别
    # 1买条件: 中枢下跌后底背离（需要在divergence_detector中配合使用）
    # 2买条件: 第一次上攻后回调不破1买低点
    # 3买条件: 突破中枢上沿 + 回调不破中枢
    if result['pivot_present'] > 0:
        ba_val = float(ba[idx]) if idx < len(ba) else 0.0
        bb = structure_info.get('breakout_below_pivot', np.zeros(1))
        bb_val = float(bb[idx]) if idx < len(bb) else 0.0

        if ba_val > 0:
            # 突破中枢上沿 = 可能的3买
            result['buy_point_type'] = 3
        if bb_val > 0:
            result['sell_point_type'] = 3

    return result
