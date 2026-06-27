"""
缠中说禅 (Chan Theory) 完整实现 — 基于 czsc 项目核心概念

核心概念层级:
1. 包含关系处理 → 顶底分型
2. 分型 → 笔 (Stroke/Bi)
3. 笔 → 线段 (Segment)
4. 线段 → 中枢 (Pivot/Hub)
5. 中枢 + 背离 → 三类买卖点
6. 买卖点 → 交易信号

与 czsc 对齐的核心规则:
- "没有趋势，没有背驰"
- "走势终完美" — 任何走势必包含至少一个中枢
- 三类买卖点的严格定义
- 区间套多级别联立

纯 NumPy 实现，与现有 factor_calculator 风格一致。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """向量化简单移动平均 (cumsum实现, O(n))"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cs = np.cumsum(np.insert(arr.astype(np.float64), 0, 0))
    result[window-1:] = (cs[window:] - cs[:-window]) / window
    return result


try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper


# ==================== 数据结构 ====================

@dataclass
class Fractal:
    """顶底分型"""
    idx: int          # 在原始K线中的索引
    f_type: int       # 1=顶分型, -1=底分型
    high: float
    low: float
    price: float      # 顶分型用high, 底分型用low


@dataclass
class Stroke:
    """笔"""
    start_idx: int
    end_idx: int
    direction: int    # 1=向上笔, -1=向下笔
    start_price: float
    end_price: float
    high: float       # 笔的最高价
    low: float        # 笔的最低价


@dataclass
class Segment:
    """线段"""
    start_idx: int
    end_idx: int
    direction: int    # 1=向上线段, -1=向下线段
    start_price: float
    end_price: float
    high: float
    low: float
    stroke_count: int


@dataclass
class Pivot:
    """中枢"""
    start_idx: int
    end_idx: int
    zg: float         # 中枢上沿 = min(线段high)
    zd: float         # 中枢下沿 = max(线段low)
    zz: float         # 中枢中轴 = (ZG + ZD) / 2
    level: int        # 中枢级别 (基于构成线段对应的级别)
    segment_count: int  # 构成中枢的线段数
    trend_dir: int = 0  # 中枢趋势方向: 1=上移(涨), -1=下移(跌), 0=无方向


# ==================== 1. 包含关系处理 ====================

@njit
def process_inclusions(
    high: np.ndarray,
    low: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    处理K线包含关系（缠论第65课）。Numba JIT加速。

    规则:
    - 如果第n根K线与第n-1根K线存在包含关系:
      - 若之前是上升序列: 取 high=max(high[n], high[n-1]), low=max(low[n], low[n-1])
      - 若之前是下降序列: 取 high=min(high[n], high[n-1]), low=min(low[n], low[n-1])
    - 方向由前一根非包含K线判定: high[n-1] > high[n-2] 则为上升

    Returns:
        new_high, new_low: 处理后的K线数组（可能比输入短）
        orig_idx_map: 每根新K线对应的原始索引
    """
    n = len(high)
    if n < 3:
        return high.copy(), low.copy(), np.arange(n)

    # 预分配数组（最大可能长度 = n），dtype 与输入一致避免 numba 类型不统一
    new_high = np.zeros(n, dtype=high.dtype)
    new_low = np.zeros(n, dtype=low.dtype)
    orig_map = np.arange(n, dtype=np.int64)

    new_high[0] = high[0]
    new_high[1] = high[1]
    new_low[0] = low[0]
    new_low[1] = low[1]
    count = 2  # 当前输出K线数

    # 初始方向
    if high[1] > high[0] and low[1] > low[0]:
        direction = 1
    elif high[1] < high[0] and low[1] < low[0]:
        direction = -1
    else:
        direction = 1 if high[1] >= high[0] else -1

    for i in range(2, n):
        curr_h, curr_l = high[i], low[i]
        prev_h, prev_l = new_high[count - 1], new_low[count - 1]

        is_contained = (curr_h <= prev_h and curr_l >= prev_l)
        is_containing = (curr_h >= prev_h and curr_l <= prev_l)

        if is_contained or is_containing:
            if direction == 1:
                new_high[count - 1] = max(curr_h, prev_h)
                new_low[count - 1] = max(curr_l, prev_l)
            else:
                new_high[count - 1] = min(curr_h, prev_h)
                new_low[count - 1] = min(curr_l, prev_l)
        else:
            if curr_h > prev_h and curr_l > prev_l:
                direction = 1
            elif curr_h < prev_h and curr_l < prev_l:
                direction = -1
            new_high[count] = curr_h
            new_low[count] = curr_l
            orig_map[count] = i
            count += 1

    return new_high[:count], new_low[:count], orig_map[:count]


# ==================== 2. 分型检测 ====================

def detect_fractals(
    high: np.ndarray,
    low: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    检测顶底分型（缠论第62课）。

    顶分型: 中间K线high最高, 三根无包含关系
    底分型: 中间K线low最低, 三根无包含关系

    分型序列必须交替（不允许多个同类型分型连续出现）。
    若出现连续同类型分型，保留更极端的那个。

    Returns:
        top_fractals: bool数组，顶分型为True
        bottom_fractals: bool数组，底分型为True
        fractal_type: int数组，1=顶, -1=底, 0=非分型
        fractal_high: float数组
        fractal_low: float数组
    """
    n = len(high)
    top_fractals = np.zeros(n, dtype=bool)
    bottom_fractals = np.zeros(n, dtype=bool)
    f_type = np.zeros(n, dtype=int)

    if n < 3:
        return {
            'top_fractals': top_fractals,
            'bottom_fractals': bottom_fractals,
            'fractal_type': f_type,
        }

    # 先处理包含关系
    proc_high, proc_low, orig_map = process_inclusions(high, low)
    pn = len(proc_high)

    if pn < 3:
        return {
            'top_fractals': top_fractals,
            'bottom_fractals': bottom_fractals,
            'fractal_type': f_type,
        }

    # 在包含处理后的K线上检测分型
    # 用 bar i 的数据确认 bar i-1 的分型（1-bar 延迟，符合实盘语义）
    raw_tops = []
    raw_bottoms = []

    for i in range(2, pn):
        h0, h1, h2 = proc_high[i-2], proc_high[i-1], proc_high[i]
        l0, l1, l2 = proc_low[i-2], proc_low[i-1], proc_low[i]

        # 顶分型 at i-1: proc_high[i-1] 是局部最高
        # P5 fix: 第三根K线low必须低于中间K线low，确认反转（避免假分型）
        if h1 > h0 and h1 > h2 and l2 < l1:
            raw_tops.append(i - 1)
        # 底分型 at i-1: proc_low[i-1] 是局部最低
        # P5 fix: 第三根K线high必须高于中间K线high，确认反转
        if l1 < l0 and l1 < l2 and h2 > h1:
            raw_bottoms.append(i - 1)

    # 分型交替过滤
    # 合并两个列表并按位置排序
    all_fractals = []
    for idx in raw_tops:
        all_fractals.append((idx, 1))  # 1=顶
    for idx in raw_bottoms:
        all_fractals.append((idx, -1))  # -1=底
    all_fractals.sort(key=lambda x: x[0])

    if not all_fractals:
        return {
            'top_fractals': top_fractals,
            'bottom_fractals': bottom_fractals,
            'fractal_type': f_type,
        }

    # 过滤: 同类型连续时保留更极端的
    filtered = [all_fractals[0]]
    for i in range(1, len(all_fractals)):
        idx, ftype = all_fractals[i]
        prev_idx, prev_ftype = filtered[-1]

        if ftype == prev_ftype:
            # 同类型: 顶分型保留更高的, 底分型保留更低的
            if ftype == 1:  # 顶
                if proc_high[idx] > proc_high[prev_idx]:
                    filtered[-1] = (idx, ftype)
            else:  # 底
                if proc_low[idx] < proc_low[prev_idx]:
                    filtered[-1] = (idx, ftype)
        else:
            filtered.append((idx, ftype))

    # 映射回原始K线索引
    for proc_idx, ftype in filtered:
        orig_idx = orig_map[proc_idx]
        if orig_idx < n:
            if ftype == 1:
                top_fractals[orig_idx] = True
                f_type[orig_idx] = 1
            else:
                bottom_fractals[orig_idx] = True
                f_type[orig_idx] = -1

    return {
        'top_fractals': top_fractals,
        'bottom_fractals': bottom_fractals,
        'fractal_type': f_type,
    }


# ==================== 3. 笔检测 ====================

def detect_strokes(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_stroke_bars: int = 4,
) -> Dict:
    """
    从分型检测笔（缠论第62-65课）。

    笔的定义: 连接一个顶分型和一个底分型
    条件:
    - 顶底之间至少间隔4根K线（不含包含处理）
    - 顶必须高于底
    - 顶分型和底分型必须交替

    Returns:
        strokes: Stroke对象列表
        stroke_direction: int数组，每个位置所属笔的方向
        stroke_idx: int数组，每个位置的笔编号
    """
    n = len(high)
    fractals = detect_fractals(high, low)
    top_f = fractals['top_fractals']
    bottom_f = fractals['bottom_fractals']
    f_type = fractals['fractal_type']

    stroke_dir = np.zeros(n, dtype=int)
    stroke_ids = np.full(n, -1, dtype=int)

    # 收集所有分型
    fractal_list = []
    for i in range(n):
        if top_f[i]:
            fractal_list.append(Fractal(i, 1, high[i], low[i], high[i]))
        elif bottom_f[i]:
            fractal_list.append(Fractal(i, -1, high[i], low[i], low[i]))

    if len(fractal_list) < 2:
        return {
            'strokes': [],
            'stroke_direction': stroke_dir,
            'stroke_id': stroke_ids,
            'top_fractals': top_f,
            'bottom_fractals': bottom_f,
            'fractal_type': f_type,
        }

    # 构建笔
    strokes = []
    i = 0
    while i < len(fractal_list) - 1:
        f1 = fractal_list[i]
        # 找下一个相反类型的分型
        j = i + 1
        while j < len(fractal_list):
            f2 = fractal_list[j]
            if f2.f_type != f1.f_type:
                # 检查间隔
                bars_between = f2.idx - f1.idx
                if bars_between >= min_stroke_bars:
                    # 检查价格关系: 顶-底笔(向下), 底-顶笔(向上)
                    if f1.f_type == 1 and f2.f_type == -1:
                        # 顶到底: 顶必须高于底(含K线高低点)
                        if f1.price > f2.price:
                            stroke = Stroke(
                                start_idx=f1.idx, end_idx=f2.idx,
                                direction=-1,
                                start_price=f1.price, end_price=f2.price,
                                high=max(f1.high, f2.high),
                                low=min(f1.low, f2.low),
                            )
                            strokes.append(stroke)
                            i = j
                            break
                    elif f1.f_type == -1 and f2.f_type == 1:
                        # 底到顶: 底必须低于顶
                        if f1.price < f2.price:
                            stroke = Stroke(
                                start_idx=f1.idx, end_idx=f2.idx,
                                direction=1,
                                start_price=f1.price, end_price=f2.price,
                                high=max(f1.high, f2.high),
                                low=min(f1.low, f2.low),
                            )
                            strokes.append(stroke)
                            i = j
                            break
                # 间隔不够，跳过当前f2，继续找下一个
                j += 1
            else:
                # 同类型分型，取更极端的
                j += 1

        if j >= len(fractal_list):
            break

    # 填充数组输出
    for sid, s in enumerate(strokes):
        for idx in range(s.start_idx, min(s.end_idx + 1, n)):
            stroke_dir[idx] = s.direction
            stroke_ids[idx] = sid

    # 向前填充未覆盖区域（笔开始之前）
    if strokes:
        first_dir = strokes[0].direction
        for idx in range(0, strokes[0].start_idx):
            stroke_ids[idx] = -1
    else:
        stroke_dir[:] = 0
        stroke_ids[:] = -1

    return {
        'strokes': strokes,
        'stroke_direction': stroke_dir,
        'stroke_id': stroke_ids,
        'top_fractals': top_f,
        'bottom_fractals': bottom_f,
        'fractal_type': f_type,
    }


# ==================== 4. 线段检测 ====================

def detect_segments(
    strokes: List[Stroke],
    n_bars: int,
) -> Tuple[List[Segment], np.ndarray, np.ndarray]:
    """
    从笔检测线段（缠论第67-69课）。

    线段至少包含3笔，方向由第一笔决定。
    使用特征序列法判断线段结束。

    Returns:
        segments: Segment对象列表
        segment_direction: 每根K线所属线段方向
        segment_id: 每根K线的线段编号
    """
    seg_dir = np.zeros(n_bars, dtype=int)
    seg_ids = np.full(n_bars, -1, dtype=int)

    if len(strokes) < 3:
        return [], seg_dir, seg_ids

    segments = []
    i = 0

    while i < len(strokes) - 2:
        s1 = strokes[i]
        s2 = strokes[i + 1]
        s3 = strokes[i + 2]

        # 前三笔必须方向交替
        if s1.direction == s2.direction or s2.direction == s3.direction:
            i += 1
            continue

        # 线段方向 = 第一笔方向
        seg_dir_val = s1.direction
        seg_start = s1.start_idx
        seg_end_idx = i + 2  # strokes索引

        # P1 fix: 特征序列法 — 替代硬编码0.95/1.05百分比
        # 向上线段：特征序列=向下笔序列，向下笔破前向下笔低点→线段结束
        # 向下线段：特征序列=向上笔序列，向上笔破前向上笔高点→线段结束
        if seg_dir_val == 1:
            # 向上线段：追踪特征序列(向下笔)的低点
            last_down_low = None
            last_down_idx = None
            for j in range(i, len(strokes)):
                sj = strokes[j]
                if sj.direction == 1:
                    # 向上笔：突破前高则延续
                    prev_up = strokes[seg_end_idx] if seg_end_idx < j else sj
                    if sj.high > prev_up.high:
                        seg_end_idx = j
                        last_down_low = None  # 新高重置特征序列
                else:  # sj.direction == -1 向下笔
                    if last_down_low is None:
                        last_down_low = sj.low
                        last_down_idx = j
                    elif sj.low < last_down_low:
                        # 向下笔破前向下笔低点 → 特征序列顶分型 → 线段结束
                        seg_end_idx = j - 1
                        break
                    else:
                        # 未破前低，更新最新向下笔
                        last_down_low = sj.low
                        last_down_idx = j
        else:  # seg_dir_val == -1 向下线段
            # 向下线段：追踪特征序列(向上笔)的高点
            last_up_high = None
            last_up_idx = None
            for j in range(i, len(strokes)):
                sj = strokes[j]
                if sj.direction == -1:
                    # 向下笔：破前低则延续
                    prev_down = strokes[seg_end_idx] if seg_end_idx < j else sj
                    if sj.low < prev_down.low:
                        seg_end_idx = j
                        last_up_high = None  # 新低重置特征序列
                else:  # sj.direction == 1 向上笔
                    if last_up_high is None:
                        last_up_high = sj.high
                        last_up_idx = j
                    elif sj.high > last_up_high:
                        # 向上笔破前向上笔高点 → 特征序列底分型 → 线段结束
                        seg_end_idx = j - 1
                        break
                    else:
                        # 未破前高，更新最新向上笔
                        last_up_high = sj.high
                        last_up_idx = j

        # 创建线段
        seg_strokes = strokes[i:seg_end_idx + 1]
        seg_highs = [s.high for s in seg_strokes]
        seg_lows = [s.low for s in seg_strokes]

        seg = Segment(
            start_idx=seg_start,
            end_idx=strokes[seg_end_idx].end_idx,
            direction=seg_dir_val,
            start_price=seg_strokes[0].start_price,
            end_price=seg_strokes[-1].end_price,
            high=max(seg_highs),
            low=min(seg_lows),
            stroke_count=len(seg_strokes),
        )
        segments.append(seg)

        i = seg_end_idx + 1  # 线段之间可以共用边界笔

        if i >= len(strokes) - 2:
            break

    # 填充数组
    for sid, seg in enumerate(segments):
        for idx in range(seg.start_idx, min(seg.end_idx + 1, n_bars)):
            seg_dir[idx] = seg.direction
            seg_ids[idx] = sid

    return segments, seg_dir, seg_ids


# ==================== 5. 中枢检测 (从线段) ====================

def detect_chan_pivots(
    segments: List[Segment],
    n_bars: int,
) -> Tuple[List[Pivot], np.ndarray, np.ndarray]:
    """
    从线段检测中枢（缠论第83-86课）。

    中枢定义: 至少三个连续线段的重叠区间
    - ZG(中枢上沿) = min(构成中枢的线段high)
    - ZD(中枢下沿) = max(构成中枢的线段low)
    - ZG > ZD 则中枢成立

    中枢方向:
    - 向上中枢: 进入段向上, 离开段向上
    - 向下中枢: 进入段向下, 离开段向下

    Returns:
        pivots: Pivot对象列表
        pivot_present: 每根K线是否在中枢内
        pivot_info: 每根K线的中枢信息(zg, zd, zz)
    """
    pivot_present = np.zeros(n_bars, dtype=bool)
    pivot_zg = np.full(n_bars, np.nan)
    pivot_zd = np.full(n_bars, np.nan)
    pivot_zz = np.full(n_bars, np.nan)
    pivot_level = np.zeros(n_bars, dtype=int)

    if len(segments) < 3:
        return [], pivot_present, {
            'pivot_present': pivot_present,
            'pivot_zg': pivot_zg,
            'pivot_zd': pivot_zd,
            'pivot_zz': pivot_zz,
            'pivot_level': pivot_level,
        }

    pivots = []

    for i in range(len(segments) - 2):
        s1, s2, s3 = segments[i], segments[i + 1], segments[i + 2]

        # 三个线段重叠区域
        zg = min(s1.high, s2.high, s3.high)
        zd = max(s1.low, s2.low, s3.low)

        if zg <= zd:
            continue  # 无重叠，非中枢

        # 检查是否与前一中枢重叠（中枢延续或扩展/趋势方向）
        trend_dir = 0  # 1=中枢上移(上涨趋势), -1=中枢下移(下跌趋势)
        if pivots:
            prev = pivots[-1]
            if zg <= prev.zg and zd >= prev.zd:
                # 新中枢被旧中枢包含 → 中枢延续（震荡区间收窄或不变）
                # ZG/ZD 不变（旧区间已经更大），只延长结束位置
                prev.end_idx = s3.end_idx
                prev.segment_count += 1
                continue
            elif zd > prev.zg:
                # 新中枢在旧中枢之上 → 中枢上移 → 上涨趋势
                trend_dir = 1
            elif zg < prev.zd:
                # 新中枢在旧中枢之下 → 中枢下移 → 下跌趋势
                trend_dir = -1
            # 部分重叠（非包含）：新中枢独立，正常创建

        # 中枢级别: 由构成线段的笔数间接反映
        # 大中枢(多段重叠) 对价格有更强的支撑/阻力
        n_segments = sum(1 for s in segments if s.high >= zd and s.low <= zg)
        level = min(30, 5 + n_segments)  # 5F ~ 30F 级别

        pivot = Pivot(
            start_idx=s1.start_idx,
            end_idx=s3.end_idx,
            zg=zg,
            zd=zd,
            zz=(zg + zd) / 2,
            level=level,
            segment_count=3,
        )
        # 记录中枢趋势方向（用于趋势分类）
        pivot.trend_dir = trend_dir
        pivots.append(pivot)

        # 尝试扩展: 后续线段若仍在重叠区间内 → 中枢延续
        # P4 fix: ZG/ZD 由前3段确定后保持不变（与原著一致），延伸仅增加段数和时间跨度
        # 通过要求后续线段的high/low必须覆盖ZG/ZD来判断是否仍在重叠区间
        for j in range(i + 3, len(segments)):
            sj = segments[j]
            # 只要有部分重叠即算中枢延续（sj.high > pivot.zd and sj.low < pivot.zg）
            if sj.high > pivot.zd and sj.low < pivot.zg:
                pivot.end_idx = sj.end_idx
                pivot.segment_count += 1
            else:
                # 脱离中枢 → 第三类买卖点可在此产生
                break

    # 填充数组
    for pivot in pivots:
        for idx in range(pivot.start_idx, min(pivot.end_idx + 1, n_bars)):
            pivot_present[idx] = True
            pivot_zg[idx] = pivot.zg
            pivot_zd[idx] = pivot.zd
            pivot_zz[idx] = pivot.zz
            pivot_level[idx] = pivot.level

    return pivots, pivot_present, {
        'pivot_present': pivot_present,
        'pivot_zg': pivot_zg,
        'pivot_zd': pivot_zd,
        'pivot_zz': pivot_zz,
        'pivot_level': pivot_level,
    }


# ==================== 6. 趋势类型分类 ====================

def classify_trend_type(
    pivots: List[Pivot],
    segments: List[Segment],
    n_bars: int,
) -> Dict[str, np.ndarray]:
    """
    分类走势类型（盘整 vs 趋势）。

    盘整: 只有一个中枢的走势
    趋势: 至少两个同向中枢（中枢之间无重叠）

    Returns:
        trend_type: 0=无走势, 1=盘整, 2=上涨趋势, -2=下跌趋势
        trend_strength: 趋势强度 [0, 1]
        consolidation_zone: 是否处于盘整区
    """
    trend_type = np.zeros(n_bars, dtype=int)
    trend_strength = np.zeros(n_bars)
    consolidation_zone = np.zeros(n_bars, dtype=bool)

    if len(pivots) == 0:
        return {
            'trend_type': trend_type,
            'trend_strength': trend_strength,
            'consolidation_zone': consolidation_zone,
        }

    # 标记中枢区间
    for pivot in pivots:
        for idx in range(pivot.start_idx, min(pivot.end_idx + 1, n_bars)):
            consolidation_zone[idx] = True

    # 分析中枢关系
    if len(pivots) == 1:
        # 单中枢 = 盘整
        p = pivots[0]
        for idx in range(p.end_idx, n_bars):
            trend_type[idx] = 1
            trend_strength[idx] = 0.3
    elif len(pivots) >= 2:
        # 多中枢: 用连续同向关系判定趋势（从中枢结束点赋值，防止前视偏差）
        consecutive_up = 0
        consecutive_down = 0
        for i in range(len(pivots) - 1):
            p1, p2 = pivots[i], pivots[i + 1]

            # 优先使用 trend_dir（在detect_chan_pivots中设置）
            td = p2.trend_dir
            fill_start = p2.end_idx  # 两个中枢都完成后趋势才确认
            if td == 1 or (td == 0 and p2.zd > p1.zg):
                # 中枢上移 = 上涨趋势
                consecutive_up += 1
                consecutive_down = 0
                strength = min(0.9, 0.5 + consecutive_up * 0.15)
                for idx in range(fill_start, n_bars):
                    trend_type[idx] = 2
                    trend_strength[idx] = max(trend_strength[idx], strength)
            elif td == -1 or (td == 0 and p2.zg < p1.zd):
                # 中枢下移 = 下跌趋势
                consecutive_down += 1
                consecutive_up = 0
                strength = min(0.9, 0.5 + consecutive_down * 0.15)
                for idx in range(fill_start, n_bars):
                    trend_type[idx] = -2
                    trend_strength[idx] = max(trend_strength[idx], strength)
            else:
                # 中枢重叠 = 级别扩展(仍为盘整)，不重置趋势计数
                # 但重置连续同向计数（方向不明确）
                consecutive_up = 0
                consecutive_down = 0
                for idx in range(p1.start_idx, min(p2.end_idx + 1, n_bars)):
                    consolidation_zone[idx] = True
                    if trend_type[idx] == 0:
                        trend_type[idx] = 1

    return {
        'trend_type': trend_type,
        'trend_strength': trend_strength,
        'consolidation_zone': consolidation_zone,
    }


# ==================== 7. 三类买卖点识别 ====================

def _compute_macd_divergence(
    close: np.ndarray,
    pivots: List[Pivot],
    last_pivot: Pivot,
    n_bars: int,
    max_idx: int = 999999,
) -> float:
    """计算背驰强度 — 比较最后两个下跌段/上涨段的MACD面积（P0: 原著MACD面积法）

    背驰 = 价格创新低(高)但MACD面积缩小 → 趋势衰竭信号

    Returns:
        divergence_strength: [0, 1], >0.3 表示有背驰, >0.5 强背驰
    """
    if len(close) < 50 or len(pivots) < 2:
        return 0.0

    # 计算MACD
    ema12 = np.zeros_like(close)
    ema26 = np.zeros_like(close)
    alpha12 = 2 / 13
    alpha26 = 2 / 27
    ema12[0] = close[0]
    ema26[0] = close[0]
    for i in range(1, len(close)):
        ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i - 1]
        ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i - 1]
    macd = ema12 - ema26

    # 找最后两个下跌段：从中枢上方到中枢下方的价格段
    p_end = last_pivot.end_idx
    p_zd = last_pivot.zd

    # 找最后两个有方向的中枢
    directional = [p for p in pivots if p.trend_dir != 0]
    if len(directional) < 2:
        return 0.0

    # 取最近两个同向中枢
    prev_p = directional[-2]
    curr_p = directional[-1]

    if curr_p.trend_dir == -1:
        # 下跌趋势：两段都从中枢上方向下运行
        seg1_start = max(0, prev_p.start_idx - 5)
        seg1_end = min(len(close) - 1, prev_p.end_idx, max_idx)
        seg2_start = max(0, curr_p.start_idx - 5)
        seg2_end = min(len(close) - 1, curr_p.end_idx, max_idx)
    elif curr_p.trend_dir == 1:
        # 上涨趋势：两段都从中枢下方向上运行
        seg1_start = max(0, prev_p.start_idx - 5)
        seg1_end = min(len(close) - 1, prev_p.end_idx, max_idx)
        seg2_start = max(0, curr_p.start_idx - 5)
        seg2_end = min(len(close) - 1, curr_p.end_idx, max_idx)
    else:
        return 0.0

    if seg2_end <= seg2_start or seg1_end <= seg1_start:
        return 0.0

    # 累积MACD面积 (取绝对值)
    area1 = np.sum(np.abs(macd[seg1_start:seg1_end + 1]))
    area2 = np.sum(np.abs(macd[seg2_start:seg2_end + 1]))

    if area1 <= 0:
        return 0.0

    area_ratio = area2 / (area1 + 1e-10)

    # 价格变化
    price1 = np.abs(close[seg1_end] - close[seg1_start])
    price2 = np.abs(close[seg2_end] - close[seg2_start])

    # 背驰条件: MACD面积减小但价格跌幅相当或更大
    # area_ratio < 1.0 = 力度减弱
    if area_ratio < 0.85 and price2 > price1 * 0.7:
        div_strength = min(1.0, (1.0 - area_ratio) * 2.0)
    elif area_ratio < 1.0:
        div_strength = (1.0 - area_ratio) * 1.2
    else:
        div_strength = 0.0

    return float(np.clip(div_strength, 0.0, 1.0))


def detect_buy_sell_points(
    pivots: List[Pivot],
    segments: List[Segment],
    strokes: List[Stroke],
    n_bars: int,
    close: np.ndarray,
    volume: np.ndarray = None,
    high: np.ndarray = None,
    low: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    识别三类买卖点（缠论第12-21课）。

    三类买点:
    - 一买 (B1): 下跌趋势最后一个中枢下方，底背离确认 → 趋势反转
    - 二买 (B2): 一买后第一次次级别回调，不破一买低点 → 二次确认
    - 三买 (B3): 突破中枢上沿(ZG)后回调，不破ZG → 趋势加速
    - 四买 (B4): 已确立上涨趋势中的回调入场 → 趋势延续（非转折点）

    三类卖点:
    - 一卖 (S1): 上涨趋势最后一个中枢上方，顶背离确认 → 趋势反转
    - 二卖 (S2): 一卖后第一次次级别反弹，不破一卖高点 → 二次确认
    - 三卖 (S3): 跌破中枢下沿(ZD)后反弹，不回ZD → 趋势加速下跌

    Returns:
        buy_point: 0=无, 1=B1一买, 2=B2二买, 3=B3三买, 4=B4趋势回调, 5=B5趋势启动,
                   6=B6均线回踩, 7=B7中枢震荡, 8=B8横盘突破(VCP)
        sell_point: 0=无, 1=一卖, 2=二卖, 3=三卖
        buy_confidence: 买点置信度 [0, 1]
        sell_confidence: 卖点置信度 [0, 1]
        pivot_position: 相对中枢的位置 (-1=下方, 0=内部, 1=上方)
    """
    buy_point = np.zeros(n_bars, dtype=int)
    sell_point = np.zeros(n_bars, dtype=int)
    buy_confidence = np.zeros(n_bars)
    sell_confidence = np.zeros(n_bars)
    pivot_position = np.zeros(n_bars, dtype=int)
    b3_trend_rank = np.zeros(n_bars, dtype=int)       # B3所处趋势的中枢序号: 1=底部首个, 2=第二个, ...
    b3_trend_confirmed = np.zeros(n_bars, dtype=bool)  # 是否满足"2个以上同向向上不重叠中枢"
    # B3质量增强指标 (量在价先 + 回调质量)
    b3_breakout_vol_ratio = np.zeros(n_bars)     # 突破段量比 (vs 20日均量), >2.0=强
    b3_pullback_vol_ratio = np.zeros(n_bars)      # 回调段量比 (vs 突破段量), <0.5=强缩量
    b3_pullback_shallowness = np.zeros(n_bars)    # 回调深度 (1.0=刚触碰ZG, 0=深跌)

    # 预计算每个中枢在上涨趋势中的序号（从底部数起，1=首个上涨中枢）
    pivot_upward_rank = {}  # pivot index -> rank
    upward_pivots = [p for p in pivots if p.trend_dir == 1]
    # 筛选非重叠的上涨中枢：每个中枢的 ZD > 前一个中枢的 ZG
    non_overlap_up = []
    for p in upward_pivots:
        if not non_overlap_up or p.zd > non_overlap_up[-1].zg:
            non_overlap_up.append(p)
    for rank, p in enumerate(non_overlap_up, 1):
        pivot_upward_rank[id(p)] = rank

    if not pivots or not segments:
        return {
            'buy_point': buy_point,
            'sell_point': sell_point,
            'buy_confidence': buy_confidence,
            'sell_confidence': sell_confidence,
            'pivot_position': pivot_position,
            'b3_trend_rank': b3_trend_rank,
            'b3_trend_confirmed': b3_trend_confirmed,
            'b3_breakout_vol_ratio': b3_breakout_vol_ratio,
            'b3_pullback_vol_ratio': b3_pullback_vol_ratio,
            'b3_pullback_shallowness': b3_pullback_shallowness,
        }

    # 为每个位置计算相对最近中枢的状态
    for idx in range(n_bars):
        c = close[idx] if idx < len(close) else 0
        if c <= 0 or np.isnan(c):
            continue

        # 找已完成的、包含当前位置的中枢（end_idx <= idx 防止前视偏差）
        relevant_pivots = [p for p in pivots if p.end_idx <= idx]  # 只取已完成中枢(start_idx<=idx隐含于end_idx<=idx)

        if not relevant_pivots:
            # 不在任何中枢附近，找最近的中枢
            nearest = None
            min_dist = float('inf')
            for p in pivots:
                if p.end_idx < idx:
                    dist = idx - p.end_idx
                    if dist < min_dist:
                        min_dist = dist
                        nearest = p
                elif p.start_idx > idx:
                    dist = p.start_idx - idx
                    if dist < min_dist:
                        min_dist = dist
                        nearest = p

            if nearest:
                if c > nearest.zg:
                    pivot_position[idx] = 1
                elif c < nearest.zd:
                    pivot_position[idx] = -1
                else:
                    pivot_position[idx] = 0
            continue

        p = relevant_pivots[-1]

        if c > p.zg:
            pivot_position[idx] = 1  # 中枢之上
        elif c < p.zd:
            pivot_position[idx] = -1  # 中枢之下
        else:
            pivot_position[idx] = 0  # 中枢内部

    # 基于中枢位置和结构识别买卖点
    # 遍历已完成的中枢，逐bar状态机（无前视偏差）
    # B3/S3 pending 状态: {pivot_index: {'bar': breakout_bar_idx, 'low'/'high': extreme_value}}
    b3_pending = {}
    s3_pending = {}

    for i in range(n_bars):
        # === B3 状态机: 逐bar处理突破→回调 ===
        _h = high if high is not None else close
        _l = low if low is not None else close
        # Step 1: 检查新的突破信号（中枢完成后5bar内, 用最高价检测真突破）
        for pi, p in enumerate(pivots):
            p_end = p.end_idx
            if i <= p_end or i > p_end + 5 or pi in b3_pending:
                continue
            if pivot_position[i] == 1 and _h[i] > p.zg * 1.02:  # 用high检测突破
                b3_pending[pi] = {'bar': i, 'low': _l[i]}  # 用low跟踪回调

        # Step 2: 检查待确认的突破是否出现回调
        resolved_b3 = []
        for pi, state in b3_pending.items():
            p = pivots[pi]
            breakout_bar = state['bar']
            if i <= breakout_bar or i > breakout_bar + 5:  # 3→5: 给更多确认时间
                continue
            # 跟踪回调最低点 (用low)
            if _l[i] < state['low']:
                state['low'] = _l[i]
            near_zg = p.zg * 0.97 < close[i] < p.zg * 1.03
            above_zg = state['low'] > p.zg * 0.98  # 放宽: 回调不低于ZG的98%
            if near_zg and above_zg and buy_point[i] == 0:
                p_rank = pivot_upward_rank.get(id(p), 0)
                p_trend_ok = len(non_overlap_up) >= 2 and p_rank >= 2
                # 额外质量检查: 突破放量 + 回调缩量
                qual_ok = True
                if volume is not None and breakout_bar >= 20:
                    vol_breakout = volume[breakout_bar]
                    vol_20_avg = np.mean(volume[max(0,breakout_bar-20):breakout_bar])
                    vol_pullback = np.mean(volume[breakout_bar+1:i+1]) if i > breakout_bar else vol_breakout
                    qual_ok = (vol_breakout > vol_20_avg * 1.1 and  # 突破放量
                              vol_pullback < vol_breakout * 0.85)     # 回调缩量
                if not qual_ok:
                    continue  # 量能不达标, 不标记B3
                buy_point[i] = 3
                b3_trend_rank[i] = p_rank
                b3_trend_confirmed[i] = p_trend_ok
                breakout_pct = (_h[breakout_bar] - p.zg) / p.zg
                pullback_pct = (close[i] - p.zg) / p.zg
                buy_confidence[i] = min(0.85, 0.4 + breakout_pct * 8 + pullback_pct * 5)
                resolved_b3.append(pi)
            elif i == breakout_bar + 5:
                resolved_b3.append(pi)  # 超时，放弃这个待确认信号
        for pi in resolved_b3:
            b3_pending.pop(pi, None)

        # === S3 状态机: 逐bar处理跌破→反弹 ===
        for pi, p in enumerate(pivots):
            p_end = p.end_idx
            if i <= p_end or i > p_end + 5 or pi in s3_pending:
                continue
            if pivot_position[i] == -1 and close[i] < p.zd * 0.98:
                s3_pending[pi] = {'bar': i, 'high': close[i]}

        resolved_s3 = []
        for pi, state in s3_pending.items():
            p = pivots[pi]
            breakdown_bar = state['bar']
            if i <= breakdown_bar or i > breakdown_bar + 3:
                continue
            if close[i] > state['high']:
                state['high'] = close[i]
            near_zd = p.zd * 0.97 < close[i] < p.zd * 1.03
            below_zd = state['high'] < p.zd
            if near_zd and below_zd and sell_point[i] == 0:
                sell_point[i] = 3
                breakdown_pct = (p.zd - close[breakdown_bar]) / p.zd
                bounce_pct = (p.zd - close[i]) / p.zd
                sell_confidence[i] = min(0.85, 0.4 + breakdown_pct * 8 + bounce_pct * 5)
                resolved_s3.append(pi)
            elif i == breakdown_bar + 3:
                resolved_s3.append(pi)
        for pi in resolved_s3:
            s3_pending.pop(pi, None)

    # === 一买/一卖: 基于累计趋势方向 ===
    if len(pivots) >= 2:
        # 统计最近N个中枢的整体走向（而非仅看最后两个）
        n_check = min(len(pivots), 5)
        recent_pivots = pivots[-n_check:]
        last_pivot = recent_pivots[-1]

        # 累计下跌/上涨计数
        down_count = 0
        up_count = 0
        for i in range(1, len(recent_pivots)):
            p_prev = recent_pivots[i - 1]
            p_curr = recent_pivots[i]
            td = p_curr.trend_dir
            if td == -1 or (td == 0 and p_curr.zg < p_prev.zd):
                down_count += 1
            elif td == 1 or (td == 0 and p_curr.zd > p_prev.zg):
                up_count += 1

        # === 一买 (B1): 标准背驰路径 ===
        # 下跌趋势确认（>=2次中枢下移）→ 背驰确认后一买
        if down_count >= 2 or (down_count >= 1 and len(recent_pivots) <= 2):
            div_strength = _compute_macd_divergence(close, pivots, last_pivot, n_bars,
                                                     max_idx=last_pivot.end_idx)
            if div_strength > 0.10:
                for idx in range(last_pivot.end_idx, min(last_pivot.end_idx + 20, n_bars)):
                    if pivot_position[idx] == -1 and buy_point[idx] == 0:
                        buy_point[idx] = 1
                        dist_pct = (last_pivot.zd - close[idx]) / (close[idx] + 1e-10)
                        base_conf = 0.35 + down_count * 0.10 + div_strength * 0.3
                        buy_confidence[idx] = min(0.9, base_conf + dist_pct * 10)

        # === B1扩展: V型反转 — 深度回撤后强力反弹 + 量能确认 ===
        # 60日最大回撤>15% + 近3日反弹>2% + 近期曾跌破MA20 + 反弹放量
        if n_bars >= 60:
            for idx in range(60, n_bars):
                if buy_point[idx] != 0:
                    continue
                max60 = np.max(close[idx - 60:idx])
                dd_from_high = (max60 - close[idx]) / max60 if max60 > 0 else 0
                bounce_3d = (close[idx] - close[max(0, idx - 3)]) / (close[max(0, idx - 3)] + 1e-10)
                # 曾跌破MA20：近20日最低价低于近20日均价(用SMA20近似)
                recent_low = np.min(close[max(0, idx - 20):idx + 1])
                recent_avg = np.mean(close[max(0, idx - 20):idx + 1])
                was_below_ma = recent_low < recent_avg * 0.97
                # 量能确认: 反弹3日均量 > 前5日均量 × 1.1（放量反弹才是真反转）
                vol_ok = True
                if volume is not None and idx >= 8:
                    vol_bounce = np.mean(volume[max(0, idx - 2):idx + 1])
                    vol_pre = np.mean(volume[max(0, idx - 7):max(0, idx - 2)])
                    vol_ok = vol_bounce > vol_pre * 1.1 if vol_pre > 0 else True
                if dd_from_high > 0.15 and bounce_3d > 0.02 and was_below_ma and vol_ok:
                    buy_point[idx] = 1
                    buy_confidence[idx] = float(np.clip(0.30 + dd_from_high * 0.5 + bounce_3d * 3, 0.25, 0.7))

        # === B1扩展: 缺口反转 — 跳空高开放量突破下跌趋势 ===
        if n_bars >= 3:
            for idx in range(3, n_bars):
                if buy_point[idx] != 0:
                    continue
                gap_up = (close[idx] - close[idx - 1]) / (close[idx - 1] + 1e-10)
                # 之前处于下跌：近5日趋势向下
                pre_trend = (close[idx - 1] - close[max(0, idx - 5)]) / (close[max(0, idx - 5)] + 1e-10)
                # 量能确认: 跳空日量比>1.0（放量跳空才是真突破，缩量跳空=假突破）
                vol_ok = True
                if volume is not None and idx >= 20:
                    avg_vol_20 = np.mean(volume[max(0, idx - 20):idx])
                    vol_ok = volume[idx] > avg_vol_20 * 1.0 if avg_vol_20 > 0 else True
                if gap_up > 0.02 and pre_trend < -0.03 and vol_ok:
                    buy_point[idx] = 1
                    buy_confidence[idx] = float(np.clip(0.30 + gap_up * 5, 0.25, 0.7))

        # === B1扩展: 横盘突破 — 中枢震荡后向上突破ZG ===
        if len(pivots) >= 1:
            last_p = pivots[-1]
            # 中枢已横盘超过30根K线 → 充分蓄力
            pivot_width = last_p.end_idx - last_p.start_idx
            if pivot_width >= 20:
                for idx in range(last_p.end_idx, min(last_p.end_idx + 15, n_bars)):
                    if buy_point[idx] != 0:
                        continue
                    if close[idx] > last_p.zg and pivot_position[idx] == 1:
                        buy_point[idx] = 3  # 突破中枢上沿→B3,非B1
                        breakout_pct = (close[idx] - last_p.zg) / last_p.zg
                        buy_confidence[idx] = float(np.clip(0.30 + breakout_pct * 8, 0.25, 0.65))

        # 上涨趋势确认（>=2次中枢上移）→ 可能一卖
        # P0+P2 fix: S1必须同时满足"中枢上方"+"背驰确认"
        if up_count >= 2 or (up_count >= 1 and len(recent_pivots) <= 2):
            div_strength = _compute_macd_divergence(close, pivots, last_pivot, n_bars,
                                                     max_idx=last_pivot.end_idx)
            if div_strength > 0.15:
                for idx in range(last_pivot.end_idx, min(last_pivot.end_idx + 20, n_bars)):
                    if pivot_position[idx] == 1 and sell_point[idx] == 0:
                        sell_point[idx] = 1
                        dist_pct = (close[idx] - last_pivot.zg) / (last_pivot.zg + 1e-10)
                        base_conf = 0.35 + up_count * 0.10 + div_strength * 0.3
                        sell_confidence[idx] = min(0.9, base_conf + dist_pct * 10)

    # === 二买/二卖: 由 detect_second_buy_point 实现 (详见函数14) ===
    # 旧的B2/S2 crude循环已删除 — 改用MACD确认+黄金分割回调+笔确认的完整实现

    # B2(二买)检测已由 detect_second_buy_point 独立函数实现, 不再在此处重复

    # === 四买 (B4): 趋势回调买点 — 必须有真实回调 + 底分型 + MACD改善 ===
    # B4只在上涨趋势中回调到位且有结构确认时触发。
    if n_bars >= 60:
        ma20 = _sma(close, 20)
        ma60 = _sma(close, 60)
        if volume is not None:
            vol_ma20 = _sma(volume, 20)
        for idx in range(60, n_bars):
            if buy_point[idx] != 0:
                continue
            if sell_point[idx] != 0:
                continue

            # 条件1: 均线多头 (MA20>MA60, 趋势确立)
            ma_bull = ma20[idx] > ma60[idx] and close[idx] > ma60[idx]
            if not ma_bull:
                continue

            # 条件2: 真实回调 (5日涨幅<1%, 不是横盘)
            mom_5d_val = (close[idx] - close[max(0,idx-5)]) / (close[max(0,idx-5)] + 1e-10)
            is_pulling_back = mom_5d_val < 0.01

            # 条件3: 回调到均线附近 (距MA20<8%)
            dist_ma20 = (close[idx] - ma20[idx]) / (ma20[idx] + 1e-10)
            near_ma = -0.05 < dist_ma20 < 0.08

            # 条件4: 真实缩量 (近5日均量 < 前5日×0.95, 量价配合)
            vol_ok = True
            if volume is not None and idx >= 10:
                vol_recent = np.mean(volume[max(0,idx-4):idx+1])
                vol_prev = np.mean(volume[max(0,idx-9):max(0,idx-4)])
                vol_ok = vol_recent < vol_prev * 0.95

            # 条件5: 非力竭 (60日涨幅<50%)
            mom_60d_val = (close[idx] - close[idx-60]) / (close[idx-60] + 1e-10)
            not_extended = mom_60d_val < 0.50

            # 条件6: 底分型确认 (回调到位, 不是悬在半空)
            has_fractal = True
            fractal_q = 0.5  # 默认质量
            try:
                if 'bottom_fractals' in dir() and bottom_fractals is not None:
                    has_fractal = (idx < len(bottom_fractals) and bottom_fractals[idx]
                                  and bottom_fractal_quality[idx] >= 0.20)
                    fractal_q = bottom_fractal_quality[idx] if has_fractal else 0.0
            except (NameError, TypeError):
                pass  # bottom_fractals不可用时放宽条件

            # 条件7: MACD改善 (动量已转, 不是死猫反弹)
            macd_ok = True  # 默认通过
            try:
                if 'macd_hist' in dir() and macd_hist is not None and idx >= 10:
                    macd_recent = np.nanmean(macd_hist[max(0, idx-5):idx+1])
                    macd_earlier = np.nanmean(macd_hist[max(0, idx-10):max(0, idx-5)])
                    macd_ok = (not np.isnan(macd_recent) and not np.isnan(macd_earlier)
                              and macd_recent > macd_earlier)
            except (NameError, TypeError):
                pass

            if (is_pulling_back and near_ma and vol_ok and not_extended
                    and has_fractal and macd_ok):
                buy_point[idx] = 4
                trend_score = min(1.0, (ma20[idx] / (ma60[idx] + 1e-10) - 1.0) * 8)
                pullback_score = max(0.0, -mom_5d_val * 8) if mom_5d_val < 0 else 0.05
                base_conf = 0.30 + trend_score * 0.3 + pullback_score * 0.25
                base_conf += fractal_q * 0.3
                buy_confidence[idx] = float(np.clip(base_conf, 0.25, 0.85))

    # === 五买 (B5): 趋势启动买点 — 捕获趋势起点（V型反转+金叉+放量） ===
    # B1-B4都要求已有结构，但趋势起点处没有任何结构。
    # B5捕获: 价格刚上穿MA20 + MA20拐头 + 放量确认 + 短动量转正。
    if n_bars >= 30:
        ma20_b5 = _sma(close, 20)
        vol_ma20_b5 = _sma(volume, 20) if volume is not None else None
        for idx in range(30, n_bars):
            if buy_point[idx] != 0:
                continue
            if sell_point[idx] != 0:
                continue

            # 条件1: 价格刚上穿MA20 (近5日内曾低于MA20，现在>MA20*0.97)
            was_below = np.any(close[max(0,idx-5):idx] < ma20_b5[max(0,idx-5):idx] * 0.99)
            near_or_above_ma20 = close[idx] > ma20_b5[idx] * 0.97

            # 条件2: MA20拐头向上 (当前MA20 > 5日前MA20)
            ma20_turning_up = idx >= 5 and ma20_b5[idx] > ma20_b5[idx-5]

            # 条件3: 短动量转正 (5日涨幅>0且<10%, 温和上涨非追高)
            mom_5d_b5 = (close[idx] - close[idx-5]) / (close[idx-5] + 1e-10)
            mild_momentum = 0.005 < mom_5d_b5 < 0.10

            # 条件4: 放量确认 (当前量>20日均量*0.85, 有资金进场)
            vol_confirm = True
            if vol_ma20_b5 is not None:
                vol_confirm = volume[idx] > vol_ma20_b5[idx] * 0.85

            # 条件5: 非追高 (10日涨幅<18%)
            mom_10d_b5 = (close[idx] - close[max(0,idx-10)]) / (close[max(0,idx-10)] + 1e-10)
            not_chasing_b5 = mom_10d_b5 < 0.18

            # 条件6: 前期有下跌或横盘 (20日最低价<当前价*0.92, 即有过>8%的回落)
            low_20d = np.min(close[max(0,idx-20):idx+1])
            had_pullback = low_20d < close[idx] * 0.92

            if (was_below and near_or_above_ma20 and ma20_turning_up and
                mild_momentum and vol_confirm and not_chasing_b5 and had_pullback):
                buy_point[idx] = 5
                # 置信度: MA20斜率+量比+金叉新鲜度
                slope_score = min(1.0, (ma20_b5[idx] / (ma20_b5[idx-5] + 1e-10) - 1.0) * 50)
                vol_score = min(1.0, volume[idx] / (vol_ma20_b5[idx] + 1e-10) - 0.9) if vol_ma20_b5 is not None else 0.5
                buy_confidence[idx] = float(np.clip(0.20 + slope_score * 0.25 + vol_score * 0.15 + mom_5d_b5 * 2, 0.15, 0.65))

    # === 六买 (B6): 均线回踩 — 首次回踩MA20/MA60不破+反弹 ===
    # 上涨趋势中价格首次回踩均线是最安全的中继买点
    if n_bars >= 60:
        ma20_b6 = _sma(close, 20)
        ma60_b6 = _sma(close, 60)
        if volume is not None:
            vol_ma20_b6 = _sma(volume, 20)
        for idx in range(60, n_bars):
            if buy_point[idx] != 0:
                continue
            if sell_point[idx] != 0:
                continue

            # 条件1: 均线多头, 价格在MA20附近（±3%）或MA60附近（±5%）
            ma_bull_b6 = ma20_b6[idx] > ma60_b6[idx]
            near_ma20 = abs(close[idx] - ma20_b6[idx]) / (ma20_b6[idx] + 1e-10) < 0.03
            near_ma60 = abs(close[idx] - ma60_b6[idx]) / (ma60_b6[idx] + 1e-10) < 0.05
            touch_ma = near_ma20 or (near_ma60 and close[idx] > ma60_b6[idx])

            # 条件2: 之前有过一段上涨（确认趋势），现在回调到均线
            high_20d = np.max(close[max(0,idx-20):idx+1])
            had_rally = high_20d > ma20_b6[idx] * 1.05  # 20日内曾高于MA20至少5%

            # 条件3: 当天收阳 AND 近3日有反弹 (两者都要, 防假信号)
            is_up_day = close[idx] > close[idx-1] if idx > 0 else False
            bounce_3d = (close[idx] - close[max(0,idx-3)]) / (close[max(0,idx-3)] + 1e-10) > -0.01

            # 条件4: 缩量回踩（回踩时缩量=健康调整, 缩到均量以下）
            vol_ok_b6 = True
            if volume is not None and idx >= 5:
                recent_vol = np.mean(volume[max(0,idx-2):idx+1])
                vol_ok_b6 = recent_vol < vol_ma20_b6[idx] * 0.95  # 从1.2收紧到0.95: 必须缩量

            if ma_bull_b6 and touch_ma and had_rally and is_up_day and bounce_3d and vol_ok_b6:
                buy_point[idx] = 6
                dist_score = 1.0 - abs(close[idx]-ma20_b6[idx])/(ma20_b6[idx]+1e-10)*10
                buy_confidence[idx] = float(np.clip(0.30 + dist_score * 0.2 + bounce_3d * 3, 0.20, 0.70))

    # === 七买 (B7): 中枢震荡买点 — 价格在中枢内部ZD附近受支撑 ===
    # 中枢是多空均衡区, ZD是核心支撑, 触碰ZD反弹是最佳B3先导信号
    if len(pivots) >= 1:
        for idx in range(60, n_bars):
            if buy_point[idx] != 0:
                continue
            if sell_point[idx] != 0:
                continue

            # 找最近已完成的中枢
            for p in reversed(pivots):
                if p.end_idx > idx:
                    continue
                # 条件1: 价格在中枢内部或略低于ZD（不超2%）
                in_pivot = p.zd <= close[idx] <= p.zg * 1.01
                near_zd = abs(close[idx] - p.zd) / (p.zd + 1e-10) < 0.03

                # 条件2: 中枢宽度适中(>3%, 有效整理区间)
                pivot_width_pct = (p.zg - p.zd) / (p.zd + 1e-10)
                wide_enough = pivot_width_pct > 0.03

                # 条件3: 中枢方向中性或向上
                ok_dir = p.trend_dir in (0, 1)

                if (in_pivot or near_zd) and wide_enough and ok_dir:
                    if idx > p.end_idx and idx < p.end_idx + 30:
                        buy_point[idx] = 7
                        # ZD越近置信度越高
                        zd_dist = abs(close[idx] - p.zd) / (p.zd + 1e-10)
                        buy_confidence[idx] = float(np.clip(0.40 - zd_dist * 5, 0.20, 0.65))
                        break

    # === 八买 (B8): 横盘突破 — 窄幅盘整后放量向上突破 ===
    # 波动收缩(VCP)后突破, Mark Minervini风格
    if n_bars >= 40:
        atr20_b8 = _sma(np.abs(np.diff(close, prepend=close[0])), 20)  # 近似ATR
        if volume is not None:
            vol_ma20_b8 = _sma(volume, 20)
        for idx in range(40, n_bars):
            if buy_point[idx] != 0:
                continue
            if sell_point[idx] != 0:
                continue

            # 条件1: 近20日波动率收窄(ATR下降>30%)
            atr_early = np.mean(atr20_b8[max(0,idx-20):max(0,idx-10)])
            atr_late = np.mean(atr20_b8[max(0,idx-10):idx+1])
            vol_contracting = atr_late < atr_early * 0.70 if atr_early > 0 else False

            # 条件2: 近期窄幅盘整(20日振幅<12%)
            range_20d = (np.max(close[max(0,idx-20):idx+1]) - np.min(close[max(0,idx-20):idx+1])) / (np.min(close[max(0,idx-20):idx+1]) + 1e-10)
            tight_range = range_20d < 0.12

            # 条件3: 今日放量突破(量>20日均量*1.3, 价格涨幅>1%)
            breakout_vol = False
            if volume is not None:
                breakout_vol = volume[idx] > vol_ma20_b8[idx] * 1.3
            price_breakout = (close[idx] - close[idx-1]) / (close[idx-1] + 1e-10) > 0.01 if idx > 0 else False

            # 条件4: 价格在MA20之上（或刚突破）
            close_above_ma20 = close[idx] > _sma(close, 20)[idx]

            if (vol_contracting or tight_range) and breakout_vol and price_breakout and close_above_ma20:
                buy_point[idx] = 8
                breakout_pct = (close[idx] - close[idx-1]) / (close[idx-1] + 1e-10) if idx > 0 else 0.02
                buy_confidence[idx] = float(np.clip(0.25 + breakout_pct * 5 + (1 if vol_contracting else 0) * 0.10, 0.15, 0.65))

    # === 九买 (B9): 涨停回调 — A股特有, 涨停板支撑确认后二次启动 ===
    if n_bars >= 10:
        daily_ret = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)
        daily_ret[0] = 0
        is_limit_up = daily_ret > 0.095  # ≥9.5%即涨停
        if volume is not None:
            vol_ma5 = _sma(volume, 5)
        for idx in range(8, n_bars):
            if buy_point[idx] != 0:
                continue
            if sell_point[idx] != 0:
                continue
            # 条件1: 近5日内有涨停
            recent = slice(max(0, idx-4), idx)
            if not np.any(is_limit_up[recent]):
                continue
            lu_idx = max(i for i in range(max(0, idx-4), idx) if is_limit_up[i])
            # 条件2: 涨停后回调1-3天, 最低价不破涨停收盘价*0.98
            lu_close = close[lu_idx]
            pullback_low = np.min(close[lu_idx+1:idx+1])
            above_support = pullback_low > lu_close * 0.98
            # 条件3: 回调缩量 (回调日均量 < 涨停日量*0.7)
            vol_ok = True
            if volume is not None and idx > lu_idx + 1:
                avg_vol_pullback = np.mean(volume[lu_idx+1:idx+1])
                vol_ok = avg_vol_pullback < volume[lu_idx] * 0.7
            # 条件4: 今日转强 (收阳或量增)
            turning_up = close[idx] > close[idx-1] if idx > 0 else False
            if above_support and vol_ok and turning_up:
                buy_point[idx] = 9
                days_pullback = idx - lu_idx
                pullback_depth = (lu_close - pullback_low) / (lu_close + 1e-10)
                conf = float(np.clip(0.35 + (3 - days_pullback) * 0.08 - pullback_depth * 3, 0.20, 0.70))
                buy_confidence[idx] = conf

    return {
        'buy_point': buy_point,
        'sell_point': sell_point,
        'buy_confidence': buy_confidence,
        'sell_confidence': sell_confidence,
        'pivot_position': pivot_position,
        'b3_trend_rank': b3_trend_rank,
        'b3_trend_confirmed': b3_trend_confirmed,
        'b3_breakout_vol_ratio': b3_breakout_vol_ratio,
        'b3_pullback_vol_ratio': b3_pullback_vol_ratio,
        'b3_pullback_shallowness': b3_pullback_shallowness,
    }


# ==================== 8. 统一Chan信号计算 ====================

def compute_chan_signal(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema20: Optional[np.ndarray] = None,
    ema60: Optional[np.ndarray] = None,
    ema120: Optional[np.ndarray] = None,
    macd_hist: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    统一Chan理论信号计算 — 模拟czsc的核心输出。

    整合所有层级: 分型 → 笔 → 线段 → 中枢 → 买卖点 → 信号

    Args:
        close, high, low: 价格数据
        ema20, ema60, ema120: EMA均线（可选）
        macd_hist: MACD柱状图（用于背离确认，可选）

    Returns:
        字典包含所有Chan理论输出（用于集成到factor_calculator和signal_engine）
    """
    n = len(close)

    # 计算EMA（若未提供）
    if ema20 is None:
        ema20 = _calc_ema(close, 20)
    if ema60 is None:
        ema60 = _calc_ema(close, 60)
    if ema120 is None:
        ema120 = _calc_ema(close, 120)

    # Layer 1: 分型
    fractals = detect_fractals(high, low)

    # Layer 2: 笔
    stroke_info = detect_strokes(high, low, close)
    strokes = stroke_info['strokes']

    # Layer 3: 线段
    segments, seg_dir, seg_ids = detect_segments(strokes, n)

    # Layer 4: 中枢
    pivots, pivot_present_arr, pivot_info_dict = detect_chan_pivots(segments, n)

    # Layer 5: 趋势分类
    trend_info = classify_trend_type(pivots, segments, n)

    # Layer 6: 买卖点
    bsp_info = detect_buy_sell_points(pivots, segments, strokes, n, close, volume, high, low)

    # === 多级别对齐 (向量化) ===
    alignment = np.zeros(n)
    valid = ~np.isnan(ema20) & ~np.isnan(ema60)
    valid_120 = valid & ~np.isnan(ema120)
    # ema20 vs ema60
    alignment[valid] = np.where(ema20[valid] > ema60[valid], 0.35, -0.35)
    # ema60 vs ema120
    above_120 = valid_120 & (ema60 > ema120)
    below_120 = valid_120 & ~(ema60 > ema120)
    alignment[above_120] += 0.35
    alignment[below_120] -= 0.35
    # 全排列
    full_bull = valid_120 & (ema20 > ema60) & (ema60 > ema120)
    full_bear = valid_120 & (ema20 < ema60) & (ema60 < ema120)
    alignment[full_bull] += 0.3
    alignment[full_bear] -= 0.3
    alignment = np.clip(alignment, -1.0, 1.0)

    # === 信号强度: 综合买卖点 + 趋势 + 对齐 (Numba JIT) ===
    buy_signal, sell_signal = _compute_chan_buy_sell(
        n,
        bsp_info['buy_point'],
        bsp_info['sell_point'],
        bsp_info['buy_confidence'],
        bsp_info['sell_confidence'],
        bsp_info['pivot_position'],
        trend_info['trend_type'],
    )

    # === 中阴状态（趋势不明）===
    zhongyin = np.zeros(n, dtype=bool)
    for i in range(20, n):
        ema_spread = abs(ema20[i] - ema60[i]) / (close[i] + 1e-10)
        if ema_spread < 0.02 and abs(alignment[i]) < 0.3:
            zhongyin[i] = True

    # === 趋势结构完整度 ===
    structure_complete = np.zeros(n, dtype=bool)
    for i in range(n):
        if trend_info['trend_type'][i] != 0:
            structure_complete[i] = True

    return {
        # 分型
        'top_fractals': fractals['top_fractals'],
        'bottom_fractals': fractals['bottom_fractals'],
        'fractal_type': fractals['fractal_type'],

        # 笔
        'strokes': strokes,  # Stroke对象列表，供增强计算复用
        'stroke_direction': stroke_info['stroke_direction'],
        'stroke_id': stroke_info['stroke_id'],
        'stroke_count': np.full(n, len(strokes)),
        'bi_direction': stroke_info['stroke_direction'],

        # 线段
        'segment_direction': seg_dir,
        'segment_id': seg_ids,
        'segment_count': np.full(n, len(segments)),

        # 中枢
        'pivot_present': pivot_info_dict['pivot_present'],
        'pivot_zg': pivot_info_dict['pivot_zg'],
        'pivot_zd': pivot_info_dict['pivot_zd'],
        'pivot_zz': pivot_info_dict['pivot_zz'],
        'chan_pivot_zg': pivot_info_dict['pivot_zg'],   # Fix#1: 兼容别名，避免下游键名错配
        'chan_pivot_zd': pivot_info_dict['pivot_zd'],
        'chan_pivot_zz': pivot_info_dict['pivot_zz'],
        'pivot_level': pivot_info_dict['pivot_level'],
        'pivot_count': np.full(n, len(pivots)),

        # 趋势
        'trend_type': trend_info['trend_type'],
        'trend_strength': trend_info['trend_strength'],
        'consolidation_zone': trend_info['consolidation_zone'],

        # 买卖点
        'buy_point': bsp_info['buy_point'],
        'sell_point': bsp_info['sell_point'],
        'buy_confidence': bsp_info['buy_confidence'],
        'sell_confidence': bsp_info['sell_confidence'],
        'pivot_position': bsp_info['pivot_position'],
        'b3_trend_rank': bsp_info['b3_trend_rank'],
        'b3_trend_confirmed': bsp_info['b3_trend_confirmed'],
        'b3_breakout_vol_ratio': bsp_info['b3_breakout_vol_ratio'],
        'b3_pullback_vol_ratio': bsp_info['b3_pullback_vol_ratio'],
        'b3_pullback_shallowness': bsp_info['b3_pullback_shallowness'],

        # 信号
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'chan_buy_score': buy_signal,
        'chan_sell_score': sell_signal,

        # 状态
        'alignment_score': alignment,
        'zhongyin': zhongyin,
        'structure_complete': structure_complete,

        # 辅助
        'breakout_above_pivot': (bsp_info['pivot_position'] == 1).astype(float),
        'breakout_below_pivot': (bsp_info['pivot_position'] == -1).astype(float),
        'hidden_bottom_divergence': np.zeros(n),  # 由divergence_detector填充
        'hidden_top_divergence': np.zeros(n),
    }


# ==================== 9. 笔趋势耗尽检测 (chanlun-pro bi_td) ====================

def check_bi_trend_depletion(
    stroke_info: Dict,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """
    笔内趋势耗尽检测 (对应 chanlun-pro 的 bi_td 方法).

    检查当前笔的内部结构是否出现动量衰竭:
    - 向上笔: 内部次级别出现高点不再抬高 → 趋势可能耗尽
    - 向下笔: 内部次级别出现低点不再降低 → 趋势可能耗尽

    Returns:
        bi_td: bool数组，True=该位置所在笔出现趋势耗尽
    """
    n = len(close)
    bi_td = np.zeros(n, dtype=bool)
    stroke_dir = stroke_info.get('stroke_direction', np.zeros(n, dtype=int))
    stroke_ids = stroke_info.get('stroke_id', np.full(n, -1, dtype=int))

    if not stroke_info.get('strokes'):
        return bi_td

    strokes = stroke_info['strokes']

    for s in strokes:
        if s.start_idx >= s.end_idx:
            continue
        seg_high = high[s.start_idx:s.end_idx + 1]
        seg_low = low[s.start_idx:s.end_idx + 1]
        seg_close = close[s.start_idx:s.end_idx + 1]
        seg_len = len(seg_close)

        if seg_len < 8:
            continue

        # 将笔内部按每3-5根K线分成"次级别段"
        sub_seg_size = max(3, seg_len // 5)
        sub_highs = []
        sub_lows = []
        for j in range(0, seg_len, sub_seg_size):
            end = min(j + sub_seg_size, seg_len)
            sub_highs.append(np.max(seg_high[j:end]))
            sub_lows.append(np.min(seg_low[j:end]))

        if len(sub_highs) < 3:
            continue

        # 检查趋势耗尽
        depletion_idx = -1
        if s.direction == 1:  # 向上笔
            # 高点不再创新高 + 最后一段回调较大
            for j in range(2, len(sub_highs)):
                if sub_highs[j] < sub_highs[j-1] * 0.995:
                    # 还有: 最后一段的低点比前一段低点更低
                    if j >= 2 and sub_lows[j] < sub_lows[j-1]:
                        depletion_idx = s.start_idx + j * sub_seg_size
                        break
        else:  # 向下笔
            # 低点不再创新低 + 最后一段反弹较大
            for j in range(2, len(sub_lows)):
                if sub_lows[j] > sub_lows[j-1] * 1.005:
                    if j >= 2 and sub_highs[j] > sub_highs[j-1]:
                        depletion_idx = s.start_idx + j * sub_seg_size
                        break

        if depletion_idx >= 0 and depletion_idx < n:
            # 标记笔末尾为趋势耗尽
            for idx in range(depletion_idx, min(s.end_idx + 1, n)):
                bi_td[idx] = True

    return bi_td


# ==================== 10. 笔级别买卖点检测 ====================

def detect_stroke_mmd(
    stroke_info: Dict,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bi_td: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    笔级别买卖点检测 (对应 chanlun-pro 的 bi.mmd_exists).

    在笔的转折处检测买卖点:
    - 向下笔结束 + 底分型 + 趋势耗尽 → 笔1买
    - 向上笔结束 + 顶分型 + 趋势耗尽 → 笔1卖
    - 回调不破前低 → 笔2买
    - 反弹不破前高 → 笔2卖

    Returns:
        bi_buy_point: int数组, 0/1/2
        bi_sell_point: int数组, 0/1/2
        bi_buy_confidence: float数组
        bi_sell_confidence: float数组
    """
    n = len(close)
    bi_buy_point = np.zeros(n, dtype=int)
    bi_sell_point = np.zeros(n, dtype=int)
    bi_buy_conf = np.zeros(n)
    bi_sell_conf = np.zeros(n)

    strokes = stroke_info.get('strokes', [])
    if len(strokes) < 3:
        return {
            'bi_buy_point': bi_buy_point,
            'bi_sell_point': bi_sell_point,
            'bi_buy_confidence': bi_buy_conf,
            'bi_sell_confidence': bi_sell_conf,
        }

    for i in range(2, len(strokes)):
        s_prev2 = strokes[i - 2]  # 前前笔
        s_prev = strokes[i - 1]   # 前笔
        s_curr = strokes[i]       # 当前笔

        # === 笔1买: 两个向下笔后，第二笔趋势耗尽 ===
        # 笔级≈30分钟级别：捕捉小时间框架背驰结构
        if (s_prev.direction == -1 and s_curr.direction == 1 and
                s_prev.low < s_prev2.low):  # 创新低
            end_zone = range(s_prev.end_idx - 5, min(s_prev.end_idx + 5, n))
            td_present = any(bi_td[idx] for idx in end_zone if 0 <= idx < n)
            # 笔力度递减: 第二笔下跌力度 < 第一笔 → 笔级背驰
            force_prev2 = (abs(s_prev2.end_price - s_prev2.start_price) / (s_prev2.start_price + 1e-10)) / max(s_prev2.end_idx - s_prev2.start_idx, 1)
            force_prev = (abs(s_prev.end_price - s_prev.start_price) / (s_prev.start_price + 1e-10)) / max(s_prev.end_idx - s_prev.start_idx, 1)
            force_decay = force_prev < force_prev2 * 0.85
            # 笔级MACD背离: 价格新低但动能减弱(≈30分钟级别MACD面积缩小)
            micro_div = False
            if s_prev.end_idx < n and s_prev2.end_idx < n:
                seg2_delta = np.mean(np.abs(np.diff(close[s_prev2.start_idx:s_prev2.end_idx + 1])))
                seg1_delta = np.mean(np.abs(np.diff(close[s_prev.start_idx:s_prev.end_idx + 1])))
                micro_div = seg1_delta < seg2_delta * 0.85 and seg1_delta > 0
            if td_present or force_decay or micro_div:
                conf = 0.65 if (td_present and force_decay) else (0.60 if force_decay else 0.50)
                for idx in range(s_curr.start_idx, min(s_curr.end_idx + 1, n)):
                    if bi_buy_point[idx] == 0:
                        bi_buy_point[idx] = 1
                        bi_buy_conf[idx] = conf

        # === 笔1卖: 两个向上笔后，第二笔趋势耗尽 ===
        if (s_prev.direction == 1 and s_curr.direction == -1 and
                s_prev.high > s_prev2.high):  # 创新高
            end_zone = range(s_prev.end_idx - 5, min(s_prev.end_idx + 5, n))
            td_present = any(bi_td[idx] for idx in end_zone if 0 <= idx < n)
            # 笔力度递减: 第二笔上涨力度 < 第一笔 → 顶背驰前兆
            force_prev2 = (abs(s_prev2.end_price - s_prev2.start_price) / (s_prev2.start_price + 1e-10)) / max(s_prev2.end_idx - s_prev2.start_idx, 1)
            force_prev = (abs(s_prev.end_price - s_prev.start_price) / (s_prev.start_price + 1e-10)) / max(s_prev.end_idx - s_prev.start_idx, 1)
            force_decay = force_prev < force_prev2 * 0.85
            if td_present or force_decay:
                conf = 0.65 if (td_present and force_decay) else 0.55
                for idx in range(s_curr.start_idx, min(s_curr.end_idx + 1, n)):
                    if bi_sell_point[idx] == 0:
                        bi_sell_point[idx] = 1
                        bi_sell_conf[idx] = conf

        # === 笔2买: 1买之后回调不破前低 ===
        # 当前笔向上，前一笔向下但不创新低
        if s_curr.direction == 1 and s_prev.direction == -1:
            if s_prev.low > s_prev2.low:  # 不创新低 → 2买
                for idx in range(s_curr.start_idx, min(s_curr.end_idx + 1, n)):
                    if bi_buy_point[idx] == 0:
                        bi_buy_point[idx] = 2
                        bi_buy_conf[idx] = 0.5

        # === 笔2卖: 1卖之后反弹不破前高 ===
        if s_curr.direction == -1 and s_prev.direction == 1:
            if s_prev.high < s_prev2.high:  # 不创新高 → 2卖
                for idx in range(s_curr.start_idx, min(s_curr.end_idx + 1, n)):
                    if bi_sell_point[idx] == 0:
                        bi_sell_point[idx] = 2
                        bi_sell_conf[idx] = 0.5

    return {
        'bi_buy_point': bi_buy_point,
        'bi_sell_point': bi_sell_point,
        'bi_buy_confidence': bi_buy_conf,
        'bi_sell_confidence': bi_sell_conf,
    }


# ==================== 11. 多级别确认信号 ====================

def compute_multi_level_confirmation(
    chan_result: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    多级别确认 (chanlun-pro 双级别背驰思路).

    要求笔级别 AND 线段级别同时出现信号才确认:
    - 笔买点 + 线段买点 = 强买信号
    - 笔卖点 + 线段卖点 = 强卖信号
    - 只有单级别 = 弱信号, 降级处理

    这是chanlun-pro最核心的思想: "大级别定方向, 小级别定买卖点"

    Returns:
        confirmed_buy: bool数组
        confirmed_sell: bool数组
        signal_level: int数组 (0=无, 1=笔级, 2=线段级, 3=双级别确认)
    """
    n = len(chan_result.get('chan_buy_score', chan_result.get('buy_signal', [])))
    if n == 0:
        # Fallback: determine n from any available array
        for key in ['buy_point', 'sell_point', 'stroke_direction', 'trend_type']:
            arr = chan_result.get(key)
            if arr is not None and len(arr) > 0:
                n = len(arr)
                break

    # 安全获取各数组
    def _safe(arr_key, default_arr=None):
        arr = chan_result.get(arr_key)
        if arr is None:
            return np.zeros(n) if default_arr is None else default_arr
        return arr

    bi_buy = _safe('bi_buy_point').astype(int)
    bi_sell = _safe('bi_sell_point').astype(int)
    seg_buy = _safe('buy_point').astype(int)
    seg_sell = _safe('sell_point').astype(int)
    bi_buy_conf = _safe('bi_buy_confidence')
    bi_sell_conf = _safe('bi_sell_confidence')
    seg_buy_conf = _safe('buy_confidence')
    seg_sell_conf = _safe('sell_confidence')
    chan_buy = _safe('chan_buy_score')
    chan_sell = _safe('chan_sell_score')

    # JIT 加速: 多级别确认循环
    confirmed_buy, confirmed_sell, signal_level, buy_strength, sell_strength = _compute_ml_confirm(
        n,
        bi_buy.astype(np.int32), bi_sell.astype(np.int32),
        seg_buy.astype(np.int32), seg_sell.astype(np.int32),
        bi_buy_conf, bi_sell_conf, seg_buy_conf, seg_sell_conf,
        chan_buy, chan_sell,
    )

    return {
        'confirmed_buy': confirmed_buy,
        'confirmed_sell': confirmed_sell,
        'signal_level': signal_level,
        'buy_strength': buy_strength,
        'sell_strength': sell_strength,
    }


# ==================== 12. 结构止损价格计算 ====================

def get_structure_stop_price(
    idx: int,
    chan_result: Dict[str, np.ndarray],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    max_loss_pct: float = 0.07,
) -> Tuple[float, str]:
    """
    基于缠论结构的动态止损价 (chanlun-pro 思路).

    买入时:
    - 优先用最近笔的低点作为止损
    - 其次用线段低点
    - 最后用固定百分比

    卖出时:
    - 优先用最近笔的高点作为止损
    - 其次用线段高点

    Returns:
        (stop_price, stop_reason)
    """
    n = len(close)
    if idx >= n or idx < 20:
        return close[idx] * (1 - max_loss_pct), 'fixed'

    stroke_dir = chan_result.get('stroke_direction', np.zeros(n, dtype=int))
    pivot_zg = chan_result.get('chan_pivot_zg', np.full(n, np.nan))
    pivot_zd = chan_result.get('chan_pivot_zd', np.full(n, np.nan))
    buy_point = chan_result.get('buy_point', np.zeros(n, dtype=int)).astype(int)
    bi_buy = chan_result.get('bi_buy_point', np.zeros(n, dtype=int)).astype(int)

    current_price = close[idx]

    # 买入信号 → 找下方支撑做止损
    if buy_point[idx] > 0 or bi_buy[idx] > 0:
        # 1. 找最近的低点 (笔/分型的低点)
        lookback = min(idx, 60)
        recent_low = np.min(low[idx - lookback:idx + 1])
        structure_stop = recent_low * 0.995  # 略低于结构低点

        # 2. 中枢下沿作为更强支撑
        if not np.isnan(pivot_zd[idx]):
            structure_stop = max(structure_stop, pivot_zd[idx] * 0.995)
            return structure_stop, 'pivot_zd'

        # 3. 固定百分比兜底
        fixed_stop = current_price * (1 - max_loss_pct)
        return max(structure_stop, fixed_stop), 'structure_low'

    # 卖出信号 → 找上方阻力做止损
    sell_point = chan_result.get('sell_point', np.zeros(n, dtype=int)).astype(int)
    bi_sell = chan_result.get('bi_sell_point', np.zeros(n, dtype=int)).astype(int)
    if sell_point[idx] > 0 or bi_sell[idx] > 0:
        lookback = min(idx, 60)
        recent_high = np.max(high[idx - lookback:idx + 1])
        structure_stop = recent_high * 1.005

        if not np.isnan(pivot_zg[idx]):
            structure_stop = min(structure_stop, pivot_zg[idx] * 1.005)
            return structure_stop, 'pivot_zg'

        fixed_stop = current_price * (1 + max_loss_pct)
        return min(structure_stop, fixed_stop), 'structure_high'

    return close[idx] * (1 - max_loss_pct), 'fixed'


# ==================== 13. 底部分型质量分析 ====================

def analyze_bottom_fractal(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    ema20: np.ndarray,
    ema60: np.ndarray,
    macd_hist: np.ndarray,
    bottom_fractals: np.ndarray,
    stroke_direction: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    底部分型质量分析.

    对每个底部分型计算多维质量评分:
    1. 分型强度 (25%): 中间K线低点相对邻居的下探深度 + 形态清晰度
    2. 均量确认 (10%): 分型处放量 vs 20日均量 = 恐慌抛售/ capitulation
    3. 前日量比 (15%): volume/volume[-1], >=3x = 量在价先, 强反转信号
    4. EMA位置 (15%): 低于EMA20的深度 = 超卖程度
    5. MACD背离 (20%): 价格新低但MACD柱上升 = 底背离
    6. 笔转折对齐 (15%): 分型是否位于向下笔结束/向上笔开始处

    质量分数从分型发生日开始指数衰减（半衰期约5根K线）。

    Returns:
        bottom_fractal_quality: 综合质量 [0, 1]，从分型日向后衰减
        bottom_fractal_strength: 原始分型强度 [0, 1]
        bottom_fractal_vol_ratio: 分型处量比 (vs 20日均量)
        bottom_fractal_vol_spike: 分型处相对前日放量倍数
        bottom_fractal_ema_dist: 分型处低于EMA20的距离
    """
    n = len(close)

    fx_quality = np.zeros(n)
    fx_strength = np.zeros(n)
    fx_vol_ratio = np.zeros(n)
    fx_vol_spike = np.zeros(n)
    fx_ema_dist = np.zeros(n)

    fx_indices = np.where(bottom_fractals)[0]
    if len(fx_indices) == 0:
        return {
            'bottom_fractal_quality': fx_quality,
            'bottom_fractal_strength': fx_strength,
            'bottom_fractal_vol_ratio': fx_vol_ratio,
            'bottom_fractal_vol_spike': fx_vol_spike,
            'bottom_fractal_ema_dist': fx_ema_dist,
        }

    # 成交量20日均值（向量化）
    vol_sma20 = _sma(volume, 20)

    for fx_idx in fx_indices:
        if fx_idx < 1 or fx_idx >= n - 1:
            continue

        # 1. 分型强度: 中间K线低点低于邻居的程度 + 形态完整度
        neighbor_low_min = min(low[fx_idx-1], low[fx_idx+1])
        bar_range = high[fx_idx] - low[fx_idx]
        if bar_range > 1e-10:
            raw_depth = (neighbor_low_min - low[fx_idx]) / bar_range
        else:
            raw_depth = 0.0
        # 形态清晰度: 中间K线的高低点都应低于邻居
        clarity = 0.0
        if low[fx_idx] < low[fx_idx-1] and low[fx_idx] < low[fx_idx+1]:
            clarity += 0.5
        if high[fx_idx] < high[fx_idx-1] and high[fx_idx] < high[fx_idx+1]:
            clarity += 0.5
        strength_score = np.clip(raw_depth * 3.0 + clarity * 0.3, 0, 1)

        # 2. 均量确认: 恐慌抛售量 vs 20日均量
        if not np.isnan(vol_sma20[fx_idx]) and vol_sma20[fx_idx] > 0:
            vol_ratio_val = volume[fx_idx] / vol_sma20[fx_idx]
        else:
            vol_ratio_val = 1.0
        vol_score = np.clip((vol_ratio_val - 0.5) / 2.0, 0, 1)

        # 3. 前日量比: volume/volume[-1] — "量在价先"的核心指标
        vol_spike_val = volume[fx_idx] / (volume[fx_idx-1] + 1e-10)
        # >=3x前日量 = 极度放量恐慌抛售，最强反转信号
        # >=2x = 显著放量，较强信号
        # >=1.5x = 温和放量
        if vol_spike_val >= 3.0:
            spike_score = 1.0
        elif vol_spike_val >= 2.0:
            spike_score = 0.7 + (vol_spike_val - 2.0) * 0.3
        elif vol_spike_val >= 1.5:
            spike_score = 0.35 + (vol_spike_val - 1.5) * 0.7
        elif vol_spike_val >= 1.0:
            spike_score = vol_spike_val * 0.35
        else:
            spike_score = 0.0

        # 4. EMA位置: 低于EMA20的百分比
        if not np.isnan(ema20[fx_idx]) and ema20[fx_idx] > 0:
            ema_dist_val = (ema20[fx_idx] - low[fx_idx]) / ema20[fx_idx]
        else:
            ema_dist_val = 0.0
        ema_score = np.clip(ema_dist_val / 0.08, 0, 1)

        # 5. MACD底背离: 价格新低 + MACD柱上升
        div_score = 0.0
        lookback = min(fx_idx, 20)
        if lookback >= 10 and not np.isnan(macd_hist[fx_idx]):
            pre_low = np.min(low[fx_idx-lookback:fx_idx])
            if low[fx_idx] <= pre_low * 0.995:
                macd_recent = macd_hist[max(0, fx_idx-5):fx_idx+1]
                macd_earlier = macd_hist[max(0, fx_idx-10):max(0, fx_idx-5)]
                r_mean = np.nanmean(macd_recent)
                e_mean = np.nanmean(macd_earlier)
                if not np.isnan(r_mean) and not np.isnan(e_mean):
                    if r_mean > e_mean:
                        div_ratio = (r_mean - e_mean) / (abs(e_mean) + 1e-10)
                        div_score = np.clip(div_ratio * 0.5 + 0.3, 0, 1)

        # 6. 笔转折对齐: 分型处笔方向由下转上
        stroke_score = 0.0
        pre_slice = stroke_direction[max(0, fx_idx-3):fx_idx]
        post_slice = stroke_direction[fx_idx:min(n, fx_idx+4)]
        pre_mode = 0
        post_mode = 0
        if len(pre_slice) > 0:
            pv = pre_slice[pre_slice != 0]
            if len(pv) > 0:
                vals, counts = np.unique(pv, return_counts=True)
                pre_mode = vals[np.argmax(counts)]
        if len(post_slice) > 0:
            pv = post_slice[post_slice != 0]
            if len(pv) > 0:
                vals, counts = np.unique(pv, return_counts=True)
                post_mode = vals[np.argmax(counts)]
        if pre_mode == -1 and post_mode == 1:
            stroke_score = 0.8
        elif pre_mode == -1:
            stroke_score = 0.4
        elif post_mode == 1:
            stroke_score = 0.3

        # 加权综合质量: 前日量比权重(15%) > 均量比(10%) — 量在价先
        quality = (
            strength_score * 0.25 +
            vol_score * 0.10 +
            spike_score * 0.15 +
            ema_score * 0.15 +
            div_score * 0.20 +
            stroke_score * 0.15
        )
        quality = np.clip(quality, 0, 1)

        fx_quality[fx_idx] = quality
        fx_strength[fx_idx] = strength_score
        fx_vol_ratio[fx_idx] = vol_ratio_val
        fx_vol_spike[fx_idx] = vol_spike_val
        fx_ema_dist[fx_idx] = ema_dist_val

    # 质量分数向后衰减（半衰期 ~5 根K线，decay=0.87）
    decay = 0.87
    for i in range(1, n):
        inherited = fx_quality[i-1] * decay
        if fx_quality[i] < inherited:
            fx_quality[i] = inherited
        if fx_strength[i] < fx_strength[i-1] * decay:
            fx_strength[i] = fx_strength[i-1] * decay
        if fx_vol_spike[i] < fx_vol_spike[i-1] * decay:
            fx_vol_spike[i] = fx_vol_spike[i-1] * decay

    return {
        'bottom_fractal_quality': fx_quality,
        'bottom_fractal_strength': fx_strength,
        'bottom_fractal_vol_ratio': fx_vol_ratio,
        'bottom_fractal_vol_spike': fx_vol_spike,
        'bottom_fractal_ema_dist': fx_ema_dist,
    }


# ==================== 14. 二买检测 (底部分型 + MACD背离) ====================

def detect_second_buy_point(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bottom_fractals: np.ndarray,
    bottom_fractal_quality: np.ndarray,
    macd_hist: np.ndarray,
    buy_point: np.ndarray,
    stroke_direction: np.ndarray,
    strokes: List[Stroke],
) -> Dict[str, np.ndarray]:
    """
    二买 (B2) 检测 — 底部分型 + MACD背离确认.

    缠论二买是最安全的买入点:
    1. 前提: 一买 (B1) 已发生 — 下跌趋势的底背驰反转
    2. B1后价格反弹，随后回落
    3. 回落处形成底部分型
    4. MACD底背离在回落处确认
    5. 回落低点 > B1低点（不破前低 = 趋势已转）

    二买的优势: 一买是抄底（可能失败），二买是回调确认（趋势已转，胜率更高）

    Returns:
        second_buy_point: bool数组
        second_buy_confidence: float [0, 1]
        second_buy_b1_ref: int, 引用的B1位置
    """
    n = len(close)

    second_buy = np.zeros(n, dtype=bool)
    second_buy_conf = np.zeros(n)
    b1_ref_idx = np.full(n, -1, dtype=int)

    if n < 60:
        return {
            'second_buy_point': second_buy,
            'second_buy_confidence': second_buy_conf,
            'second_buy_b1_ref': b1_ref_idx,
        }

    # Step 1: 找到所有 B1 zone (连续 buy_point==1 的bar归为一个zone)
    b1_zones = []
    i = 20
    while i < n:
        if buy_point[i] == 1:
            zone_start = i
            zone_low = low[i]
            zone_end = i
            i += 1
            while i < n and buy_point[i] == 1:
                zone_low = min(zone_low, low[i])
                zone_end = i
                i += 1
            b1_zones.append({
                'start': zone_start,
                'end': zone_end,
                'low': zone_low,
                'ref_idx': zone_start + np.argmin(low[zone_start:zone_end+1]),
            })
        else:
            i += 1

    # Step 2: 补充隐式B1 — 强底分型 + MACD背离（即使 buy_point 没标记）
    for i in range(40, n):
        if not bottom_fractals[i] or bottom_fractal_quality[i] < 0.30:
            continue
        # 确认有MACD背离
        lookback = min(i, 20)
        if lookback < 10:
            continue
        macd_recent = macd_hist[max(0, i-5):i+1]
        macd_earlier = macd_hist[max(0, i-10):max(0, i-5)]
        r_mean = np.nanmean(macd_recent)
        e_mean = np.nanmean(macd_earlier)
        if np.isnan(r_mean) or np.isnan(e_mean) or r_mean <= e_mean:
            continue
        # 检查不被已有B1 zone覆盖
        already_covered = any(abs(zone['ref_idx'] - i) < 10 for zone in b1_zones)
        if not already_covered:
            b1_zones.append({
                'start': i, 'end': i,
                'low': low[i], 'ref_idx': i,
            })

    if not b1_zones:
        return {
            'second_buy_point': second_buy,
            'second_buy_confidence': second_buy_conf,
            'second_buy_b1_ref': b1_ref_idx,
        }

    # Step 3: 对每个B1 zone，向前扫描B2
    b1_zones.sort(key=lambda z: z['ref_idx'])

    for zone in b1_zones:
        b1_idx = zone['ref_idx']
        b1_low = zone['low']
        b1_price = close[b1_idx]

        # 反弹确认: B1后价格需有回升（>1%），否则未形成有效反弹
        post_high_idx = min(b1_idx + 15, n)
        post_high = np.max(high[b1_idx:post_high_idx])
        if post_high <= b1_price * 1.01:
            continue

        # B2搜索窗口: B1后5~50根bar
        search_start = b1_idx + 5
        search_end = min(b1_idx + 50, n)
        if search_start >= search_end:
            continue

        for i in range(search_start, search_end):
            if not bottom_fractals[i]:
                continue
            if bottom_fractal_quality[i] < 0.30:
                continue

            # 条件1: 不破前低 (B2核心要求)
            if low[i] <= b1_low * 0.98:
                continue

            # 条件2: 当前价格低于反弹高点（确实在回调）
            if close[i] > post_high * 0.97:
                continue  # 没回调够，还在高位

            # 条件2b: 验证趋势已反转 — B1后必须形成向上的笔（否则只是下跌中继）
            upward_stroke_found = False
            for s in strokes:
                if s.direction == 1 and s.start_idx > b1_idx and s.end_idx < i:
                    # 向上的笔在B1和B2之间，且幅度>1%才算有效反弹
                    if (s.end_price - s.start_price) / (s.start_price + 1e-10) > 0.01:
                        upward_stroke_found = True
                        break
            if not upward_stroke_found:
                continue  # 没有有效向上笔 → 趋势未反转，不是B2

            # 条件3: MACD背离确认
            div_confirmed = False
            lookback = min(i, 20)
            if lookback >= 10:
                macd_recent = macd_hist[max(0, i-5):i+1]
                macd_earlier = macd_hist[max(0, i-10):max(0, i-5)]
                r_mean = np.nanmean(macd_recent)
                e_mean = np.nanmean(macd_earlier)
                if not np.isnan(r_mean) and not np.isnan(e_mean) and r_mean > e_mean:
                    div_confirmed = True

            # 条件4: MACD比B1时改善（更强的底部确认）
            macd_improved = False
            if b1_idx >= 5:
                macd_b1 = np.nanmean(macd_hist[max(0, b1_idx-3):b1_idx+1])
                macd_now = np.nanmean(macd_hist[max(0, i-3):i+1])
                if not np.isnan(macd_b1) and not np.isnan(macd_now) and macd_now > macd_b1:
                    macd_improved = True

            # Step 4: B2置信度评分
            conf = 0.30  # 基础分
            conf += bottom_fractal_quality[i] * 0.25       # 分型质量
            if div_confirmed:
                conf += 0.20                                # MACD背离
            if macd_improved:
                conf += 0.10                                # MACD改善

            # 回调深度: 回调到反弹幅度的38%~62% (黄金分割) 最佳
            pullback_range = post_high - b1_low
            if pullback_range > 0:
                retrace_pct = (post_high - low[i]) / pullback_range
                if 0.38 <= retrace_pct <= 0.62:
                    conf += 0.10                            # 黄金分割回调
                elif 0.3 <= retrace_pct <= 0.7:
                    conf += 0.05

            # 不破前低的距离（越大越安全）
            dist_from_b1 = (low[i] - b1_low) / (b1_low + 1e-10)
            conf += min(0.10, dist_from_b1 * 0.5)

            conf = min(conf, 0.95)

            if conf >= 0.35:
                second_buy[i] = True
                second_buy_conf[i] = conf
                b1_ref_idx[i] = b1_idx
                break  # 每个B1只标记第一个符合条件的B2

    return {
        'second_buy_point': second_buy,
        'second_buy_confidence': second_buy_conf,
        'second_buy_b1_ref': b1_ref_idx,
    }


def detect_second_sell_point(
    close, high, low,
    top_fractals, top_fractal_quality, macd_hist,
    sell_point, stroke_direction, strokes,
):
    """二卖 (S2) 检测 — 对称于 B2，顶部分型 + MACD 顶背离确认.

    缠论 S2: S1后反弹不破前高 → 顶部确认，最可靠的卖点之一.
    此前系统中线段级 S2 完全缺失，导致趋势末端没有标准卖点.

    Returns:
        second_sell_point: bool数组
        second_sell_confidence: float [0, 1]
        second_sell_s1_ref: int, 引用的S1位置
    """
    n = len(close)
    second_sell = np.zeros(n, dtype=bool)
    second_sell_conf = np.zeros(n)
    s1_ref_idx = np.full(n, -1, dtype=int)
    if n < 60:
        return {'second_sell_point': second_sell, 'second_sell_confidence': second_sell_conf, 'second_sell_s1_ref': s1_ref_idx}

    # Step 1: 找到所有 S1 zone
    s1_zones = []
    i = 20
    while i < n:
        if sell_point[i] == 1:
            zone_start = i; zone_high = high[i]; zone_end = i
            i += 1
            while i < n and sell_point[i] == 1:
                zone_high = max(zone_high, high[i]); zone_end = i; i += 1
            s1_zones.append({'start': zone_start, 'end': zone_end, 'high': zone_high,
                           'ref_idx': zone_start + np.argmax(high[zone_start:zone_end+1])})
        else:
            i += 1

    # Step 2: 补充隐式S1
    for i in range(40, n):
        if not top_fractals[i] or top_fractal_quality[i] < 0.45:
            continue
        lb = min(i, 20)
        if lb < 10: continue
        mr = np.nanmean(macd_hist[max(0,i-5):i+1])
        me = np.nanmean(macd_hist[max(0,i-10):max(0,i-5)])
        if np.isnan(mr) or np.isnan(me) or mr >= me: continue
        if any(abs(z['ref_idx']-i) < 10 for z in s1_zones): continue
        s1_zones.append({'start': i, 'end': i, 'high': high[i], 'ref_idx': i})

    if not s1_zones:
        return {'second_sell_point': second_sell, 'second_sell_confidence': second_sell_conf, 'second_sell_s1_ref': s1_ref_idx}

    # Step 3: 对每个S1 zone扫描S2
    for zone in sorted(s1_zones, key=lambda z: z['ref_idx']):
        s1_idx = zone['ref_idx']; s1_high = zone['high']
        post_low_idx = min(s1_idx + 15, n)
        post_low = np.min(low[s1_idx:post_low_idx])
        if post_low >= close[s1_idx] * 0.98: continue

        search_start = s1_idx + 5; search_end = min(s1_idx + 50, n)
        if search_start >= search_end: continue

        for i in range(search_start, search_end):
            if not top_fractals[i] or top_fractal_quality[i] < 0.30: continue
            if high[i] >= s1_high * 1.02: continue
            if close[i] < post_low * 1.03: continue

            down_stroke_ok = any(
                s.direction == -1 and s.start_idx > s1_idx and s.end_idx < i and
                (s.start_price - s.end_price) / (s.start_price + 1e-10) > 0.01
                for s in strokes
            )
            if not down_stroke_ok: continue

            lb2 = min(i, 20)
            div_ok = False
            if lb2 >= 10:
                mr2 = np.nanmean(macd_hist[max(0,i-5):i+1])
                me2 = np.nanmean(macd_hist[max(0,i-10):max(0,i-5)])
                div_ok = not np.isnan(mr2) and not np.isnan(me2) and mr2 < me2

            macd_worse = False
            if s1_idx >= 5:
                ms1 = np.nanmean(macd_hist[max(0,s1_idx-3):s1_idx+1])
                mnw = np.nanmean(macd_hist[max(0,i-3):i+1])
                macd_worse = not np.isnan(ms1) and not np.isnan(mnw) and mnw < ms1

            conf = 0.30 + top_fractal_quality[i] * 0.25
            if div_ok: conf += 0.20
            if macd_worse: conf += 0.10

            ret_range = s1_high - post_low
            if ret_range > 0:
                ret_pct = (high[i] - post_low) / ret_range
                if 0.38 <= ret_pct <= 0.62: conf += 0.10
                elif 0.3 <= ret_pct <= 0.7: conf += 0.05

            dist = (s1_high - high[i]) / (s1_high + 1e-10)
            conf += min(0.10, dist * 0.5)
            conf = min(conf, 0.95)

            if conf >= 0.45:
                second_sell[i] = True; second_sell_conf[i] = conf; s1_ref_idx[i] = s1_idx
                break

    return {'second_sell_point': second_sell, 'second_sell_confidence': second_sell_conf, 'second_sell_s1_ref': s1_ref_idx}


# ==================== 15. 增强信号计算 (整合所有chanlun-pro优化) ====================

def compute_enhanced_chan_output(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema20: Optional[np.ndarray] = None,
    ema60: Optional[np.ndarray] = None,
    ema120: Optional[np.ndarray] = None,
    macd_hist: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    增强版Chan信号计算 — 整合 chanlun-pro 所有优化.

    相比 compute_chan_signal, 额外增加:
    1. bi_td (笔趋势耗尽)
    2. 笔级别买卖点
    3. 多级别确认信号
    4. 底部分型质量分析
    """
    # 基础计算
    base = compute_chan_signal(close, high, low, ema20, ema60, ema120, macd_hist, volume)

    # 笔趋势耗尽 — 复用 base 中已计算的 strokes，避免重复调用 detect_strokes
    strokes = base.get('strokes', [])
    stroke_info = {
        'strokes': strokes,
        'stroke_direction': base.get('stroke_direction', np.zeros(len(close), dtype=int)),
        'stroke_id': base.get('stroke_id', np.full(len(close), -1, dtype=int)),
    }
    bi_td = check_bi_trend_depletion(stroke_info, close, high, low)

    # 笔级别买卖点
    stroke_mmd = detect_stroke_mmd(stroke_info, close, high, low, bi_td)

    # 多级别确认
    base['bi_buy_point'] = stroke_mmd['bi_buy_point']
    base['bi_sell_point'] = stroke_mmd['bi_sell_point']
    base['bi_buy_confidence'] = stroke_mmd['bi_buy_confidence']
    base['bi_sell_confidence'] = stroke_mmd['bi_sell_confidence']
    base['bi_td'] = bi_td.astype(float)

    ml_confirm = compute_multi_level_confirmation(base)
    base['confirmed_buy'] = ml_confirm['confirmed_buy']
    base['confirmed_sell'] = ml_confirm['confirmed_sell']
    base['signal_level'] = ml_confirm['signal_level']
    base['buy_strength'] = ml_confirm['buy_strength']
    base['sell_strength'] = ml_confirm['sell_strength']

    # 底部分型质量分析
    if ema20 is None:
        ema20 = _calc_ema(close, 20)
    if ema60 is None:
        ema60 = _calc_ema(close, 60)
    if macd_hist is None:
        macd_hist = np.zeros(len(close))

    if volume is not None:
        fx_analysis = analyze_bottom_fractal(
            close, high, low, volume,
            ema20, ema60, macd_hist,
            base['bottom_fractals'],
            base['stroke_direction'],
        )
        base['bottom_fractal_quality'] = fx_analysis['bottom_fractal_quality']
        base['bottom_fractal_strength'] = fx_analysis['bottom_fractal_strength']
        base['bottom_fractal_vol_ratio'] = fx_analysis['bottom_fractal_vol_ratio']
        base['bottom_fractal_vol_spike'] = fx_analysis['bottom_fractal_vol_spike']
        base['bottom_fractal_ema_dist'] = fx_analysis['bottom_fractal_ema_dist']
    else:
        n = len(close)
        base['bottom_fractal_quality'] = np.zeros(n)
        base['bottom_fractal_strength'] = np.zeros(n)
        base['bottom_fractal_vol_ratio'] = np.zeros(n)
        base['bottom_fractal_vol_spike'] = np.zeros(n)
        base['bottom_fractal_ema_dist'] = np.zeros(n)

    # 二买检测: 底部分型 + MACD背离 → B2
    b2_info = detect_second_buy_point(
        close, high, low,
        base['bottom_fractals'],
        base.get('bottom_fractal_quality', np.zeros(len(close))),
        macd_hist if macd_hist is not None else np.zeros(len(close)),
        base['buy_point'],
        base['stroke_direction'],
        strokes,
    )
    base['second_buy_point'] = b2_info['second_buy_point']
    base['second_buy_confidence'] = b2_info['second_buy_confidence']
    base['second_buy_b1_ref'] = b2_info['second_buy_b1_ref']

    # 合并 B2 (second_buy_point → buy_point) 以替代已删除的crude B2循环
    b2_mask = b2_info['second_buy_point'] & (base['buy_point'] == 0)
    base['buy_point'] = np.where(b2_mask, 2, base['buy_point'])
    # B2置信度: 仅填充尚未有置信度的位置
    b2_conf_mask = b2_mask & (base['buy_confidence'] == 0)
    base['buy_confidence'] = np.where(b2_conf_mask, b2_info['second_buy_confidence'], base['buy_confidence'])

    n = len(close)

    # 顶部分型质量 (用于S2检测 — 对称于 analyze_bottom_fractal)
    top_fx_quality = np.zeros(n)
    top_fx_idx = np.where(base['top_fractals'])[0]
    if len(top_fx_idx) > 0:
        vol_sma20 = _sma(volume, 20) if volume is not None else _sma(np.ones(n), 20)
        for fx_idx in top_fx_idx:
            if fx_idx < 1 or fx_idx >= n - 1:
                continue
            nh_max = max(high[fx_idx-1], high[fx_idx+1])
            bar_rng = high[fx_idx] - low[fx_idx]
            raw_d = (high[fx_idx] - nh_max) / (bar_rng + 1e-10) if bar_rng > 1e-10 else 0.0
            strength = np.clip(raw_d, 0.0, 1.0) * 0.6
            if high[fx_idx] > high[fx_idx-1] and high[fx_idx] > high[fx_idx+1]:
                strength += 0.2
            if close[fx_idx] < high[fx_idx]:
                strength += 0.2
            vr = volume[fx_idx] / (vol_sma20[fx_idx] + 1e-10) if volume is not None else 1.0
            ema20_val = ema20[fx_idx] if ema20 is not None else close[fx_idx]
            ema_d = abs(close[fx_idx] - ema20_val) / (ema20_val + 1e-10) if ema20_val > 0 else 0.0
            top_fx_quality[fx_idx] = strength * 0.35 + min(vr / 3.0, 1.0) * 0.15 + min(ema_d / 0.15, 1.0) * 0.15
            decay = 0.85
            for k in range(fx_idx + 1, min(fx_idx + 8, n)):
                top_fx_quality[k] = max(top_fx_quality[k], top_fx_quality[fx_idx] * decay)
                decay *= 0.85

    # 二卖检测: 顶部分型 + MACD顶背离 → S2
    s2_info = detect_second_sell_point(
        close, high, low,
        base['top_fractals'], top_fx_quality,
        macd_hist if macd_hist is not None else np.zeros(n),
        base['sell_point'], base['stroke_direction'], strokes,
    )
    base['second_sell_point'] = s2_info['second_sell_point']
    base['second_sell_confidence'] = s2_info['second_sell_confidence']
    base['second_sell_s1_ref'] = s2_info['second_sell_s1_ref']
    s2_mask = s2_info['second_sell_point'] & (base['sell_point'] == 0)
    base['sell_point'] = np.where(s2_mask, 2, base['sell_point'])
    s2_conf_mask = s2_mask & (base['sell_confidence'] == 0)
    base['sell_confidence'] = np.where(s2_conf_mask, s2_info['second_sell_confidence'], base['sell_confidence'])

    # 结构止损价 (向量化: 预计算滚动min/max，避免per-bar函数调用)
    structure_stop = np.full(n, np.nan)
    max_loss_pct = 0.07

    # 预取所有需要的数组
    buy_pt = base['buy_point']
    sell_pt = base['sell_point']
    bi_buy = base.get('bi_buy_point', np.zeros(n, dtype=int))
    bi_sell = base.get('bi_sell_point', np.zeros(n, dtype=int))
    pivot_zg = base.get('chan_pivot_zg', np.full(n, np.nan))
    pivot_zd = base.get('chan_pivot_zd', np.full(n, np.nan))

    # 预计算60日滚动最低价/最高价 + 前缀累计min/max (处理窗口不满的情况)
    from numpy.lib.stride_tricks import sliding_window_view
    roll_low_60 = np.full(n, np.nan)
    roll_high_60 = np.full(n, np.nan)
    if n >= 60:
        sw_low = sliding_window_view(low, 60)
        roll_low_60[59:] = sw_low.min(axis=1)
        sw_high = sliding_window_view(high, 60)
        roll_high_60[59:] = sw_high.max(axis=1)
    cummin_low = np.minimum.accumulate(low)
    cummax_high = np.maximum.accumulate(high)

    fixed_stop_buy = close * (1 - max_loss_pct)
    fixed_stop_sell = close * (1 + max_loss_pct)

    for i in range(20, n):
        has_buy = buy_pt[i] > 0 or bi_buy[i] > 0
        has_sell = sell_pt[i] > 0 or bi_sell[i] > 0

        if has_buy:
            # 与原 lookback=min(i,60) 逻辑一致: 不满60日用cummin, 满后用rolling min
            recent_low = roll_low_60[i] if i >= 59 else cummin_low[i]
            struct_stop = recent_low * 0.995
            if not np.isnan(pivot_zd[i]):
                structure_stop[i] = max(struct_stop, pivot_zd[i] * 0.995)
            else:
                structure_stop[i] = max(struct_stop, fixed_stop_buy[i])
        elif has_sell:
            recent_high = roll_high_60[i] if i >= 59 else cummax_high[i]
            struct_stop = recent_high * 1.005
            if not np.isnan(pivot_zg[i]):
                structure_stop[i] = min(struct_stop, pivot_zg[i] * 1.005)
            else:
                structure_stop[i] = min(struct_stop, fixed_stop_sell[i])
        else:
            structure_stop[i] = fixed_stop_buy[i]

    base['structure_stop_price'] = structure_stop

    return base


# ==================== 辅助函数 ====================

@njit
def _compute_chan_buy_sell(
    n: int,
    buy_point: np.ndarray,
    sell_point: np.ndarray,
    buy_confidence: np.ndarray,
    sell_confidence: np.ndarray,
    pivot_position: np.ndarray,
    trend_type: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """综合买卖点 + 趋势 + 对齐 → buy/sell 信号 (Numba JIT)"""
    buy_signal = np.zeros(n)
    sell_signal = np.zeros(n)

    for i in range(n):
        bp = buy_point[i]
        sp = sell_point[i]
        bc = buy_confidence[i]
        sc = sell_confidence[i]
        pp = pivot_position[i]
        tt = trend_type[i]

        if bp == 1:
            buy_signal[i] = 0.7 + 0.3 * bc
        elif bp == 2:
            buy_signal[i] = 0.5 + 0.3 * bc
        elif bp == 3:
            buy_signal[i] = 0.6 + 0.3 * bc

        if sp == 1:
            sell_signal[i] = 0.7 + 0.3 * sc
        elif sp == 2:
            sell_signal[i] = 0.5 + 0.3 * sc
        elif sp == 3:
            sell_signal[i] = 0.6 + 0.3 * sc

        if tt == 2 and buy_signal[i] > 0:
            buy_signal[i] *= 1.2
        elif tt == -2 and sell_signal[i] > 0:
            sell_signal[i] *= 1.2

        if pp == -1 and tt >= 1:
            if buy_signal[i] > 0.3:
                buy_signal[i] *= 1.15

    return np.clip(buy_signal, 0.0, 1.0), np.clip(sell_signal, 0.0, 1.0)


@njit
def _compute_ml_confirm(
    n: int,
    bi_buy: np.ndarray,
    bi_sell: np.ndarray,
    seg_buy: np.ndarray,
    seg_sell: np.ndarray,
    bi_buy_conf: np.ndarray,
    bi_sell_conf: np.ndarray,
    seg_buy_conf: np.ndarray,
    seg_sell_conf: np.ndarray,
    chan_buy: np.ndarray,
    chan_sell: np.ndarray,
) -> tuple:
    """多级别确认信号 (Numba JIT)

    Returns:
        (confirmed_buy, confirmed_sell, signal_level, buy_strength, sell_strength)
    """
    confirmed_buy = np.zeros(n, dtype=np.bool_)
    confirmed_sell = np.zeros(n, dtype=np.bool_)
    signal_level = np.zeros(n, dtype=np.int32)
    buy_strength = np.zeros(n)
    sell_strength = np.zeros(n)

    for i in range(n):
        has_bi_buy = bi_buy[i] > 0
        has_bi_sell = bi_sell[i] > 0
        has_seg_buy = seg_buy[i] > 0
        has_seg_sell = seg_sell[i] > 0
        has_chan_buy = chan_buy[i] > 0.3
        has_chan_sell = chan_sell[i] > 0.3
        bi_buy_strong = bi_buy_conf[i] >= 0.45
        bi_sell_strong = bi_sell_conf[i] >= 0.45
        seg_buy_strong = seg_buy_conf[i] >= 0.35
        seg_sell_strong = seg_sell_conf[i] >= 0.35

        # level 4: 笔+线段强共振 (≈30min+日线双级别背驰)
        if has_bi_buy and has_seg_buy and bi_buy_strong and seg_buy_strong:
            confirmed_buy[i] = True
            signal_level[i] = 4
            buy_strength[i] = 0.80 + 0.10 * bi_buy_conf[i] + 0.10 * seg_buy_conf[i]
        elif has_bi_sell and has_seg_sell and bi_sell_strong and seg_sell_strong:
            confirmed_sell[i] = True
            signal_level[i] = -4
            sell_strength[i] = 0.80 + 0.10 * bi_sell_conf[i] + 0.10 * seg_sell_conf[i]
        # level 3: 笔+线段普通确认
        elif has_bi_buy and has_seg_buy:
            confirmed_buy[i] = True
            signal_level[i] = 3
            buy_strength[i] = 0.65 + 0.15 * bi_buy_conf[i] + 0.15 * seg_buy_conf[i]
        elif has_bi_sell and has_seg_sell:
            confirmed_sell[i] = True
            signal_level[i] = -3
            sell_strength[i] = 0.65 + 0.15 * bi_sell_conf[i] + 0.15 * seg_sell_conf[i]
        # level 2: 线段级 (日线)
        elif has_seg_buy and not has_bi_sell:
            confirmed_buy[i] = True
            signal_level[i] = 2
            buy_strength[i] = 0.45 + 0.25 * seg_buy_conf[i]
        elif has_seg_sell and not has_bi_buy:
            confirmed_sell[i] = True
            signal_level[i] = -2
            sell_strength[i] = 0.45 + 0.25 * seg_sell_conf[i]
        # level 1: 笔级 (≈30分钟级别)
        elif has_bi_buy and not has_seg_sell:
            signal_level[i] = 1
            buy_strength[i] = 0.25 + 0.25 * bi_buy_conf[i]
            if has_chan_buy or bi_buy_strong:
                confirmed_buy[i] = True
        elif has_bi_sell and not has_seg_buy:
            signal_level[i] = -1
            sell_strength[i] = 0.25 + 0.25 * bi_sell_conf[i]
            if has_chan_sell or bi_sell_strong:
                confirmed_sell[i] = True

    return confirmed_buy, confirmed_sell, signal_level, buy_strength, sell_strength


@njit
def _calc_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """计算EMA (Numba JIT)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < span:
        return result
    alpha = 2 / (span + 1)
    result[span - 1] = np.mean(arr[:span])
    for i in range(span, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result
