# core/gate_scorer.py
"""
门控评分模块 — 4门控分级打分 + 硬拒绝检测

替代旧的乘法叠加流水线（factor × chan × MTF × concept × ti_floor × tech_dip）
改为：factor_rank * gate_quality，每层输出Grade而非连续乘数。

Gate定义:
  Gate 1 - Chan Structure (缠论结构): 是否有买点/背离/分型确认
  Gate 2 - MTF Alignment  (多时间框架): 周线/月线方向是否共振
  Gate 3 - Concept Heat   (题材热度):   概念板块是否活跃
  Gate 4 - Trend Direction (趋势方向):  均线多头/空头排列

Grade定义:
  A = 1.0  (强烈看涨)
  B = 0.8  (温和看涨)
  C = 0.55 (中性/不确定/中阴)
  D = 0.3  (偏空)
  默认 = 0.5 (无数据/一般)

硬拒绝 (任一触发=不能买入):
  - 跌停 (daily_ret <= -0.095)
  - Chan强卖 (signal_level <= -3 AND confirmed_sell)
  - 涨停 (daily_ret >= 0.095, 追涨停风险太高)
"""

import numpy as np
from typing import Dict, Tuple


def _safe_arr(d: dict, key: str, n: int, default=0.0):
    """安全取数组，缺失时返回默认填充数组"""
    arr = d.get(key)
    if arr is not None:
        arr = np.asarray(arr)
        if len(arr) == n:
            return arr
    if isinstance(default, (int, float)):
        return np.full(n, default, dtype=np.float64)
    elif isinstance(default, bool):
        return np.full(n, default, dtype=bool)
    return np.full(n, default)


# ── Gate 1: 缠论结构 ──

def gate_chan_structure(grades: np.ndarray, indicators: dict, n: int):
    """缠论结构门控（原地写入 grades 列）

    A (1.0): 线段级确认买点 (signal_level>=2 AND confirmed_buy)
    B (0.8): 有买点/背离+趋势/趋势起点(底分型+放量+ti确认)
    C (0.55): 中阴阶段(方向不明)
    D (0.3): 下跌趋势+无买点信号
    """
    sl = _safe_arr(indicators, 'signal_level', n, 0).astype(int)
    bp = _safe_arr(indicators, 'buy_point', n, 0).astype(int)
    cb = _safe_arr(indicators, 'confirmed_buy', n, False).astype(bool)
    bd = _safe_arr(indicators, 'bottom_divergence', n, 0.0)
    zy = _safe_arr(indicators, 'zhongyin', n, 0.0)
    tt = _safe_arr(indicators, 'trend_type', n, 0).astype(int)
    pp = _safe_arr(indicators, 'chan_pivot_present', n, 0.0)
    ti = _safe_arr(indicators, 'trend_initiation', n, 0.0)
    fx_q = _safe_arr(indicators, 'bottom_fractal_quality', n, 0.0)
    fx_vol = _safe_arr(indicators, 'bottom_fractal_vol_spike', n, 1.0)

    cond_a = (sl >= 2) & cb
    # B1底部反转 → A-(0.85), 胜率最高(55.9%)需高权重
    # B2/B3/有信号级别 → B(0.72)
    cond_b1 = (bp == 1)
    cond_b_struct = ((bp >= 2) | (sl >= 1)) & ~cond_b1
    cond_b_trend_init = ((bd > 0.3) & (zy < 0.5) & (tt != 0)) | \
                        ((ti > 0.2) & (fx_q > 0.25) & (fx_vol > 1.5))
    cond_b = cond_b_struct | cond_b_trend_init
    # B-: 结构确认但强度不够 → 0.62（新增中间档，提升区分度）
    cond_bm = (bp >= 1) & ~cb & (sl < 2)
    cond_c = (zy > 0.5) | (pp > 0.5)
    cond_d = (tt == -2) & (bp == 0) & (sl <= 0) & (bd < 0.2) & ~(cond_b)

    grades[:] = np.select(
        [cond_a, cond_b1, cond_b_struct, cond_b_trend_init, cond_c, cond_d],
        [1.0, 0.85, 0.72, 0.62, 0.50, 0.25],
        default=0.45
    )


# ── Gate 2: 多时间框架 ──

def gate_mtf_alignment(grades: np.ndarray, indicators: dict, n: int):
    """多时间框架门控（原地写入）"""
    align = _safe_arr(indicators, 'mtf_alignment_score', n, 0.0)
    w_up = _safe_arr(indicators, 'weekly_trend_up', n, False).astype(bool)
    m_up = _safe_arr(indicators, 'monthly_trend_up', n, False).astype(bool)

    cond_a = (align > 0.35) | (w_up & m_up)
    cond_b = (align > 0.05) | w_up | m_up
    cond_c = align > -0.10
    cond_cminus = align > -0.25

    grades[:] = np.select(
        [cond_a, cond_b, cond_c, cond_cminus],
        [1.0, 0.72, 0.50, 0.35],
        default=0.25
    )


# ── Gate 3: 题材热度 ──

def gate_concept_heat(grades: np.ndarray, indicators: dict, n: int):
    """题材热度门控（原地写入）"""
    ch = _safe_arr(indicators, 'concept_heat', n, 0.5)

    cond_a = ch > 0.7
    cond_b = ch > 0.5
    cond_c = ch > 0.35

    grades[:] = np.select(
        [cond_a, cond_b, cond_c],
        [1.0, 0.8, 0.6],
        default=0.4
    )


# ── Gate 4: 趋势方向 ──

def gate_trend_direction(grades: np.ndarray, indicators: dict, n: int):
    """趋势方向门控（原地写入）"""
    close = _safe_arr(indicators, 'close', n, 0.0)
    ma20 = _safe_arr(indicators, 'ma20', n, 0.0)
    ema20 = _safe_arr(indicators, 'ema20', n, 0.0)
    ema60 = _safe_arr(indicators, 'ema60', n, 0.0)
    ema120 = _safe_arr(indicators, 'ema120', n, 0.0)
    ema250 = _safe_arr(indicators, 'ema250', n, 0.0)

    valid = (close > 0) & (ma20 > 0) & (ema20 > 0) & (ema60 > 0)
    above_ma20 = close > ma20
    ema_bull = ema20 > ema60
    ema_full_bull = ema_bull & (ema60 > ema120)

    cond_a = valid & above_ma20 & ema_full_bull
    cond_b = valid & above_ma20 & ema_bull
    cond_c = valid & above_ma20
    cond_cm = valid & ~above_ma20 & ema_bull  # 均线下但EMA多头(可能回调中)
    cond_d = valid & ~above_ma20 & ~ema_bull

    grades[:] = np.select(
        [cond_a, cond_b, cond_c, cond_cm, cond_d],
        [1.0, 0.72, 0.50, 0.38, 0.25],
        default=0.40
    )


# ── 硬拒绝检测 ──

def detect_hard_rejects(indicators: dict, n: int) -> np.ndarray:
    """检测硬拒绝条件，返回bool数组 (True=拒绝买入)"""
    rejects = np.zeros(n, dtype=bool)

    close = _safe_arr(indicators, 'close', n, 0.0)
    close_prev = np.zeros_like(close)
    close_prev[1:] = close[:-1]
    daily_ret = np.zeros(n)
    valid_ret = close_prev > 0
    daily_ret[valid_ret] = (close[valid_ret] - close_prev[valid_ret]) / close_prev[valid_ret]

    rejects[daily_ret <= -0.095] = True
    rejects[daily_ret >= 0.095] = True

    sl = _safe_arr(indicators, 'signal_level', n, 0).astype(int)
    confirmed_sell = _safe_arr(indicators, 'confirmed_sell', n, False).astype(bool)
    rejects[(sl <= -3) & confirmed_sell] = True

    return rejects


# ── 综合接口 ──

GATE_FUNCTIONS = [gate_chan_structure, gate_mtf_alignment, gate_concept_heat, gate_trend_direction]


def compute_all_gates(indicators: dict, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算所有门控Grade和硬拒绝标记（单次分配，4门控原地写入）"""
    grades = np.empty((n, 4), dtype=np.float64)
    for j, fn in enumerate(GATE_FUNCTIONS):
        fn(grades[:, j], indicators, n)
    hard_rejects = detect_hard_rejects(indicators, n)
    return grades, hard_rejects


# 各Gate默认值均值: G1=0.45 G2=0.25 G3=0.4 G4=0.40 → 0.375
# 归一化除数取0.55而非0.375: 使得全默认信号 gate_quality≈0.68 < GATE_FLOOR_NEW=0.70
# 确保至少一个Gate有明确正向信号才能通过新入场门槛
_GATE_DEFAULT_MEAN = 0.55


def compute_gate_quality(grades: np.ndarray) -> np.ndarray:
    """4个Gate Grade取均值 → 归一化到中性≈1.0的质量系数

    各Gate默认值: G1=0.45 G2=0.25 G3=0.4 G4=0.40 → 均值0.375。
    无结构股票≈1.0（中性），有结构股票获得boost，弱结构股票得到折扣。
    几何均值过于激进(Sharpe 1.79 < 1.97)，算术均值保留合理的信号区分度。
    """
    raw_avg = grades.mean(axis=1)
    return np.clip(raw_avg / _GATE_DEFAULT_MEAN, 0.5, 2.0)
