# core/factor_calculator.py
"""
统一的因子计算模块 - 所有因子计算集中在这里

两个调用方:
1. signal_engine - 用于实盘信号生成
2. factor_preparer - 用于IC验证的因子预计算

设计原则:
- 纯numpy实现，最大性能
- 两个调用方使用相同的计算逻辑
"""

import numpy as np
from typing import Dict, Optional

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

from .divergence_detector import compute_divergence
from .chan_theory import compute_enhanced_chan_output
from .capital_flow import compute_capital_flow_signal
from .news_sentiment import compute_news_sentiment_signal


# ==================== 基本面因子压缩（单点真源） ====================

def compress_fundamental_factor(raw_value: float, factor_name: str) -> float:
    """统一的基本面因子压缩 - 单点真源

    所有模块（signal_engine, factor_preparer, offline_calibration）
    都调用此函数，确保压缩逻辑一致。

    Args:
        raw_value: 基本面因子原始值
        factor_name: 因子名（如 'fund_score', 'fund_roe' 等）

    Returns:
        压缩后的值，通常在 (-1, 1) 范围
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return 0.0
    compressors = {
        'fund_score': lambda v: np.tanh((np.clip(v, 0, 1.0) - 0.5) * 3),  # [0,1.0]→[-0.9,0.9]
        'fund_profit_growth': lambda v: np.tanh(np.clip(v, -100, 100)),
        'fund_revenue_growth': lambda v: np.tanh(np.clip(v, -100, 100)),
        'fund_roe': lambda v: np.tanh((np.clip(v, -50, 50) - 10) / 20),
        'fund_cf_to_profit': lambda v: np.tanh(np.clip(v, -5, 5) - 1),
        'fund_gross_margin': lambda v: np.tanh((np.clip(v, -20, 80) - 30) / 30),
        'fund_eps': lambda v: np.tanh(np.clip(v, -10, 10)),
        'fund_debt_ratio': lambda v: np.tanh((50 - np.clip(v, 0, 100)) / 50),
        'fund_pg_improve': lambda v: np.tanh(np.clip(v, -5, 5) * 2),
        'fund_rg_improve': lambda v: np.tanh(np.clip(v, -5, 5) * 2),
        'fund_pe': lambda v: np.tanh((15 - np.clip(v, 1, 200)) / 30),   # PE<15=便宜→正信号
        'fund_pb': lambda v: np.tanh((1.5 - np.clip(v, 0.1, 50)) / 5),  # PB<1.5=便宜→正信号
    }
    compressor = compressors.get(factor_name)
    if compressor:
        return float(compressor(raw_value))
    return float(np.tanh(raw_value))  # fallback


def compute_fundamental_score(roe=None, profit_growth=None, revenue_growth=None,
                              eps=None, gross_margin=None, cf_to_profit=None) -> float:
    """基本面综合评分 — 单点真源

    signal_engine._compute_fund_score_from_row 和 _get_fundamental_score
    都调用此函数，确保评分逻辑一致。

    Returns:
        float in [0, 1.0]
    """
    score = 0.0
    if roe is not None:
        try:
            roe = float(roe)
            if roe > 0.15:       score += 0.35
            elif roe > 0.10:     score += 0.25
            elif roe > 0.05:     score += 0.15
        except (ValueError, TypeError):
            pass

    if profit_growth is not None:
        try:
            profit_growth = float(profit_growth)
            if profit_growth > 0.50:   score += 0.30
            elif profit_growth > 0.20: score += 0.20
            elif profit_growth > 0:    score += 0.10
        except (ValueError, TypeError):
            pass

    if revenue_growth is not None:
        try:
            revenue_growth = float(revenue_growth)
            if revenue_growth > 0.30:   score += 0.20
            elif revenue_growth > 0.15: score += 0.12
            elif revenue_growth > 0:    score += 0.05
        except (ValueError, TypeError):
            pass

    if eps is not None:
        try:
            eps = float(eps)
            if eps > 1.0:   score += 0.20
            elif eps > 0.5: score += 0.12
        except (ValueError, TypeError):
            pass

    return min(1.0, score)


# ==================== 基础指标计算 ====================

def calculate_indicators(
    close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    turnover_rate: Optional[np.ndarray] = None,
    amplitude: Optional[np.ndarray] = None,
    open_arr: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    计算所有技术指标（与signal_engine._calculate_indicators逻辑一致）

    Args:
        close: 收盘价数组
        high: 最高价数组
        low: 最低价数组
        volume: 成交量数组
        params: 参数字典

    Returns:
        包含所有技术指标的字典
    """
    if params is None:
        params = get_default_params()

    n = len(close)
    result = {
        'close': close,
        'high': high if high is not None else close,
        'low': low if low is not None else close,
        'volume': volume if volume is not None else np.ones(n),
        'open': open_arr if open_arr is not None else close,
    }

    close_arr = close
    high_arr = result['high']
    low_arr = result['low']
    vol_arr = result['volume']

    # === EMA ===
    for span in params.get('ema_periods', [5, 10, 20, 60]):
        result[f'ema{span}'] = _ema(close_arr, span)
    # 120日EMA（缠论大级别方向判断）
    result['ema120'] = _ema(close_arr, 120)
    # 250日EMA（年线 — A股机构资金核心趋势参考线）
    result['ema250'] = _ema(close_arr, 250)

    # === MA ===
    for span in params.get('ma_periods', [5, 10, 20, 30, 60]):
        result[f'ma{span}'] = _sma(close_arr, span)

    # === RSI (多周期) ===
    for period in params.get('rsi_periods', [6, 8, 10, 14]):
        result[f'rsi_{period}'] = _rsi(close_arr, period)
    result['rsi'] = result['rsi_14']

    # === 布林带 ===
    bb_window = params.get('bb_window', 20)
    bb_std = params.get('bb_std', 2)
    result['bb_upper'], result['bb_middle'], result['bb_lower'] = _bollinger(close_arr, bb_window, bb_std)

    # 布林带宽度
    bb_std_arr = np.zeros(n)
    for i in range(bb_window, n):
        bb_std_arr[i] = np.std(close_arr[max(0, i-bb_window):i])
    result['bb_width_20'] = 4 * bb_std_arr / (result['bb_middle'] + 1e-10)

    # 布林带位置
    result['bb_pos_30'] = (close_arr - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)

    # === 成交量 ===
    vol_ma_period = params.get('volume_ma_period', 20)
    result['volume_ma20'] = _sma(vol_arr, vol_ma_period)
    # 成交量比 - 使用arctan压缩防止极端值
    # 根因：停牌后复牌或异常交易会导致成交量比极大
    raw_vol_ratio = vol_arr / (result['volume_ma20'] + 1e-6)
    result['volume_ratio'] = np.arctan(raw_vol_ratio - 1) / (np.pi / 2)  # 相对于1的偏离，压缩到(-1, 1)
    result['volume_ratio_raw'] = raw_vol_ratio  # 原始量比，供阈值判断使用

    # === 妖股成交量放大因子 (volume_surge) ===
    # 逻辑：计算今日成交量 / 20日平均成交量，>2 表示显著放量
    # 使用 arctan 压缩防止极端值，放量>2x时为正信号
    vol_ma20 = result['volume_ma20']
    vol_surge_raw = vol_arr / (vol_ma20 + 1e-6)
    result['volume_surge'] = np.tanh((vol_surge_raw - 1.5) * 2)  # 以1.5倍为基准

    # === 换手率突破因子 (turnover_burst) ===
    # 逻辑：今日换手率 / 20日平均换手率，>2 表示资金进场
    if turnover_rate is not None and len(turnover_rate) == n:
        to_ma20 = _sma(turnover_rate, 20)
        to_burst_raw = turnover_rate / (to_ma20 + 1e-10)
        result['turnover_ma20'] = to_ma20
        result['turnover_burst'] = np.tanh((to_burst_raw - 1.5) * 2)
    else:
        result['turnover_ma20'] = np.ones(n)
        result['turnover_burst'] = np.zeros(n)

    # === ATR ===
    for period in params.get('atr_periods', [10, 14, 20]):
        result[f'atr_{period}'] = _atr(high_arr, low_arr, close_arr, period)
    result['atr'] = result['atr_14']
    result['atr_ratio'] = result['atr_14'] / (close_arr + 1e-10)

    # === 动量（收益率）===
    for period in params.get('momentum_periods', [3, 5, 10, 20, 30]):
        # 动量 = 过去period天的收益率
        shifted_close = _shift(close_arr, period, safe=True)
        result[f'mom_{period}'] = (close_arr - shifted_close) / (shifted_close + 1e-10)

    # === 价格位置 ===
    result['high_20'] = _rolling_max(high_arr, 20)
    result['low_20'] = _rolling_min(low_arr, 20)
    result['price_position_20'] = (close_arr - result['low_20']) / (result['high_20'] - result['low_20'] + 1e-10)

    # === 波动率 ===
    returns = np.zeros(n)
    returns[1:] = (close_arr[1:] - close_arr[:-1]) / (close_arr[:-1] + 1e-10)
    result['ret'] = returns

    # === 涨停频率因子 (limit_up_freq) ===
    # 逻辑: 过去20日中涨停(>9.5%)的次数，捕捉妖股"持续涨停"特征
    daily_ret = returns.copy()
    daily_ret[0] = 0
    limit_up_mask = daily_ret >= 0.095
    limit_up_sum_20 = _sma(limit_up_mask.astype(float), 20) * 20
    result['limit_up_freq'] = np.tanh(limit_up_sum_20 * 0.5)

    # === 连板数因子 (consec_limit_up) ===
    # 连续涨停天数 — A股最强动量信号之一
    consec = np.zeros(n, dtype=np.int32)
    cnt = 0
    for i in range(n):
        if daily_ret[i] >= 0.095:
            cnt += 1
        else:
            cnt = 0
        consec[i] = cnt
    result['consec_limit_up'] = np.tanh(consec.astype(float) * 0.3)

    # === 题材热度因子 (concept_heat) ===
    # 由 concept_heat.py 的 ConceptHeatCalculator 基于真实概念数据计算
    # calculate_indicators() 不再本地合成 — 避免与 volume_surge/turnover_burst 信息重叠
    result['concept_heat'] = np.full(n, 0.5)  # 默认中等, 由调用方覆盖

    for period in params.get('volatility_periods', [5, 10, 20]):
        result[f'volatility_{period}'] = _rolling_std(returns, period)

    # === 下行风险因子 (downside deviation) ===
    # 使用Numba批量计算: skewness_20, kurtosis_20, volatility_skew, downside_dev, residual_momentum
    skew_20, kurt_20, vol_skew, down_dev_10, down_dev_20, residual_mom = \
        _njit_compute_alpha_factors(returns, n)
    result['downside_dev_10'] = down_dev_10
    result['downside_dev_20'] = down_dev_20
    # 低下行风险因子: 值越大=下行风险越低=越好
    result['low_downside'] = np.tanh(-result['downside_dev_20'] * 30)

    # === 均线关系 ===
    result['ema5_above_20'] = result['ema5'] > result['ema20']
    result['ema20_above_60'] = result['ema20'] > result['ema60']
    result['full_golden'] = (result['ema5'] > result['ema20']) & (result['ema20'] > result['ema60'])
    result['full_death'] = (result['ema5'] < result['ema20']) & (result['ema20'] < result['ema60'])

    # === MA趋势质量 (替代冗余二元均线因子) ===
    # 三维度综合: 排列度(40%) + 通道宽度(30%) + 斜率方向(30%)
    # 相比纯排列度，加入价差和斜率使其:
    #   - 更灵敏: 价差收窄先于排列破坏 (Fix#5)
    #   - 更独立: 不同于trend_vol和mom_x_vol (Fix#6)
    ma5, ma10, ma20, ma60 = result['ma5'], result['ma10'], result['ma20'], result['ma60']
    # 维度1: MA排列度 (0-5)
    alignment = np.zeros(n, dtype=np.float32)
    alignment += (close_arr > ma5).astype(np.float32)
    alignment += (ma5 > ma10).astype(np.float32)
    alignment += (ma10 > ma20).astype(np.float32)
    alignment += (ma20 > ma60).astype(np.float32)
    alignment += (ma5 > ma20).astype(np.float32)
    alignment_score = alignment / 5.0
    # 维度2: MA通道宽度 (价差越大=趋势越强，使用tanh压缩)
    ma_spread = (ma5 - ma60) / (ma60 + 1e-10)
    spread_score = np.tanh(ma_spread * 3)
    # 维度3: MA20斜率方向（复用下述 ema20_slope，提前计算）
    ema20_slope = result['ema20'] / _shift(result['ema20'], 10, safe=True) - 1
    slope_score = np.tanh(ema20_slope * 10)
    # 综合
    result['ma_alignment'] = (
        alignment_score * 0.4 + spread_score * 0.3 + slope_score * 0.3
    )

    # === 趋势强度 ===
    result['trend_strength'] = (result['ema20'] - result['ema60']) / (result['ema60'] + 1e-10)

    # === 斜率 ===
    result['ema20_slope'] = ema20_slope

    # === MACD ===
    result['macd'], result['macd_signal'], result['macd_hist'] = _macd(close_arr)

    # === 行业因子配置中需要的指标 ===
    result['price_ma_30'] = close_arr / (result['ma30'] + 1e-10) - 1
    result['price_ma_60'] = close_arr / (result['ma60'] + 1e-10) - 1

    ma10, ma20, ma60 = result['ma10'], result['ma20'], result['ma60']
    result['ma_cross_10_60'] = ma10 / (ma60 + 1e-10) - 1
    result['ma_cross_10_20'] = ma10 / (ma20 + 1e-10) - 1
    result['ma_golden_20_60'] = (ma20 > ma60).astype(float)
    result['ma_all'] = ((result['ma5'] > ma20) & (ma20 > ma60)).astype(float)

    result['trend_30'] = (close_arr - result['ma30']) / (result['ma30'] + 1e-10)
    result['price_pos_20'] = result['price_position_20']

    # === 复合因子（与backtest_factors配置一致）===
    # 动量×低波动 - 使用tanh压缩
    # 值域分析：mom约[-0.5, 0.5], vol约[0.01, 0.05]，乘积约[-0.025, 0.025]
    # 为了统一量纲，放大后压缩
    raw_mlv_10_10 = result['mom_10'] * result['volatility_10']
    raw_mlv_10_20 = result['mom_10'] * result['volatility_20']
    raw_mlv_20_10 = result['mom_20'] * result['volatility_10']
    raw_mlv_20_20 = result['mom_20'] * result['volatility_20']
    result['mom_x_vol_10_10'] = np.tanh(raw_mlv_10_10 * 20)  # 放大后压缩
    result['mom_x_vol_10_20'] = np.tanh(raw_mlv_10_20 * 20)
    result['mom_x_vol_20_10'] = np.tanh(raw_mlv_20_10 * 20)
    result['mom_x_vol_20_20'] = np.tanh(raw_mlv_20_20 * 20)

    # RSI+波动率组合 - tanh压缩
    result['rsi_vol_combo'] = np.tanh((50 - result['rsi_14']) / 100 - result['volatility_20'] * 0.5)
    # 布林带+RSI组合 - tanh压缩
    result['bb_rsi_combo'] = np.tanh((50 - result['rsi_14']) / 100 - result['bb_pos_30'] * 0.3)

    # 收益波动率比 - 使用arctan压缩防止极端值
    # 根因：当volatility很小时，简单的除法会产生极大值
    # 解决：使用np.arctan将任意值压缩到(-π/2, π/2)，然后缩放到(-1, 1)
    raw_ratio_10 = result['mom_10'] / (result['volatility_10'] + 1e-6)  # 增大保护值
    raw_ratio_20 = result['mom_20'] / (result['volatility_20'] + 1e-6)
    result['ret_vol_ratio_10'] = np.arctan(raw_ratio_10) / (np.pi / 2)  # 压缩到(-1, 1)
    result['ret_vol_ratio_20'] = np.arctan(raw_ratio_20) / (np.pi / 2)

    # 动量反转 - tanh压缩
    result['momentum_reversal'] = np.tanh(-result.get('mom_20', np.zeros(n)) * 3)
    if 'mom_10' in result and 'mom_20' in result:
        result['momentum_acceleration'] = np.tanh((result['mom_10'] - result['mom_20']) * 5)

    result['return_risk_ratio'] = result.get('ret_vol_ratio_10', np.zeros(n))

    # 波动率因子 - 使用tanh压缩（与compute_composite_factors一致）
    result['volatility'] = np.tanh(-result['volatility_20'] * 20)

    # 趋势低波动因子: 强趋势+低波动=稳定上涨
    trend_str = result.get('trend_strength', np.zeros(n))
    atr_ratio = result.get('atr_ratio', np.zeros(n))
    result['trend_vol'] = np.tanh(trend_str * atr_ratio * 50)

    # 量价确认因子: 近20天量价正相关+动量方向一致 (向量化)
    vp_corr = _rolling_corr(result['ret'], result['volume'], 20)
    mom20 = result.get('mom_20', np.zeros(n))
    result['vol_confirm'] = np.tanh(vp_corr * mom20 * 10)

    # === 趋势起点突破因子 (捕捉趋势初期，避免等动量积累20天才触发) ===

    # 1. MA20斜率: 5日变化率，正值=均线拐头向上
    ma20 = result['ma20']
    ma20_slope_5d = (ma20 - _shift(ma20, 5, safe=True)) / (_shift(ma20, 5, safe=True) + 1e-10)
    # MA20斜率加速度: 当前斜率 vs 5日前斜率，正加速=趋势启动
    ma20_slope_accel = ma20_slope_5d - _shift(ma20_slope_5d, 5, safe=True)
    result['ma20_slope'] = np.tanh(ma20_slope_5d * 200)

    # 2. 趋势突破: 价格刚站上MA20 + MA20拐头 + 放量确认
    price_vs_ma20 = (close_arr - ma20) / (ma20 + 1e-10)
    # 接近MA20(0~3%以内) = 新鲜突破, 非追高
    proximity = 1.0 - np.clip(np.abs(price_vs_ma20) / 0.05, 0.0, 1.0)
    above_ma20 = (price_vs_ma20 > 0).astype(float)
    vol_ratio_20 = result['volume_ratio_raw']
    vol_spike = np.clip((vol_ratio_20 - 0.8) / 1.2, 0.0, 1.0)  # 量比>0.8开始给分, >2.0满分
    # 突破强度: 价格位置 + 均线拐头 + 量能
    breakout_score = (above_ma20 * proximity * 0.35 +
                      np.tanh(ma20_slope_5d * 100 + 0.5) * 0.35 +
                      vol_spike * 0.30)
    result['trend_breakout'] = np.tanh(breakout_score * 3)

    # 3. 短期金叉: MA5上穿MA20, 趋势启动的最早信号
    ma5 = result['ma5']
    cross_up = ((ma5 > ma20) & (_shift(ma5, 1, safe=True) <= _shift(ma20, 1, safe=True))).astype(float)
    # 金叉后3天内仍有效(窗口期)
    cross_recent = np.zeros(n)
    for i in range(5, n):
        if cross_up[i]:
            cross_recent[i] = 1.0
        elif i >= 1 and cross_recent[i-1] > 0 and i - np.argmax(cross_up[max(0,i-4):i+1]) <= 3:
            cross_recent[i] = 0.7  # 金叉后第2-3天衰减
    # 金叉强度 = 金叉时MA5-MA20的差距
    cross_gap = (ma5 - ma20) / (ma20 + 1e-10)
    result['ma_golden_cross'] = cross_recent * np.tanh(cross_gap * 50 + 0.5)

    # 4. 趋势启动综合: 突破+金叉+均线斜率+RSI恢复
    rsi_14 = result['rsi_14']
    # RSI从40-55区域回升=趋势初期; >70=过热扣分
    rsi_recovery = np.where(rsi_14 < 40, 0.0,
                    np.where(rsi_14 < 55, (rsi_14 - 40) / 15,
                    np.where(rsi_14 < 70, 1.0,
                    1.0 - (rsi_14 - 70) / 30)))
    result['trend_initiation'] = (
        result['trend_breakout'] * 0.30 +
        result['ma_golden_cross'] * 0.25 +
        result['ma20_slope'] * 0.25 +
        rsi_recovery * 0.20
    )

    # === 换手率因子 ===
    if turnover_rate is not None:
        tr_ma20 = _sma(turnover_rate, 20)
        # 低换手率溢价: 1/turnover → 高值=低换手=未来正收益
        raw_inv_turnover = 1.0 / (turnover_rate + 0.1)
        raw_inv_turnover_ma = 1.0 / (tr_ma20 + 0.1)
        result['inv_turnover'] = np.tanh((raw_inv_turnover_ma - 5) / 5)  # 压缩
        # 换手率变化: 当前/MA20-1, 缩量=正信号
        tr_ratio = turnover_rate / (tr_ma20 + 1e-6)
        result['turnover_shrink'] = np.tanh(-(tr_ratio - 1) * 3)  # 缩量为正

        # === 流动性因子 ===
        # Amihud非流动性: |ret| / (close * volume)
        daily_illiq = np.abs(result['ret']) / (close_arr * vol_arr + 1e-10)
        illiq_20 = _sma(daily_illiq, 20)
        # 高流动性 = 低illiq = 正因子值
        result['illiq_20'] = np.tanh(-illiq_20 * 1e10)

        # 换手率稳定性: 换手率变异系数的负值 (稳定=正信号)
        tr_std_20 = _rolling_std(turnover_rate, 20)
        tr_ma20_safe = tr_ma20 + 1e-6
        turnover_cv = tr_std_20 / tr_ma20_safe
        result['turnover_stability'] = np.tanh(-turnover_cv * 2)
    else:
        n_pts = len(close)
        result['inv_turnover'] = np.zeros(n_pts)
        result['turnover_shrink'] = np.zeros(n_pts)
        result['illiq_20'] = np.zeros(n_pts)
        result['turnover_stability'] = np.zeros(n_pts)

    # === 新增Alpha因子 ===

    # 1. 隔夜收益 vs 日内收益：隔夜跳空是信息驱动的，日内是噪音
    # 高隔夜收益=强基本面信号，压缩到(-1, 1)
    open_arr = result.get('open', close_arr)
    if open_arr is not None:
        overnight_ret = np.zeros(n)
        overnight_ret[1:] = (open_arr[1:] - close_arr[:-1]) / (close_arr[:-1] + 1e-10)
        # 20日累计隔夜收益 (向量化 cumsum)
        overnight_cum = np.zeros(n)
        if n > 20:
            cs = np.cumsum(np.insert(overnight_ret, 0, 0))
            overnight_cum[20:] = cs[21:] - cs[1:n-19]  # sum of 20 elements
        result['overnight_ret'] = np.tanh(overnight_cum * 15)

        intraday_ret = np.zeros(n)
        intraday_ret[1:] = (close_arr[1:] - open_arr[1:]) / (open_arr[1:] + 1e-10)
        # 20日累计日内收益 (向量化 cumsum)
        intraday_cum = np.zeros(n)
        if n > 20:
            cs2 = np.cumsum(np.insert(intraday_ret, 0, 0))
            intraday_cum[20:] = cs2[21:] - cs2[1:n-19]
        result['intraday_ret'] = np.tanh(intraday_cum * 15)
    else:
        result['overnight_ret'] = np.zeros(n)
        result['intraday_ret'] = np.zeros(n)

    # 2-4. 收益偏度/峰度/波动率偏度/残差动量 — 已由 _njit_compute_alpha_factors 批量计算
    result['skewness_20'] = np.tanh(skew_20)
    result['kurtosis_20'] = np.tanh(-kurt_20 / 3)
    result['volatility_skew'] = np.tanh(vol_skew)
    result['residual_momentum'] = np.tanh(residual_mom * 2)

    # 4. 最大单日涨幅（20日）：捕捉动量突破 (向量化)
    max_ret_20 = np.zeros(n)
    if n >= 21:
        from numpy.lib.stride_tricks import sliding_window_view
        sw_ret = sliding_window_view(returns, 20)
        max_ret_20[20:] = sw_ret.max(axis=1)[:n-20]
    result['max_ret_20'] = np.tanh(max_ret_20 * 3)

    # 5. 尾部风险（CVaR-like）：过去20日最差5日平均收益 (向量化)
    tail_risk = np.full(n, np.nan)
    if n >= 20:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(returns, 20)
        tail_risk[19:] = np.mean(np.partition(sw, 5, axis=1)[:, :5], axis=1)[:n - 19]
    result['tail_risk'] = np.tanh(tail_risk * 5)

    # 6. 量价相关性（20日）：价格变化 vs 成交量变化的相关性 (向量化)
    price_chg = np.diff(close_arr, prepend=close_arr[0])
    vol_chg = np.diff(vol_arr, prepend=vol_arr[0])
    pv_corr_20 = _rolling_corr(price_chg, vol_chg, 20)
    # 量价正相关+上涨趋势=正信号；负相关+下跌趋势=负信号
    result['price_volume_corr_20'] = np.tanh(pv_corr_20 * mom20 * 10)

    # 8. 跳空缺口比率：向上跳空 / (向上+向下跳空)
    if open_arr is not None:
        gap_ratio = np.zeros(n)
        for i in range(20, n):
            gaps = overnight_ret[i-20:i]
            up_gaps = np.sum(gaps[gaps > 0.01])
            down_gaps = np.sum(np.abs(gaps[gaps < -0.01]))
            total = up_gaps + down_gaps
            gap_ratio[i] = (up_gaps - down_gaps) / (total + 1e-10)
        result['gap_ratio'] = np.tanh(gap_ratio * 3)
    else:
        result['gap_ratio'] = np.zeros(n)

    # === 小说灵感因子：成交量+价格结构分析 ===
    # 使用Numba批量计算: wash_sale_score, vol_price_breakout, volume_contraction, relative_strength, consolidation_breakout
    wash_score, vol_breakout, vol_contract, rs_score, cb_score = \
        _njit_compute_novel_factors(close_arr, high_arr, low_arr, vol_arr, returns, n)
    result['wash_sale_score'] = np.tanh(wash_score * 2)
    result['vol_price_breakout'] = np.tanh(vol_breakout * 2)
    result['volume_contraction'] = np.tanh(vol_contract * 3)
    result['relative_strength'] = np.tanh(rs_score * 0.5)
    result['consolidation_breakout'] = np.tanh(cb_score * 2)

    # 13. 主力资金流向推断因子 (smart_money_flow)
    mf_score = _calc_smart_money_flow(returns, vol_arr, n)
    result['smart_money_flow'] = np.tanh(mf_score * 3)

    # === 15-16. 残差动量 + 短期反转 已在 _njit_compute_alpha_factors 中计算 ===

    # === 16. 短期反转因子：单日极端涨跌(>4%)后均值回归 ===
    short_reversal = np.zeros(n)
    for i in range(5, n):
        daily_ret = returns[i]
        if abs(daily_ret) > 0.04:
            avg_range = np.mean(high_arr[i-4:i+1] - low_arr[i-4:i+1]) / (close_arr[i] + 1e-10)
            if avg_range > 0.03:
                short_reversal[i] = -daily_ret * 2  # 反转信号
    result['short_reversal'] = np.tanh(short_reversal)

    # === 17. 异质波动率：去除自身趋势后的残差波动率 ===
    # 低异质波动率在A股有显著正溢价（低波动率异象）
    if n >= 20:
        rolling_mean_ret = _sma(returns, 20)
        residual_ret = np.zeros(n)
        residual_ret[20:] = returns[20:] - rolling_mean_ret[20:]
        idio_vol = _rolling_std(residual_ret, 20)
        result['idiosyncratic_volatility'] = np.tanh(-idio_vol * 30)
    else:
        result['idiosyncratic_volatility'] = np.zeros(n)

    # === 18. 盈利质量因子：低波动率+稳定换手=高质量盈利 ===
    if turnover_rate is not None:
        inv_turnover = result.get('inv_turnover', np.zeros(n))
        vol_20 = result.get('volatility', np.zeros(n))
        eq_quality = inv_turnover * (1.0 / (1.0 + np.abs(vol_20) * 10))
        result['earnings_quality'] = np.tanh(eq_quality * 3)
    else:
        result['earnings_quality'] = np.zeros(n)

    # === 缠论：MACD背离检测 ===
    # MACD已在上面计算（macd, macd_signal, macd_hist）
    divergence_params = params.get('divergence', {})
    div_result = compute_divergence(
        close_arr,
        result['macd_hist'],
        ema20=result['ema20'],
        ema60=result['ema60'],
        lookback=divergence_params.get('lookback', 20),
        peak_trough_lookback=divergence_params.get('peak_trough_lookback', 5),
        strength_threshold=divergence_params.get('strength_threshold', 0.3),
        verify_trend=divergence_params.get('verify_trend', True),
    )
    result['top_divergence'] = div_result['top_divergence']
    result['bottom_divergence'] = div_result['bottom_divergence']
    result['hidden_top_divergence'] = div_result['hidden_top']
    result['hidden_bottom_divergence'] = div_result['hidden_bottom']
    result['divergence_active'] = div_result['divergence_active'].astype(float)

    # === 缠论增强 (chanlun-pro风格): 分型→笔→线段→中枢→买卖点 + 双级别确认 ===
    # 统一使用 chan_theory 的完整层级体系，替代旧的简单K线重叠法
    chan_result = compute_enhanced_chan_output(
        close_arr, high_arr, low_arr,
        ema20=result['ema20'],
        ema60=result['ema60'],
        ema120=result['ema120'],
        macd_hist=result['macd_hist'],
        volume=volume if volume is not None else None,
    )

    # === 中枢数据（来自chan_theory: 分型→笔→线段→中枢层级）===
    result['pivot_present'] = chan_result['pivot_present'].astype(float)
    result['pivot_zg'] = chan_result['pivot_zg']
    result['pivot_zd'] = chan_result['pivot_zd']
    result['pivot_zz'] = chan_result['pivot_zz']
    result['pivot_level'] = chan_result['pivot_level']
    result['pivot_count'] = chan_result['pivot_count']
    # 兼容旧字段名
    result['chan_pivot_present'] = chan_result['pivot_present']
    result['chan_pivot_zg'] = chan_result['pivot_zg']
    result['chan_pivot_zd'] = chan_result['pivot_zd']
    result['chan_pivot_zz'] = chan_result['pivot_zz']
    result['chan_pivot_level'] = chan_result['pivot_level']
    # 中枢位置
    result['pivot_position'] = chan_result['pivot_position']
    result['breakout_above_pivot'] = chan_result['breakout_above_pivot']
    result['breakout_below_pivot'] = chan_result['breakout_below_pivot']
    result['consolidation_zone'] = chan_result['consolidation_zone'].astype(float)

    # === 走势结构（统一来自chan_theory，不再使用旧的classify_trend_structure）===
    result['zhongyin'] = chan_result['zhongyin'].astype(float)
    result['structure_complete'] = chan_result['structure_complete'].astype(float)
    result['alignment_score'] = chan_result['alignment_score']
    result['trend_type'] = chan_result['trend_type']
    result['trend_strength'] = chan_result['trend_strength']

    # === 分型/笔/线段 ===
    result['top_fractals'] = chan_result['top_fractals'].astype(float)
    result['bottom_fractals'] = chan_result['bottom_fractals'].astype(float)
    result['fractal_type'] = chan_result['fractal_type'].astype(float)
    result['stroke_direction'] = chan_result['stroke_direction']
    result['stroke_id'] = chan_result['stroke_id']
    result['stroke_count'] = chan_result['stroke_count']
    result['segment_direction'] = chan_result['segment_direction']
    result['segment_id'] = chan_result['segment_id']
    result['segment_count'] = chan_result['segment_count']

    # === chanlun-pro 增强字段 ===
    result['bi_td'] = chan_result.get('bi_td', np.zeros(n))
    result['bi_buy_point'] = chan_result.get('bi_buy_point', np.zeros(n, dtype=int))
    result['bi_sell_point'] = chan_result.get('bi_sell_point', np.zeros(n, dtype=int))
    result['bi_buy_confidence'] = chan_result.get('bi_buy_confidence', np.zeros(n))
    result['bi_sell_confidence'] = chan_result.get('bi_sell_confidence', np.zeros(n))
    result['confirmed_buy'] = chan_result.get('confirmed_buy', np.zeros(n, dtype=bool))
    result['confirmed_sell'] = chan_result.get('confirmed_sell', np.zeros(n, dtype=bool))
    result['signal_level'] = chan_result.get('signal_level', np.zeros(n, dtype=int))
    result['buy_strength'] = chan_result.get('buy_strength', np.zeros(n))
    result['sell_strength'] = chan_result.get('sell_strength', np.zeros(n))
    result['structure_stop_price'] = chan_result.get('structure_stop_price', np.full(n, np.nan))

    # === 底部分型质量分析 ===
    result['bottom_fractal_quality'] = chan_result.get('bottom_fractal_quality', np.zeros(n))
    result['bottom_fractal_strength'] = chan_result.get('bottom_fractal_strength', np.zeros(n))
    result['bottom_fractal_vol_ratio'] = chan_result.get('bottom_fractal_vol_ratio', np.zeros(n))
    result['bottom_fractal_vol_spike'] = chan_result.get('bottom_fractal_vol_spike', np.zeros(n))
    result['bottom_fractal_ema_dist'] = chan_result.get('bottom_fractal_ema_dist', np.zeros(n))

    # === 买卖点（三类买卖点 + 二买 + B3趋势元数据）===
    result['buy_point'] = chan_result['buy_point']
    result['sell_point'] = chan_result['sell_point']
    result['buy_confidence'] = chan_result['buy_confidence']
    result['sell_confidence'] = chan_result['sell_confidence']
    result['second_buy_point'] = chan_result.get('second_buy_point', np.zeros(n, dtype=bool))
    result['second_buy_confidence'] = chan_result.get('second_buy_confidence', np.zeros(n))
    result['second_buy_b1_ref'] = chan_result.get('second_buy_b1_ref', np.full(n, -1, dtype=int))
    result['second_sell_point'] = chan_result.get('second_sell_point', np.zeros(n, dtype=bool))
    result['second_sell_confidence'] = chan_result.get('second_sell_confidence', np.zeros(n))
    result['second_sell_s1_ref'] = chan_result.get('second_sell_s1_ref', np.full(n, -1, dtype=int))
    result['b3_trend_rank'] = chan_result.get('b3_trend_rank', np.zeros(n, dtype=int))
    result['b3_trend_confirmed'] = chan_result.get('b3_trend_confirmed', np.zeros(n, dtype=bool))
    result['b3_breakout_vol_ratio'] = chan_result.get('b3_breakout_vol_ratio', np.zeros(n))
    result['b3_pullback_vol_ratio'] = chan_result.get('b3_pullback_vol_ratio', np.zeros(n))
    result['b3_pullback_shallowness'] = chan_result.get('b3_pullback_shallowness', np.zeros(n))

    # === 统一信号强度 ===
    result['chan_buy_score'] = chan_result['chan_buy_score']
    result['chan_sell_score'] = chan_result['chan_sell_score']

    # === 缠论系统2: 资金流向 (独立系统) ===
    if volume is not None and open_arr is not None:
        cf_signal = compute_capital_flow_signal(
            close_arr, high_arr if high is not None else close_arr,
            low_arr if low is not None else close_arr,
            volume, open_arr, turnover_rate=turnover_rate,
        )
        result['capital_flow_score'] = cf_signal['capital_flow_score']
        result['capital_flow_direction'] = cf_signal['capital_flow_direction']
    else:
        result['capital_flow_score'] = np.zeros(n)
        result['capital_flow_direction'] = np.zeros(n, dtype=int)

    # === 缠论系统3: 资讯热点 (独立系统, 价量代理) ===
    if volume is not None and open_arr is not None:
        ns_signal = compute_news_sentiment_signal(
            close_arr, high_arr if high is not None else close_arr,
            low_arr if low is not None else close_arr,
            volume, open_arr, amplitude=amplitude,
        )
        result['news_sentiment_score'] = ns_signal['news_sentiment_score']
        result['news_sentiment_direction'] = ns_signal['news_sentiment_direction']
    else:
        result['news_sentiment_score'] = np.zeros(n)
        result['news_sentiment_direction'] = np.zeros(n, dtype=int)

    # ==================== 新因子：跳空+量能+笔阶段+力竭风险+顶分型量能 ====================

    # --- 1. gap_breakout_confirm: 跳空缺口B3确认因子 (向量化) ---
    gap_breakout = np.zeros(n)
    if open_arr is not None and n > 1:
        raw_vr_arr = result.get('volume_ratio_raw', np.ones(n))
        gap = np.zeros(n)
        gap[1:] = (open_arr[1:] - close_arr[:-1]) / (close_arr[:-1] + 1e-10)

        gap_up = gap > 0.02
        gap_down = gap < -0.02
        # 高开+爆量(强突破)
        mask = gap_up & (raw_vr_arr > 2.0)
        gap_breakout[mask] = 0.9
        # 高开+温和放量
        mask = gap_up & (raw_vr_arr > 1.3) & (raw_vr_arr <= 2.0)
        gap_breakout[mask] = 0.5
        # 高开+缩量(假突破)
        mask = gap_up & (raw_vr_arr < 0.7) & ~(gap_breakout > 0)
        gap_breakout[mask] = -0.5
        # 高开+正常量
        mask = gap_up & (gap_breakout == 0)
        gap_breakout[mask] = 0.2
        # 低开+放量(恐慌)
        mask = gap_down & (raw_vr_arr > 2.0)
        gap_breakout[mask] = -0.9
        # 低开+缩量(洗盘嫌疑)
        mask = gap_down & (raw_vr_arr < 0.5)
        gap_breakout[mask] = 0.1
        # 低开+正常量
        mask = gap_down & (gap_breakout == 0)
        gap_breakout[mask] = -0.4
        # 连续同向跳空强化
        if n >= 3:
            prev_gap = np.zeros(n)
            prev_gap[2:] = (open_arr[1:-1] - close_arr[:-2]) / (close_arr[:-2] + 1e-10)
            same_dir = (gap * prev_gap > 0) & (np.abs(gap) > 0.01)
            gap_breakout[same_dir] *= 1.3
    result['gap_breakout_confirm'] = np.clip(gap_breakout, -1, 1)

    # --- 1.5. vol_opening_confirm: "放量+次日开盘确认"多K线形态 ---
    # 五维评估（不只两根K线）:
    #   前文: Day T前5日趋势(下跌→反转加分/上涨→加速加分/横盘→突破加分)
    #   量能: Day T量比>1.5(基础), >2.0(强), >3.0(极强)
    #   实体: Day T阳线实体大小(>3%=强阳, <1%=小阳)
    #   确认: Day T+1开盘>前收 AND T+1收盘位置(继续收阳更可靠)
    #   位置: 20日高低位(底部放量=真反转, 高位放量=出货嫌疑)
    vol_open_confirm = np.zeros(n)
    vol_open_strength = np.zeros(n)
    if open_arr is not None and n > 1:
        raw_vr = result.get('volume_ratio_raw', np.ones(n))
        # 预计算上下文指标
        pp20 = result.get('price_position_20', np.full(n, 0.5))
        ret_5d = np.zeros(n)
        ret_5d[5:] = (close_arr[5:] - close_arr[:-5]) / (close_arr[:-5] + 1e-10)
        # 前5日振幅(判断横盘)
        amp_5d = np.zeros(n)
        if high is not None and low is not None and n >= 5:
            for i in range(5, n):
                h5 = np.max(high[i-5:i+1])
                l5 = np.min(low[i-5:i+1])
                amp_5d[i] = (h5 - l5) / (close_arr[i] + 1e-10)

        for i in range(1, n):
            if raw_vr[i-1] < 1.5:
                continue
            if open_arr[i] <= close_arr[i-1]:
                continue

            vol_open_confirm[i] = 1.0
            is_bullish_prev = close_arr[i-1] > open_arr[i-1]
            body_pct = (close_arr[i-1] - open_arr[i-1]) / (open_arr[i-1] + 1e-10)
            strength = 0.0

            # ── 1. 前文趋势 (±0.15) ──
            if i >= 6:
                if ret_5d[i-1] < -0.04:
                    strength += 0.15  # 下跌后放量=反转信号(最强)
                elif ret_5d[i-1] > 0.04:
                    strength += 0.05  # 上涨后放量=加速但位置偏高需谨慎
                else:
                    strength += 0.10  # 横盘后放量=突破蓄力

            # ── 2. 前文振幅 — 横盘突破 (±0.08) ──
            if i >= 5 and amp_5d[i-1] < 0.05:
                strength += 0.08  # 窄幅横盘后放量→蓄力突破
            elif i >= 5 and amp_5d[i-1] > 0.15:
                strength -= 0.05  # 宽幅震荡后放量→方向不确定

            # ── 3. 20日位置 — 底部vs高位 (±0.12) ──
            pos = pp20[i-1] if i >= 20 else 0.5
            if pos < 0.25:
                strength += 0.12  # 底部放量=主力建仓
            elif pos > 0.80:
                strength -= 0.10  # 高位放量=出货嫌疑

            # ── 4. 量能级别 (基础0.35~0.65) ──
            if raw_vr[i-1] > 3.0:
                strength += 0.30
            elif raw_vr[i-1] > 2.0:
                strength += 0.20
            else:
                strength += 0.12

            # ── 5. Day T阳线实体 (+0.05~0.15) ──
            if is_bullish_prev:
                strength += 0.05 + min(0.10, max(0, body_pct) * 2)
            else:
                strength -= 0.10  # 放量收阴=分歧大

            # ── 6. Day T+1收盘位置确认 (±0.10) ──
            # T+1继续收阳(收盘>开盘)=更强确认；T+1收阴=开盘确认但盘中回落
            if close_arr[i] > open_arr[i]:
                strength += 0.08
                # T+1收盘>T收盘=资金继续流入
                if close_arr[i] > close_arr[i-1]:
                    strength += 0.05
            else:
                strength -= 0.06  # 高开低走=诱多嫌疑

            vol_open_strength[i] = float(np.clip(strength, 0.15, 1.0))

    result['vol_opening_confirm'] = vol_open_confirm
    result['vol_opening_strength'] = vol_open_strength

    # --- 2. stroke_phase: 笔阶段识别因子 ---
    # 原理: 价格位置+量能趋势+振幅+趋势方向=判断笔处于形成/加速/力竭阶段
    # 上涨趋势: 正值=笔早期(可买), 负值=笔晚期(风险高)
    # 下跌趋势: 正值=下跌早期(可卖), 负值=下跌晚期(潜在B1)
    stroke_phase = np.zeros(n)
    if n >= 20:
        pp20 = result.get('price_position_20', np.full(n, 0.5))
        ma20 = result.get('ma20', close_arr)
        ma60 = result.get('ma60', close_arr)
        amp_arr = np.zeros(n)
        if high is not None and low is not None:
            amp_arr[1:] = (high_arr[1:] - low_arr[1:]) / (close_arr[1:] + 1e-10)

        for i in range(20, n):
            pp = pp20[i]
            # 趋势方向: 上涨趋势=True, 下跌/横盘=False
            is_uptrend = close_arr[i] > ma20[i] and ma20[i] > ma60[i]

            if is_uptrend:
                # 上涨笔阶段
                if pp < 0.25:
                    pp_score = 0.8  # 低位=笔起点, 买入时机好
                elif pp < 0.60:
                    pp_score = 0.4  # 中位=笔加速, 顺势
                elif pp < 0.85:
                    pp_score = -0.2  # 偏高位=需谨慎
                else:
                    pp_score = -0.7  # 极高位=笔末端风险

                # 量能趋势
                vol_trend = np.mean(vol_arr[i-5:i+1]) / (np.mean(vol_arr[i-15:i-5]) + 1e-10)
                ret_5 = (close_arr[i] - close_arr[i-5]) / (close_arr[i-5] + 1e-10)
                if ret_5 > 0.03 and vol_trend > 1.3:
                    vol_score = 0.3
                elif ret_5 > 0.03 and vol_trend < 0.7:
                    vol_score = -0.5  # 量价背离=力竭
                else:
                    vol_score = 0.0

                # 振幅趋势
                amp_recent = np.mean(amp_arr[i-5:i+1])
                amp_base = np.mean(amp_arr[i-15:i-5]) + 1e-10
                amp_ratio = amp_recent / amp_base
                if amp_ratio > 2.5:
                    amp_score = -0.4
                elif amp_ratio > 1.5:
                    amp_score = 0.3
                else:
                    amp_score = 0.0
            else:
                # 下跌笔阶段: 方向相反
                if pp > 0.75:
                    pp_score = 0.8  # 高位=下跌笔起点, 卖出时机
                elif pp > 0.40:
                    pp_score = 0.4  # 中高位=下跌加速
                elif pp > 0.15:
                    pp_score = -0.2  # 偏低=下跌末端
                else:
                    pp_score = -0.7  # 极低位=超跌, 潜在B1

                # 量能趋势: 下跌放量=加速, 下跌缩量=衰竭
                vol_trend = np.mean(vol_arr[i-5:i+1]) / (np.mean(vol_arr[i-15:i-5]) + 1e-10)
                ret_5 = (close_arr[i] - close_arr[i-5]) / (close_arr[i-5] + 1e-10)
                if ret_5 < -0.03 and vol_trend > 1.3:
                    vol_score = 0.3  # 放量下跌=加速
                elif ret_5 < -0.03 and vol_trend < 0.7:
                    vol_score = -0.5  # 缩量下跌=衰竭
                else:
                    vol_score = 0.0

                # 振幅趋势
                amp_recent = np.mean(amp_arr[i-5:i+1])
                amp_base = np.mean(amp_arr[i-15:i-5]) + 1e-10
                amp_ratio = amp_recent / amp_base
                if amp_ratio > 2.5:
                    amp_score = -0.4  # 振幅极值+下跌=恐慌底
                elif amp_ratio > 1.5:
                    amp_score = 0.3   # 振幅放大+下跌=加速
                else:
                    amp_score = 0.0

            stroke_phase[i] = pp_score + vol_score + amp_score

    result['stroke_phase'] = np.clip(stroke_phase, -1, 1)

    # --- 3. exhaustion_risk: 力竭风险因子 ---
    # 原理: 高位+量能背离+高振幅+动量减速 = S2/S3卖出预警
    exhaustion = np.zeros(n)
    if n >= 20:
        for i in range(20, n):
            pp = pp20[i]
            if pp < 0.7:
                continue  # 非高位, 无力竭风险

            risk = 0.0

            # 信号1: 价格在新高但量能萎缩(量价背离)
            if close_arr[i] >= np.max(close_arr[i-10:i]):
                vol_now = np.mean(vol_arr[i-3:i+1])
                vol_peak = np.max(vol_arr[i-15:i-5])
                if vol_peak > 0 and vol_now / vol_peak < 0.5:
                    risk += 0.4

            # 信号2: 振幅异常放大(近期振幅超历史2倍)
            amp_now = np.mean(amp_arr[i-3:i+1]) if i >= 3 else amp_arr[i]
            amp_hist = np.mean(amp_arr[i-20:i-5]) + 1e-10
            if amp_now / amp_hist > 2.0:
                risk += 0.3

            # 信号3: 价格连涨但MACD柱缩短(顶背离)
            if i >= 8 and result.get('macd_hist') is not None:
                macd_hist_arr = result['macd_hist']
                if close_arr[i] > close_arr[i-8] and macd_hist_arr[i] < macd_hist_arr[i-8]:
                    risk += 0.3

            # 信号4: 跳空高开后回落(高位衰竭缺口)
            if i >= 1 and open_arr is not None:
                gap_i = (close_arr[i] - close_arr[i-1]) / (close_arr[i-1] + 1e-10)
                intra_ret = (close_arr[i] - open_arr[i]) / (open_arr[i] + 1e-10) if open_arr[i] > 0 else 0
                if gap_i > 0.02 and intra_ret < -0.02:
                    risk += 0.2  # 高开后低走=衰竭缺口

            exhaustion[i] = min(risk, 1.0)

    result['exhaustion_risk'] = exhaustion

    # --- 4. top_fractal_volume: 顶分型量能背离因子 ---
    # 原理: 顶分型形成时, 对比本次高点量能与前次高点量能
    # 量能递减=背离=高概率反转 (缠论: "量在价先")
    tf_volume = np.zeros(n)
    if n >= 30:
        top_fx = result.get('top_fractals', np.zeros(n, dtype=bool))
        for i in range(30, n):
            if not top_fx[i]:
                continue
            # 找到前两个顶分型的位置
            prev_tops = []
            for j in range(i-1, max(0, i-60), -1):
                if top_fx[j]:
                    prev_tops.append(j)
                    if len(prev_tops) >= 2:
                        break
            if len(prev_tops) < 1:
                continue

            cur_start = max(0, i-3)
            cur_vol = np.max(vol_arr[cur_start:i+1])  # 当前分型附近的最大量
            prev_start = max(0, prev_tops[0]-3)
            prev_vol = np.max(vol_arr[prev_start:prev_tops[0]+1])  # 前一个分型的量

            if prev_vol > 0:
                vol_ratio_vs_prev = cur_vol / prev_vol
                # 量能递减+价格新高=顶背离
                if vol_ratio_vs_prev < 0.7:
                    if len(prev_tops) >= 2 and close_arr[i] > close_arr[prev_tops[0]]:
                        tf_volume[i] = -0.8  # 强顶背离: 价创新高量递减
                    else:
                        tf_volume[i] = -0.4
                elif vol_ratio_vs_prev > 2.0:
                    tf_volume[i] = 0.5  # 放量过前高=突破有效, 顶分型可能失败

    result['top_fractal_volume'] = tf_volume

    # === 新因子增强缠论买卖信号 ===
    # Apply boosts/reductions to chan_buy_score and chan_sell_score
    _apply_chan_signal_boosts(result, n)

    # === 熊市防御因子 (Phase 2.2) ===
    # 1. downside_vol_ratio: 下跌波动占比, 高值=大部分波动是下跌=风险大
    dd20 = result.get('downside_dev_20', np.zeros(n))
    vol20 = result.get('volatility_20', np.ones(n) * 0.01)
    result['downside_vol_ratio'] = np.tanh(-(dd20 / (vol20 + 1e-8) - 0.5) * 5)

    # 2. vol_stability: 波动率稳定性, 短/长期波动率比越接近1越稳定
    vol60_raw = _rolling_std(result['ret'], 60) * np.sqrt(252)
    vol_stab = np.zeros(n)
    valid = (vol60_raw > 0.01)
    vol_stab[valid] = 1.0 - np.abs(result['volatility_20'][valid] / vol60_raw[valid] - 1.0)
    vol_stab = np.clip(vol_stab, 0.0, 1.0)
    result['vol_stability'] = np.tanh((vol_stab - 0.5) * 5)

    # 3. ret_asymmetry: 收益不对称性, 正收益日占比 > 负收益 = 健康
    ret_arr = result['ret']
    pos_days = _sma((ret_arr > 0).astype(np.float32), 60)
    neg_days = _sma((ret_arr < 0).astype(np.float32), 60)
    asym = np.zeros(n)
    total_days = pos_days + neg_days + 1e-10
    asym = (pos_days - neg_days) / total_days  # [-1, 1]
    result['ret_asymmetry'] = np.tanh(asym * 3)

    # 4. price_quality: 价格质量 = 低回撤 + 紧贴均线 + 低波动
    dd60 = np.zeros(n)
    if n > 60:
        roll_max_60 = _rolling_max(close_arr, 60)
        dd60 = (close_arr - roll_max_60) / (roll_max_60 + 1e-10)
    ma_dist = np.abs(close_arr / (ma20 + 1e-10) - 1.0)  # 偏离MA20程度
    pq = (1.0 + dd60) * 0.4 + (1.0 - ma_dist * 5).clip(0, 1) * 0.4 + (1.0 - result['volatility_20']).clip(0, 1) * 0.2
    result['price_quality'] = np.tanh((pq - 0.5) * 5)

    return result


def _apply_chan_signal_boosts(result: Dict[str, np.ndarray], n: int) -> None:
    """用新因子增强/削弱缠论买卖信号强度

    gap_breakout_confirm: 跳空放量=中枢突破确认 → B3加分
    stroke_phase: 笔早期=买点质量高, 笔晚期=风险大
    exhaustion_risk: 高位力竭 → 削弱买入, 增强卖出
    top_fractal_volume: 顶分型量能背离 → 增强卖出
    """
    chan_buy = result.get('chan_buy_score')
    chan_sell = result.get('chan_sell_score')
    if chan_buy is None or chan_sell is None:
        return

    gbc = result.get('gap_breakout_confirm', np.zeros(n))
    sp = result.get('stroke_phase', np.zeros(n))
    er = result.get('exhaustion_risk', np.zeros(n))
    tfv = result.get('top_fractal_volume', np.zeros(n))

    for i in range(n):
        # --- 买入信号调整 ---
        if chan_buy[i] > 0.05:
            multiplier = 1.0

            # gap_breakout_confirm: 跳空放量=强突破, B3加分; 假突破=减分
            g = gbc[i]
            if g > 0.3:
                multiplier += g * 0.25
            elif g < -0.3:
                multiplier += g * 0.15  # 跳空下跌=买入风险

            # stroke_phase: 上涨笔早期(>0.3)=买点质量高; 笔晚期(< -0.3)=追高风险
            s = sp[i]
            if s > 0.3:
                multiplier += s * 0.15
            elif s < -0.3:
                multiplier += s * 0.15  # s为负, 减少乘数

            # exhaustion_risk: 高位力竭=买入风险大
            e = er[i]
            if e > 0.3:
                multiplier -= e * 0.4

            # top_fractal_volume: 量能背离=顶部信号
            t = tfv[i]
            if t < -0.3:
                multiplier += t * 0.25  # t为负, 相当于减分

            result['chan_buy_score'][i] = float(np.clip(
                chan_buy[i] * max(multiplier, 0.25), 0.0, 1.0))

        # --- 卖出信号调整 ---
        if chan_sell[i] > 0.05:
            multiplier = 1.0

            # exhaustion_risk: 力竭信号强烈=卖出加分
            e = er[i]
            if e > 0.3:
                multiplier += e * 0.35

            # top_fractal_volume: 顶背离=强烈卖出信号
            t = tfv[i]
            if t < -0.5:
                multiplier += abs(t) * 0.3
            elif t < -0.2:
                multiplier += abs(t) * 0.15

            # gap_breakout_confirm: 向下跳空放量=恐慌抛售, 卖出加分
            g = gbc[i]
            if g < -0.3:
                multiplier += abs(g) * 0.2

            # stroke_phase: 下跌笔早期(>0.3)=卖出时机好; 笔晚期(< -0.3)=超跌可能反弹
            s = sp[i]
            if s > 0.3:
                multiplier += s * 0.15   # 下跌早期+卖出=正确方向
            elif s < -0.3:
                multiplier += s * 0.15   # s为负, 减少卖出强度(底部不杀跌)

            result['chan_sell_score'][i] = float(np.clip(
                chan_sell[i] * min(multiplier, 1.6), 0.0, 1.0))


def get_default_params() -> Dict:
    """获取默认参数"""
    return {
        'ema_periods': [5, 10, 20, 60],
        'ma_periods': [5, 10, 20, 30, 60],
        'rsi_periods': [6, 8, 10, 14],
        'bb_window': 20,
        'bb_std': 2,
        'momentum_periods': [3, 5, 10, 20, 30],
        'volatility_periods': [5, 10, 20],
        'atr_periods': [10, 14, 20],
        'volume_ma_period': 20,
        # 缠论参数
        'divergence': {
            'lookback': 20,
            'peak_trough_lookback': 5,
            'strength_threshold': 0.3,
            'verify_trend': True,
        },
        'structure': {
            'pivot_min_overlap': 3,
            'pivot_zone_buffer': 0.02,
            'min_trend_bars': 8,
            'zhongyin_threshold': 0.02,
        },
    }


# ==================== 底层计算函数 ====================

def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均 (cumsum实现, 比Python循环快50-100x)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cs = np.cumsum(np.insert(arr.astype(float), 0, 0))
    result[window - 1:] = (cs[window:] - cs[:-window]) / window
    return result


@njit
def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均 (Numba JIT)"""
    result = np.zeros_like(arr, dtype=np.float64)
    result[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    """RSI指标 (向量化)"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = _sma(gain, window)
    avg_loss = _sma(loss, window)
    rs = avg_gain / (avg_loss + 1e-10)
    result = np.full(n, np.nan)
    result[window - 1:] = 100 - (100 / (1 + rs[window - 1:]))
    return result


def _bollinger(close: np.ndarray, window: int, num_std: float) -> tuple:
    """布林带 (向量化, 含当天, 与原loop close[i-window+1:i+1]行为一致)"""
    middle = _sma(close, window)
    # 布林带使用含当天的std，与_rolling_std(不含当天)不同
    sma_sq = _sma(close * close, window)
    std = np.full(len(close), np.nan)
    std[window - 1:] = np.sqrt(np.maximum(0, sma_sq[window - 1:] - middle[window - 1:] ** 2))
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """ATR指标"""
    n = len(close)
    tr = np.zeros(n)
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - np.roll(close, 1)[1:]),
            np.abs(low[1:] - np.roll(close, 1)[1:])
        )
    )
    tr[0] = high[0] - low[0]
    return _sma(tr, window)


def _shift(arr: np.ndarray, periods: int, safe: bool = False) -> np.ndarray:
    """数组位移

    Args:
        arr: 输入数组
        periods: 位移周期
        safe: 如果为True，位移后价格为0或负数的位置设为0
    """
    result = np.zeros_like(arr, dtype=float)
    result[periods:] = arr[:-periods]
    result[:periods] = np.nan

    if safe:
        shifted_vals = arr[:-periods]
        result[periods:][shifted_vals <= 0] = np.nan

    return result


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动Pearson相关系数 (向量化, 不含当天 arr[i-window:i], 与原loop行为一致)"""
    n = len(x)
    result = np.full(n, np.nan)
    if n < window + 1:
        return result
    ex = _sma(x, window)
    ey = _sma(y, window)
    exy = _sma(x * y, window)
    sx = _rolling_std(x, window)
    sy = _rolling_std(y, window)
    # 原语义: corr[i] = (sma_xy[i-1] - sma_x[i-1]*sma_y[i-1]) / (std_x[i]*std_y[i])
    denom = sx[window:] * sy[window:]
    numer = exy[window-1:n-1] - ex[window-1:n-1] * ey[window-1:n-1]
    valid = (denom > 1e-15) & ~np.isnan(denom)
    result[window:][valid] = (numer[valid] / denom[valid])
    result = np.clip(result, -1, 1)
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差 (不含当天, arr[i-window:i], 与原Python循环行为一致)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window + 1:
        return result
    sma = _sma(arr, window)
    sma_sq = _sma(arr * arr, window)
    # _sma含当天(sma[window-1]=mean(arr[0:window]))，原语义需result[window]=std(arr[0:window])
    result[window:] = np.sqrt(np.maximum(0, sma_sq[window-1:n-1] - sma[window-1:n-1] ** 2))
    return result


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最大值 (向量化)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window + 1:
        return result
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(arr, window)
    result[window:] = sw.max(axis=1)[:n - window]
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最小值 (向量化)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window + 1:
        return result
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(arr, window)
    result[window:] = sw.min(axis=1)[:n - window]
    return result


@njit
def _calc_smart_money_flow(returns: np.ndarray, vol_arr: np.ndarray, n: int) -> np.ndarray:
    """主力资金流向推断 (Numba JIT) — 与原Python循环逻辑完全一致"""
    mf_score = np.zeros(n)
    for i in range(5, n):
        score = 0.0
        for j in range(i - 4, i + 1):
            ret_day = returns[j] if j > 0 else 0.0
            start = max(0, j - 20)
            vol_mean = np.mean(vol_arr[start:j + 1])
            vr = vol_arr[j] / (vol_mean + 1e-6)
            if ret_day > 0 and vr > 1.2:
                if vr > 3.0:
                    vr = 3.0
                score += ret_day * vr * 2.0
            elif ret_day > 0 and vr < 0.7:
                score += ret_day * 0.5
            elif ret_day < 0 and vr > 1.5:
                score += ret_day * 0.3
            elif ret_day < 0 and vr < 0.7:
                score += ret_day * 1.0
            else:
                score += ret_day
        mf_score[i] = score / 5.0
    return mf_score


@njit
def _njit_compute_alpha_factors(returns, n):
    """Numba JIT: 在单次遍历中计算 skewness_20, kurtosis_20, downside_dev_10/20, volatility_skew, residual_momentum

    逻辑与原Python循环完全一致，仅用Numba加速。
    """
    skew_20 = np.zeros(n)
    kurt_20 = np.zeros(n)
    vol_skew = np.zeros(n)
    down_dev_10 = np.zeros(n)
    down_dev_20 = np.zeros(n)
    residual_mom = np.zeros(n)

    for i in range(20, n):
        r = np.empty(20)
        for j in range(20):
            r[j] = returns[i-20+j]

        # 均值和标准差 (population)
        mean_r = 0.0
        for j in range(20):
            mean_r += r[j]
        mean_r /= 20.0

        var_r = 0.0
        for j in range(20):
            diff = r[j] - mean_r
            var_r += diff * diff
        var_r /= 20.0
        std_r = var_r ** 0.5

        if std_r > 1e-10:
            # 偏度: mean((x-mean)^3) / std^3
            skew_sum = 0.0
            for j in range(20):
                diff = r[j] - mean_r
                skew_sum += diff * diff * diff
            skew_20[i] = (skew_sum / 20.0) / (std_r * std_r * std_r + 1e-10)

            # 峰度: mean((x-mean)^4) / std^4 - 3
            kurt_sum = 0.0
            for j in range(20):
                diff = r[j] - mean_r
                kurt_sum += diff * diff * diff * diff
            kurt_20[i] = (kurt_sum / 20.0) / (var_r * var_r) - 3.0

            # 波动率偏度: up_std / down_std - 1 (与原版一致: np.std(up_r) / np.std(down_r) - 1)
            up_sum = 0.0
            up_cnt = 0
            down_sum = 0.0
            down_cnt = 0
            for j in range(20):
                if r[j] > 0:
                    up_sum += r[j]
                    up_cnt += 1
                elif r[j] < 0:
                    down_sum += r[j]
                    down_cnt += 1

            up_std = 0.0
            if up_cnt > 2:
                up_mean = up_sum / up_cnt
                up_var = 0.0
                for j in range(20):
                    if r[j] > 0:
                        up_var += (r[j] - up_mean) ** 2
                up_std = (up_var / up_cnt) ** 0.5  # np.std 默认 ddof=0

            down_std = 0.0
            if down_cnt > 2:
                down_mean = down_sum / down_cnt
                down_var = 0.0
                for j in range(20):
                    if r[j] < 0:
                        down_var += (r[j] - down_mean) ** 2
                down_std = (down_var / down_cnt) ** 0.5

            if down_std > 1e-10:
                vol_skew[i] = up_std / down_std - 1.0

            # 残差动量: sum(residual[-5:]) / std(residual)
            # 与原版一致: stock_rets = returns[i-19:i+1], residual = stock_rets - mean(stock_rets)
            # 取最后5个残差的均值 / std(residual)
            res_arr = np.empty(20)
            res_mean = 0.0
            for j in range(20):
                val = returns[i-19+j]
                res_arr[j] = val
                res_mean += val
            res_mean /= 20.0

            res_std = 0.0
            for j in range(20):
                diff = res_arr[j] - res_mean
                res_std += diff * diff
            res_std = (res_std / 20.0) ** 0.5

            if res_std > 1e-10:
                res_sum = 0.0
                for j in range(15, 20):  # 最后5个残差
                    res_sum += res_arr[j] - res_mean
                residual_mom[i] = res_sum / res_std

    # 下行偏差 (与原版一致: np.std(neg_r) where neg_r = r[r < 0])
    for i in range(10, n):
        neg_sum = 0.0
        neg_cnt = 0
        for j in range(10):
            val = returns[i-10+j]
            if val < 0:
                neg_sum += val
                neg_cnt += 1
        if neg_cnt > 2:
            neg_mean = neg_sum / neg_cnt
            neg_var = 0.0
            for j in range(10):
                val = returns[i-10+j]
                if val < 0:
                    neg_var += (val - neg_mean) ** 2
            down_dev_10[i] = (neg_var / neg_cnt) ** 0.5

    for i in range(20, n):
        neg_sum = 0.0
        neg_cnt = 0
        for j in range(20):
            val = returns[i-20+j]
            if val < 0:
                neg_sum += val
                neg_cnt += 1
        if neg_cnt > 2:
            neg_mean = neg_sum / neg_cnt
            neg_var = 0.0
            for j in range(20):
                val = returns[i-20+j]
                if val < 0:
                    neg_var += (val - neg_mean) ** 2
            down_dev_20[i] = (neg_var / neg_cnt) ** 0.5

    return skew_20, kurt_20, vol_skew, down_dev_10, down_dev_20, residual_mom


@njit
def _njit_compute_novel_factors(close_arr, high_arr, low_arr, vol_arr, returns, n):
    """Numba JIT: 在单次遍历中计算 wash_sale_score, vol_price_breakout, volume_contraction, relative_strength, consolidation_breakout

    逻辑与原Python循环完全一致，仅用Numba加速。
    """
    wash_score = np.zeros(n)
    vol_breakout = np.zeros(n)
    vol_contract = np.zeros(n)
    rs_score = np.zeros(n)
    cb_score = np.zeros(n)

    for i in range(20, n):
        # === wash_sale_score (与原版完全一致) ===
        # 找过去20天内的最大单日跌幅
        min_ret = 1e10
        min_ret_idx = 0
        for j in range(20):
            ret_val = returns[i-20+j]
            if ret_val < min_ret:
                min_ret = ret_val
                min_ret_idx = j

        if min_ret < -0.03:
            avg_vol = 0.0
            for j in range(20):
                avg_vol += vol_arr[i-20+j]
            avg_vol /= 20.0

            if vol_arr[i-20+min_ret_idx] > avg_vol * 1.5:
                post_start = min_ret_idx
                # post_low = recent_close[min_ret_idx:]
                if post_start < 20:
                    post0 = close_arr[i-20+post_start]
                    post_last = close_arr[i-1]
                    recovery = (post_last - post0) / (abs(post0) + 1e-10)
                    if recovery > 0.5 * abs(min_ret):
                        # vol_contracting 检查 (与原版一致)
                        vol_after_len = 20 - min_ret_idx
                        vol_contracting = False
                        if vol_after_len >= 6:
                            early_sum = 0.0
                            early_cnt = min(3, vol_after_len)
                            for jj in range(early_cnt):
                                early_sum += vol_arr[i-20+min_ret_idx+jj]
                            late_sum = 0.0
                            late_start = max(0, vol_after_len - 3)
                            for jj in range(late_start, vol_after_len):
                                late_sum += vol_arr[i-20+min_ret_idx+jj]
                            late_cnt = vol_after_len - late_start
                            if late_cnt > 0 and early_cnt > 0:
                                vol_contracting = (late_sum / late_cnt) < (early_sum / early_cnt)
                        wash_score[i] = abs(min_ret) * 3.0
                        if vol_contracting:
                            wash_score[i] += 0.2

        # === vol_price_breakout (与原版完全一致) ===
        avg_vol_20 = 0.0
        for j in range(20):
            avg_vol_20 += vol_arr[i-20+j]
        avg_vol_20 /= 20.0

        if vol_arr[i] > avg_vol_20 * 2.0:
            # np.max(recent_high_20[:-1]) = max(high_arr[i-20:i-1])
            max_high_excl = 0.0
            for j in range(19):
                val = high_arr[i-20+j]
                if val > max_high_excl:
                    max_high_excl = val
            if close_arr[i] > max_high_excl:
                vol_breakout[i] = 1.0
            elif close_arr[i] > max_high_excl * 0.98:
                vol_breakout[i] = 0.5

        # === volume_contraction (与原版完全一致) ===
        # ma20_now = np.mean(close_arr[i-19:i+1])
        ma20_now = 0.0
        for j in range(i-19, i+1):
            ma20_now += close_arr[j]
        ma20_now /= 20.0

        # ma20_before = np.mean(close_arr[i-24:i-4]) if i >= 24 else ma20_now
        ma20_before = ma20_now
        if i >= 24:
            ma20_before = 0.0
            for j in range(i-24, i-4):
                ma20_before += close_arr[j]
            ma20_before /= 20.0

        trend_up = ma20_now > ma20_before

        # recent_5vol = vol_arr[i-4:i+1]
        recent_5v = 0.0
        for j in range(i-4, i+1):
            recent_5v += vol_arr[j]
        recent_5v /= 5.0

        # prev_5vol = vol_arr[i-9:i-4] if i >= 9 else recent_5vol
        prev_5v = recent_5v
        if i >= 9:
            prev_5v = 0.0
            for j in range(i-9, i-4):
                prev_5v += vol_arr[j]
            prev_5v /= 5.0

        vol_shrinking = recent_5v < prev_5v * 0.8

        # price_drawdown = (np.max(close_arr[i-10:i+1]) - close_arr[i]) / ...
        # 原版: i-10:i+1 → 11 elements
        max_close_10 = 0.0
        for j in range(i-10, i+1):
            if close_arr[j] > max_close_10:
                max_close_10 = close_arr[j]

        price_dd = (max_close_10 - close_arr[i]) / (max_close_10 + 1e-10)
        mild_pullback = price_dd > 0.0 and price_dd < 0.05

        if trend_up and vol_shrinking and mild_pullback:
            vol_contract[i] = 1.0 - price_dd * 10.0

        # === relative_strength (与原版完全一致) ===
        ret_20d = (close_arr[i] - close_arr[i-20]) / (close_arr[i-20] + 1e-10)
        ret_10d = (close_arr[i] - close_arr[i-10]) / (close_arr[i-10] + 1e-10)
        ret_5d = (close_arr[i] - close_arr[i-5]) / (close_arr[i-5] + 1e-10)

        # momentum_consistency (与原版一致: 1 for positive, -0.5 for negative)
        mc = 0.0
        if ret_5d > 0.0:
            mc += 1.0
        else:
            mc -= 0.5
        if ret_10d > 0.0:
            mc += 1.0
        else:
            mc -= 0.5
        if ret_20d > 0.0:
            mc += 1.0
        else:
            mc -= 0.5

        # vol_20d = np.std(returns[i-19:i+1]) (与原版一致: population std)
        vol_mean = 0.0
        for j in range(i-19, i+1):
            vol_mean += returns[j]
        vol_mean /= 20.0

        vol_var = 0.0
        for j in range(i-19, i+1):
            diff = returns[j] - vol_mean
            vol_var += diff * diff
        vol_20d = (vol_var / 20.0) ** 0.5

        if vol_20d > 1e-6:
            rs_score[i] = (ret_20d / vol_20d) * (0.5 + 0.5 * max(0.0, mc / 3.0))

        # === consolidation_breakout (与原版完全一致) ===
        # recent range: close[i-9:i+1], high[i-9:i+1], low[i-9:i+1]
        rec10_h = 0.0
        rec10_l = 1e10
        rec10_v = 0.0
        rec10_c = 0.0
        for j in range(i-9, i+1):
            rec10_c += close_arr[j]
            rec10_v += vol_arr[j]
            if high_arr[j] > rec10_h:
                rec10_h = high_arr[j]
            if low_arr[j] < rec10_l:
                rec10_l = low_arr[j]
        rec10_c /= 10.0
        rec10_v /= 10.0
        recent_range = (rec10_h - rec10_l) / (rec10_c + 1e-10)

        # prev range: close[i-19:i-9], high[i-19:i-9], low[i-19:i-9]
        prev_range = recent_range * 2.0
        if i >= 19:
            prev_h = 0.0
            prev_l = 1e10
            prev_c = 0.0
            prev_v = 0.0
            for j in range(i-19, i-9):
                prev_c += close_arr[j]
                prev_v += vol_arr[j]
                if high_arr[j] > prev_h:
                    prev_h = high_arr[j]
                if low_arr[j] < prev_l:
                    prev_l = low_arr[j]
            prev_c /= 10.0
            prev_v /= 10.0
            prev_range = (prev_h - prev_l) / (prev_c + 1e-10)

        range_narrowing = recent_range < prev_range * 0.7
        vol_shrink = rec10_v < prev_v * 0.8

        # price flatness: std(close[i-9:i+1])
        c_std = 0.0
        for j in range(i-9, i+1):
            diff = close_arr[j] - rec10_c
            c_std += diff * diff
        c_std = (c_std / 10.0) ** 0.5
        price_flat = c_std / (rec10_c + 1e-10) < 0.03

        # trend_ok = ma20 > ma60
        ma20_val = 0.0
        for j in range(i-19, i+1):
            ma20_val += close_arr[j]
        ma20_val /= 20.0

        ma60_val = ma20_val
        if i >= 59:
            ma60_val = 0.0
            for j in range(i-59, i+1):
                ma60_val += close_arr[j]
            ma60_val /= 60.0

        trend_ok = ma20_val > ma60_val

        if range_narrowing and vol_shrink and price_flat and trend_ok:
            cb_score[i] = 1.0
        elif range_narrowing and (vol_shrink or price_flat):
            cb_score[i] = 0.4

    return wash_score, vol_breakout, vol_contract, rs_score, cb_score


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD指标"""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ==================== 涨停缩量回踩5日线低吸 ====================

def _compute_limit_pullback_score(ind: Dict[str, np.ndarray], idx: int) -> float:
    """涨停后缩量回踩5日线低吸因子

    模式：涨停启动 → 缩量整理（非出货）→ 回踩MA5 → 低吸机会
    得分范围：tanh压缩到(-1, 1)，正值为低吸买入信号
    """
    close = ind['close']
    high = ind.get('high', close)
    volume = ind.get('volume', None)
    if volume is None:
        turnover = ind.get('turnover_rate', None)
        vol_arr = turnover if turnover is not None else np.ones(len(close))
    else:
        vol_arr = volume

    n = len(close)
    if idx < 25:
        return 0.0

    # 1. 寻找最近涨停日（20天内）
    limit_up_idx = -1
    lookback = min(idx - 1, 20)
    for i in range(idx - 1, idx - lookback - 1, -1):
        if close[i - 1] <= 0 or high[i] <= 0:
            continue
        daily_ret = (close[i] - close[i - 1]) / close[i - 1]
        close_high_ratio = close[i] / high[i]
        # A股涨停: 涨幅>=9.5%且收盘接近最高价(>=97%)
        if daily_ret >= 0.095 and close_high_ratio >= 0.97:
            limit_up_idx = i
            break

    if limit_up_idx < 0:
        return 0.0

    days_since = idx - limit_up_idx
    if days_since < 1 or days_since > 15:
        return 0.0  # 涨停当日不追/太久远失效

    # 2. 缩量判断: 涨停后均量 vs 涨停日量
    limit_vol = vol_arr[limit_up_idx]
    post_vol = np.mean(vol_arr[limit_up_idx + 1:idx + 1]) if idx > limit_up_idx else limit_vol
    vol_score = 0.0
    if limit_vol > 0:
        ratio = post_vol / limit_vol
        if ratio < 0.5:
            vol_score = 1.0
        elif ratio < 0.8:
            vol_score = 1.0 - (ratio - 0.5) / 0.3
        else:
            vol_score = 0.0

    # 3. 回踩MA5判断
    ma5_arr = ind.get('ma5', None)
    if ma5_arr is None:
        ma5_arr = np.array([np.mean(close[max(0, j - 4):j + 1]) for j in range(n)])
    ma5_val = ma5_arr[idx]
    if ma5_val <= 0:
        return 0.0

    price_vs_ma5 = close[idx] / ma5_val
    if 0.97 <= price_vs_ma5 <= 1.02:
        dist = abs(price_vs_ma5 - 1.0)
        pullback_score = 1.0 - dist / 0.03
    elif 0.95 <= price_vs_ma5 < 0.97:
        pullback_score = (price_vs_ma5 - 0.95) / 0.02 * 0.5
    elif 1.02 < price_vs_ma5 <= 1.05:
        pullback_score = (1.05 - price_vs_ma5) / 0.03 * 0.5
    else:
        pullback_score = 0.0

    # 4. MA5趋势: 走平或上行才能低吸
    ma5_past = ma5_arr[max(0, idx - 5)]
    ma5_slope = (ma5_val - ma5_past) / ma5_past if ma5_past > 0 else 0
    trend_score = 1.0 if ma5_slope >= -0.005 else max(0.0, 1.0 + ma5_slope * 100)

    # 5. 综合得分
    score = vol_score * 0.35 + pullback_score * 0.40 + trend_score * 0.25
    return float(np.tanh(score * 3))


# ==================== 组合因子计算 ====================

def compute_composite_factors(ind: Dict[str, np.ndarray], idx: int, fund_score: float = 0.0) -> Dict[str, float]:
    """
    计算组合因子（用于单个时间点）

    Args:
        ind: calculate_indicators返回的指标字典
        idx: 时间点索引
        fund_score: 基本面评分（压缩后值，来自compress_fundamental_factor）

    Returns:
        组合因子字典
    """
    n = len(ind['close'])
    if idx < 20:
        return {}

    result = {}

    # 基础值
    m10 = ind['mom_10'][idx] if not np.isnan(ind['mom_10'][idx]) else 0
    m20 = ind['mom_20'][idx] if not np.isnan(ind['mom_20'][idx]) else 0
    v10 = ind['volatility_10'][idx] if not np.isnan(ind['volatility_10'][idx]) else 0
    v20 = ind['volatility_20'][idx] if not np.isnan(ind['volatility_20'][idx]) else 0
    rsi = ind['rsi_14'][idx] if not np.isnan(ind['rsi_14'][idx]) else 50

    # 趋势动量 - 使用arctan压缩防止极端值
    # 根因：m20可能很大（如50%涨幅），乘以2.1后更大
    # 解决：使用arctan压缩，保留单调性同时限制值域
    # 用m10方向的sigmoid做连续过渡, 避免if/else硬截断导致m10略负时完全归零
    m10_gate = 1.0 / (1.0 + np.exp(-m10 * 30))  # sigmoid: m10>0→≈1, m10<0→≈0
    result['trend_mom_v41'] = np.tanh(m20 * 3) * m10_gate
    result['trend_mom_v24'] = np.tanh(m20 * 3) * m10_gate
    result['trend_mom_v46'] = np.tanh(m20 * 2.5) * m10_gate

    # 动量×低波动 - tanh压缩
    result['mom_x_vol_20_20'] = np.tanh(m20 * v20 * 20)
    result['mom_x_vol_20_10'] = np.tanh(m20 * v10 * 20)
    result['mom_x_vol_10_20'] = np.tanh(m10 * v20 * 20)
    result['mom_x_vol_10_10'] = np.tanh(m10 * v10 * 20)

    # 动量差异 - tanh压缩
    mom_5 = ind['mom_5'][idx] if not np.isnan(ind['mom_5'][idx]) else 0
    result['mom_diff_5_20'] = np.tanh((mom_5 - m20) * 5)
    result['mom_diff_10_20'] = np.tanh((m10 - m20) * 5)

    # RSI因子 - 已经在[-0.5, 0.5]范围，无需压缩
    result['rsi_factor'] = (rsi - 50) / 100

    # 波动率因子 - 使用tanh压缩
    result['volatility'] = np.tanh(-v20 * 20)

    # 成交量因子 - 从ind获取（已在calculate_indicators中压缩）
    result['volume_ratio'] = ind['volume_ratio'][idx] if not np.isnan(ind['volume_ratio'][idx]) else 0.0

    # 布林带宽度 - 使用tanh压缩
    bb_w = ind['bb_width_20'][idx] if not np.isnan(ind['bb_width_20'][idx]) else 0
    result['bb_width_20'] = np.tanh(bb_w * 2)

    # 复合因子
    trend_mom = result.get('trend_mom_v41', 0)
    rsi_f = result.get('rsi_factor', 0)
    result['V41_RSI_915'] = trend_mom * 0.915 + rsi_f * 0.085

    # 其他复合因子
    if 'rsi_vol_combo' in ind:
        result['rsi_vol_combo'] = ind['rsi_vol_combo'][idx]
    if 'bb_rsi_combo' in ind:
        result['bb_rsi_combo'] = ind['bb_rsi_combo'][idx]
    if 'ret_vol_ratio_10' in ind:
        result['ret_vol_ratio_10'] = ind['ret_vol_ratio_10'][idx]
    if 'ret_vol_ratio_20' in ind:
        result['ret_vol_ratio_20'] = ind['ret_vol_ratio_20'][idx]
    if 'momentum_reversal' in ind:
        result['momentum_reversal'] = ind['momentum_reversal'][idx]
    if 'momentum_acceleration' in ind:
        result['momentum_acceleration'] = ind['momentum_acceleration'][idx]

    # tech_fund_combo: 技术+基本面组合
    result['tech_fund_combo'] = result.get('trend_mom_v41', 0) * 0.7 + result.get('rsi_factor', 0) * 0.1 + fund_score * 0.2

    # 新因子: trend_vol + vol_confirm
    if 'trend_vol' in ind:
        result['trend_vol'] = ind['trend_vol'][idx]
    if 'vol_confirm' in ind:
        result['vol_confirm'] = ind['vol_confirm'][idx]

    # 换手率因子
    if 'inv_turnover' in ind:
        result['inv_turnover'] = ind['inv_turnover'][idx]
    if 'turnover_shrink' in ind:
        result['turnover_shrink'] = ind['turnover_shrink'][idx]

    # 流动性因子
    if 'illiq_20' in ind:
        result['illiq_20'] = ind['illiq_20'][idx] if not np.isnan(ind['illiq_20'][idx]) else 0.0
    if 'turnover_stability' in ind:
        result['turnover_stability'] = ind['turnover_stability'][idx] if not np.isnan(ind['turnover_stability'][idx]) else 0.0

    # 下行风险因子
    if 'low_downside' in ind:
        result['low_downside'] = ind['low_downside'][idx]

    # === 新增Alpha因子 ===
    if 'overnight_ret' in ind:
        result['overnight_ret'] = ind['overnight_ret'][idx] if not np.isnan(ind['overnight_ret'][idx]) else 0.0
    if 'intraday_ret' in ind:
        result['intraday_ret'] = ind['intraday_ret'][idx] if not np.isnan(ind['intraday_ret'][idx]) else 0.0
    if 'skewness_20' in ind:
        result['skewness_20'] = ind['skewness_20'][idx] if not np.isnan(ind['skewness_20'][idx]) else 0.0
    if 'kurtosis_20' in ind:
        result['kurtosis_20'] = ind['kurtosis_20'][idx] if not np.isnan(ind['kurtosis_20'][idx]) else 0.0
    if 'max_ret_20' in ind:
        result['max_ret_20'] = ind['max_ret_20'][idx] if not np.isnan(ind['max_ret_20'][idx]) else 0.0
    if 'tail_risk' in ind:
        result['tail_risk'] = ind['tail_risk'][idx] if not np.isnan(ind['tail_risk'][idx]) else 0.0
    if 'price_volume_corr_20' in ind:
        result['price_volume_corr_20'] = ind['price_volume_corr_20'][idx] if not np.isnan(ind['price_volume_corr_20'][idx]) else 0.0
    if 'volatility_skew' in ind:
        result['volatility_skew'] = ind['volatility_skew'][idx] if not np.isnan(ind['volatility_skew'][idx]) else 0.0
    if 'gap_ratio' in ind:
        result['gap_ratio'] = ind['gap_ratio'][idx] if not np.isnan(ind['gap_ratio'][idx]) else 0.0

    # === 小说灵感因子（计算与 ind 中的值一致）===
    if 'wash_sale_score' in ind:
        result['wash_sale_score'] = ind['wash_sale_score'][idx] if not np.isnan(ind['wash_sale_score'][idx]) else 0.0
    if 'vol_price_breakout' in ind:
        result['vol_price_breakout'] = ind['vol_price_breakout'][idx] if not np.isnan(ind['vol_price_breakout'][idx]) else 0.0
    if 'volume_contraction' in ind:
        result['volume_contraction'] = ind['volume_contraction'][idx] if not np.isnan(ind['volume_contraction'][idx]) else 0.0
    if 'relative_strength' in ind:
        result['relative_strength'] = ind['relative_strength'][idx] if not np.isnan(ind['relative_strength'][idx]) else 0.0
    if 'smart_money_flow' in ind:
        result['smart_money_flow'] = ind['smart_money_flow'][idx] if not np.isnan(ind['smart_money_flow'][idx]) else 0.0
    if 'consolidation_breakout' in ind:
        result['consolidation_breakout'] = ind['consolidation_breakout'][idx] if not np.isnan(ind['consolidation_breakout'][idx]) else 0.0
    if 'residual_momentum' in ind:
        result['residual_momentum'] = ind['residual_momentum'][idx] if not np.isnan(ind['residual_momentum'][idx]) else 0.0
    if 'short_reversal' in ind:
        result['short_reversal'] = ind['short_reversal'][idx] if not np.isnan(ind['short_reversal'][idx]) else 0.0
    if 'earnings_quality' in ind:
        result['earnings_quality'] = ind['earnings_quality'][idx] if not np.isnan(ind['earnings_quality'][idx]) else 0.0
    if 'idiosyncratic_volatility' in ind:
        result['idiosyncratic_volatility'] = ind['idiosyncratic_volatility'][idx] if not np.isnan(ind['idiosyncratic_volatility'][idx]) else 0.0

    # === 新增缠论增强因子 ===
    if 'gap_breakout_confirm' in ind:
        result['gap_breakout_confirm'] = ind['gap_breakout_confirm'][idx] if not np.isnan(ind['gap_breakout_confirm'][idx]) else 0.0
    if 'stroke_phase' in ind:
        result['stroke_phase'] = ind['stroke_phase'][idx] if not np.isnan(ind['stroke_phase'][idx]) else 0.0
    if 'exhaustion_risk' in ind:
        result['exhaustion_risk'] = ind['exhaustion_risk'][idx] if not np.isnan(ind['exhaustion_risk'][idx]) else 0.0
    if 'top_fractal_volume' in ind:
        result['top_fractal_volume'] = ind['top_fractal_volume'][idx] if not np.isnan(ind['top_fractal_volume'][idx]) else 0.0
    if 'vol_opening_confirm' in ind:
        result['vol_opening_confirm'] = ind['vol_opening_confirm'][idx] if not np.isnan(ind['vol_opening_confirm'][idx]) else 0.0
    if 'vol_opening_strength' in ind:
        result['vol_opening_strength'] = ind['vol_opening_strength'][idx] if not np.isnan(ind['vol_opening_strength'][idx]) else 0.0

    # === 均线系统因子 ===
    if 'ma_alignment' in ind:
        result['ma_alignment'] = ind['ma_alignment'][idx] if not np.isnan(ind['ma_alignment'][idx]) else 0.0
    if 'ema20_slope' in ind:
        result['ema20_slope'] = ind['ema20_slope'][idx] if not np.isnan(ind['ema20_slope'][idx]) else 0.0

    # === 涨停缩量回踩5日线低吸 ===
    result['limit_pullback_score'] = _compute_limit_pullback_score(ind, idx)

    return result
