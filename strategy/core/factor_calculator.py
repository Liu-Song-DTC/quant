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

from .divergence_detector import compute_divergence
from .structure_analyzer import (
    detect_pivot_zone,
    classify_trend_structure,
    compute_multi_level_alignment,
)


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
        'fund_score': lambda v: np.tanh((np.clip(v, -100, 100) - 50) / 50),
        'fund_profit_growth': lambda v: np.tanh(np.clip(v, -100, 100)),
        'fund_revenue_growth': lambda v: np.tanh(np.clip(v, -100, 100)),
        'fund_roe': lambda v: np.tanh((np.clip(v, -50, 50) - 10) / 20),
        'fund_cf_to_profit': lambda v: np.tanh(np.clip(v, -5, 5) - 1),
        'fund_gross_margin': lambda v: np.tanh((np.clip(v, -20, 80) - 30) / 30),
        'fund_eps': lambda v: np.tanh(np.clip(v, -10, 10)),
        'fund_debt_ratio': lambda v: np.tanh((50 - np.clip(v, 0, 100)) / 50),
        'fund_pg_improve': lambda v: np.tanh(np.clip(v, -5, 5) * 2),
        'fund_rg_improve': lambda v: np.tanh(np.clip(v, -5, 5) * 2),
    }
    compressor = compressors.get(factor_name)
    if compressor:
        return float(compressor(raw_value))
    return float(np.tanh(raw_value))  # fallback


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
    for period in params.get('volatility_periods', [5, 10, 20]):
        result[f'volatility_{period}'] = _rolling_std(returns, period)

    # === 下行风险因子 (downside deviation) ===
    # 仅计算负收益的标准差，与volatility不同
    for period in [10, 20]:
        down_dev = np.zeros(n)
        for i in range(period, n):
            r = returns[i-period:i]
            neg_r = r[r < 0]
            if len(neg_r) > 2:
                down_dev[i] = np.std(neg_r)
        result[f'downside_dev_{period}'] = down_dev
    # 低下行风险因子: 值越大=下行风险越低=越好
    result['low_downside'] = np.tanh(-result['downside_dev_20'] * 30)

    # === 均线关系 ===
    result['ema5_above_20'] = result['ema5'] > result['ema20']
    result['ema20_above_60'] = result['ema20'] > result['ema60']
    result['full_golden'] = (result['ema5'] > result['ema20']) & (result['ema20'] > result['ema60'])
    result['full_death'] = (result['ema5'] < result['ema20']) & (result['ema20'] < result['ema60'])

    # === 趋势强度 ===
    result['trend_strength'] = (result['ema20'] - result['ema60']) / (result['ema60'] + 1e-10)

    # === 斜率 ===
    result['ema20_slope'] = result['ema20'] / _shift(result['ema20'], 10, safe=True) - 1

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
    raw_mlv_10_10 = result['mom_10'] * (-result['volatility_10'])
    raw_mlv_10_20 = result['mom_10'] * (-result['volatility_20'])
    raw_mlv_20_10 = result['mom_20'] * (-result['volatility_10'])
    raw_mlv_20_20 = result['mom_20'] * (-result['volatility_20'])
    result['mom_x_lowvol_10_10'] = np.tanh(raw_mlv_10_10 * 20)  # 放大后压缩
    result['mom_x_lowvol_10_20'] = np.tanh(raw_mlv_10_20 * 20)
    result['mom_x_lowvol_20_10'] = np.tanh(raw_mlv_20_10 * 20)
    result['mom_x_lowvol_20_20'] = np.tanh(raw_mlv_20_20 * 20)

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
    result['trend_lowvol'] = np.tanh(trend_str * (-atr_ratio) * 50)

    # 量价确认因子: 近20天量价正相关+动量方向一致
    # 量价齐升>0, 量价齐跌<0
    vp_corr = np.zeros(n)
    for i in range(20, n):
        r = result['ret'][i-20:i]
        v = result['volume'][i-20:i]
        if np.std(r) > 0 and np.std(v) > 0:
            vp_corr[i] = np.corrcoef(r, v)[0, 1]
    mom20 = result.get('mom_20', np.zeros(n))
    result['vol_confirm'] = np.tanh(vp_corr * mom20 * 10)

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
        # 20日累计隔夜收益
        overnight_cum = np.zeros(n)
        for i in range(20, n):
            overnight_cum[i] = np.sum(overnight_ret[i-19:i+1])
        result['overnight_ret'] = np.tanh(overnight_cum * 15)

        intraday_ret = np.zeros(n)
        intraday_ret[1:] = (close_arr[1:] - open_arr[1:]) / (open_arr[1:] + 1e-10)
        intraday_cum = np.zeros(n)
        for i in range(20, n):
            intraday_cum[i] = np.sum(intraday_ret[i-19:i+1])
        result['intraday_ret'] = np.tanh(intraday_cum * 15)
    else:
        result['overnight_ret'] = np.zeros(n)
        result['intraday_ret'] = np.zeros(n)

    # 2. 收益偏度（20日）：正偏度=上涨集中，负偏度=下跌集中
    skew_20 = np.zeros(n)
    for i in range(20, n):
        r = returns[i-20:i]
        if np.std(r) > 1e-10:
            skew_20[i] = ((r - np.mean(r)) ** 3).mean() / (np.std(r) ** 3 + 1e-10)
    result['skewness_20'] = np.tanh(skew_20)

    # 3. 收益峰度（20日）：高峰度=尾部风险高
    kurt_20 = np.zeros(n)
    for i in range(20, n):
        r = returns[i-20:i]
        if np.std(r) > 1e-10:
            kurt_20[i] = ((r - np.mean(r)) ** 4).mean() / (np.std(r) ** 4 + 1e-10) - 3
    result['kurtosis_20'] = np.tanh(-kurt_20 / 3)  # 低峰度=正信号

    # 4. 最大单日涨幅（20日）：捕捉动量突破
    max_ret_20 = np.zeros(n)
    for i in range(20, n):
        max_ret_20[i] = np.max(returns[i-20:i])
    result['max_ret_20'] = np.tanh(max_ret_20 * 3)

    # 5. 尾部风险（CVaR-like）：过去20日最差5日平均收益
    tail_risk = np.zeros(n)
    for i in range(20, n):
        r = returns[i-20:i]
        tail_risk[i] = np.mean(np.sort(r)[:5])
    result['tail_risk'] = np.tanh(tail_risk * 5)

    # 6. 量价相关性（20日）：价格变化 vs 成交量变化的相关性
    pv_corr_20 = np.zeros(n)
    for i in range(20, n):
        price_chg = np.diff(close_arr[i-20:i+1])
        vol_chg = np.diff(vol_arr[i-20:i+1])
        if np.std(price_chg) > 0 and np.std(vol_chg) > 0 and len(price_chg) >= 5:
            corr = np.corrcoef(price_chg, vol_chg)[0, 1]
            pv_corr_20[i] = corr if not np.isnan(corr) else 0
    # 量价正相关+上涨趋势=正信号；负相关+下跌趋势=负信号
    result['price_volume_corr_20'] = np.tanh(pv_corr_20 * mom20 * 10)

    # 7. 波动率偏度：上行波动/下行波动 - 1
    vol_skew = np.zeros(n)
    for i in range(20, n):
        r = returns[i-20:i]
        up_r = r[r > 0]
        down_r = r[r < 0]
        up_vol = np.std(up_r) if len(up_r) > 2 else 0
        down_vol = np.std(down_r) if len(down_r) > 2 else 0
        if down_vol > 1e-10:
            vol_skew[i] = up_vol / down_vol - 1
    result['volatility_skew'] = np.tanh(vol_skew)

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

    # 9. 洗盘检测因子 (wash_sale_score)
    # 核心逻辑（来自小说）：放量下跌+高换手+快速回升=主力洗盘而非出货
    # 特征：过去N天内出现放量大跌(>3%)，但随后价格快速回升到跌幅的50%以上
    # 值越大越像洗盘，正值为看涨信号
    wash_score = np.zeros(n)
    for i in range(20, n):
        recent_ret = returns[i-20:i]
        recent_vol = vol_arr[i-20:i]
        recent_close = close_arr[i-20:i]
        # 找过去20天内的最大单日跌幅
        min_ret_idx = np.argmin(recent_ret)
        min_ret = recent_ret[min_ret_idx]
        # 条件1：有显著下跌（>3%）
        if min_ret < -0.03:
            # 条件2：下跌日成交量放大（>1.5倍均量）
            avg_vol = np.mean(recent_vol)
            if recent_vol[min_ret_idx] > avg_vol * 1.5:
                # 条件3：下跌后价格回升
                post_low = recent_close[min_ret_idx:]
                recovery = (post_low[-1] - post_low[0]) / (abs(post_low[0]) + 1e-10)
                if recovery > 0.5 * abs(min_ret):  # 回升超过跌幅的50%
                    # 条件4（来自小说）：换手率特征 - 高换手+缩量回升
                    # 用成交量变化率作为换手率替代
                    vol_after = recent_vol[min_ret_idx:]
                    vol_contracting = np.mean(vol_after[-3:]) < np.mean(vol_after[:3]) if len(vol_after) >= 6 else False
                    wash_score[i] = abs(min_ret) * 3 + (0.2 if vol_contracting else 0)
    result['wash_sale_score'] = np.tanh(wash_score * 2)

    # 10. 量价突破因子 (vol_price_breakout)
    # 核心逻辑（来自小说）：缩量盘整后的放量突破 = 主力拉升信号
    # "量在价先" - 成交量先行于价格
    vol_breakout = np.zeros(n)
    for i in range(20, n):
        recent_vol_20 = vol_arr[i-20:i]
        recent_high_20 = high_arr[i-20:i]
        avg_vol_20 = np.mean(recent_vol_20)
        # 当日成交量 > 2倍 20日均量（放量）
        if vol_arr[i] > avg_vol_20 * 2.0:
            # 价格突破20日高点
            if close_arr[i] > np.max(recent_high_20[:-1]):  # 不包含今天
                vol_breakout[i] = 1.0
            # 或接近突破（在2%以内）
            elif close_arr[i] > np.max(recent_high_20[:-1]) * 0.98:
                vol_breakout[i] = 0.5
    result['vol_price_breakout'] = np.tanh(vol_breakout * 2)

    # 11. 缩量回调因子 (volume_contraction)
    # 核心逻辑（来自小说）：上升趋势中缩量回调 = 洗盘结束，准备拉升
    # "缩量回调不破支撑" = 买入时机
    vol_contract = np.zeros(n)
    for i in range(20, n):
        # 中期趋势向上（20日均线向上）
        ma20_now = np.mean(close_arr[i-19:i+1])
        ma20_before = np.mean(close_arr[i-24:i-4]) if i >= 24 else ma20_now
        trend_up = ma20_now > ma20_before
        # 近5日缩量（成交量递减）
        recent_5vol = vol_arr[i-4:i+1]
        prev_5vol = vol_arr[i-9:i-4] if i >= 9 else recent_5vol
        vol_shrinking = np.mean(recent_5vol) < np.mean(prev_5vol) * 0.8
        # 价格回调但幅度不大（<5%）
        price_drawdown = (np.max(close_arr[i-10:i+1]) - close_arr[i]) / (np.max(close_arr[i-10:i+1]) + 1e-10)
        mild_pullback = 0 < price_drawdown < 0.05
        if trend_up and vol_shrinking and mild_pullback:
            vol_contract[i] = 1.0 - price_drawdown * 10  # 回调越小信号越强
    result['volume_contraction'] = np.tanh(vol_contract * 3)

    # 12. 强者恒强因子 (relative_strength)
    # 核心逻辑（来自小说）：强势股在大盘涨时涨更多，大盘跌时跌更少
    # 用相对强度衡量：近期收益/近期波动，捕捉独立行情
    rs_score = np.zeros(n)
    for i in range(20, n):
        ret_20d = (close_arr[i] - close_arr[i-20]) / (close_arr[i-20] + 1e-10) if i >= 20 else 0
        ret_10d = (close_arr[i] - close_arr[i-10]) / (close_arr[i-10] + 1e-10) if i >= 10 else 0
        ret_5d = (close_arr[i] - close_arr[i-5]) / (close_arr[i-5] + 1e-10) if i >= 5 else 0
        # 各周期动量一致性：短中长期都为正更可靠
        momentum_consistency = (1 if ret_5d > 0 else -0.5) + (1 if ret_10d > 0 else -0.5) + (1 if ret_20d > 0 else -0.5)
        # 波动调整收益
        vol_20d = np.std(returns[i-19:i+1]) if i >= 19 else 0.02
        if vol_20d > 1e-6:
            rs_score[i] = (ret_20d / vol_20d) * (0.5 + 0.5 * max(0, momentum_consistency / 3))
    result['relative_strength'] = np.tanh(rs_score * 0.5)

    # 13. 主力资金流向推断因子 (smart_money_flow)
    # 核心逻辑（来自小说）：通过价量关系推断主力动向
    # 量价齐升（放量上涨）= 主力买入；量价背离（缩量上涨/放量下跌）= 警惕
    # 综合5日量价关系
    mf_score = np.zeros(n)
    for i in range(5, n):
        score_5d = 0.0
        for j in range(i-4, i+1):
            ret_day = returns[j] if j > 0 else 0
            vol_ratio_day = vol_arr[j] / (np.mean(vol_arr[max(0,j-20):j+1]) + 1e-6)
            if ret_day > 0 and vol_ratio_day > 1.2:  # 放量上涨：主力买入
                score_5d += ret_day * min(vol_ratio_day, 3) * 2
            elif ret_day > 0 and vol_ratio_day < 0.7:  # 缩量上涨：趋势持续但力度减弱
                score_5d += ret_day * 0.5
            elif ret_day < 0 and vol_ratio_day > 1.5:  # 放量下跌：可能是洗盘
                score_5d += ret_day * 0.3  # 轻罚（洗盘可能性）
            elif ret_day < 0 and vol_ratio_day < 0.7:  # 缩量下跌：正常调整
                score_5d += ret_day * 1.0
            else:
                score_5d += ret_day
        mf_score[i] = score_5d / 5
    result['smart_money_flow'] = np.tanh(mf_score * 3)

    # 14. 盘整突破准备因子 (consolidation_breakout)
    # 核心逻辑（来自小说）：小5浪调整后缩量走平 = 即将突破
    # 特征：近10天振幅收窄 + 成交量萎缩 + 价格走平
    cb_score = np.zeros(n)
    for i in range(20, n):
        recent_10_high = high_arr[i-9:i+1]
        recent_10_low = low_arr[i-9:i+1]
        recent_10_vol = vol_arr[i-9:i+1]
        recent_10_close = close_arr[i-9:i+1]
        # 振幅收窄：近期振幅 < 前期振幅的70%
        recent_range = (np.mean(recent_10_high) - np.mean(recent_10_low)) / (np.mean(recent_10_close) + 1e-10)
        prev_range = (np.mean(high_arr[i-19:i-9]) - np.mean(low_arr[i-19:i-9])) / (np.mean(close_arr[i-19:i-9]) + 1e-10) if i >= 19 else recent_range * 2
        range_narrowing = recent_range < prev_range * 0.7
        # 成交量萎缩：近期量 < 前期量的80%
        vol_shrink = np.mean(recent_10_vol) < np.mean(vol_arr[i-19:i-9]) * 0.8 if i >= 19 else False
        # 价格走平：近期价格标准差很小
        price_flat = np.std(recent_10_close) / (np.mean(recent_10_close) + 1e-10) < 0.03
        # 趋势向上（中长期均线多头）
        ma20_val = np.mean(close_arr[i-19:i+1])
        ma60_val = np.mean(close_arr[i-59:i+1]) if i >= 59 else ma20_val
        trend_ok = ma20_val > ma60_val
        if range_narrowing and vol_shrink and price_flat and trend_ok:
            cb_score[i] = 1.0
        elif range_narrowing and (vol_shrink or price_flat):
            cb_score[i] = 0.4
    result['consolidation_breakout'] = np.tanh(cb_score * 2)

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

    # === 缠论：多级别结构分析 ===
    structure_params = params.get('structure', {})
    pivot_info = detect_pivot_zone(
        high_arr, low_arr, close_arr,
        min_overlap=structure_params.get('pivot_min_overlap', 3),
        zone_buffer=structure_params.get('pivot_zone_buffer', 0.02),
    )
    structure_info = classify_trend_structure(
        close_arr, high_arr, low_arr,
        result['ema20'], result['ema60'], pivot_info,
        min_trend_bars=structure_params.get('min_trend_bars', 8),
        zhongyin_threshold=structure_params.get('zhongyin_threshold', 0.02),
    )
    alignment = compute_multi_level_alignment(
        close_arr, result['ema20'], result['ema60'], result['ema120'],
    )

    result['pivot_present'] = pivot_info['pivot_present'].astype(float)
    result['pivot_top'] = pivot_info['pivot_top']
    result['pivot_bottom'] = pivot_info['pivot_bottom']
    result['pivot_mid'] = pivot_info['pivot_mid']
    result['pivot_level'] = pivot_info['pivot_level'].astype(float)
    result['pivot_count'] = pivot_info['pivot_count'].astype(float)
    result['structure_complete'] = structure_info['structure_complete'].astype(float)
    result['zhongyin'] = structure_info['zhongyin'].astype(float)
    result['breakout_above_pivot'] = structure_info['breakout_above_pivot'].astype(float)
    result['breakout_below_pivot'] = structure_info['breakout_below_pivot'].astype(float)
    result['pivot_distance'] = structure_info['pivot_distance']
    result['alignment_score'] = alignment

    return result


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
    """简单移动平均"""
    n = len(arr)
    result = np.zeros(n)
    result[:] = np.nan
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    result = np.zeros_like(arr, dtype=float)
    result[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    """RSI指标"""
    n = len(close)
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    result = np.zeros(n)
    result[:] = np.nan

    avg_gain = _sma(gain, window)
    avg_loss = _sma(loss, window)

    for i in range(window - 1, n):
        if avg_loss[i] == 0:
            result[i] = 100
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            result[i] = 100 - (100 / (1 + rs))
    return result


def _bollinger(close: np.ndarray, window: int, num_std: float) -> tuple:
    """布林带"""
    n = len(close)
    middle = _sma(close, window)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan

    for i in range(window - 1, n):
        std = np.std(close[i - window + 1:i + 1])
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std

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


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差"""
    n = len(arr)
    result = np.zeros(n)
    result[:] = np.nan
    for i in range(window, n):
        result[i] = np.std(arr[i - window:i])
    return result


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最大值"""
    n = len(arr)
    result = np.zeros(n)
    result[:] = np.nan
    for i in range(window, n):
        result[i] = np.max(arr[i - window:i])
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最小值"""
    n = len(arr)
    result = np.zeros(n)
    result[:] = np.nan
    for i in range(window, n):
        result[i] = np.min(arr[i - window:i])
    return result


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD指标"""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


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
    if m10 > 0:
        result['trend_mom_v41'] = np.tanh(m20 * 3)  # 压缩到(-1, 1)
        result['trend_mom_v24'] = np.tanh(m20 * 3)
        result['trend_mom_v46'] = np.tanh(m20 * 2.5)
    else:
        result['trend_mom_v41'] = 0.0
        result['trend_mom_v24'] = np.tanh(m20 * 0.5)  # 下跌时轻微负值
        result['trend_mom_v46'] = 0.0

    # 动量×低波动 - tanh压缩
    result['mom_x_lowvol_20_20'] = np.tanh(m20 * (-v20) * 20)
    result['mom_x_lowvol_20_10'] = np.tanh(m20 * (-v10) * 20)
    result['mom_x_lowvol_10_20'] = np.tanh(m10 * (-v20) * 20)
    result['mom_x_lowvol_10_10'] = np.tanh(m10 * (-v10) * 20)

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

    # 新因子: trend_lowvol + vol_confirm
    if 'trend_lowvol' in ind:
        result['trend_lowvol'] = ind['trend_lowvol'][idx]
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

    return result
