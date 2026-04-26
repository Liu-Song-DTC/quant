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
    }

    close_arr = close
    high_arr = result['high']
    low_arr = result['low']
    vol_arr = result['volume']

    # === EMA ===
    for span in params.get('ema_periods', [5, 10, 20, 60]):
        result[f'ema{span}'] = _ema(close_arr, span)

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
    else:
        n_pts = len(close)
        result['inv_turnover'] = np.zeros(n_pts)
        result['turnover_shrink'] = np.zeros(n_pts)

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

    # 下行风险因子
    if 'low_downside' in ind:
        result['low_downside'] = ind['low_downside'][idx]

    return result
