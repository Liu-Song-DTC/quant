# core/factors.py
"""
因子定义模块 - 统一管理所有单因子和组合因子
"""
import numpy as np
import pandas as pd


class FactorRegistry:
    """因子注册表"""
    _factors = {}
    _factor_funcs = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册因子"""
        def decorator(func):
            cls._factors[name] = func
            cls._factor_funcs[name] = func
            return func
        return decorator

    @classmethod
    def get_factor(cls, name: str):
        """获取因子函数"""
        return cls._factor_funcs.get(name)

    @classmethod
    def list_factors(cls):
        """列出所有注册的因子"""
        return list(cls._factors.keys())

    @classmethod
    def get_all_factors(cls):
        """获取所有因子函数"""
        return cls._factor_funcs.copy()


# ====================== 基础指标计算 ======================

def calc_base_indicators(close, high=None, low=None, volume=None):
    """计算基础技术指标（供因子函数使用）

    Args:
        close: 收盘价数组
        high: 最高价数组（可选）
        low: 最低价数组（可选）
        volume: 成交量数组（可选）

    Returns:
        dict: 包含所有基础指标的字典
    """
    result = {}

    # 动量指标
    for period in [3, 5, 10, 20, 60]:
        result[f'mom_{period}'] = close / np.roll(close, period) - 1
        result[f'mom_{period}'][np.isnan(result[f'mom_{period}'])] = 0

    # 均线
    for span in [5, 10, 12, 20, 26, 60]:
        result[f'ema{span}'] = pd.Series(close).ewm(span=span, adjust=False).mean().values

    for span in [5, 10, 20, 60]:
        result[f'ma{span}'] = pd.Series(close).rolling(span).mean().values

    # RSI 多周期
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    for p in [6, 10, 14]:
        avg_gain = pd.Series(gain).rolling(p).mean().values
        avg_loss = pd.Series(loss).rolling(p).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        result[f'rsi_{p}'] = 100 - (100 / (1 + rs))
    result['rsi'] = result['rsi_14']  # 保持兼容性

    # RSI变化
    if 'rsi_14' in result:
        result['rsi_change'] = result['rsi_14'] - pd.Series(result['rsi_14']).shift(5).values

    # RSI位置
    if 'rsi_14' in result:
        rsi_high = pd.Series(result['rsi_14']).rolling(20).max().values
        rsi_low = pd.Series(result['rsi_14']).rolling(20).min().values
        result['rsi_position'] = (result['rsi_14'] - rsi_low) / (rsi_high - rsi_low + 1e-10)

    # 波动率
    returns = pd.Series(close).pct_change().values
    for p in [5, 10, 20, 60]:
        result[f'volatility_{p}'] = pd.Series(returns).rolling(p).std().values

    if high is not None and low is not None:
        # 价格位置
        for period in [10, 20, 60]:
            high_p = pd.Series(high).rolling(period).max().values
            low_p = pd.Series(low).rolling(period).min().values
            result[f'price_position_{period}'] = (close - low_p) / (high_p - low_p + 1e-10)

    if volume is not None:
        # 成交量均线
        for p in [5, 10, 20]:
            vol_ma = pd.Series(volume).rolling(p).mean().values
            result[f'vol_ma_{p}'] = volume / (vol_ma + 1e-10)
            result[f'vol_change_{p}'] = volume / (np.roll(volume, p) + 1e-10) - 1
        # 保持兼容性
        volume_ma20 = result['vol_ma_20']
        result['volume_ratio'] = volume / (volume_ma20 + 1e-10)
        result['volume_change'] = result['vol_change_5']

    # MACD
    ema_12 = result['ema12']
    ema_26 = result['ema26']
    ema_9 = pd.Series(ema_12 - ema_26).ewm(span=9, adjust=False).mean().values
    result['macd_value'] = ema_12 - ema_26
    result['macd_signal_diff'] = (ema_12 - ema_26) - ema_9
    result['macd_hist_change'] = (ema_12 - ema_26 - ema_9) - pd.Series(ema_12 - ema_26 - ema_9).shift(1).values

    # 布林带
    ma20 = result['ma20']
    std20 = pd.Series(close).rolling(20).std().values
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    result['bb_percent_b'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    result['bb_width'] = (bb_upper - bb_lower) / (ma20 + 1e-10)

    # 均线交叉
    result['ma_cross'] = (result['ma5'] > result['ma20']).astype(float)
    result['ema_trend'] = result['ema10'] - result['ema20']

    return result


def calc_all_factors_for_validation(close, high=None, low=None, volume=None, fundamental_data=None, code=None, eval_date=None):
    """计算所有用于验证的因子（供验证脚本使用）

    基于 calc_base_indicators 的结果，计算所有复合因子

    Args:
        close: 收盘价数组
        high: 最高价数组
        low: 最低价数组
        volume: 成交量数组
        fundamental_data: 基本面数据对象（可选）
        code: 股票代码（基本面数据用）
        eval_date: 评估日期（基本面数据用）

    Returns:
        dict: 包含所有因子的字典
    """
    # 先计算基础指标
    base = calc_base_indicators(close, high, low, volume)
    factors = base.copy()

    # 复合因子计算
    # 动量差
    if 'mom_3' in factors and 'mom_5' in factors:
        factors['mom_diff_3_5'] = factors['mom_3'] - factors['mom_5']
    if 'mom_5' in factors and 'mom_10' in factors:
        factors['mom_diff_5_10'] = factors['mom_5'] - factors['mom_10']
    if 'mom_5' in factors and 'mom_20' in factors:
        factors['mom_diff_5_20'] = factors['mom_5'] - factors['mom_20']
    if 'mom_10' in factors and 'mom_20' in factors:
        factors['mom_diff_10_20'] = factors['mom_10'] - factors['mom_20']

    # 动量×低波动
    if 'mom_10' in factors and 'volatility_10' in factors:
        factors['mom_x_lowvol_10_10'] = factors['mom_10'] * (-factors['volatility_10'])
    if 'mom_10' in factors and 'volatility_20' in factors:
        factors['mom_x_lowvol_10_20'] = factors['mom_10'] * (-factors['volatility_20'])
    if 'mom_20' in factors and 'volatility_10' in factors:
        factors['mom_x_lowvol_20_10'] = factors['mom_20'] * (-factors['volatility_10'])
    if 'mom_20' in factors and 'volatility_20' in factors:
        factors['mom_x_lowvol_20_20'] = factors['mom_20'] * (-factors['volatility_20'])

    # RSI + 动量
    if 'rsi_10' in factors and 'mom_5' in factors:
        factors['rsi_mom_10_5'] = (50 - factors['rsi_10']) / 100 + factors['mom_5']
    if 'rsi_14' in factors and 'mom_5' in factors:
        factors['rsi_mom_14_5'] = (50 - factors['rsi_14']) / 100 + factors['mom_5']

    # 低波动+高位置
    if 'price_position_20' in factors and 'volatility_10' in factors:
        factors['low_vol_high_pos'] = -factors['volatility_10'] + factors['price_position_20']

    # 收益波动率比
    if 'mom_10' in factors and 'volatility_10' in factors:
        factors['ret_vol_ratio_10'] = factors['mom_10'] / (factors['volatility_10'] + 1e-10)
    if 'mom_20' in factors and 'volatility_20' in factors:
        factors['ret_vol_ratio_20'] = factors['mom_20'] / (factors['volatility_20'] + 1e-10)

    # MA交叉
    if 'ma5' in factors and 'ma10' in factors and 'ma20' in factors:
        factors['ma_cross_5_10'] = (factors['ma5'] - factors['ma10']) / (factors['ma10'] + 1e-10)
        factors['ma_cross_10_20'] = (factors['ma10'] - factors['ma20']) / (factors['ma20'] + 1e-10)
        if 'ma60' in factors:
            factors['ma_golden_count'] = ((factors['ma5'] > factors['ma10']).astype(float) +
                                       (factors['ma10'] > factors['ma20']).astype(float) +
                                       (factors['ma20'] > factors['ma60']).astype(float))

    # ===== 新增组合因子 =====
    # RSI + 波动率
    if 'rsi_14' in factors and 'volatility_20' in factors:
        factors['rsi_vol_combo'] = (50 - factors['rsi_14']) / 100 - factors['volatility_20'] * 0.5

    # 布林带 + RSI
    if 'bb_percent_b' in factors and 'rsi_14' in factors:
        factors['bb_rsi_combo'] = (50 - factors['rsi_14']) / 100 - factors['bb_percent_b'] * 0.3

    # 动量 + 成交量
    if 'mom_20' in factors and 'volume_ratio' in factors:
        vol_factor = np.clip(factors['volume_ratio'] - 1, -0.5, 1)
        factors['price_mom_volume'] = factors['mom_20'] * (1 + vol_factor)

    # 均线交叉 + 成交量
    if 'ma_cross_5_10' in factors and 'volume_ratio' in factors:
        vol_factor = np.clip(factors['volume_ratio'] - 1, -0.3, 0.5)
        factors['ma_cross_volume'] = factors['ma_cross_5_10'] * (1 + vol_factor)

    # 高动量 + 低波动
    if 'mom_20' in factors and 'volatility_20' in factors:
        factors['highmom_lowvol'] = factors['mom_20'] - factors['volatility_20'] * 0.5

    # 趋势强度
    if 'mom_5' in factors and 'mom_10' in factors and 'mom_20' in factors:
        score = np.zeros_like(factors['mom_5'])
        score = np.where(factors['mom_5'] > 0, score + 1, score)
        score = np.where(factors['mom_10'] > 0, score + 1, score)
        score = np.where(factors['mom_20'] > 0, score + 1, score)
        factors['trend_strength'] = score / 3

    # 动量反转
    if 'mom_5' in factors and 'mom_20' in factors:
        factors['momentum_reversal'] = factors['mom_5'] - factors['mom_20']

    # ===== 基本面因子 =====
    if fundamental_data is not None and code is not None and eval_date is not None:
        try:
            roe = fundamental_data.get_roe(code, eval_date)
            if roe is not None:
                factors['fund_roe'] = roe

            profit_growth = fundamental_data.get_profit_growth(code, eval_date)
            if profit_growth is not None:
                factors['fund_profit_growth'] = profit_growth

            revenue_growth = fundamental_data.get_revenue_growth(code, eval_date)
            if revenue_growth is not None:
                factors['fund_revenue_growth'] = revenue_growth

            eps = fundamental_data.get_eps(code, eval_date)
            if eps is not None:
                factors['fund_eps'] = eps

            debt_ratio = fundamental_data.get_debt_ratio(code, eval_date)
            if debt_ratio is not None:
                factors['fund_debt_ratio'] = debt_ratio

            gross_margin = fundamental_data.get_gross_margin(code, eval_date)
            if gross_margin is not None:
                factors['fund_gross_margin'] = gross_margin

            # 经营性现金流/净利润（质量因子）
            operating_cf = fundamental_data.get_operating_cash_flow(code, eval_date)
            profit = fundamental_data.get_profit(code, eval_date)
            if operating_cf is not None and profit is not None and profit > 0:
                factors['fund_cf_to_profit'] = operating_cf / profit

            fund_score = fundamental_data.get_fundamental_score(code, eval_date)
            if fund_score is not None:
                factors['fund_score'] = fund_score
        except:
            pass

    # ===== 量价背离因子 =====
    # 价涨量缩（动量向上但成交量向下）
    if 'mom_5' in factors and 'vol_change_5' in factors:
        factors['price_volume_divergence_up'] = factors['mom_5'] - np.clip(factors['vol_change_5'], -0.5, 0.5)

    # 价跌量增（动量向下但成交量向上）
    if 'mom_5' in factors and 'vol_change_5' in factors:
        factors['price_volume_divergence_down'] = -factors['mom_5'] + np.clip(factors['vol_change_5'], -0.5, 0.5)

    # 综合背离信号
    if 'price_volume_divergence_up' in factors and 'price_volume_divergence_down' in factors:
        factors['divergence_signal'] = factors['price_volume_divergence_up'] - factors['price_volume_divergence_down']

    # 收盘价 vs 成交量背离（20日）
    if 'mom_20' in factors and 'vol_change_20' in factors:
        factors['price_vol_divergence_20'] = factors['mom_20'] - np.clip(factors['vol_change_20'], -0.5, 0.5)

    # ===== 新增组合因子 =====
    # 动量 + 质量（基本面）
    if 'mom_10' in factors and 'fund_score' in factors:
        # 需要在有基本面数据时才能计算
        pass

    # 低波动 + 高RSI位置
    if 'volatility_10' in factors and 'rsi_position' in factors:
        factors['lowvol_high_rsi_pos'] = -factors['volatility_10'] + factors['rsi_position']

    # 趋势强度 + 波动率
    if 'trend_strength' in factors and 'volatility_10' in factors:
        factors['trend_volatility_combo'] = factors['trend_strength'] - factors['volatility_10'] * 0.5

    # 收益风险比（动量/波动率）
    if 'mom_20' in factors and 'volatility_20' in factors:
        factors['return_risk_ratio'] = factors['mom_20'] / (factors['volatility_20'] + 1e-10)

    # 动量加速度
    if 'mom_5' in factors and 'mom_20' in factors:
        factors['momentum_acceleration'] = factors['mom_5'] - factors['mom_20']

    # RSI超卖反弹
    if 'rsi_14' in factors and 'mom_5' in factors:
        factors['rsi_oversold_rebound'] = (30 - np.clip(factors['rsi_14'], 0, 30)) / 30 + np.clip(factors['mom_5'], -0.1, 0.2)

    # 布林带突破 + 成交量确认
    if 'bb_percent_b' in factors and 'volume_ratio' in factors:
        bb_signal = np.where(factors['bb_percent_b'] > 0.8, 1, np.where(factors['bb_percent_b'] < 0.2, -1, 0))
        factors['bb_volume_confirm'] = bb_signal * np.clip(factors['volume_ratio'], 0.5, 2)

    return factors


# ====================== 单因子定义 ======================

@FactorRegistry.register('trend_mom_v41')
def trend_mom_v41(mom_20, mom_10):
    """
    趋势动量V41因子 (最优单因子)
    公式: 10日动量>0时，20日动量×2.1；否则为0
    IC: 约8.27%
    """
    return np.where(mom_10 > 0, mom_20 * 2.1, 0)


@FactorRegistry.register('trend_mom_v24')
def trend_mom_v24(mom_20, mom_10):
    """
    趋势动量V24因子
    公式: 10日动量>0时，20日动量×2.1；否则为20日动量×0.04
    IC: 约8.23%
    """
    return np.where(mom_10 > 0, mom_20 * 2.1, mom_20 * 0.04)


@FactorRegistry.register('trend_mom_v46')
def trend_mom_v46(mom_20, mom_10):
    """
    趋势动量V46因子
    公式: 10日动量>0时，20日动量×2.05；否则为0
    """
    return np.where(mom_10 > 0, mom_20 * 2.05, 0)


@FactorRegistry.register('trend_mom_v49')
def trend_mom_v49(mom_20, mom_10):
    """
    趋势动量V49因子
    公式: 10日动量>0时，20日动量×2.03；否则为0
    """
    return np.where(mom_10 > 0, mom_20 * 2.03, 0)


@FactorRegistry.register('rsi_factor')
def rsi_factor(rsi):
    """
    RSI因子
    公式: (RSI - 50) / 100
    """
    return (rsi - 50) / 100


@FactorRegistry.register('rsi_6')
def rsi_6_factor(rsi_6):
    """
    RSI-6日因子
    """
    return (rsi_6 - 50) / 100


@FactorRegistry.register('price_position')
def price_position_factor(price_position_20):
    """
    价格位置因子
    公式: 收盘价在20日高低点位置
    """
    return price_position_20


@FactorRegistry.register('volatility')
def volatility_factor(volatility_20):
    """
    波动率因子（负波动率，低波动率更好）
    """
    return -volatility_20


@FactorRegistry.register('bb_width')
def bb_width_factor(bb_width):
    """
    布林带宽度因子
    """
    return bb_width


@FactorRegistry.register('volume_ratio')
def volume_ratio_factor(volume_ratio):
    """
    成交量比率因子
    """
    return volume_ratio - 1


# ====================== 组合因子函数 ======================

def combine_factors(factor1, factor2, weight1=0.9, weight2=0.1):
    """组合两个因子

    Args:
        factor1: 第一个因子值
        factor2: 第二个因子值
        weight1: 第一个因子权重
        weight2: 第二个因子权重

    Returns:
        组合后的因子值
    """
    return factor1 * weight1 + factor2 * weight2


def combine_3_factors(factor1, factor2, factor3, w1=0.85, w2=0.1, w3=0.05):
    """组合三个因子

    Args:
        factor1, factor2, factor3: 因子值
        w1, w2, w3: 对应权重

    Returns:
        组合后的因子值
    """
    return factor1 * w1 + factor2 * w2 + factor3 * w3


# ====================== 预定义组合因子 ======================

@FactorRegistry.register('V41_RSI_915')
def V41_RSI_915(mom_20, mom_10, rsi):
    """V41 + RSI组合 (0.915:0.085)"""
    trend_mom = trend_mom_v41(mom_20, mom_10)
    rsi_f = rsi_factor(rsi)
    return trend_mom * 0.915 + rsi_f * 0.085


@FactorRegistry.register('V41_PricePos_915')
def V41_PricePos_915(mom_20, mom_10, price_position_20):
    """V41 + 价格位置组合 (0.915:0.085)"""
    trend_mom = trend_mom_v41(mom_20, mom_10)
    return trend_mom * 0.915 + price_position_20 * 0.085


@FactorRegistry.register('V46_RSI_915')
def V46_RSI_915(mom_20, mom_10, rsi):
    """V46 + RSI组合"""
    trend_mom = trend_mom_v46(mom_20, mom_10)
    rsi_f = rsi_factor(rsi)
    return trend_mom * 0.915 + rsi_f * 0.085


@FactorRegistry.register('V49_RSI_915')
def V49_RSI_915(mom_20, mom_10, rsi):
    """V49 + RSI组合"""
    trend_mom = trend_mom_v49(mom_20, mom_10)
    rsi_f = rsi_factor(rsi)
    return trend_mom * 0.915 + rsi_f * 0.085


# ====================== 便捷函数 ======================

def get_factor(factor_name: str, **kwargs):
    """获取并计算指定因子

    Args:
        factor_name: 因子名称
        **kwargs: 需要的参数（如mom_20, mom_10, rsi等）

    Returns:
        因子值数组
    """
    func = FactorRegistry.get_factor(factor_name)
    if func is None:
        raise ValueError(f"Unknown factor: {factor_name}")
    return func(**kwargs)


# ====================== 基本面因子 ======================

@FactorRegistry.register('fundamental_score')
def fundamental_score(roe=None, profit_growth=None, revenue_growth=None, eps=None):
    """基本面综合评分因子

    整合ROE、净利润增长、营收增长、EPS

    Args:
        roe: 净资产收益率 (小数形式，如0.12表示12%)，支持数组
        profit_growth: 净利润增长率 (小数形式)，支持数组
        revenue_growth: 营收增长率 (小数形式)，支持数组
        eps: 每股收益，支持数组

    Returns:
        综合基本面评分 (0-1之间)
    """
    # 确定输出形状
    if roe is not None and hasattr(roe, '__len__'):
        n = len(roe)
        score = np.zeros(n)
    elif profit_growth is not None and hasattr(profit_growth, '__len__'):
        n = len(profit_growth)
        score = np.zeros(n)
    elif revenue_growth is not None and hasattr(revenue_growth, '__len__'):
        n = len(revenue_growth)
        score = np.zeros(n)
    elif eps is not None and hasattr(eps, '__len__'):
        n = len(eps)
        score = np.zeros(n)
    else:
        score = 0.0

    # ROE评分 - 最重要的因子 (权重最高)
    if roe is not None:
        if hasattr(roe, '__len__'):
            score = np.where(roe > 0.15, score + 0.35,
                    np.where(roe > 0.10, score + 0.25,
                    np.where(roe > 0.05, score + 0.15, score)))
        else:
            if roe > 0.15:
                score += 0.35
            elif roe > 0.10:
                score += 0.25
            elif roe > 0.05:
                score += 0.15

    # 净利润增长 - 重要
    if profit_growth is not None:
        if hasattr(profit_growth, '__len__'):
            score = np.where(profit_growth > 0.50, score + 0.30,
                    np.where(profit_growth > 0.20, score + 0.20,
                    np.where(profit_growth > 0, score + 0.10, score)))
        else:
            if profit_growth > 0.50:
                score += 0.30
            elif profit_growth > 0.20:
                score += 0.20
            elif profit_growth > 0:
                score += 0.10

    # 营业收入增长
    if revenue_growth is not None:
        if hasattr(revenue_growth, '__len__'):
            score = np.where(revenue_growth > 0.30, score + 0.20,
                    np.where(revenue_growth > 0.15, score + 0.12,
                    np.where(revenue_growth > 0, score + 0.05, score)))
        else:
            if revenue_growth > 0.30:
                score += 0.20
            elif revenue_growth > 0.15:
                score += 0.12
            elif revenue_growth > 0:
                score += 0.05

    # 每股收益
    if eps is not None:
        if hasattr(eps, '__len__'):
            score = np.where(eps > 1.0, score + 0.20,
                    np.where(eps > 0.5, score + 0.12, score))
        else:
            if eps > 1.0:
                score += 0.20
            elif eps > 0.5:
                score += 0.12

    # 归一化到0-1
    if hasattr(score, '__len__'):
        return np.clip(score / 1.0, 0, 1)
    return min(score / 1.0, 1.0)


@FactorRegistry.register('roe_factor')
def roe_factor(roe):
    """ROE因子 (负ROE penalize)"""
    if roe is None:
        return 0
    if hasattr(roe, '__len__'):
        return np.maximum(roe, 0)
    return max(roe, 0)


@FactorRegistry.register('profit_growth_factor')
def profit_growth_factor(profit_growth):
    """净利润增长因子"""
    if profit_growth is None:
        return 0
    return np.clip(profit_growth, -1, 3)


@FactorRegistry.register('revenue_growth_factor')
def revenue_growth_factor(revenue_growth):
    """营收增长因子"""
    if revenue_growth is None:
        return 0
    return np.clip(revenue_growth, -1, 3)


@FactorRegistry.register('eps_factor')
def eps_factor(eps):
    """EPS因子"""
    if eps is None:
        return 0
    if hasattr(eps, '__len__'):
        return np.minimum(np.maximum(eps, 0), 5)
    return min(max(eps, 0), 5)


@FactorRegistry.register('debt_ratio_factor')
def debt_ratio_factor(debt_ratio):
    """资产负债率因子 (低负债率更好)"""
    if debt_ratio is None:
        return 0
    if hasattr(debt_ratio, '__len__'):
        return 0.5 - debt_ratio
    return 0.5 - debt_ratio


@FactorRegistry.register('gross_margin_factor')
def gross_margin_factor(gross_margin):
    """销售毛利率因子"""
    if gross_margin is None:
        return 0
    return gross_margin


# ====================== 资金流向因子 ======================

@FactorRegistry.register('money_flow_5d')
def money_flow_5d(close, volume):
    """5日资金流向因子

    上涨日放量=资金流入，下跌日缩量=资金观望
    """
    if len(close) < 5:
        return np.zeros_like(close)

    price_change = np.diff(close, prepend=close[0])
    flow = np.where(price_change > 0, volume, -volume)

    result = np.zeros_like(close, dtype=float)
    for i in range(4, len(close)):
        result[i] = np.mean(flow[i-4:i+1]) / (np.mean(volume[i-4:i+1]) + 1e-10)

    return result


@FactorRegistry.register('money_flow_20d')
def money_flow_20d(close, volume):
    """20日资金流向因子"""
    if len(close) < 20:
        return np.zeros_like(close)

    price_change = np.diff(close, prepend=close[0])
    flow = np.where(price_change > 0, volume, -volume)

    result = np.zeros_like(close, dtype=float)
    for i in range(19, len(close)):
        result[i] = np.mean(flow[i-19:i+1]) / (np.mean(volume[i-19:i+1]) + 1e-10)

    return result


@FactorRegistry.register('volume_price_trend')
def volume_price_trend_factor(close, volume):
    """量价配合因子

    上涨且放量为正向，下跌且缩量为正向
    """
    if len(close) < 10:
        return np.zeros_like(close)

    returns = np.zeros_like(close)
    vol_change = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)
    vol_change[1:] = (volume[1:] - volume[:-1]) / (volume[:-1] + 1e-10)

    result = np.zeros_like(close, dtype=float)
    for i in range(9, len(close)):
        score = 0
        for j in range(i-9, i+1):
            if returns[j] > 0 and vol_change[j] > 0:
                score += 0.1
            elif returns[j] < 0 and vol_change[j] < 0:
                score += 0.05
            elif returns[j] > 0 and vol_change[j] < 0:
                score -= 0.05
        result[i] = score

    return result


# ====================== 筹码分布因子 ======================

@FactorRegistry.register('price_concentration')
def price_concentration_factor(close, volume):
    """价格集中度因子

    近期成交量集中的价格区间
    """
    if len(close) < 20:
        return np.zeros_like(close)

    result = np.zeros_like(close, dtype=float)

    for i in range(19, len(close)):
        # 取最近20日
        prices_20 = close[i-19:i+1]
        vols_20 = volume[i-19:i+1]

        # 计算成交量加权平均价格
        vwap = np.sum(prices_20 * vols_20) / (np.sum(vols_20) + 1e-10)

        # 价格与VWAP的偏离度
        result[i] = (close[i] - vwap) / (vwap + 1e-10)

    return result


@FactorRegistry.register('cost_basis')
def cost_basis_factor(close, volume):
    """持仓成本因子

    基于成交量加权的平均成本
    """
    if len(close) < 20:
        return np.zeros_like(close)

    result = np.zeros_like(close, dtype=float)

    for i in range(19, len(close)):
        prices_20 = close[i-19:i+1]
        vols_20 = volume[i-19:i+1]

        # 成交量加权平均成本
        cost = np.sum(prices_20 * vols_20) / (np.sum(vols_20) + 1e-10)

        # 当前价格相对成本的位置
        result[i] = (close[i] - cost) / (cost + 1e-10)

    return result


@FactorRegistry.register('volume_distribution')
def volume_distribution_factor(close, volume):
    """成交量分布因子

    放量突破时信号增强
    """
    if len(close) < 20:
        return np.zeros_like(close)

    vol_ma20 = pd.Series(volume).rolling(20).mean().values
    result = np.zeros_like(close, dtype=float)

    for i in range(19, len(close)):
        # 放量程度
        vol_ratio = volume[i] / (vol_ma20[i] + 1e-10)

        # 价格变化
        price_change = (close[i] - close[i-1]) / (close[i-1] + 1e-10)

        # 放量突破
        if vol_ratio > 1.5 and price_change > 0.01:
            result[i] = 1.0
        elif vol_ratio > 2.0:
            result[i] = 0.5
        elif vol_ratio > 1.2:
            result[i] = 0.2

    return result


# ====================== 波动率改进因子 ======================

@FactorRegistry.register('volatility_ratio')
def volatility_ratio_factor(close):
    """波动率比值因子 (短期/长期)

    波动率从高向低变化可能预示趋势形成
    """
    if len(close) < 60:
        return np.zeros_like(close)

    returns = pd.Series(close).pct_change().values
    vol_5 = pd.Series(returns).rolling(5).std().values
    vol_20 = pd.Series(returns).rolling(20).std().values

    result = vol_5 / (vol_20 + 1e-10)
    return result


@FactorRegistry.register('return_skewness')
def return_skewness_factor(close):
    """收益偏度因子

    正偏度（上涨时波动大）可能预示下跌风险
    """
    if len(close) < 20:
        return np.zeros_like(close)

    result = np.zeros_like(close, dtype=float)
    returns = pd.Series(close).pct_change()

    for i in range(19, len(close)):
        ret_20 = returns[i-19:i+1].values
        if len(ret_20) > 0 and np.std(ret_20) > 0:
            # 计算偏度 (简化版)
            mean_ret = np.mean(ret_20)
            std_ret = np.std(ret_20)
            if std_ret > 0:
                skew = np.mean(((ret_20 - mean_ret) / std_ret) ** 3)
                result[i] = -skew  # 负偏度更好（下跌时波动大）
            else:
                result[i] = 0
        else:
            result[i] = 0

    return result


# ====================== 均线组合因子 ======================

@FactorRegistry.register('ma_momentum')
def ma_momentum_factor(close):
    """均线动量因子

    多条均线的方向一致性
    """
    if len(close) < 60:
        return np.zeros_like(close)

    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    # 各均线方向
    ma5_dir = np.sign(ma5 - np.roll(ma5, 5))
    ma10_dir = np.sign(ma10 - np.roll(ma10, 10))
    ma20_dir = np.sign(ma20 - np.roll(ma20, 20))

    # 方向一致性
    result = (ma5_dir + ma10_dir + ma20_dir) / 3
    return result


@FactorRegistry.register('ema_momentum')
def ema_momentum_factor(close):
    """EMA动量因子"""
    if len(close) < 60:
        return np.zeros_like(close)

    ema10 = pd.Series(close).ewm(span=10, adjust=False).mean().values
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema60 = pd.Series(close).ewm(span=60, adjust=False).mean().values

    # 多头排列且向上
    result = np.zeros_like(close, dtype=float)
    for i in range(59, len(close)):
        score = 0
        if ema10[i] > ema20[i] > ema60[i]:
            score += 1
        if ema10[i] > ema10[i-10]:
            score += 0.5
        if ema20[i] > ema20[i-20]:
            score += 0.3
        if ema60[i] > ema60[i-60]:
            score += 0.2
        result[i] = score

    return result


# ====================== 行业动量因子 ======================

def calc_industry_momentum(close, industry_returns):
    """行业动量因子

    Args:
        close: 个股收盘价
        industry_returns: 行业收益率序列

    Returns:
        行业动量因子值
    """
    if industry_returns is None or len(industry_returns) < 20:
        return np.zeros_like(close)

    # 行业20日动量
    ind_mom = industry_returns[-1] / (industry_returns[-20] + 1e-10) - 1

    result = np.full(len(close), ind_mom)
    return result


# ====================== 复合因子 ======================

@FactorRegistry.register('mom_diff_3_5')
def mom_diff_3_5(mom_3, mom_5):
    """动量差因子: 3日动量 - 5日动量"""
    return mom_3 - mom_5


@FactorRegistry.register('mom_diff_5_10')
def mom_diff_5_10(mom_5, mom_10):
    """动量差因子: 5日动量 - 10日动量"""
    return mom_5 - mom_10


@FactorRegistry.register('mom_diff_5_20')
def mom_diff_5_20(mom_5, mom_20):
    """动量差因子: 5日动量 - 20日动量"""
    return mom_5 - mom_20


@FactorRegistry.register('mom_diff_10_20')
def mom_diff_10_20(mom_10, mom_20):
    """动量差因子: 10日动量 - 20日动量"""
    return mom_10 - mom_20


@FactorRegistry.register('mom_x_lowvol_10_10')
def mom_x_lowvol_10_10(mom_10, volatility_10):
    """动量×低波动因子: 10日动量 × (-波动率)"""
    return mom_10 * (-volatility_10)


@FactorRegistry.register('mom_x_lowvol_10_20')
def mom_x_lowvol_10_20(mom_10, volatility_20):
    """动量×低波动因子: 10日动量 × (-波动率)"""
    return mom_10 * (-volatility_20)


@FactorRegistry.register('mom_x_lowvol_20_10')
def mom_x_lowvol_20_10(mom_20, volatility_10):
    """动量×低波动因子: 20日动量 × (-波动率)"""
    return mom_20 * (-volatility_10)


@FactorRegistry.register('mom_x_lowvol_20_20')
def mom_x_lowvol_20_20(mom_20, volatility_20):
    """动量×低波动因子: 20日动量 × (-波动率)"""
    return mom_20 * (-volatility_20)


@FactorRegistry.register('rsi_mom_10_5')
def rsi_mom_10_5(rsi_10, mom_5):
    """RSI+动量组合: (50-RSI)/100 + 动量"""
    return (50 - rsi_10) / 100 + mom_5


@FactorRegistry.register('rsi_mom_14_5')
def rsi_mom_14_5(rsi_14, mom_5):
    """RSI+动量组合: (50-RSI)/100 + 动量"""
    return (50 - rsi_14) / 100 + mom_5


@FactorRegistry.register('low_vol_high_pos')
def low_vol_high_pos(volatility_10, price_position_20):
    """低波动+高位置因子"""
    return -volatility_10 + price_position_20


@FactorRegistry.register('ret_vol_ratio_10')
def ret_vol_ratio_10(mom_10, volatility_10):
    """收益波动率比: 动量/波动率"""
    return mom_10 / (volatility_10 + 1e-10)


@FactorRegistry.register('ret_vol_ratio_20')
def ret_vol_ratio_20(mom_20, volatility_20):
    """收益波动率比: 动量/波动率"""
    return mom_20 / (volatility_20 + 1e-10)


@FactorRegistry.register('ma_golden_count')
def ma_golden_count(ma5, ma10, ma20, ma60):
    """均线多头排列计数"""
    return ((ma5 > ma10).astype(float) +
            (ma10 > ma20).astype(float) +
            (ma20 > ma60).astype(float))


@FactorRegistry.register('ma_cross_5_10')
def ma_cross_5_10(ma5, ma10):
    """均线交叉: (MA5-MA10)/MA10"""
    return (ma5 - ma10) / (ma10 + 1e-10)


@FactorRegistry.register('ma_cross_10_20')
def ma_cross_10_20(ma10, ma20):
    """均线交叉: (MA10-MA20)/MA20"""
    return (ma10 - ma20) / (ma20 + 1e-10)


# ====================== 综合多因子 ======================

@FactorRegistry.register('tech_fund_combo')
def tech_fund_combo(mom_20, mom_10, rsi, fundamental_score_val):
    """技术+基本面组合因子

    Args:
        mom_20: 20日动量
        mom_10: 10日动量
        rsi: RSI值
        fundamental_score_val: 基本面评分 (0-1)

    Returns:
        组合因子值
    """
    # 趋势动量
    trend_mom = np.where(mom_10 > 0, mom_20 * 2.1, 0)

    # 技术因子 (80%)
    tech_score = trend_mom * 0.7 + (rsi - 50) / 100 * 0.1

    # 基本面因子 (20%)
    fund_score = fundamental_score_val * 0.2

    return tech_score + fund_score


# ====================== 新增组合因子 ======================

@FactorRegistry.register('mom_roe_combo')
def mom_roe_combo(mom_20, roe_factor_val):
    """动量+ROE组合因子

    Args:
        mom_20: 20日动量
        roe_factor_val: ROE因子值

    Returns:
        组合因子值 (动量70% + ROE 30%)
    """
    return mom_20 * 0.7 + roe_factor_val * 0.3


@FactorRegistry.register('mom_profit_combo')
def mom_profit_combo(mom_20, profit_growth_val):
    """动量+净利润增长组合因子

    Args:
        mom_20: 20日动量
        profit_growth_val: 净利润增长因子值

    Returns:
        组合因子值
    """
    return mom_20 * 0.7 + profit_growth_val * 0.3


@FactorRegistry.register('rsi_vol_combo')
def rsi_vol_combo(rsi_14, volatility_20):
    """RSI+波动率组合

    Args:
        rsi_14: 14日RSI
        volatility_20: 20日波动率

    Returns:
        RSI超卖且低波动时更强
    """
    # RSI低于30为超卖(正面)，波动率低为正面
    rsi_signal = (50 - rsi_14) / 100  # 转正值
    return rsi_signal - volatility_20 * 0.5


@FactorRegistry.register('bb_rsi_combo')
def bb_rsi_combo(bb_percent_b, rsi_14):
    """布林带+RSI组合

    Args:
        bb_percent_b: 布林带%B
        rsi_14: 14日RSI

    Returns:
        价格在布林带低位且RSI超卖时更强
    """
    bb_signal = bb_percent_b  # 0-1之间，越低越好
    rsi_signal = (50 - rsi_14) / 100  # 转正值
    return rsi_signal - bb_signal * 0.3


@FactorRegistry.register('price_mom_volume')
def price_mom_volume(mom_20, volume_ratio):
    """价格动量+成交量组合

    Args:
        mom_20: 20日动量
        volume_ratio: 成交量比率

    Returns:
        动量向上且放量时更强
    """
    return mom_20 * (1 + np.clip(volume_ratio - 1, -0.5, 1))


@FactorRegistry.register('ma_cross_volume')
def ma_cross_volume(ma_cross, volume_ratio):
    """均线交叉+成交量组合

    Args:
        ma_cross: 均线交叉信号
        volume_ratio: 成交量比率

    Returns:
        金叉且放量时更强
    """
    return ma_cross * (1 + np.clip(volume_ratio - 1, -0.3, 0.5))


@FactorRegistry.register('lowvol_highroe')
def lowvol_highroe(volatility_20, roe_factor_val):
    """低波动+高ROE组合 (经典smart beta)

    Args:
        volatility_20: 20日波动率
        roe_factor_val: ROE因子值

    Returns:
        低波动且高ROE的股票
    """
    return -volatility_20 * 0.6 + roe_factor_val * 0.4


@FactorRegistry.register('highmom_lowvol')
def highmom_lowvol(mom_20, volatility_20):
    """高动量+低波动组合

    Args:
        mom_20: 20日动量
        volatility_20: 20日波动率

    Returns:
        动量强且波动低的股票
    """
    return mom_20 - volatility_20 * 0.5


@FactorRegistry.register('trend_strength')
def trend_strength(mom_5, mom_10, mom_20):
    """趋势强度因子

    多周期动量一致向上时更强

    Args:
        mom_5: 5日动量
        mom_10: 10日动量
        mom_20: 20日动量

    Returns:
        趋势强度得分
    """
    score = 0
    if mom_5 > 0:
        score += 1
    if mom_10 > 0:
        score += 1
    if mom_20 > 0:
        score += 1
    return score / 3


@FactorRegistry.register('reversal_strength')
def reversal_strength(rsi_14, volatility_20):
    """反转强度因子

    RSI超卖且波动率高时可能反转

    Args:
        rsi_14: 14日RSI
        volatility_20: 20日波动率

    Returns:
        反转强度得分
    """
    # RSI低于30为超卖，波动率高可能反弹
    oversold = np.where(rsi_14 < 30, 1, 0)
    high_vol = np.where(volatility_20 > np.mean(volatility_20), 1, 0)
    return oversold * 0.6 + high_vol * 0.4


@FactorRegistry.register('momentum_reversal')
def momentum_reversal(mom_5, mom_20):
    """动量反转因子

    短期动量强于长期动量可能延续，短期弱于长期可能反转

    Args:
        mom_5: 5日动量
        mom_20: 20日动量

    Returns:
        动量差
    """
    return mom_5 - mom_20


# ====================== 导出的因子列表 ======================
SINGLE_FACTORS = [
    # 趋势动量因子
    'trend_mom_v41',
    'trend_mom_v24',
    'trend_mom_v46',
    'trend_mom_v49',
    # 技术因子
    'rsi_factor',
    'rsi_6',
    'price_position',
    'volatility',
    'bb_width',
    'volume_ratio',
    # 基本面因子
    'fundamental_score',
    'roe_factor',
    'profit_growth_factor',
    'revenue_growth_factor',
    'eps_factor',
    'debt_ratio_factor',
    'gross_margin_factor',
    # 资金流向因子
    'money_flow_5d',
    'money_flow_20d',
    'volume_price_trend',
    # 筹码分布因子
    'price_concentration',
    'cost_basis',
    'volume_distribution',
    # 波动率因子
    'volatility_ratio',
    'return_skewness',
    # 均线因子
    'ma_momentum',
    'ema_momentum',
    # 复合因子
    'mom_diff_3_5',
    'mom_diff_5_10',
    'mom_diff_5_20',
    'mom_diff_10_20',
    'mom_x_lowvol_10_10',
    'mom_x_lowvol_10_20',
    'mom_x_lowvol_20_10',
    'mom_x_lowvol_20_20',
    'rsi_mom_10_5',
    'rsi_mom_14_5',
    'low_vol_high_pos',
    'ret_vol_ratio_10',
    'ret_vol_ratio_20',
    'ma_golden_count',
    'ma_cross_5_10',
    'ma_cross_10_20',
]

COMBO_FACTORS = [
    'V41_RSI_915',
    'V41_PricePos_915',
    'V46_RSI_915',
    'V49_RSI_915',
    'tech_fund_combo',
]

ALL_FACTORS = SINGLE_FACTORS + COMBO_FACTORS
