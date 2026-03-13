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

    # RSI (14日)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    result['rsi'] = rsi

    # RSI (6日)
    avg_gain6 = pd.Series(gain).rolling(6).mean().values
    avg_loss6 = pd.Series(loss).rolling(6).mean().values
    rs6 = avg_gain6 / (avg_loss6 + 1e-10)
    result['rsi_6'] = 100 - (100 / (1 + rs6))

    # RSI变化
    result['rsi_change'] = rsi - pd.Series(rsi).shift(5).values

    # RSI位置
    rsi_high = pd.Series(rsi).rolling(20).max().values
    rsi_low = pd.Series(rsi).rolling(20).min().values
    result['rsi_position'] = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)

    # 波动率
    returns = pd.Series(close).pct_change().values
    result['volatility_20'] = pd.Series(returns).rolling(20).std().values
    result['volatility_60'] = pd.Series(returns).rolling(60).std().values

    if high is not None and low is not None:
        # 价格位置
        for period in [10, 20, 60]:
            high_p = pd.Series(high).rolling(period).max().values
            low_p = pd.Series(low).rolling(period).min().values
            result[f'price_position_{period}'] = (close - low_p) / (high_p - low_p + 1e-10)

    if volume is not None:
        # 成交量因子
        volume_ma20 = pd.Series(volume).rolling(20).mean().values
        result['volume_ratio'] = volume / (volume_ma20 + 1e-10)

        # 成交量变化率
        vol_ma5 = pd.Series(volume).rolling(5).mean().values
        result['volume_change'] = volume / (vol_ma5 + 1e-10) - 1

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


# 导出常用因子名称
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
]

COMBO_FACTORS = [
    'V41_RSI_915',
    'V41_PricePos_915',
    'V46_RSI_915',
    'V49_RSI_915',
    'tech_fund_combo',
]

ALL_FACTORS = SINGLE_FACTORS + COMBO_FACTORS
