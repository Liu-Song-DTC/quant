# core/signal_types.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    """
    Strategy 对某只股票在某一天的"偏好表达"
    """
    buy: bool
    sell: bool
    score: float = 0.0
    risk_vol: float = 0.0


@dataclass
class RawSignal:
    """
    原始信号组件 - 解耦不同的信号来源
    """
    # 技术面信号
    tech_score: float = 0.0  # 技术面综合得分
    rsi: Optional[float] = None  # RSI值
    momentum: Optional[float] = None  # 动量

    # 基本面信号
    fund_score: float = 0.0  # 基本面得分
    roe: Optional[float] = None  # ROE
    profit_growth: Optional[float] = None  # 净利润增长

    # 趋势信号
    trend_score: float = 0.0  # 趋势得分
    ma_cross: int = 0  # 均线金叉/死叉: 1=金叉, -1=死叉, 0=无

    # 成交量信号
    volume_score: float = 0.0  # 成交量得分


@dataclass
class AdjustedSignal:
    """
    调整后的信号 - 考虑市场状态和风险
    """
    raw: RawSignal

    # 市场状态
    market_regime: int = 0  # -1=熊, 0=震荡, 1=牛

    # 调整后的得分
    final_score: float = 0.0
    buy: bool = False
    sell: bool = False

    # 风险指标
    risk_vol: float = 0.03
    position_size: float = 0.0  # 建议仓位


@dataclass
class MarketRegime:
    """
    市场状态
    """
    regime: int  # -1=熊, 0=震荡, 1=牛
    confidence: float = 0.0  # 判断置信度 0-1

    # 详细指标
    trend: str = "sideways"  # up, down, sideways
    volatility: float = 0.0  # 波动率
    momentum: float = 0.0  # 动量

    # 历史状态（用于平滑）
    prev_regime: int = 0
    regime_streak: int = 0  # 连续同状态天数
