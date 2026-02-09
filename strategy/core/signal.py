# core/signal.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Signal:
    """
    Strategy 对某只股票在某一天的“偏好表达”
    """
    buy: bool
    sell: bool
    score: float = 0.0
    risk_vol: float = 0.0

