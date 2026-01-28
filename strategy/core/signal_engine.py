# core/signal_engine.py
from abc import ABC, abstractmethod
from .signal import Signal

class SignalEngine(ABC):
    """
    负责：市场信息 → Signal
    """

    @abstractmethod
    def generate(self, code, date, market_data) -> Signal:
        pass

