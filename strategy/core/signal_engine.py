# core/signal_engine.py
from .signal import Signal
from .signal_store import SignalStore

class SignalEngine:
    """
    负责：市场信息 → Signal
    """

    def generate(self, code, market_data, signal_store):
        return

