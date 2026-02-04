from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor

# core/strategy.py
class Strategy:
    """
    Strategy = 各组件的组合
    """

    def __init__(self, init_cash, max_position):
        self.signal_engine = SignalEngine()
        self.portfolio = PortfolioConstructor(max_position=max_position)
        self.signal_store = SignalStore()
        self.init_cash = init_cash

    def generate_signal(self, code, market_data):
        """
        生成所有 signal
        """
        self.signal_engine.generate(code, market_data, self.signal_store)

    def generate_positions(
        self,
        date,
        universe,
        current_positions,
        cash,
        prices,
    ):
        """
        根据signal 生成持仓target
        """
        return self.portfolio.build(
            date,
            universe,
            current_positions,
            self.signal_store,
            cash,
            prices,
            self.init_cash
        )
