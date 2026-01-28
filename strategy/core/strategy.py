# core/strategy.py
class Strategy:
    """
    Strategy = 各组件的组合
    """

    def __init__(self, signal_engine, portfolio):
        self.signal_engine = signal_engine
        self.portfolio = portfolio
        self.signal_store = SignalStore()

    def run_day(self, date, universe, market_data):
        """
        生成当天所有可执行 signal
        """
        for code in universe:
            sig = self.signal_engine.generate(code, date, market_data)
            self.signal_store.set(code, date, sig)

        return self.signal_store

