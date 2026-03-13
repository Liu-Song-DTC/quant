import pandas as pd

from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor
from .market_regime_detector import MarketRegimeDetector
from .stock_pool_filter import StockPoolFilter


class Strategy:
    """带股灾检测的市场状态判断"""

    def __init__(self, init_cash, max_position, fundamental_data=None):
        self.signal_engine = SignalEngine()
        if fundamental_data:
            self.signal_engine.set_fundamental_data(fundamental_data)
        self.portfolio = PortfolioConstructor(
            max_position=max_position,
        )
        self.market_regime = []
        self.signal_store = SignalStore()
        self.init_cash = init_cash

        self.index_data = None

        # 使用独立的市场状态检测器
        self.regime_detector = MarketRegimeDetector()

        # 季度股票池筛选器（暂时关闭）
        self.stock_pool_filter = None

    def generate_market_regime(self, index_df):
        # 使用独立的市场状态检测器
        self.index_data = self.regime_detector.generate(index_df)

    def generate_signal(self, code, market_data):
        self.signal_engine.generate(code, market_data, self.signal_store)

    def generate_positions(
        self,
        date,
        universe,
        current_positions,
        cash,
        prices,
        cost,
        rebalance,
    ):
        # 季度股票池筛选
        if self.stock_pool_filter:
            universe = self.stock_pool_filter.filter_quarterly(universe, date)

        market_regime = 0
        if self.index_data is not None:
            row = self.index_data[self.index_data["datetime"].dt.date == date]
            if not row.empty:
                market_regime = int(row["regime"].values[0])
        return self.portfolio.build(
            date=date,
            universe=universe,
            current_positions=current_positions,
            signal_store=self.signal_store,
            cash=cash,
            prices=prices,
            market_regime=market_regime,
            cost=cost,
            rebalance=rebalance
        )
