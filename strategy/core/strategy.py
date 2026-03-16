import pandas as pd

from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor
from .market_regime_detector import MarketRegimeDetector
from .config_loader import load_config


class Strategy:
    """带股灾检测的市场状态判断"""

    def __init__(self, init_cash, max_position=None, fundamental_data=None):
        self.signal_engine = SignalEngine()
        if fundamental_data:
            self.signal_engine.set_fundamental_data(fundamental_data)

        # 从配置加载组合参数
        config = load_config()
        portfolio_config = config.get_portfolio_config()

        # 使用传入的 max_position 或配置文件中的值
        final_max_position = max_position if max_position is not None else portfolio_config.get('max_position', 10)

        self.portfolio = PortfolioConstructor(
            max_position=final_max_position,
        )
        self.market_regime = []
        self.signal_store = SignalStore()
        self.init_cash = init_cash

        self.index_data = None

        # 使用独立的市场状态检测器
        self.regime_detector = MarketRegimeDetector()

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
