import pandas as pd

from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor


class Strategy:
    """带股灾检测的市场状态判断"""

    def __init__(self, init_cash, max_position):
        self.signal_engine = SignalEngine()
        self.portfolio = PortfolioConstructor(
            max_position=max_position,
        )
        self.market_regime = []
        self.signal_store = SignalStore()
        self.init_cash = init_cash

        self.index_data = None

    def generate_market_regime(self, index_df):

        self.index_data = index_df.copy()
        close = index_df["close"]

        # EMA
        ema20 = close.ewm(span=20).mean()
        ema60 = close.ewm(span=60).mean()
        ema120 = close.ewm(span=120).mean()

        # 趋势
        trend_up = ema20 > ema60
        long_up = ema60 > ema120

        # 动量
        momentum = close / close.shift(20) - 1

        # 快速下跌检测
        mom_5 = close / close.shift(5) - 1

        regime = []

        for i in range(len(index_df)):

            if i < 120:
                regime.append(0)
                continue

            # 紧急熊市检测 - 快速下跌
            if mom_5.iloc[i] < -0.10:
                regime.append(-1)
                continue

            # 显著下跌
            if momentum.iloc[i] < -0.15:
                regime.append(-1)
                continue

            if momentum.iloc[i] < -0.08 and not trend_up.iloc[i]:
                regime.append(-1)
                continue

            # 牛市
            if trend_up.iloc[i] and long_up.iloc[i] and momentum.iloc[i] > 0:
                regime.append(1)
                continue

            if momentum.iloc[i] > 0.05:
                regime.append(1)
                continue

            regime.append(0)

        self.index_data["regime"] = regime

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
