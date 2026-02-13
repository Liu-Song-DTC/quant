import pandas as pd

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
        high = index_df["high"]
        low = index_df["low"]
        volume = index_df["volume"]

        # =====================================================
        #  长周期趋势结构
        # =====================================================

        ema60 = close.ewm(span=60).mean()
        ema150 = close.ewm(span=150).mean()

        long_trend_up = ema60 > ema150

        # =====================================================
        #  趋势斜率（避免死猫反弹）
        # =====================================================

        slope = ema60 - ema60.shift(20)

        # =====================================================
        #  波动风险（ATR 归一化）
        # =====================================================

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()

        atr_ratio = atr / close

        # 风险基准
        atr_baseline = atr_ratio.rolling(200).mean()

        high_volatility = atr_ratio > 1.3 * atr_baseline

        # =====================================================
        #  成交量结构
        # =====================================================

        vol_ma20 = volume.rolling(20).mean()
        vol_ma60 = volume.rolling(60).mean()

        volume_weak = vol_ma20 < vol_ma60

        # =====================================================
        #  综合判断
        # =====================================================

        regime = []

        for i in range(len(index_df)):

            if i < 200:
                regime.append(0)
                continue

            # 基础趋势判断
            if long_trend_up.iloc[i] and slope.iloc[i] > 0:
                r = 1
            elif not long_trend_up.iloc[i]:
                r = -1
            else:
                r = 0

            # 波动惩罚（风险扩张）
            if high_volatility.iloc[i]:
                r -= 1

            # 缩量上涨惩罚
            if r == 1 and volume_weak.iloc[i]:
                r = 0

            # 限制区间
            r = max(-1, min(1, r))

            regime.append(r)

        self.index_data["regime"] = regime

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
        cost,
        rebalance,
    ):
        """
        根据signal 生成持仓target
        """
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
