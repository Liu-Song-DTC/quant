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

        ema20 = close.ewm(span=20).mean()
        ema60 = close.ewm(span=60).mean()
        ema120 = close.ewm(span=120).mean()

        # 趋势判断
        mid_trend_up = ema20 > ema60
        long_trend_up = ema60 > ema120

        # =====================================================
        #  趋势斜率
        # =====================================================

        slope_20 = (ema20 - ema20.shift(10)) / ema20
        slope_60 = (ema60 - ema60.shift(20)) / ema60

        # =====================================================
        #  回撤检测（关键改进）
        # =====================================================

        # 计算60日内最高点的回撤
        rolling_high = close.rolling(60).max()
        drawdown = (close - rolling_high) / rolling_high

        # =====================================================
        #  波动风险（ATR 归一化）
        # =====================================================

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()

        atr_ratio = atr / close
        atr_baseline = atr_ratio.rolling(120).mean()

        high_volatility = atr_ratio > 1.5 * atr_baseline

        # =====================================================
        #  综合判断 - 增加下跌保护
        # =====================================================

        regime = []

        for i in range(len(index_df)):

            if i < 120:
                regime.append(0)
                continue

            # 快速下跌检测 - 优先级最高
            if drawdown.iloc[i] < -0.12:  # 从高点下跌超过12%
                r = -1
                regime.append(r)
                continue

            # 剧烈下跌中，即使均线还没完全转空，也要保守
            if drawdown.iloc[i] < -0.08 and slope_20.iloc[i] < -0.01:
                r = -1
                regime.append(r)
                continue

            # 基础趋势判断
            if mid_trend_up.iloc[i] and long_trend_up.iloc[i]:
                # 中期和长期都向上
                if slope_20.iloc[i] > 0.005:
                    r = 1  # 牛市
                elif slope_20.iloc[i] > 0:
                    r = 0  # 温和上涨，震荡对待
                else:
                    r = 0  # 震荡
            elif not mid_trend_up.iloc[i] and not long_trend_up.iloc[i]:
                # 中期和长期都向下
                if slope_60.iloc[i] < -0.008:
                    r = -1  # 熊市
                else:
                    r = 0  # 震荡
            else:
                r = 0  # 趋势不一致，震荡

            # 高波动惩罚
            if high_volatility.iloc[i]:
                if r == 1:
                    r = 0  # 高波动牛市降为震荡
                elif r == 0:
                    r = -1  # 高波动震荡降为熊市

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
