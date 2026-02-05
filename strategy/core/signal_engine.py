# core/signal_engine.py
import numpy as np
import pandas as pd

from .signal import Signal
from .signal_store import SignalStore

class SignalEngine:
    """
    多因子趋势 / 动量 Signal
    """

    def __init__(
        self,
        fast_ma=20,
        slow_ma=60,
        mom_window=20,
        vol_window=20,
        slope_window=5,
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.mom_window = mom_window
        self.vol_window = vol_window
        self.slope_window = slope_window

    def generate(self, code, market_data, signal_store):
        close = market_data["close"]
        dates = market_data["datetime"].values

        # =========================
        # 1. 趋势
        # =========================
        ma_fast = close.rolling(self.fast_ma).mean()
        ma_slow = close.rolling(self.slow_ma).mean()
        trend = ma_fast / ma_slow - 1.0

        # =========================
        # 2. 动量
        # =========================
        momentum = close / close.shift(self.mom_window) - 1.0

        # =========================
        # 3. 趋势斜率（过滤震荡）
        # =========================
        trend_slope = ma_fast.diff(self.slope_window)

        # =========================
        # 4. 风险
        # =========================
        logret = np.log(close / close.shift(1))
        vol = logret.rolling(self.vol_window).std()

        # =========================
        # 5. 生成 Signal
        # =========================
        for idx in close.index:
            if (
                np.isnan(trend.loc[idx])
                or np.isnan(momentum.loc[idx])
                or np.isnan(trend_slope.loc[idx])
                or np.isnan(vol.loc[idx])
            ):
                continue

            score = 0.6 * trend.loc[idx] + 0.4 * momentum.loc[idx]

            buy = (
                trend.loc[idx] > 0
                and momentum.loc[idx] > 0
                and trend_slope.loc[idx] > 0
            )

            sell = (
                trend.loc[idx] < 0
                or momentum.loc[idx] < 0
            )

            s = Signal(
                buy=buy,
                sell=sell,
                score=float(score),
                vol=float(vol.loc[idx]),
            )

            date = pd.to_datetime(dates[idx]).date()
            signal_store.set(code, date, s)

