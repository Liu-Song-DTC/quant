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
        vol_window=20,
        slope_window=5,
        buy_th=0.02,
        sell_th=-0.01
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.vol_window = vol_window
        self.slope_window = slope_window
        self.buy_th = buy_th
        self.sell_th = sell_th

    def generate(self, code, market_data, signal_store):
        dates = market_data["datetime"].values
        close = market_data["close"]
        volume = market_data["volume"]
        amplitude = market_data["amplitude"] / 100.0
        turnover = market_data["turnover_rate"]

        self.ma_fast = close.rolling(self.fast_ma).mean()
        self.ma_slow = close.rolling(self.slow_ma).mean()

        ret_20 = close / close.shift(20) - 1
        ret_60 = close / close.shift(60) - 1
        momentum = 0.6 * ret_20 + 0.4 * ret_60
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume / vol_ma
        vol_ratio = vol_ratio.clip(0.5, 2.0)
        noise_penalty = np.exp(-amplitude)
        raw_score = momentum * vol_ratio * noise_penalty

        turn_confirm = (turnover / turnover.rolling(20).mean()).clip(0.7, 1.3)
        score = raw_score * turn_confirm
        score = score.rolling(5).mean()

        # =========================
        # 4. 风险
        # =========================
        logret = np.log(close / close.shift(1))
        self.vol = logret.rolling(self.vol_window).std()

        # =========================
        # 5. 生成 Signal
        # =========================
        for idx in close.index:
            if np.isnan(score.loc[idx]) or np.isnan(self.vol.loc[idx]):
                continue

            buy = score.loc[idx] > self.buy_th
            sell = score.loc[idx] < self.sell_th

            s = Signal(
                buy=buy,
                sell=sell,
                score=float(score[idx]),
                vol=float(self.vol.loc[idx]),
            )

            date = pd.to_datetime(dates[idx]).date()
            signal_store.set(code, date, s)

