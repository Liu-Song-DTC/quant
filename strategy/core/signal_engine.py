# core/signal_engine.py
import numpy as np
import pandas as pd

from .signal import Signal
from .signal_store import SignalStore

class SignalEngine:
    """
    负责：市场信息 → Signal
    """
    def __init__(self, ma_window=20, vol_window=20):
        self.ma_window = ma_window
        self.vol_window = vol_window

    def generate(self, code, market_data, signal_store):
        close = market_data["close"]
        dates = market_data["datetime"].values

        # =========
        # 1. alpha
        # =========
        ma = close.rolling(self.ma_window).mean()
        score = close / ma - 1.0

        # =========
        # 2. risk
        # =========
        logret = np.log(close / close.shift(1))
        vol = logret.rolling(self.vol_window).std()

        # =========
        # 3. 生成 Signal
        # =========
        for idx in close.index:
            if (
                np.isnan(ma.loc[idx])
                or np.isnan(vol.loc[idx])
            ):
                continue

            s = Signal(
                buy=score.loc[idx] > 0,
                sell=score.loc[idx] < 0,
                score=float(score.loc[idx]),
                vol=float(vol.loc[idx]),
            )

            date = pd.to_datetime(dates[idx]).date()

            signal_store.set(code, date, s)

