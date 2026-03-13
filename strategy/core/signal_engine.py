# core/signal_engine.py
import numpy as np
import pandas as pd

from .signal import Signal
from .signal_store import SignalStore

import warnings
warnings.filterwarnings('ignore')


class SignalEngine:
    """优化的均值回归+趋势策略"""

    def __init__(self, config=None):
        self.config = config or {}
        self.min_score = 0.32
        self.rsi_oversold = 28
        self.rsi_overbought = 72

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        dates = market_data["datetime"].values
        close = market_data['close']

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)

        last_sig = None
        for idx in close.index:
            sig = self._generate_signal(indicators, idx, last_sig)
            last_sig = sig
            date = pd.to_datetime(dates[idx]).date()
            signal_store.set(code, date, sig)

    def _calculate_indicators(self, data: pd.DataFrame):
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        result = {'close': close, 'high': high, 'low': low, 'volume': volume}

        # EMA
        for span in [5, 10, 20, 60]:
            result[f'ema{span}'] = self._ema(close, span)

        # MA
        for span in [5, 10, 20, 60]:
            result[f'ma{span}'] = self._sma(close, span)

        # RSI
        result['rsi'] = self._rsi(close, 14)

        # 布林带
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = self._bollinger(close, 20, 2)

        # 成交量
        result['volume_ma20'] = self._sma(volume, 20)
        result['volume_ratio'] = volume / (result['volume_ma20'] + 1e-10)

        # ATR
        result['atr'] = self._atr(high, low, close, 14)
        result['atr_ratio'] = result['atr'] / close

        # 动量
        for period in [5, 10, 20]:
            result[f'mom_{period}'] = close / self._shift(close, period) - 1

        # 价格位置
        result['high_20'] = self._rolling_max(high, 20)
        result['low_20'] = self._rolling_min(low, 20)

        # 均线关系
        result['ema5_above_20'] = result['ema5'] > result['ema20']
        result['ema20_above_60'] = result['ema20'] > result['ema60']
        result['full_golden'] = (result['ema5'] > result['ema20']) & (result['ema20'] > result['ema60'])
        result['full_death'] = (result['ema5'] < result['ema20']) & (result['ema20'] < result['ema60'])

        # 趋势强度
        result['trend_strength'] = (result['ema20'] - result['ema60']) / result['ema60']

        # 斜率
        result['ema20_slope'] = result['ema20'] / self._shift(result['ema20'], 10) - 1

        # MACD
        result['macd'], result['macd_signal'], result['macd_hist'] = self._macd(close)

        return result

    def _generate_signal(self, ind, idx, last_sig):
        if idx < 60:
            return Signal(buy=False, sell=False, score=0.0, risk_vol=0.03)

        close = ind['close'][idx]
        rsi = ind['rsi'][idx]

        buy_score = 0.0
        sell_score = 0.0

        # ==================== 买入信号 ====================

        # 1. 超跌反弹 - 更敏感
        if rsi < self.rsi_oversold:
            buy_score += 0.25

        # 2. 均线多头排列
        if ind['full_golden'][idx]:
            buy_score += 0.20

        # 3. 动量反弹
        if ind['mom_5'][idx] > 0 and ind['mom_10'][idx] > -0.02:
            buy_score += 0.15

        # 4. 站上均线
        if ind['ema5_above_20'][idx]:
            buy_score += 0.12

        # 5. 趋势向上
        if ind['trend_strength'][idx] > 0:
            buy_score += 0.10

        # 6. MACD金叉
        if idx > 0:
            if ind['macd_hist'][idx] > 0 and ind['macd_hist'][idx-1] <= 0:
                buy_score += 0.15

        # 7. 价格在低位
        if close <= ind['low_20'][idx] * 1.05:
            buy_score += 0.10

        # 8. 成交量放大
        if ind['volume_ratio'][idx] > 1.3:
            buy_score += 0.10

        # ==================== 卖出信号 ====================

        # 1. 超买
        if rsi > self.rsi_overbought:
            sell_score += 0.25

        # 2. 均线死叉
        if ind['full_death'][idx]:
            sell_score += 0.20

        # 3. 动量转跌
        if ind['mom_10'][idx] < -0.05:
            sell_score += 0.20

        # 4. 跌破均线
        if not ind['ema5_above_20'][idx] and idx > 0 and ind['ema5_above_20'][idx-1]:
            sell_score += 0.15

        # 5. 趋势向下
        if ind['trend_strength'][idx] < -0.02:
            sell_score += 0.10

        # 6. MACD死叉
        if idx > 0:
            if ind['macd_hist'][idx] < 0 and ind['macd_hist'][idx-1] >= 0:
                sell_score += 0.10

        score = buy_score - sell_score

        if score < self.min_score:
            buy = False
            score = max(0, score)
        else:
            buy = True

        sell = score < -0.15

        risk_vol = ind['atr_ratio'][idx] * 2

        return Signal(buy=buy, sell=sell, score=score, risk_vol=risk_vol)

    # 辅助函数
    def _sma(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        result[window-1:] = np.convolve(arr, np.ones(window)/window, mode='valid')
        return result

    def _ema(self, arr, span):
        result = np.zeros_like(arr, dtype=float)
        result[0] = arr[0]
        alpha = 2 / (span + 1)
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result

    def _rsi(self, close, window):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = self._sma(gain, window)
        avg_loss = self._sma(loss, window)
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _bollinger(self, close, window, num_std):
        middle = self._sma(close, window)
        std = np.array([np.std(close[i-window:i]) if i >= window else 0 for i in range(len(close))])
        return middle + num_std * std, middle, middle - num_std * std

    def _atr(self, high, low, close, window):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        return self._sma(tr, window)

    def _rolling_max(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result

    def _rolling_min(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result

    def _shift(self, arr, periods):
        result = np.zeros_like(arr, dtype=float)
        result[periods:] = arr[:-periods]
        result[:periods] = np.nan
        return result

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
