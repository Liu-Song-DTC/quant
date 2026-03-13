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
        self.min_score = 0.35  # 提高最低分数
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.fundamental_data = None  # 基本面数据

    def set_fundamental_data(self, fundamental_data):
        """设置基本面数据"""
        self.fundamental_data = fundamental_data

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        dates = market_data["datetime"].values
        close = market_data['close']

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)

        last_sig = None
        for idx in close.index:
            sig = self._generate_signal(indicators, idx, last_sig, dates[idx], code)
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

    def _generate_signal(self, ind, idx, last_sig, current_date=None, code=None):
        """
        信号生成 - 趋势动量V24因子
        公式: 如果10日动量>0，用20日动量*2.1；否则用20日动量*0.04
        IC = 8.23%
        """
        if idx < 60:
            return Signal(buy=False, sell=False, score=0.0, risk_vol=0.03, factor_value=0.0)

        mom_20 = ind['mom_20'][idx]
        mom_10 = ind['mom_10'][idx]

        # 趋势动量V24：如果10日动量>0，用20日动量*2.1；否则用20日动量*0.04
        if mom_10 > 0:
            trend_mom = mom_20 * 2.1
        else:
            trend_mom = mom_20 * 0.04

        # 原始因子值（用于IC计算）
        factor_value = trend_mom

        # 交易分数（用于排序）
        score = max(0, trend_mom)

        # 买入：趋势动量>0
        # 卖出：趋势动量<-0.03
        buy = trend_mom >= 0
        sell = trend_mom <= -0.03

        risk_vol = ind['atr_ratio'][idx] * 2

        return Signal(buy=buy, sell=sell, score=score, risk_vol=risk_vol, factor_value=factor_value)

    def _get_fundamental_score(self, code, current_date):
        """获取基本面因子评分

        注意：基本面数据是百分比形式（如ROE=12.16表示12.16%）
        """
        if not self.fundamental_data or not code:
            return 0

        score = 0.0

        # ROE评分 - 最重要的因子（数据是百分比形式，如5.79表示5.79%）
        roe = self.fundamental_data.get_roe(code, current_date)
        if roe is not None:
            if roe > 15:  # > 15%
                score += 0.35
            elif roe > 10:  # > 10%
                score += 0.25
            elif roe > 5:  # > 5%
                score += 0.15

        # 净利润增长 - 重要（数据是百分比，如66.39表示66.39%）
        profit_growth = self.fundamental_data.get_profit_growth(code, current_date)
        if profit_growth is not None:
            # 转换为小数形式比较
            if profit_growth > 50:  # > 50%
                score += 0.30
            elif profit_growth > 20:  # > 20%
                score += 0.20
            elif profit_growth > 0:  # > 0%
                score += 0.10

        # 营业收入增长
        revenue_growth = self.fundamental_data.get_revenue_growth(code, current_date)
        if revenue_growth is not None:
            if revenue_growth > 30:  # > 30%
                score += 0.20
            elif revenue_growth > 15:  # > 15%
                score += 0.12
            elif revenue_growth > 0:  # > 0%
                score += 0.05

        # 每股收益
        eps = self.fundamental_data.get_eps(code, current_date)
        if eps is not None and eps > 0:
            if eps > 1.0:
                score += 0.20
            elif eps > 0.5:
                score += 0.12

        return score

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
