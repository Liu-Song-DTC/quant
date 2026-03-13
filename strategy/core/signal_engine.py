# core/signal_engine.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .signal import Signal
from .signal_store import SignalStore
from .factors import trend_mom_v41

import warnings
warnings.filterwarnings('ignore')


class SignalEngine:
    """优化的均值回归+趋势策略 - 支持市场状态信息"""

    def __init__(self, config=None):
        self.config = config or {}
        self.min_score = 0.35
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.fundamental_data = None

        # 市场状态信息
        self.market_regime_data = None  # DataFrame: datetime, regime, confidence, momentum_score, trend_score, volatility, is_extreme
        self.current_idx = 0

    def set_fundamental_data(self, fundamental_data):
        """设置基本面数据"""
        self.fundamental_data = fundamental_data

    def set_market_regime(self, regime_df: pd.DataFrame):
        """
        设置市场状态数据
        DataFrame应包含列: datetime, regime, confidence, momentum_score, trend_score, volatility, is_extreme
        """
        if 'datetime' in regime_df.columns:
            regime_df = regime_df.copy()
            regime_df['datetime'] = pd.to_datetime(regime_df['datetime'])
            self.market_regime_data = regime_df.set_index('datetime')

    def _get_market_info(self, date) -> Dict[str, Any]:
        """获取指定日期的市场状态信息"""
        if self.market_regime_data is None:
            return {
                'regime': 0,
                'confidence': 0.0,
                'momentum_score': 0.0,
                'trend_score': 0.0,
                'volatility': 0.15,
                'is_extreme': False,
            }

        dt = pd.to_datetime(date)
        if dt in self.market_regime_data.index:
            row = self.market_regime_data.loc[dt]
            return {
                'regime': int(row.get('regime', 0)),
                'confidence': float(row.get('confidence', 0.0)),
                'momentum_score': float(row.get('momentum_score', 0.0)),
                'trend_score': float(row.get('trend_score', 0.0)),
                'volatility': float(row.get('volatility', 0.15)),
                'is_extreme': bool(row.get('is_extreme', False)),
            }
        return {
            'regime': 0,
            'confidence': 0.0,
            'momentum_score': 0.0,
            'trend_score': 0.0,
            'volatility': 0.15,
            'is_extreme': False,
        }

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
        信号生成 - 优化版
        保持主因子IC，同时用负IC因子做风险过滤
        """
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0,
                factor_value=0.0, factor_name='V41_only',
                risk_vol=0.03, risk_regime=0, risk_confidence=0.0,
                risk_extreme=False, adjusted_score=0.0
            )

        # === 1. 主因子: trend_mom_v41 (IC=3.59%) ===
        trend_mom = trend_mom_v41(ind['mom_20'][idx], ind['mom_10'][idx])
        factor_value = trend_mom
        score = max(0, trend_mom)

        # === 2. 负IC因子仅用于风险标记，不直接扣分 ===
        # 记录风险状态供后续使用
        vol_20 = self._safe_get(ind, 'volatility_20', idx, 0.02)
        is_high_vol = vol_20 > 0.03  # 高波动标志

        vpt = self._safe_get(ind, 'volume_price_trend', idx, 0)
        is_vpt_risk = abs(vpt) > 3  # 价量背离风险

        skew = self._safe_get(ind, 'return_skewness', idx, 0)
        is_skew_risk = skew > 0.5  # 偏度风险

        # === 3. 获取市场状态信息 ===
        market_info = self._get_market_info(current_date)
        risk_regime = market_info['regime']
        risk_confidence = market_info['confidence']
        risk_extreme = market_info['is_extreme']

        # === 4. 波动率风险 ===
        risk_vol = ind['atr_ratio'][idx] * 2

        # === 5. 风险调整后的分数 ===
        regime_weight = 1.0
        if risk_regime == -1:  # 熊市
            regime_weight = 0.6
        elif risk_regime == 1:  # 牛市
            regime_weight = 1.1

        if risk_extreme:
            regime_weight *= 0.8

        adjusted_score = score * regime_weight

        # === 6. 交易信号 ===
        # 买入：主因子为正，且非高风险状态
        buy = trend_mom >= 0 and adjusted_score > 0.01

        # 卖出：主因子转负
        sell = trend_mom <= -0.03

        return Signal(
            buy=buy,
            sell=sell,
            score=score,
            factor_value=factor_value,
            factor_name='V41_only',
            risk_vol=risk_vol,
            risk_regime=risk_regime,
            risk_confidence=risk_confidence,
            risk_extreme=risk_extreme or is_high_vol,
            adjusted_score=adjusted_score
        )

    def _safe_get(self, ind: dict, key: str, idx: int, default: float = 0.0) -> float:
        """安全获取数组元素"""
        arr = ind.get(key)
        if arr is None:
            return default
        if isinstance(arr, (int, float)):
            return default
        if hasattr(arr, '__len__'):
            if len(arr) <= idx:
                return default
            val = arr[idx]
            if isinstance(val, (int, float)) and not np.isnan(val):
                return val
        return default

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
