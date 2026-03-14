# core/signal_engine.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .signal import Signal
from .signal_store import SignalStore
from .factor_library import calc_factor_volatility_10, calc_factor_rsi_8, calc_factor_bb_width_20, calc_factor_mom1_rsi7_bb10

import warnings
warnings.filterwarnings('ignore')


class SignalEngine:
    """信号生成引擎 - 使用高质量因子库"""

    def __init__(self, config=None):
        self.config = config or {}
        self.min_score = 0.35

        # 市场状态信息
        self.market_regime_data = None
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

        # RSI - 更多周期
        result['rsi'] = self._rsi(close, 14)
        result['rsi_14'] = self._rsi(close, 14)
        result['rsi_8'] = self._rsi(close, 8)
        result['rsi_6'] = self._rsi(close, 6)
        result['rsi_10'] = self._rsi(close, 10)

        # 布林带
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = self._bollinger(close, 20, 2)
        # 布林带宽度
        bb_std = np.zeros_like(close)
        bb_std[20:] = np.array([np.std(close[max(0,i-20):i]) for i in range(20, len(close))])
        bb_middle = result['bb_middle']
        result['bb_width_20'] = 4 * bb_std / (bb_middle + 1e-10)

        # 成交量
        result['volume_ma20'] = self._sma(volume, 20)
        result['volume_ratio'] = volume / (result['volume_ma20'] + 1e-10)

        # ATR
        result['atr'] = self._atr(high, low, close, 14)
        result['atr_ratio'] = result['atr'] / close

        # 动量 - 更多周期
        for period in [3, 5, 10, 20, 30]:
            result[f'mom_{period}'] = close / self._shift(close, period) - 1

        # 价格位置
        result['high_20'] = self._rolling_max(high, 20)
        result['low_20'] = self._rolling_min(low, 20)
        result['price_position_20'] = (close - result['low_20']) / (result['high_20'] - result['low_20'] + 1e-10)

        # 波动率
        returns = np.diff(close, prepend=close[0]) / close
        result['ret'] = returns
        result['volatility_10'] = self._rolling_std(returns, 10)
        result['volatility_5'] = self._rolling_std(returns, 5)
        result['volatility_20'] = self._rolling_std(returns, 20)

        # 均线关系
        result['ema5_above_20'] = result['ema5'] > result['ema20']
        result['ema20_above_60'] = result['ema20'] > result['ema60']
        result['full_golden'] = (result['ema5'] > result['ema20']) & (result['ema20'] > result['ema60'])
        result['full_death'] = (result['ema5'] < result['ema20']) & (result['ema20'] < result['ema60'])

        # 趋势强度
        result['trend_strength'] = (result['ema20'] - result['ema60']) / (result['ema60'] + 1e-10)

        # 斜率
        result['ema20_slope'] = result['ema20'] / self._shift(result['ema20'], 10) - 1

        # MACD
        result['macd'], result['macd_signal'], result['macd_hist'] = self._macd(close)

        return result

    def _rolling_std(self, arr, window):
        """计算滚动标准差"""
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window, len(arr)):
            result[i] = np.std(arr[i-window:i])
        return result

    def _generate_signal(self, ind, idx, last_sig, current_date=None, code=None):
        """
        信号生成 - 纯因子组合版
        不依赖市场状态判断，依赖因子本身的自适应性
        """
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0,
                factor_value=0.0, factor_name='V41',
                risk_vol=0.03, risk_regime=0, risk_confidence=0.0,
                risk_extreme=False, adjusted_score=0.0
            )

        # 获取市场状态（仅用于记录，不影响因子计算）
        market_info = self._get_market_info(current_date)
        risk_regime = market_info['regime']
        risk_confidence = market_info['confidence']
        risk_extreme = market_info['is_extreme']

        # === 使用纯因子组合，不依赖市场状态 ===
        factor_name, factor_value, risk_info = self._select_factor(ind, idx, risk_regime)

        score = max(0, factor_value)
        risk_vol = ind['atr_ratio'][idx] * 2

        # === 极端波动时降低仓位 ===
        regime_weight = 1.0
        if risk_extreme:
            regime_weight = 0.85

        adjusted_score = score * regime_weight

        # === 交易信号 ===
        # 买入阈值
        buy = factor_value > 0.15 and adjusted_score > 0.05
        sell = factor_value <= -0.05

        return Signal(
            buy=buy,
            sell=sell,
            score=score,
            factor_value=factor_value,
            factor_name=factor_name,
            risk_vol=risk_vol,
            risk_regime=risk_regime,
            risk_confidence=risk_confidence,
            risk_extreme=risk_extreme or risk_info.get('is_high_vol', False),
            adjusted_score=adjusted_score
        )

    def _select_factor(self, ind, idx, regime: int):
        """
        根据市场状态选择因子组合

        使用高质量因子库:
        - volatility_10: IC=4.02%, IR=1.07 (A股最佳)
        - volatility_5: IC=3.21%
        - rsi_8: IC=2.37%
        - bb_width_20: IC=2.33%
        """
        # 获取各因子值
        vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
        vol_5 = self._safe_get(ind, 'volatility_5', idx, 0.02)
        vol_20 = self._safe_get(ind, 'volatility_20', idx, 0.02)
        rsi = self._safe_get(ind, 'rsi_14', idx, 50)
        rsi_8 = self._safe_get(ind, 'rsi_8', idx, 50)
        rsi_10 = self._safe_get(ind, 'rsi_10', idx, 50)
        rsi_6 = self._safe_get(ind, 'rsi_6', idx, 50)
        bb_width = self._safe_get(ind, 'bb_width_20', idx, 0.05)
        mom_20 = self._safe_get(ind, 'mom_20', idx, 0)
        mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
        mom_3 = self._safe_get(ind, 'mom_3', idx, 0)
        price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)

        # 计算各因子
        # 波动率因子 (放大到类似动量) - 使用vol_10
        vol_factor = vol_10 * 10

        # RSI因子 (-1 到 1)
        rsi_val = (rsi_8 - 50) / 50

        # 多周期RSI平均
        rsi_avg = (rsi_6 + rsi_8 + rsi_10) / 3
        rsi_avg_val = (rsi_avg - 50) / 50

        # 布林带因子
        bb_val = bb_width

        # 动量因子
        mom_val = mom_10 * 2

        # 风险信息
        risk_info = {
            'is_high_vol': vol_20 > 0.05,  # 5%以上才算高波动
            'vol_factor': vol_factor,
        }

        # === 核心组合: 波动率(30%) + 多周期RSI平均(25%) + 布林带(15%) + 动量(30%) ===
        factor_value = vol_factor * 0.30 + rsi_avg_val * 0.25 + bb_val * 0.15 + mom_val * 0.30
        factor_name = 'V30_RSIavg25_BB15_Mom30'

        # === 熊市: 降低权重 ===
        if regime == -1:
            factor_value = factor_value * 0.7
            factor_name = 'V30_RSIavg25_BB15_Mom30_bear'

        return factor_name, factor_value, risk_info

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
