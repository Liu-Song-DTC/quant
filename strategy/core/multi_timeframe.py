"""
统一多时间框架分析器 (Unified Multi-Timeframe Analyzer)

替代原有的固定步长降采样方案，提供:
1. 日历感知的周线/月线重采样 (向量化实现)
2. 完整OHLCV数据，不只是收盘价
3. K线形态识别 (吞没、锤子线、十字星等)
4. 基于高级别结构的支撑/阻力位
5. 统一的折扣因子计算 (合并原来分散的三层机制)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MTFResult:
    """多时间框架分析结果"""
    weekly_trend_up: np.ndarray
    monthly_trend_up: np.ndarray
    weekly_trend_strength: np.ndarray
    monthly_trend_strength: np.ndarray
    alignment_score: np.ndarray
    discount_factor: np.ndarray
    weekly_pattern_signal: np.ndarray
    nearest_resistance_pct: np.ndarray
    nearest_support_pct: np.ndarray
    weekly_ema20: np.ndarray
    weekly_ema60: np.ndarray
    monthly_ema10: np.ndarray
    monthly_ema30: np.ndarray


class MultiTimeframeAnalyzer:
    """统一多时间框架分析器

    在日线数据基础上，通过日历感知重采样构建周线和月线数据，
    计算趋势、形态、支撑阻力，并输出统一的折扣因子。
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        mtf = cfg.get('multi_timeframe', {})

        self.enabled = mtf.get('enabled', True)
        self.weekly_ema_fast = mtf.get('weekly_ema_fast', 20)
        self.weekly_ema_slow = mtf.get('weekly_ema_slow', 60)
        self.monthly_ema_fast = mtf.get('monthly_ema_fast', 5)
        self.monthly_ema_slow = mtf.get('monthly_ema_slow', 15)

        discount = mtf.get('discount', {})
        self.discount_counter_trend = discount.get('counter_trend', 0.50)
        self.discount_partial = discount.get('partial', 0.72)
        self.discount_full = discount.get('full', 1.0)
        self.discount_weak_trend = discount.get('weak_trend', 0.95)

        pattern_cfg = mtf.get('pattern', {})
        self.pattern_weight = pattern_cfg.get('weight', 0.15)

        self.min_weekly_bars = mtf.get('min_weekly_bars', 12)
        self.min_monthly_bars = mtf.get('min_monthly_bars', 8)
        self.min_daily_bars = mtf.get('min_daily_bars', 60)

        structure = mtf.get('structure', {})
        self.pivot_lookback = structure.get('pivot_lookback', 5)
        self.support_resistance_enabled = structure.get('enabled', True)

    # ===================================================================
    # 公共接口
    # ===================================================================

    def analyze(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
        dates: np.ndarray,
    ) -> MTFResult:
        """主分析入口"""
        n = len(close)

        if not self.enabled:
            return MTFResult(
                weekly_trend_up=np.zeros(n, dtype=bool),
                monthly_trend_up=np.zeros(n, dtype=bool),
                weekly_trend_strength=np.zeros(n),
                monthly_trend_strength=np.zeros(n),
                alignment_score=np.zeros(n),
                discount_factor=np.ones(n),
                weekly_pattern_signal=np.zeros(n),
                nearest_resistance_pct=np.zeros(n),
                nearest_support_pct=np.zeros(n),
                weekly_ema20=np.full(n, np.nan),
                weekly_ema60=np.full(n, np.nan),
                monthly_ema10=np.full(n, np.nan),
                monthly_ema30=np.full(n, np.nan),
            )

        dates = pd.DatetimeIndex(dates)

        # Step 1: 整数索引重采样（避免 strftime 开销）
        weekly = self._resample_weekly_fast(dates, close, high, low, open_, volume)
        monthly = self._resample_monthly_fast(dates, close, high, low, open_, volume)

        # Step 2: 计算各时间框架的趋势
        wt_result = self._compute_timeframe_trend(
            weekly['close'].values, weekly['high'].values,
            weekly['low'].values, weekly['volume'].values,
            self.weekly_ema_fast, self.weekly_ema_slow,
            min_bars=self.min_weekly_bars,
        )
        mt_result = self._compute_timeframe_trend(
            monthly['close'].values, monthly['high'].values,
            monthly['low'].values, monthly['volume'].values,
            self.monthly_ema_fast, self.monthly_ema_slow,
            min_bars=self.min_monthly_bars,
        )

        # Step 3: 映射回日线
        weekly_daily_map = self._build_daily_map_fast(dates, weekly['date'].values, n)
        monthly_daily_map = self._build_daily_map_fast(dates, monthly['date'].values, n)

        weekly_trend_up = self._map_to_daily(wt_result['trend_up'], weekly_daily_map, n)
        weekly_trend_strength = self._map_to_daily(wt_result['trend_strength'], weekly_daily_map, n)
        monthly_trend_up = self._map_to_daily(mt_result['trend_up'], monthly_daily_map, n)
        monthly_trend_strength = self._map_to_daily(mt_result['trend_strength'], monthly_daily_map, n)

        weekly_ema20 = self._map_to_daily(wt_result['ema_fast'], weekly_daily_map, n)
        weekly_ema60 = self._map_to_daily(wt_result['ema_slow'], weekly_daily_map, n)
        monthly_ema10 = self._map_to_daily(mt_result['ema_fast'], monthly_daily_map, n)
        monthly_ema30 = self._map_to_daily(mt_result['ema_slow'], monthly_daily_map, n)

        # Step 4: K线形态识别 (周线级别)
        weekly_pattern_signal = self._compute_weekly_patterns(weekly, weekly_daily_map, n)

        # Step 5: 支撑/阻力位 (可选，开销较大)
        if self.support_resistance_enabled:
            resistance_pct, support_pct = self._compute_support_resistance_fast(
                weekly, monthly, weekly_daily_map, monthly_daily_map, close, n
            )
        else:
            resistance_pct = np.zeros(n)
            support_pct = np.zeros(n)

        # Step 6: 向量化对齐分数和折扣因子
        alignment_score = self._compute_alignment_score_vec(
            weekly_trend_up, monthly_trend_up,
            weekly_trend_strength, monthly_trend_strength,
        )
        discount_factor = self._compute_unified_discount_vec(
            alignment_score, weekly_trend_strength, monthly_trend_strength,
            weekly_pattern_signal, resistance_pct, support_pct,
        )

        return MTFResult(
            weekly_trend_up=weekly_trend_up,
            monthly_trend_up=monthly_trend_up,
            weekly_trend_strength=weekly_trend_strength,
            monthly_trend_strength=monthly_trend_strength,
            alignment_score=alignment_score,
            discount_factor=discount_factor,
            weekly_pattern_signal=weekly_pattern_signal,
            nearest_resistance_pct=resistance_pct,
            nearest_support_pct=support_pct,
            weekly_ema20=weekly_ema20,
            weekly_ema60=weekly_ema60,
            monthly_ema10=monthly_ema10,
            monthly_ema30=monthly_ema30,
        )

    # ===================================================================
    # 快速重采样 (整数索引, 避免 strftime)
    # ===================================================================

    @staticmethod
    def _resample_weekly_fast(
        dates: pd.DatetimeIndex,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """基于整数周索引的快速周线重采样

        使用 (year-2000)*52 + week_of_year 作为分组键，避免 strftime 字符串开销。
        """
        n = len(dates)
        iso = dates.isocalendar()
        year = iso['year'].values.astype(np.int64)
        week = iso['week'].values.astype(np.int64)
        # 整数编码: year*100 + week, 如 2024*100+1 = 202401
        week_key = year * 100 + week

        # 找到分组边界 (week_key 变化的位置)
        boundaries = np.concatenate([[0], np.where(week_key[:-1] != week_key[1:])[0] + 1, [n]])

        n_bars = len(boundaries) - 1
        w_open = np.empty(n_bars)
        w_high = np.empty(n_bars)
        w_low = np.empty(n_bars)
        w_close = np.empty(n_bars)
        w_volume = np.empty(n_bars)
        w_date = dates.values[boundaries[1:] - 1]  # 每周最后一天

        for i in range(n_bars):
            start, end = boundaries[i], boundaries[i + 1]
            w_open[i] = open_[start]
            w_high[i] = np.max(high[start:end])
            w_low[i] = np.min(low[start:end])
            w_close[i] = close[end - 1]
            w_volume[i] = np.sum(volume[start:end])

        return pd.DataFrame({
            'open': w_open, 'high': w_high, 'low': w_low,
            'close': w_close, 'volume': w_volume, 'date': w_date,
        })

    @staticmethod
    def _resample_monthly_fast(
        dates: pd.DatetimeIndex,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """基于整数年月索引的快速月线重采样"""
        n = len(dates)
        month_key = dates.year.values.astype(np.int64) * 100 + dates.month.values.astype(np.int64)

        boundaries = np.concatenate([[0], np.where(month_key[:-1] != month_key[1:])[0] + 1, [n]])

        n_bars = len(boundaries) - 1
        m_open = np.empty(n_bars)
        m_high = np.empty(n_bars)
        m_low = np.empty(n_bars)
        m_close = np.empty(n_bars)
        m_volume = np.empty(n_bars)
        m_date = dates.values[boundaries[1:] - 1]

        for i in range(n_bars):
            start, end = boundaries[i], boundaries[i + 1]
            m_open[i] = open_[start]
            m_high[i] = np.max(high[start:end])
            m_low[i] = np.min(low[start:end])
            m_close[i] = close[end - 1]
            m_volume[i] = np.sum(volume[start:end])

        return pd.DataFrame({
            'open': m_open, 'high': m_high, 'low': m_low,
            'close': m_close, 'volume': m_volume, 'date': m_date,
        })

    # ===================================================================
    # 趋势计算
    # ===================================================================

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        n = len(arr)
        result = np.full(n, np.nan)
        if n == 0:
            return result
        alpha = 2.0 / (period + 1)
        result[0] = arr[0]
        for i in range(1, n):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _compute_timeframe_trend(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        fast_period: int,
        slow_period: int,
        min_bars: int = 10,
    ) -> dict:
        n = len(close)
        if n < min_bars:
            return {
                'trend_up': np.zeros(n, dtype=bool),
                'trend_strength': np.zeros(n),
                'ema_fast': np.full(n, np.nan),
                'ema_slow': np.full(n, np.nan),
            }

        ema_fast = MultiTimeframeAnalyzer._ema(close, fast_period)
        ema_slow = MultiTimeframeAnalyzer._ema(close, slow_period)

        trend_up = np.zeros(n, dtype=bool)
        trend_strength = np.zeros(n)

        start = max(fast_period, slow_period)
        valid = ~(np.isnan(ema_fast[start:]) | np.isnan(ema_slow[start:]))

        trend_up[start:] = np.where(valid, ema_fast[start:] > ema_slow[start:], False)
        spread = np.abs(ema_fast[start:] - ema_slow[start:]) / (ema_slow[start:] + 1e-10)
        trend_strength[start:] = np.where(valid, np.tanh(spread * 50), 0.0)

        return {
            'trend_up': trend_up,
            'trend_strength': trend_strength,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
        }

    # ===================================================================
    # 日线映射
    # ===================================================================

    @staticmethod
    def _build_daily_map_fast(
        daily_dates: pd.DatetimeIndex,
        bar_end_dates: np.ndarray,
        n_daily: int,
    ) -> np.ndarray:
        """使用 searchsorted 建立日线→高级别bar的映射（向量化）"""
        bar_dt = pd.DatetimeIndex(bar_end_dates).asi8
        daily_dt = daily_dates.asi8

        # searchsorted: 找到每日对应的最后一个bar
        mapping = np.searchsorted(bar_dt, daily_dt, side='right') - 1
        mapping = np.clip(mapping, 0, len(bar_dt) - 1)

        # 日线在第一个bar之前的 → 映射到-1
        before_first = daily_dt < bar_dt[0]
        mapping[before_first] = -1

        return mapping

    @staticmethod
    def _map_to_daily(
        bar_array: np.ndarray,
        daily_map: np.ndarray,
        n_daily: int,
        fill_value=0.0,
    ) -> np.ndarray:
        """将高级别bar的值映射回日线频率（向量化）"""
        valid = daily_map >= 0
        n_bars = len(bar_array)

        if bar_array.dtype == np.bool_:
            result = np.zeros(n_daily, dtype=bool)
        elif np.issubdtype(bar_array.dtype, np.integer):
            result = np.full(n_daily, fill_value if not isinstance(fill_value, bool) else int(fill_value))
        else:
            result = np.full(n_daily, fill_value, dtype=float)

        safe_map = np.clip(daily_map[valid], 0, n_bars - 1)
        result[valid] = bar_array[safe_map]
        return result

    # ===================================================================
    # K线形态识别 (周线级别)
    # ===================================================================

    @staticmethod
    def _compute_weekly_patterns(
        weekly: pd.DataFrame,
        daily_map: np.ndarray,
        n_daily: int,
    ) -> np.ndarray:
        n_bars = len(weekly)
        bar_signal = np.zeros(n_bars)

        if n_bars < 3:
            return np.zeros(n_daily)

        open_ = weekly['open'].values
        high = weekly['high'].values
        low = weekly['low'].values
        close = weekly['close'].values

        body = close - open_
        upper_shadow = high - np.maximum(open_, close)
        lower_shadow = np.minimum(open_, close) - low
        total_range = high - low

        valid_range = total_range > 0
        body_ratio = np.zeros(n_bars)
        body_ratio[valid_range] = np.abs(body[valid_range]) / total_range[valid_range]

        for i in range(1, n_bars):
            if total_range[i] <= 0:
                continue

            prev_body = body[i - 1]
            # 吞没形态
            if prev_body < 0 and body[i] > 0 and close[i] > open_[i - 1] and open_[i] < close[i - 1]:
                bar_signal[i] = 0.8
            elif prev_body > 0 and body[i] < 0 and close[i] < open_[i - 1] and open_[i] > close[i - 1]:
                bar_signal[i] = -0.8
            # 锤子线/上吊线
            elif lower_shadow[i] > total_range[i] * 0.6 and body_ratio[i] < 0.3:
                if i >= 2 and close[i - 1] < close[i - 2]:
                    bar_signal[i] = max(bar_signal[i], 0.6)
                elif i >= 2 and close[i - 1] > close[i - 2]:
                    bar_signal[i] = min(bar_signal[i], -0.5)
            # 十字星
            elif body_ratio[i] < 0.1:
                bar_signal[i] *= 0.5

        return MultiTimeframeAnalyzer._map_to_daily(bar_signal, daily_map, n_daily, fill_value=0.0)

    # ===================================================================
    # 支撑/阻力位 (向量化优化)
    # ===================================================================

    @staticmethod
    def _find_pivot_levels(high: np.ndarray, low: np.ndarray, lookback: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """使用 sliding_window_view 向量化寻找枢轴点"""
        n = len(high)
        if n < lookback * 2 + 1:
            return np.array([]), np.array([])

        from numpy.lib.stride_tricks import sliding_window_view

        # 对局部高点: 看长度为 2*lookback+1 的窗口
        win_size = lookback * 2 + 1
        h_win = sliding_window_view(high, win_size)
        l_win = sliding_window_view(low, win_size)

        # 中心元素是最大值 → 阻力位
        center_h = h_win[:, lookback]
        is_max = np.all(h_win <= center_h[:, None], axis=1)
        # 确保没有并列最大值
        max_count = np.sum(h_win == center_h[:, None], axis=1)
        is_unique_max = is_max & (max_count == 1)
        resistance_levels = high[lookback:n - lookback][is_unique_max]

        # 中心元素是最小值 → 支撑位
        center_l = l_win[:, lookback]
        is_min = np.all(l_win >= center_l[:, None], axis=1)
        min_count = np.sum(l_win == center_l[:, None], axis=1)
        is_unique_min = is_min & (min_count == 1)
        support_levels = low[lookback:n - lookback][is_unique_min]

        return resistance_levels, support_levels

    @staticmethod
    def _merge_nearby_levels(levels: np.ndarray, threshold_pct: float = 0.03) -> np.ndarray:
        if len(levels) <= 1:
            return levels
        sorted_levels = np.sort(levels)
        merged = [sorted_levels[0]]
        for lvl in sorted_levels[1:]:
            if abs(lvl - merged[-1]) / (merged[-1] + 1e-10) > threshold_pct:
                merged.append(lvl)
            else:
                merged[-1] = (merged[-1] + lvl) / 2
        return np.array(merged)

    @staticmethod
    def _compute_support_resistance_fast(
        weekly: pd.DataFrame,
        monthly: pd.DataFrame,
        weekly_daily_map: np.ndarray,
        monthly_daily_map: np.ndarray,
        daily_close: np.ndarray,
        n_daily: int,
        lookback: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """向量化计算最近支撑/阻力位距离

        使用 searchsorted 替代逐日 boolean indexing 循环。
        """
        resistance_pct = np.zeros(n_daily)
        support_pct = np.zeros(n_daily)

        if len(weekly) < lookback * 2 + 1:
            return resistance_pct, support_pct

        w_resist, w_support = MultiTimeframeAnalyzer._find_pivot_levels(
            weekly['high'].values, weekly['low'].values, lookback
        )
        m_resist, m_support = MultiTimeframeAnalyzer._find_pivot_levels(
            monthly['high'].values, monthly['low'].values, lookback
        )

        all_resist = np.concatenate([w_resist, m_resist]) if len(w_resist) + len(m_resist) > 0 else np.array([])
        all_support = np.concatenate([w_support, m_support]) if len(w_support) + len(m_support) > 0 else np.array([])

        all_resist = MultiTimeframeAnalyzer._merge_nearby_levels(all_resist)
        all_support = MultiTimeframeAnalyzer._merge_nearby_levels(all_support)

        if len(all_resist) == 0 and len(all_support) == 0:
            return resistance_pct, support_pct

        # 使用 searchsorted 向量化查找
        valid = daily_close > 0
        prices = daily_close[valid]

        if len(all_resist) > 0:
            sorted_resist = np.sort(all_resist)
            # searchsorted: 找到第一个 > price 的阻力位索引
            idx_r = np.searchsorted(sorted_resist, prices, side='right')
            has_resist = idx_r < len(sorted_resist)
            nearest_r = np.full(len(prices), np.nan)
            nearest_r[has_resist] = sorted_resist[idx_r[has_resist]]
            resistance_pct[valid] = np.where(has_resist, (nearest_r - prices) / prices, 0.0)

        if len(all_support) > 0:
            sorted_support = np.sort(all_support)
            # searchsorted: 找到第一个 >= price 的支撑位索引, 支撑位在它左边
            idx_s = np.searchsorted(sorted_support, prices, side='left') - 1
            has_support = idx_s >= 0
            nearest_s = np.full(len(prices), np.nan)
            nearest_s[has_support] = sorted_support[idx_s[has_support]]
            support_pct[valid] = np.where(has_support, (prices - nearest_s) / prices, 0.0)

        return resistance_pct, support_pct

    # ===================================================================
    # 对齐分数 & 统一折扣因子 (向量化)
    # ===================================================================

    @staticmethod
    def _compute_alignment_score_vec(
        weekly_trend_up: np.ndarray,
        monthly_trend_up: np.ndarray,
        weekly_strength: np.ndarray,
        monthly_strength: np.ndarray,
    ) -> np.ndarray:
        """向量化计算多时间框架对齐分数 [-1, 1]"""
        w_up = weekly_trend_up
        m_up = monthly_trend_up
        w_str = weekly_strength
        m_str = monthly_strength

        both_up = w_up & m_up
        both_down = ~w_up & ~m_up
        mixed = ~(both_up | both_down)

        alignment = np.zeros(len(w_up))
        alignment[both_up] = (w_str[both_up] + m_str[both_up]) / 2
        alignment[both_down] = -(w_str[both_down] + m_str[both_down]) / 2

        # 方向不一致: 月线权重更高
        mixed_m_up = mixed & m_up
        mixed_m_down = mixed & ~m_up
        alignment[mixed_m_up] = m_str[mixed_m_up] * 0.5 - w_str[mixed_m_up] * 0.3
        alignment[mixed_m_down] = -m_str[mixed_m_down] * 0.5 + w_str[mixed_m_down] * 0.3

        return np.clip(alignment, -1.0, 1.0)

    def _compute_unified_discount_vec(
        self,
        alignment_score: np.ndarray,
        weekly_strength: np.ndarray,
        monthly_strength: np.ndarray,
        pattern_signal: np.ndarray,
        resistance_pct: np.ndarray,
        support_pct: np.ndarray,
    ) -> np.ndarray:
        """向量化计算统一折扣因子"""
        n = len(alignment_score)
        align = alignment_score
        w_str = weekly_strength
        m_str = monthly_strength

        # 基础折扣: 基于对齐分数分段线性
        base = np.ones(n)
        high_align = align >= 0.3
        low_align = align <= -0.3
        mid_align = ~(high_align | low_align)

        base[high_align] = self.discount_partial + (self.discount_full - self.discount_partial) * (align[high_align] - 0.3) / 0.7
        base[low_align] = self.discount_counter_trend + (self.discount_partial - self.discount_counter_trend) * (align[low_align] + 1.0) / 0.7
        # mid段从partial开始(对齐分数-0.3处与low段连续), 到partial+0.06(对齐分数+0.3处与high段连续)
        base[mid_align] = self.discount_partial + 0.06 * (align[mid_align] + 0.3) / 0.6
        base = np.clip(base, self.discount_counter_trend, self.discount_full)

        # 趋势强度调整: 高级别趋势弱 → 额外折扣
        avg_strength = (w_str + m_str) / 2
        weak_trend = avg_strength < 0.3
        base[weak_trend] *= self.discount_weak_trend

        # 形态确认
        bullish_pattern = pattern_signal > 0.3
        bearish_pattern = pattern_signal < -0.3
        base[bullish_pattern] *= (1.0 + self.pattern_weight * np.minimum(pattern_signal[bullish_pattern], 1.0))
        base[bearish_pattern] *= (1.0 - self.pattern_weight * np.minimum(np.abs(pattern_signal[bearish_pattern]), 1.0))

        # 支撑/阻力调整
        near_support = (support_pct > 0) & (support_pct < 0.05)
        near_resistance = (resistance_pct > 0) & (resistance_pct < 0.03)
        base[near_support] *= 1.05
        base[near_resistance] *= 0.92

        return np.clip(base, 0.30, 1.10)
