# core/signal_store.py
import pandas as pd
import numpy as np
from collections import OrderedDict


class SignalStore:
    """信号存储 — 内存高效的两阶段设计。

    Phase 1 (生成期): 使用 dict 存储 Signal 对象（快速写入）。
    Phase 2 (查询期): 调用 finalize(csv_path) 加载CSV到DataFrame，
                      释放dict内存，get()从DataFrame构建Signal。
                      内存: ~200MB DataFrame vs ~1GB Python对象。
    """

    def __init__(self):
        self._store = {}  # Phase 1: (code, date) -> Signal
        self._df = None   # Phase 2: DataFrame with MultiIndex (code, date)
        self._cache = OrderedDict()  # Phase 2: LRU cache for constructed Signals
        self._cache_max = 512

    def set(self, code, date, signal):
        """写入信号（仅 Phase 1 可用）"""
        if self._df is not None:
            raise RuntimeError("Cannot set() after finalize() — SignalStore is in read-only mode")
        self._store[(code, date)] = signal

    def get(self, code, date):
        """获取信号（Phase 1 或 Phase 2）"""
        if self._df is not None:
            return self._get_from_df(code, date)
        return self._store.get((code, date))

    def finalize(self, csv_path):
        """从CSV加载信号到DataFrame，释放dict内存。

        调用后 SignalStore 进入只读模式，_store dict 被释放。
        内存节省: ~800MB → ~200MB (对于 ~700K 条信号)。
        """
        from .signal import Signal

        dtypes = {
            'code': str, 'buy': bool, 'sell': bool,
            'score': float, 'pre_discount_score': float,
            'factor_value': float, 'factor_name': str, 'industry': str,
            'factor_quality': float,
            'chan_divergence_type': str, 'chan_divergence_strength': float,
            'chan_structure_score': float, 'chan_buy_point': 'Int64',
            'chan_sell_point': 'Int64', 'signal_level': 'Int64',
            'trend_type': 'Int64',
            'chan_pivot_zg': float, 'chan_pivot_zd': float,
            'mom_60d': float, 'dist_ma60': float, 'max_dd_20d': float,
            'vol_regime': float, 'mtf_discount_factor': float,
            'mtf_alignment_score': float, 'avg_trend_strength': float,
            'risk_vol': float, 'daily_return': float,
            'volume_ratio': float, 'stroke_phase': float,
            'exhaustion_risk': float, 'gap_breakout_confirm': float,
            'profit_declining': bool, 'ma_trend_up': bool,
        }
        use_cols = list(dtypes.keys()) + ['date']
        self._df = pd.read_csv(
            csv_path, dtype=dtypes, usecols=use_cols,
        )
        # 用字符串索引避免 datetime.date vs pd.Timestamp 类型不匹配
        self._df.set_index(['code', 'date'], inplace=True)
        self._df.sort_index(inplace=True)

        # 释放 dict 内存
        self._store.clear()
        del self._store
        self._store = None

        print(f"SignalStore finalized: {len(self._df):,} rows in DataFrame "
              f"({self._df.memory_usage(deep=True).sum() / 1024**2:.0f} MB)")

    def _get_from_df(self, code, date):
        """从 DataFrame 查找并构造 Signal 对象（带 LRU 缓存）。"""
        # 统一日期格式：DataFrame index 使用字符串 (YYYY-MM-DD)
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        cache_key = (code, date_str)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        try:
            row = self._df.loc[(code, date_str)]
        except KeyError:
            return None

        # 处理重复索引 (同一 code+date 有多行)
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        sig = self._build_signal(row)
        self._cache[cache_key] = sig
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return sig

    @staticmethod
    def _build_signal(row):
        """从 DataFrame 行构建 Signal 对象。"""
        from .signal import Signal

        def _f(key, default=0.0):
            v = row.get(key)
            if pd.isna(v) if not isinstance(v, (bool, str)) else v is None:
                return default
            if isinstance(default, float) and not isinstance(v, (int, float, bool)):
                return default
            return v

        def _i(key, default=0):
            v = row.get(key)
            if pd.isna(v) if not isinstance(v, (bool, str)) else v is None:
                return default
            try:
                return int(v)
            except (ValueError, TypeError):
                return default

        def _b(key, default=False):
            v = row.get(key)
            if pd.isna(v) if not isinstance(v, (bool, str)) else v is None:
                return default
            return bool(v) if v is not None else default

        return Signal(
            buy=_b('buy'),
            sell=_b('sell'),
            score=_f('score'),
            factor_value=_f('factor_value'),
            factor_name=str(_f('factor_name', '')) if row.get('factor_name') and not (isinstance(row.get('factor_name'), float) and pd.isna(row.get('factor_name'))) else '',
            industry=str(_f('industry', '')) if row.get('industry') and not (isinstance(row.get('industry'), float) and pd.isna(row.get('industry'))) else '',
            factor_quality=_f('factor_quality'),
            chan_divergence_type=str(_f('chan_divergence_type', '')) if row.get('chan_divergence_type') and not (isinstance(row.get('chan_divergence_type'), float) and pd.isna(row.get('chan_divergence_type'))) else '',
            chan_divergence_strength=_f('chan_divergence_strength'),
            chan_structure_score=_f('chan_structure_score'),
            chan_buy_point=_i('chan_buy_point', 0),
            chan_sell_point=_i('chan_sell_point', 0),
            signal_level=_i('signal_level', 0),
            trend_type=_i('trend_type', 0),
            chan_pivot_zg=_f('chan_pivot_zg', float('nan')),
            chan_pivot_zd=_f('chan_pivot_zd', float('nan')),
            mom_60d=_f('mom_60d'),
            dist_ma60=_f('dist_ma60'),
            max_dd_20d=_f('max_dd_20d'),
            vol_regime=_f('vol_regime', 1.0),
            pre_discount_score=_f('pre_discount_score'),
            mtf_discount_factor=_f('mtf_discount_factor', 1.0),
            mtf_alignment_score=_f('mtf_alignment_score'),
            weekly_trend_strength=_f('avg_trend_strength'),
            monthly_trend_strength=_f('avg_trend_strength'),
            risk_vol=_f('risk_vol', 0.0),
            daily_return=_f('daily_return'),
            volume_ratio=_f('volume_ratio', 1.0),
            stroke_phase=_f('stroke_phase'),
            exhaustion_risk=_f('exhaustion_risk'),
            gap_breakout_confirm=_f('gap_breakout_confirm'),
            profit_declining=_b('profit_declining'),
            ma_trend_up=_b('ma_trend_up'),
        )
