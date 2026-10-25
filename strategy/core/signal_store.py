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
        self._dates = None  # Phase 2: 惰性日期集合, has_date() 用
        self._date_positions = None
        self._buy_day = None
        self._buy_day_values = {}

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

    def finalize(self, csv_path, membership_map=None):
        """从CSV加载信号到DataFrame，释放dict内存。

        调用后 SignalStore 进入只读模式，_store dict 被释放。
        内存节省: ~800MB → ~200MB (对于 ~700K 条信号)。

        membership_map: 0f季度日历池成员映射 {boundary_ts: set(codes)} —
            非成员日期的信号行丢弃(universe选择与实盘同构: 实盘=target_date
            日历池+安全滤镜)。None=不闸(off模式)。指数码恒保留。
        """
        from .signal import Signal

        # 注意: 浮点列必须保持float64 — 2026-10-09实验证明float32读入会在
        # 主导行业中位数分数的razor-edge处翻转产业链聚焦分支(524→21候选→全灭
        # Gate floor→0订单), 出单路径语义被破坏。内存优化用其他手段。
        dtypes = {
            'code': str, 'buy': bool, 'sell': bool,
            'score': float, 'pre_discount_score': float,
            'factor_value': float, 'factor_name': str, 'industry': str,
            'factor_quality': float,
            'chan_divergence_type': str, 'chan_divergence_strength': float,
            'chan_structure_score': float, 'chan_buy_point': 'Int64',
            'chan_sell_point': 'Int64', 'bi_buy_point': 'Int64', 'signal_level': 'Int64',
            'trend_type': 'Int64',
            'chan_pivot_zg': float, 'chan_pivot_zd': float,
            'mom_60d': float, 'dist_ma60': float, 'max_dd_20d': float,
            'vol_regime': float, 'mtf_discount_factor': float,
            'mtf_alignment_score': float, 'avg_trend_strength': float,
            'risk_vol': float, 'daily_return': float,
            'volume_ratio': float, 'stroke_phase': float,
            'exhaustion_risk': float, 'gap_breakout_confirm': float,
            'vol_opening_confirm': float, 'vol_opening_strength': float,
            'bom_quality_score': float, 'gate_quality': float,
            'profit_declining': bool, 'ma_trend_up': bool,
            'ml_score': float, 'adjusted_score': float,
        }
        # 只读取CSV中存在的列（向后兼容旧格式，缺少新字段时用默认值）
        available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        effective_dtypes = {k: v for k, v in dtypes.items() if k in available_cols}
        use_cols = [c for c in available_cols if c in effective_dtypes or c == 'date']
        # 缓存的是 CSV 解析后的 float64 原值，不直接缓存生成期浮点数，避免
        # CSV解析末位舍入差异改变排名。membership 掩码在读取后每次重施。
        import json
        import os
        from pathlib import Path
        import tempfile
        source = Path(csv_path)
        cache = Path(str(source) + '.store.parquet')
        meta = Path(str(source) + '.store.json')
        stat = source.stat()
        identity = {'version': 1, 'csv_stat': [stat.st_size, stat.st_mtime_ns], 'columns': use_cols}
        frame = None
        if cache.exists() and meta.exists():
            try:
                if json.loads(meta.read_text()) == identity:
                    frame = pd.read_parquet(cache, columns=use_cols).astype(effective_dtypes)
            except (OSError, ValueError, KeyError, TypeError):
                frame = None
        if frame is None:
            frame = pd.read_csv(csv_path, dtype=effective_dtypes, usecols=use_cols)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError('信号CSV在读取过程中变化，拒绝混合快照')
            temp = None
            try:
                fd, temp = tempfile.mkstemp(dir=source.parent, prefix=cache.name + '.', suffix='.tmp')
                os.close(fd)
                frame.to_parquet(temp, index=False, compression='zstd')
                os.replace(temp, cache)
                from .live_state import atomic_json
                atomic_json(meta, identity)
            except (OSError, ValueError, TypeError) as exc:
                print(f'[SignalStore] 列式缓存写入失败，沿用CSV解析结果: {exc}')
            finally:
                if temp and os.path.exists(temp):
                    os.unlink(temp)
        final_stat = source.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError('信号CSV在缓存读取过程中变化，拒绝混合快照')
        self._df = frame
        # 用字符串索引避免 datetime.date vs pd.Timestamp 类型不匹配
        self._df.set_index(['code', 'date'], inplace=True)
        self._df.sort_index(inplace=True)
        self._cache_max = max(512, 2 * len(self._df.index.levels[0]))

        # 0f日历池membership闸 (2026-09-17): 非成员日期信号行丢弃。
        # 磁盘CSV保持未过滤全量(闸在消费时每次重施), 故闸代码变更无需信号重生成。
        if membership_map is not None and len(self._df) > 0:
            _IDX_CODES = {'sh000001', 'sh000852', '000001', '399006'}
            _boundaries = sorted(pd.Timestamp(b) for b in membership_map.keys())
            _dates = pd.to_datetime(self._df.index.get_level_values('date'))
            _b_idx = np.searchsorted(_boundaries, _dates.values, side='right') - 1
            _keep = np.zeros(len(self._df), dtype=bool)
            _codes_all = self._df.index.get_level_values('code').values
            for _bi, _b in enumerate(_boundaries):
                _rows = np.where(_b_idx == _bi)[0]
                if len(_rows) == 0:
                    continue
                _codes = _codes_all[_rows]
                _keep[_rows] = (np.isin(_codes, list(membership_map[_b]))
                                | np.isin(_codes, list(_IDX_CODES)))
            # 首边界之前的日期: 指数保留, 其余丢弃 (正常构造下应为空)
            _pre = _b_idx < 0
            _keep[_pre] = np.isin(_codes_all[_pre], list(_IDX_CODES))
            _dropped = int((~_keep).sum())
            self._df = self._df.iloc[np.where(_keep)[0]]
            if _dropped:
                print(f"SignalStore membership闸: 丢弃 {_dropped:,} 行非成员信号 "
                      f"({len(self._df):,} 行保留)")

        # 释放 dict 内存
        self._store.clear()
        del self._store
        self._store = None

        print(f"SignalStore finalized: {len(self._df):,} rows in DataFrame "
              f"({self._df.memory_usage(deep=True).sum() / 1024**2:.0f} MB)")

    def buy_signals_for_date(self, date, universe):
        """Only buy/industry are needed for sector density, not thousands of Signals.

        Preserve universe order and the same first-row semantics as get(). Date
        positions are indexed once; only one day's small buy map is retained.
        """
        if self._df is None:
            values = [(code, self.get(code, date)) for code in universe]
            return [(code, sig.industry) for code, sig in values if sig and sig.buy]
        day = (date.strftime('%Y-%m-%d') if hasattr(date, 'strftime')
               else str(date)[:10])
        if self._buy_day != day:
            if self._date_positions is None:
                # 键统一取前10位, 与 has_date 同口径: 上游日期格式一变
                # (带时分秒) 时不能静默返回空 (板块信号密度归零)。
                self._date_positions = {}
                for _k, _pos in self._df.groupby(level='date', sort=False).indices.items():
                    self._date_positions.setdefault(str(_k)[:10], []).extend(_pos)
            positions = self._date_positions.get(day)
            available = {}
            if positions is not None and 'buy' in self._df.columns:
                columns = [c for c in ('buy', 'industry') if c in self._df.columns]
                group = self._df[columns].iloc[positions]
                group = group.loc[~group.index.duplicated(keep='first')]
                group = group.loc[group['buy'].fillna(False)]
                industries = (group['industry'].fillna('').values if 'industry' in group
                              else [''] * len(group))
                available = dict(zip(group.index.get_level_values('code'), industries))
            self._buy_day, self._buy_day_values = day, available
        return [(code, self._buy_day_values[code]) for code in universe if code in self._buy_day_values]

    def max_date(self):
        """信号数据中的最新日期 ('YYYY-MM-DD'), 空→None. 出单层信号新鲜度闸用."""
        if self._df is None or len(self._df) == 0:
            return None
        return str(self._df.index.get_level_values('date').max())[:10]

    def has_date(self, date):
        """信号数据中是否存在该日期. 出单层"当日有无信号"判定用.

        2026-10-14新增: 替代 generate_trade_orders.py 原"抽任意单只股票 get()!=None"
        的试探法 — 该法在抽中股票恰好无信号(新股/停牌/不在信号universe)时误判为
        "当日无信号", 触发静默回退到旧日期出单. 本方法直接查日期索引, 与抽样无关.
        """
        if self._df is None or len(self._df) == 0:
            return False
        date_str = (date.strftime('%Y-%m-%d') if hasattr(date, 'strftime')
                    else str(date)[:10])
        if self._dates is None:  # 惰性构建, 单次~5000个日期
            self._dates = set(
                self._df.index.get_level_values('date').astype(str).str[:10])
        return date_str in self._dates

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
            bi_buy_point=_i('bi_buy_point', 0),
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
            vol_opening_confirm=_f('vol_opening_confirm'),
            vol_opening_strength=_f('vol_opening_strength'),
            bom_quality_score=_f('bom_quality_score', 0.3),
            _gate_quality=_f('gate_quality', 0.5),
            profit_declining=_b('profit_declining'),
            ma_trend_up=_b('ma_trend_up'),
            ml_score=_f('ml_score'),
            adjusted_score=_f('adjusted_score'),
        )
