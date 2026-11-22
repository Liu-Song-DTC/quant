"""Verified as-of raw/frozen-QFQ conversion for live state adapters."""
from functools import lru_cache
import math
from pathlib import Path

import pandas as pd


class LivePriceBasis:
    def __init__(self, raw_paths, qfq_paths):
        self.raw_paths = dict(raw_paths)
        self.qfq_paths = dict(qfq_paths)

    @lru_cache(maxsize=8192)
    def raw_per_qfq(self, code, day):
        """Use the latest COMMON traded date <= day, including suspensions.

        A missing file, nonfinite/nonpositive quote, duplicate bar or mismatch
        between the two latest dates is an error. There is no fallback to1.
        The ratio at the actual fill day is frozen by the lifecycle adapter.
        """
        end = pd.Timestamp(day).normalize()
        try:
            raw_path, qfq_path = self.raw_paths[code], self.qfq_paths[code]
        except KeyError as error:
            raise ValueError(f'Missing verified price files: {code}') from error
        raw = pd.read_csv(raw_path, encoding='utf-8-sig',usecols=['日期','收盘'])
        qfq = pd.read_csv(qfq_path,usecols=['datetime','close'])
        raw['日期'] = pd.to_datetime(raw['日期'])
        qfq['datetime'] = pd.to_datetime(qfq['datetime'])
        if raw['日期'].duplicated().any() or qfq.datetime.duplicated().any():
            raise ValueError(f'Duplicate raw/QFQ dates: {code}')
        raw = raw.loc[raw['日期'].le(end)].sort_values('日期')
        qfq = qfq.loc[qfq.datetime.le(end)].sort_values('datetime')
        if raw.empty or qfq.empty or raw['日期'].iloc[-1] != qfq.datetime.iloc[-1]:
            raise ValueError(f'Unmatched as-of raw/QFQ dates: {code} {end.date()}')
        a, b = float(raw['收盘'].iloc[-1]), float(qfq.close.iloc[-1])
        if not all(math.isfinite(v) and v>0 for v in (a,b)):
            raise ValueError(f'Invalid raw/QFQ quotes: {code}')
        return a/b
