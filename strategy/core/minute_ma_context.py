"""Shared, feature-only 30m context for an explicitly timed EOD decision.

This reader is not wired into production scoring. It returns no buy/sell advice
or score and accepts no buy-type filter. Calendar-date lookup avoids comparing
integer timestamps with incompatible units. Availability is never backfilled
from a later feature row.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


FEATURE_NAMES = ('spread', 'slope', 'distance_atr', 'width_atr', 'crossings_24',
                 'above_last4', 'recovery', 'contraction', 'bull_aligned')
ALLOWED_COLUMNS = frozenset((*FEATURE_NAMES, 'date', 'code', 'valid',
                            'price_match_today', 'basis_bridge_today'))


@dataclass(frozen=True)
class MinuteMAContext:
    status: str
    session: str
    values: Mapping[str, float]


class MinuteMAContextStore:
    def __init__(self, frame):
        unexpected = set(frame.columns)-ALLOWED_COLUMNS
        required = {'date', 'code', 'valid', *FEATURE_NAMES}
        if unexpected or not required.issubset(frame.columns):
            raise ValueError('Expected feature-only minute schema, without scores or return labels')
        data = frame.copy(deep=True)
        dates = pd.to_datetime(data['date'])
        if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
            raise ValueError('Feature dates must be local session-date labels')
        codes = data.code.astype(str).str.zfill(6)
        if not codes.str.fullmatch(r'\d{6}').all():
            raise ValueError('Invalid stock code')
        if not data.valid.isin([True, False]).all():
            raise ValueError('Availability must be explicit for every feature row')
        data['date'], data['code'] = dates.dt.date, codes
        if data.duplicated(['code', 'date']).any():
            raise ValueError('Duplicate minute feature keys')
        numeric = data.loc[:, list(FEATURE_NAMES)].apply(pd.to_numeric, errors='raise')
        valid = data.valid.astype(bool)
        if not np.isfinite(numeric.loc[valid].to_numpy()).all():
            raise ValueError('Valid rows must contain finite features')
        data.loc[:, list(FEATURE_NAMES)] = numeric
        self._data = data.set_index(['code', 'date']).sort_index()

    @classmethod
    def from_parquet(cls, path, *, expected_sha256):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError('Minute feature artifact hash mismatch')
        import io
        return cls(pd.read_parquet(io.BytesIO(payload)))

    def at_decision(self, code, decision_time):
        stamp = pd.Timestamp(decision_time)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError('Use an explicit timezone-aware decision timestamp')
        stamp = stamp.tz_convert('Asia/Shanghai')
        session = stamp.date()
        empty = MappingProxyType({})
        if stamp.hour*60+stamp.minute < 15*60:
            return MinuteMAContext('not_closed', session.isoformat(), empty)
        key = (str(code).zfill(6), session)
        if key not in self._data.index:
            return MinuteMAContext('missing', session.isoformat(), empty)
        row = self._data.loc[key]
        if not bool(row.valid):
            return MinuteMAContext('invalid_window', session.isoformat(), empty)
        return MinuteMAContext('available', session.isoformat(),
            MappingProxyType({name: float(row[name]) for name in FEATURE_NAMES}))
