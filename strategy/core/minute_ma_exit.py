"""Optional EOD 30m MA cross-down observation for a next-session exit.

No holding-period target, buy-type filter, future labels or intraday execution.
The caller decides whether a position exists and preserves execution limits.
"""
from dataclasses import dataclass
from datetime import date
import hashlib
import io
from pathlib import Path

import pandas as pd

from .minute_ma_context import MinuteMAContextStore


@dataclass(frozen=True)
class MinuteMAExitObservation:
    status: str
    exit_requested: bool
    session: str
    previous_session: str | None = None


class MinuteMACrossDown:
    def __init__(self, store, calendar):
        self.store = store
        dates = pd.DatetimeIndex(calendar)
        if (dates.tz is not None or dates.hasnans or dates.has_duplicates
                or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
            raise ValueError('Expected sorted unique local exchange-session dates')
        self.calendar = dates

    @classmethod
    def from_paths(cls, feature_path, feature_sha256, calendar_path, calendar_sha256):
        store = MinuteMAContextStore.from_parquet(feature_path, expected_sha256=feature_sha256)
        payload = Path(calendar_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != calendar_sha256:
            raise ValueError('Exchange-calendar artifact hash mismatch')
        calendar = pd.read_csv(io.BytesIO(payload), usecols=['datetime'], parse_dates=['datetime']).datetime
        return cls(store, calendar)

    def at_decision(self, code, decision_time):
        # Store validates the timezone and blocks all pre-close observations.
        current = self.store.at_decision(code, decision_time)
        if current.status != 'available':
            return MinuteMAExitObservation(current.status, False, current.session)
        session = pd.Timestamp(current.session)
        index = self.calendar.get_indexer([session])[0]
        if index < 0:
            return MinuteMAExitObservation('not_exchange_session', False, current.session)
        if index == 0:
            return MinuteMAExitObservation('no_previous_session', False, current.session)
        previous_day = self.calendar[index-1]
        previous_time = previous_day.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
        previous = self.store.at_decision(code, previous_time)
        if previous.status != 'available':
            return MinuteMAExitObservation('previous_'+previous.status, False,
                                           current.session, previous.session)
        cross = previous.values['spread'] > 0 and current.values['spread'] <= 0
        return MinuteMAExitObservation('available', cross, current.session, previous.session)

    def at_daily_close(self, code, session):
        """Explicit daily-engine adapter; never advance an intraday timestamp."""
        if type(session) is not date:
            raise TypeError('Daily-close observation requires a date, not a timestamp')
        decision = pd.Timestamp(session).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
        return self.at_decision(code, decision)
