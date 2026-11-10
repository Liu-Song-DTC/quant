"""Observed ST changes appended after the legacy frozen prehistory baseline."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import pickle
import re
import tempfile
from zoneinfo import ZoneInfo

import numpy as np


MARKET_ZONE = ZoneInfo('Asia/Shanghai')


def is_st_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('security name is missing')
    return bool(re.match(r'^S?\*?ST', name.strip().upper()))


def load_history(path):
    path = Path(path)
    if not path.exists():
        return {'schema': 1, 'observations': {}}
    with path.open('rb') as stream:
        history = pickle.load(stream)
    if not isinstance(history, dict) or history.get('schema') != 1 \
            or not isinstance(history.get('observations'), dict):
        raise ValueError('invalid ST observation ledger')
    for code, rows in history['observations'].items():
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit() or not rows:
            raise ValueError('invalid ST observation identity/rows')
        previous = ''
        for row in rows:
            day = row['date']
            if not isinstance(day, str) or len(day) != 8 \
                    or datetime.strptime(day, '%Y%m%d').strftime('%Y%m%d') != day \
                    or day <= previous or type(row['is_st']) is not bool:
                raise ValueError('invalid ST observation ordering/state')
            previous = day
    return history


def record_observations(path, names, observed_at=None):
    """Record observed states; historical dates cannot be revised or backfilled.

    Same-day refreshes can revise today's observation; decisions must retain
    their input snapshot to distinguish separate intraday refreshes.
    """
    when = datetime.now(MARKET_ZONE) if observed_at is None else datetime.fromisoformat(str(observed_at))
    when = when.replace(tzinfo=MARKET_ZONE) if when.tzinfo is None else when.astimezone(MARKET_ZONE)
    day = when.strftime('%Y%m%d')
    if not names:
        raise ValueError('empty ST source observation')
    states = {}
    for code, name in names.items():
        code = str(code)
        if len(code) != 6 or not code.isascii() or not code.isdigit():
            raise ValueError(f'invalid ST source identity: {code!r}')
        states[code] = (is_st_name(name), name.strip())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history = load_history(path)
        all_rows = history['observations']
        last_date = max((rows[-1]['date'] for rows in all_rows.values()), default='')
        if day < last_date:
            raise ValueError('refusing backdated ST observation')
        changed = 0
        for code, (state, name) in states.items():
            rows = all_rows.setdefault(code, [])
            if rows and rows[-1]['is_st'] == state:
                continue
            row = {'date': day, 'is_st': state, 'name': name,
                   'observed_at': when.isoformat(), 'source': 'exchange-qualified A-share listing snapshot'}
            if rows and rows[-1]['date'] == day:
                rows[-1] = row
            else:
                rows.append(row)
            changed += 1
        if not changed:
            return 0
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                pickle.dump(history, stream)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return changed


def merge_timeline(base, observations):
    """Observed history supersedes the legacy baseline only from first sighting."""
    if not observations:
        return base
    first = observations[0]['date']
    timeline = {}
    if base is not None:
        dates, states = base
        timeline.update((str(day), bool(state)) for day, state in zip(dates, states) if str(day) < first)
    timeline.update((row['date'], row['is_st']) for row in observations)
    dates = sorted(timeline)
    return np.asarray(dates), np.asarray([timeline[day] for day in dates], dtype=bool)
