"""Trading-session arithmetic shared by research and live decisions."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache(maxsize=4)
def _read_calendar(path, mtime):
    frame = pd.read_csv(path, usecols=['datetime'])
    return pd.DatetimeIndex(pd.to_datetime(frame['datetime']).dropna().unique()).sort_values()


def load_trading_calendar():
    from .config_loader import load_config
    cfg = load_config()
    default = Path(__file__).resolve().parents[2] / 'data/stock_data/backtrader_data'
    path = Path(cfg.get('paths.data', str(default))) / 'sh000001_qfq.csv'
    return _read_calendar(str(path), path.stat().st_mtime_ns)


def sessions_elapsed(start, end, calendar=None):
    """Count sessions in (start, end]; refuse to invent holidays outside the calendar."""
    cal = load_trading_calendar() if calendar is None else pd.DatetimeIndex(calendar).sort_values().unique()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        return -sessions_elapsed(end, start, cal)
    if not len(cal) or start < cal[0] or end > cal[-1]:
        raise ValueError(f'trading calendar does not cover {start.date()}..{end.date()}')
    return int(np.searchsorted(cal, end, side='right') - np.searchsorted(cal, start, side='right'))
