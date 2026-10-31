"""Strict unadjusted-price loader for an isolated execution research arm.

Missing bars remain missing: an index trading session does not imply a stock
traded, and a forward-filled open would invent an executable price.
"""
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {'开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close'}


def load_raw_ohlc(path, calendar):
    """Align one ``raw_data/{code}/none.csv`` with an actual trading calendar.

    The caller must explicitly decide how to value suspended positions; this
    loader never invents an open/low/close price or substitutes a QFQ bar.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'missing unadjusted stock prices: {path}')
    calendar = pd.DatetimeIndex(calendar).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('invalid execution calendar')
    source = pd.read_csv(path, encoding='utf-8-sig',
                         usecols=['股票代码', '日期', *RAW_COLUMNS],
                         dtype={'股票代码': str, '日期': str})
    if source.empty:
        raise ValueError(f'empty raw stock prices: {path}')
    expected_code = path.parent.name
    if (len(expected_code) == 6 and expected_code.isdigit() and
            source['股票代码'].str.zfill(6).ne(expected_code).any()):
        raise ValueError(f'raw stock identity mismatch: {path}')
    source['日期'] = pd.to_datetime(source['日期'], errors='raise').dt.normalize()
    if source['日期'].duplicated().any():
        raise ValueError(f'duplicate raw stock date: {path}')
    frame = source.set_index('日期')[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).reindex(calendar)
    for column in RAW_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    # An existing row with all empty prices is corrupt, not a suspension gap.
    present = frame.index.isin(source['日期'])
    bad = present & (frame[['open', 'high', 'low', 'close']].isna().any(axis=1) |
                     frame[['open', 'high', 'low', 'close']].le(0).any(axis=1) |
                     frame['low'].gt(frame['high']) |
                     frame['open'].gt(frame['high']) | frame['open'].lt(frame['low']) |
                     frame['close'].gt(frame['high']) | frame['close'].lt(frame['low']))
    if bad.any():
        raise ValueError(f'invalid raw OHLC on {frame.index[bad][0].date()}: {path}')
    return frame.astype(np.float64)
