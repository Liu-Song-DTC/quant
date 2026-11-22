"""Canonical accepted465 peer evidence, independent of signals and holdings.

The three peer kernels retain the archived NumPy/pandas operation order. In
particular, rolling calculations need their full archived calendar prefix for
bit-for-bit parity; a mathematically equivalent short window is insufficient.
"""
from pathlib import Path
import hashlib
import io

import numpy as np
import pandas as pd
import pyarrow as pa

from .fundamental import FundamentalData
from .stock_pool import _prefix_nonstock_flags


SCHEMA = pa.schema([
    ('code', pa.string()), ('date', pa.timestamp('ns')), ('sector', pa.string()),
    ('peer_count', pa.float64()), ('peer_return5', pa.float64()),
    ('peer_relative5', pa.float64()), ('peer_breadth5', pa.float64()),
    ('peer_known5', pa.bool_()), ('peer_ebb', pa.bool_()),
])
PEER_FIELDS = ['peer_known5', 'peer_ebb', 'peer_return5', 'peer_relative5', 'peer_breadth5']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def peer_without_self(total_return,advancing,count,own_return):
    count=np.asarray(count,dtype=float);own=np.asarray(own_return,dtype=float)
    eligible=(count>=6)&np.isfinite(count)&np.isfinite(own)&np.isfinite(total_return)&np.isfinite(advancing)
    mean=np.full(len(count),np.nan);breadth=mean.copy()
    mean[eligible]=(np.asarray(total_return)[eligible]-own[eligible])/(count[eligible]-1)
    breadth[eligible]=(np.asarray(advancing)[eligible]-(own[eligible]>0))/(count[eligible]-1)
    return mean,breadth


def peer_deterioration(peer,calendar,benchmark):
    panel=peer.reindex(calendar).copy()
    same=panel.sector.eq(panel.sector.shift())
    stable5=same.rolling(4,min_periods=4).sum().eq(4)&panel.sector.notna()
    r=panel.peer_ret.where(panel.peer_count.ge(5))
    panel['peer_return5']=np.expm1(np.log1p(r).rolling(5,min_periods=5).sum())
    panel['peer_relative5']=np.expm1((np.log1p(r)-np.log1p(benchmark)).rolling(5,min_periods=5).sum())
    panel['peer_breadth5']=panel.peer_breadth.where(panel.peer_count.ge(5)).rolling(5,min_periods=5).mean()
    known=stable5&panel[['peer_return5','peer_relative5','peer_breadth5']].notna().all(axis=1)
    panel['peer_ebb']=(panel.peer_return5.lt(0)&panel.peer_relative5.lt(0)&panel.peer_breadth5.lt(.5)).where(known).astype('boolean')
    return panel


def canonical_batch(daily,calendar,benchmark):
    daily=daily.copy()
    daily['peer_ret'],daily['peer_breadth']=peer_without_self(daily.total_ret.to_numpy(),daily.advancing.to_numpy(),daily.observed.to_numpy(),daily.ret_1d.to_numpy())
    daily['peer_count']=daily.observed-1;panels=[]
    for code,g in daily.groupby('code',sort=False):
        panel=peer_deterioration(g.set_index('date'),calendar,benchmark)
        # Preserve missing keys before first observation and on suspended days.
        panel=panel.loc[g.date].copy();panel['date']=panel.index;panel['code']=code
        panel['peer_known5']=panel.peer_ebb.notna()
        panels.append(panel[[field.name for field in SCHEMA]])
    return pd.concat(panels,ignore_index=True)


def is_report_stock(code):
    """Exactly the report archive's file universe, before quote eligibility."""
    return len(code) == 6 and code.isdigit() and not code.startswith(('4', '8', '92', '39'))


def report_industries_asof(code, dates, fundamental_path, end):
    """Replay every supplied report-availability event, including revisions.

    This is the original report archive's provider/event algorithm. Unknown
    industry stays unknown. It does not consult thematic or portfolio mappings.
    """
    provider = FundamentalData(str(fundamental_path), stock_codes=[code])
    provider._load_stock(code)
    reports = provider.stock_data.get(code)
    events = []
    invalid = 0

    def clean(value):
        return value.strip() if isinstance(value, str) and value.strip().lower() not in {'', 'nan', 'none'} else None

    try:
        if reports is not None and len(reports) and '数据可用日期' in reports:
            raw = reports['数据可用日期'].astype(str)
            valid = raw.str.fullmatch(r'\d{8}')
            invalid = int((~valid).sum())
            available = pd.to_datetime(raw[valid], format='%Y%m%d', errors='raise')
            for day in sorted(available[available <= pd.Timestamp(end)].unique()):
                events.append((pd.Timestamp(day), clean(provider.get_industry(code, pd.Timestamp(day)))))
        event_dates = np.array([r[0].to_datetime64() for r in events], dtype='datetime64[ns]')
        industries = np.array([r[1] for r in events], dtype=object)
        positions = np.searchsorted(event_dates, pd.DatetimeIndex(dates).to_numpy(dtype='datetime64[ns]'), side='right') - 1
        assigned = np.full(len(dates), None, dtype=object)
        known = positions >= 0
        assigned[known] = industries[positions[known]]
        for position in sorted({0, len(dates)//2, len(dates)-1}) if len(dates) else []:
            if assigned[position] != clean(provider.get_industry(code, pd.Timestamp(dates[position]))):
                raise ValueError(f'Report industry event/provider mismatch: {code} {dates[position]}')
        return assigned, dict(events=len(events), invalid_availability_dates=invalid)
    finally:
        provider.clear_stock_cache(code)


def read_report_stock(task):
    """Read full real QFQ history, return only the requested source tail.

    Prefix classification, the 60-observation warmup, positive quote filters,
    and return arithmetic match observe_sector_before_entry.read_stock.
    """
    code, path, start, end, fundamental_path = task
    path = Path(path)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Price file changed while reading: {path}')
    frame = pd.read_csv(io.BytesIO(payload), usecols=['datetime', 'close', 'volume', 'amount'])
    frame['datetime'] = pd.to_datetime(frame.datetime)
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise ValueError(f'Unordered/duplicate price dates: {code}')
    # Future rows must not participate even in the prefix classifier.
    frame = frame.loc[frame.datetime.le(end)].reset_index(drop=True)
    flags = _prefix_nonstock_flags(frame)
    close = frame.close
    result = pd.DataFrame(dict(code=code, date=frame.datetime, amount=frame.amount,
        ret_1d=close.pct_change(fill_method=None)))
    valid = (np.arange(len(frame)) >= 59) & ~flags & close.gt(0) & frame.volume.gt(0) & frame.amount.gt(0)
    result = result.loc[valid & frame.datetime.between(start, end)].copy()
    if not np.isfinite(result[['amount', 'ret_1d']].to_numpy(float)).all():
        raise ValueError(f'Nonfinite report-context source: {code}')
    fundamental = Path(fundamental_path)/f'{code}.csv'
    fundamental_hash = sha256(fundamental) if fundamental.exists() else None
    sectors, events = report_industries_asof(code, result.date.to_numpy(), fundamental_path, end)
    if (sha256(fundamental) if fundamental.exists() else None) != fundamental_hash:
        raise ValueError(f'Financial file changed while reading: {fundamental}')
    result['sector'] = sectors
    result['active'] = 1
    return result, dict(code=code, price_path=str(path), price_sha256=hashlib.sha256(payload).hexdigest(),
        size=after.st_size, mtime_ns=after.st_mtime_ns, financial_path=str(fundamental),
        financial_sha256=fundamental_hash, **events)


def require_exact_rows(actual, expected, fields, label):
    """Check keys, null masks and values without a floating-point tolerance."""
    keys = ['code', 'date']
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f'{label}: duplicate stock/date')
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError(f'{label}: source key coverage changed (actual={len(left)}, expected={len(right)})')
    differences = {}
    for field in fields:
        a, b = left[field], right[field]
        same = a.eq(b).fillna(False) | (a.isna() & b.isna())
        if not same.all():
            differences[field] = int((~same).sum())
    if differences:
        raise ValueError(f'{label}: exact value mismatches {differences}')
    return dict(rows=len(left), fields=list(fields), exact=True)
