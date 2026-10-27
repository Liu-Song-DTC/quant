"""Pure reporting-period and merge rules shared by the daily fundamental refresh."""
import pandas as pd
from fundamental_pit import pit_available_date, normalize_report_period


def recent_completed_quarters(as_of=None, count=3):
    """Last completed calendar quarters, oldest first; publication is separate."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('quarter count must be a positive integer')
    day = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    if pd.isna(day):
        raise ValueError('invalid as-of date')
    quarter = day.to_period('Q')
    if quarter.end_time.normalize() > day:
        quarter -= 1
    return [(quarter - offset).end_time.strftime('%Y%m%d')
            for offset in range(count - 1, -1, -1)]


def recent_refresh_quarters(as_of=None, count=3):
    """Roll the existing deadline-proxy policy without freezing partial reports.

    Only select periods whose proxy deadline has finished. Actual publication
    timestamps and revision histories require a separate data contract.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('quarter count must be a positive integer')
    day = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    candidates = recent_completed_quarters(day, count + 5)
    ready = [q for q in candidates if pd.Timestamp(pit_available_date(q)) < day]
    if len(ready) < count:
        raise ValueError('insufficient matured reporting periods')
    return ready[-count:]


def merge_report_rows(old, incoming):
    """Replace only periods actually received; keep all other historical rows."""
    if incoming.empty:
        return old.copy()
    if '报告期' not in incoming:
        raise ValueError('incoming financial data has no report period')
    old, incoming = old.copy(), incoming.copy()
    for frame in (old, incoming):
        if frame.empty:
            continue
        if '报告期' not in frame:
            raise ValueError('existing financial data has no report period')
        normalized = frame['报告期'].map(normalize_report_period)
        if normalized.eq('').any():
            raise ValueError('invalid financial report period')
        frame['报告期'] = normalized
        if frame['报告期'].duplicated().any():
            raise ValueError('ambiguous duplicate report periods; revision handling required')
    if not old.empty:
        old = old[~old['报告期'].isin(incoming['报告期'])]
    result = pd.concat([old, incoming], ignore_index=True, sort=False)
    return result.sort_values('报告期', kind='stable').reset_index(drop=True)
