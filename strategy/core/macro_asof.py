"""Release-aware macro observations for research and shared live/replay use.

Daily observation dates are NOT release timestamps. Inputs retain revisions;
callers supply verified publication or first-observed times, with explicit proxy
labels where historical publication coverage is unavailable. No trading rule or
exposure multiplier is enabled by this module.
"""
import math

import pandas as pd


def _instant(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('macro timestamps must include their time zone')
    return result.tz_convert('UTC')


def known_history(records, decision_at, allow_proxy=False):
    """Keep only vintages available at decision time; never back-fill releases.

    Required columns: series, observation_at, available_at, value, source,
    availability_basis (verified_release / observed / proxy).
    """
    cutoff = _instant(decision_at)
    if records.empty:
        return records.copy()
    frame = records.copy()
    required = ['series', 'observation_at', 'available_at', 'value', 'source', 'availability_basis']
    if not set(required).issubset(frame):
        raise ValueError('missing macro release metadata')
    for column in ['series', 'source']:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError('missing macro source/series')
    for column in ['observation_at', 'available_at']:
        frame[column] = pd.to_datetime(frame[column].map(_instant), utc=True)
    frame['value'] = pd.to_numeric(frame['value'], errors='raise')
    if not frame['value'].map(math.isfinite).all():
        raise ValueError('nonfinite macro value')
    if (frame['available_at'] < frame['observation_at']).any():
        raise ValueError('macro observation released before observation time')
    if not frame['availability_basis'].isin(['verified_release', 'observed', 'proxy']).all():
        raise ValueError('unknown macro availability basis')
    frame = frame[frame['available_at'] <= cutoff]
    if not allow_proxy:
        frame = frame[frame['availability_basis'] != 'proxy']
    keys = ['series', 'observation_at', 'available_at']
    frame = frame.drop_duplicates(subset=required)
    if frame.duplicated(keys).any():
        raise ValueError('conflicting macro versions at the same release time')
    # Later revisions supersede an observation only once they become available.
    return (frame.sort_values(['available_at', 'observation_at'], kind='stable')
            .drop_duplicates(['series', 'observation_at'], keep='last')
            .sort_values(['series', 'observation_at'], kind='stable').reset_index(drop=True))


def macro_warning_features(records, decision_at, max_age_days, allow_proxy=False):
    """Return raw candidate warning features, freshness and provenance.

    us10y is yield in percentage points; usdcny is CNY per USD, so a positive
    return denotes CNY depreciation. vix is US implied volatility, not A-share
    market volatility. Observation-count windows do not imply Chinese sessions.
    Stale/uncovered series remain missing, not a manufactured zero-risk signal.
    """
    history = known_history(records, decision_at, allow_proxy)
    cutoff = _instant(decision_at)
    result = {'features': {}, 'coverage': {}}
    for series in ['us10y', 'usdcny', 'vix', 'a_realized_vol20']:
        if series not in max_age_days or not math.isfinite(max_age_days[series]) or max_age_days[series] <= 0:
            raise ValueError(f'explicit positive freshness limit required for {series}')
        rows = history[history['series'] == series] if not history.empty else history
        if rows.empty:
            result['coverage'][series] = {'state': 'missing'}
            continue
        latest = rows.iloc[-1]
        # A newly released correction to an old observation is still old data.
        age = (cutoff - latest['observation_at']).total_seconds() / 86400
        coverage = {'state': 'fresh' if age <= max_age_days[series] else 'stale',
                    'observation_age_days': age, 'source': latest['source'],
                    'available_at': latest['available_at'].isoformat(),
                    'availability_basis': latest['availability_basis']}
        result['coverage'][series] = coverage
        if coverage['state'] != 'fresh':
            continue
        values = rows['value'].to_numpy(dtype=float)
        result['features'][series + '_level'] = float(values[-1])
        for window in [5, 20]:
            if len(values) <= window:
                continue
            if series == 'us10y':
                result['features'][f'us10y_change_{window}_bp'] = float((values[-1] - values[-1-window]) * 100)
            elif values[-1-window] > 0:
                result['features'][f'{series}_return_{window}'] = float(values[-1] / values[-1-window] - 1)
    return result
