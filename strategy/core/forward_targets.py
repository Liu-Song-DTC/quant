"""Optional short-horizon ML labels aligned with next-open execution.

Only labels change; feature rows, prediction coverage and stock eligibility never
depend on future outcomes. Exits are net marks, including suspended holdings.
"""
import numpy as np
import pandas as pd


def short_net_targets(factors, stock_file_map, calendar, commission=.0005, slippage=.001,
                      horizons=(3, 5), return_details=False):
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    horizons = tuple(int(h) for h in horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError('positive target horizons required')
    labels = np.full(len(factors), np.nan)
    details = None
    if return_details:
        details = {f'net_return_{h}': np.full(len(factors), np.nan) for h in horizons}
        details['entry_available'] = np.zeros(len(factors), dtype=bool)
        details['label_matured'] = np.zeros(len(factors), dtype=bool)
    ends = np.full(len(factors), np.datetime64('NaT'), dtype='datetime64[ns]')
    dates = pd.to_datetime(factors['date']).to_numpy(dtype='datetime64[ns]')
    for code, positions in factors.groupby('code', sort=False).indices.items():
        if code not in stock_file_map:
            continue
        frame = pd.read_csv(stock_file_map[code], parse_dates=['datetime'],
                            usecols=['datetime', 'open', 'close', 'volume'])
        frame = frame.sort_values('datetime').drop_duplicates('datetime', keep='last').set_index('datetime').reindex(cal)
        close = frame['close'].ffill().to_numpy(dtype=float)
        opening = frame['open'].to_numpy(dtype=float)
        volume = frame['volume'].fillna(0).to_numpy(dtype=float)
        origin = cal.searchsorted(dates[positions])
        valid = (origin < len(cal) - max(horizons))
        selected = positions[valid]; origin = origin[valid]
        if not len(origin):
            continue
        if not np.array_equal(cal.values[origin], dates[selected]):
            raise ValueError('factor dates missing from trading calendar')
        entry = origin + 1
        limit = .20 if str(code).startswith(('30', '68')) else .30 if str(code)[0] in '48' else .10
        available = (np.isfinite(opening[entry]) & (opening[entry] > 0) & (volume[entry] > 0)
                     & (opening[entry] / close[origin] - 1 < limit - .003))
        if details is not None:
            details['entry_available'][selected] = available
            details['label_matured'][selected] = True
        net = []
        for horizon in horizons:
            exit_day = origin + horizon
            stamp = np.where(cal.values[exit_day] < np.datetime64('2023-08-28'), .001, .0005)
            returns = (close[exit_day] * (1 - slippage) * (1 - commission - stamp)
                       / (opening[entry] * (1 + slippage) * (1 + commission)) - 1)
            net.append(returns)
            if details is not None:
                details[f'net_return_{horizon}'][selected] = np.where(
                    available & np.isfinite(returns), returns, np.nan)
        mean = np.mean(net, axis=0)
        labels[selected] = np.where(available & np.isfinite(mean), mean, np.nan)
        ends[selected] = cal.values[origin + max(horizons)]
    result = factors.copy(deep=False)
    result['future_ret'] = labels
    result['label_end_date'] = ends
    if return_details:
        # Keep diagnostics separate: future execution availability is never an
        # input feature or a rule for choosing prediction candidates.
        return result, pd.DataFrame(details, index=factors.index)
    return result
