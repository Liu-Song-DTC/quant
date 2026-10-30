"""Read-only audit of event-day adjusted close against raw close and frozen actions.

This checks consistency with the supplied actions, not completeness/accuracy of
third-party corporate actions or executable real-price/cash settlement.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def audit(data_dir: Path, codes=None):
    actions = pd.read_pickle(data_dir / 'alternative_data/events_table.pkl')
    actions['ex_date'] = pd.to_datetime(actions['ex_date'])
    actions['code'] = actions['code'].astype(str).str.zfill(6)
    by_code = {code: group for code, group in actions.groupby('code', sort=False)}
    price_files = sorted((data_dir / 'stock_data/backtrader_data').glob('*_qfq.csv'))
    if codes is not None:
        chosen = set(codes)
        price_files = [f for f in price_files if f.stem[:-4] in chosen]
    counts = defaultdict(int)
    exceptions = []
    max_over_bound = 0.0
    deltas = []
    largest_rounding_errors = []
    recent_deltas = []
    missing_raw_ids = []
    shifted_event_examples = []
    largest_non_event_errors = []
    largest_non_event_rounding_errors = []
    largest_non_event_errors_since_2021 = []
    non_event_max_since_2021 = 0.0
    for path in price_files:
        code = path.stem[:-4]
        raw_path = data_dir / 'stock_data/raw_data' / code / 'none.csv'
        if not raw_path.exists():
            counts['missing_raw_stock'] += 1
            missing_raw_ids.append(code)
            continue
        try:
            raw = pd.read_csv(raw_path, usecols=['日期', '收盘'])
            adjusted = pd.read_csv(path, usecols=['datetime', 'close'])
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            counts['unreadable_stock'] += 1
            exceptions.append({'code': code, 'reason': 'unreadable', 'error': str(exc)})
            continue
        raw['date'] = pd.to_datetime(raw['日期'])
        adjusted['date'] = pd.to_datetime(adjusted['datetime'])
        raw_dates = raw['date'].to_numpy(dtype='datetime64[ns]')
        adj_dates = adjusted['date'].to_numpy(dtype='datetime64[ns]')
        if (len(raw_dates) > 1 and (np.diff(raw_dates) <= np.timedelta64(0, 'ns')).any()) or (len(adj_dates) > 1 and (np.diff(adj_dates) <= np.timedelta64(0, 'ns')).any()):
            counts['bad_date_order_stock'] += 1
            exceptions.append({'code': code, 'reason': 'duplicate_or_unsorted_dates'})
            continue
        frame = raw[['date', '收盘']].merge(adjusted[['date', 'close']], on='date', how='inner')
        if len(frame) != len(raw) or len(frame) != len(adjusted):
            counts['date_mismatch_stock'] += 1
            exceptions.append({'code': code, 'reason': 'raw_adjusted_calendar_mismatch', 'raw_rows': len(raw), 'adjusted_rows': len(adjusted), 'common_rows': len(frame)})
        if len(frame) < 2:
            counts['too_short_stock'] += 1
            continue
        counts['audited_stock'] += 1
        dates = frame.date.to_numpy(dtype='datetime64[ns]')
        raw_close = pd.to_numeric(frame['收盘'], errors='coerce').to_numpy(float)
        adj_close = pd.to_numeric(frame['close'], errors='coerce').to_numpy(float)
        mapped = {}
        for action in by_code.get(code, pd.DataFrame(columns=['ex_date', 's', 'D'])).itertuples(index=False):
            ix = int(np.searchsorted(dates, np.datetime64(action.ex_date)))
            if ix >= len(dates):
                counts['event_after_price_history'] += 1
                continue
            if ix == 0:
                counts['event_on_or_before_first_bar'] += 1
                continue
            if dates[ix] != np.datetime64(action.ex_date):
                counts['event_shifted_to_next_raw_bar'] += 1
                if len(shifted_event_examples) < 40:
                    shifted_event_examples.append({'code': code,
                        'ex_date': str(action.ex_date.date()),
                        'mapped_bar': str(frame.date.iat[ix].date())})
            previous_shares, previous_cash, previous_mult = mapped.get(ix, (0.0, 0.0, None))
            shares = float(action.s) if np.isfinite(action.s) else 0.0
            cash = float(action.D) if np.isfinite(action.D) else 0.0
            mult = getattr(action, 'mult', None)
            mult = float(mult) if mult is not None and np.isfinite(mult) else None
            mapped[ix] = ((1 + previous_shares) * (1 + shares) - 1,
                          previous_cash + cash, mult if mult is not None else previous_mult)
        ordinary = np.ones(len(frame), dtype=bool)
        ordinary[0] = False
        if mapped:
            ordinary[list(mapped)] = False
        ordinary_ix = np.flatnonzero(ordinary)
        if len(ordinary_ix):
            yesterday = ordinary_ix - 1
            finite = (np.isfinite(raw_close[yesterday]) & np.isfinite(raw_close[ordinary_ix]) &
                      np.isfinite(adj_close[yesterday]) & np.isfinite(adj_close[ordinary_ix]) &
                      (raw_close[yesterday] > 0) & (adj_close[yesterday] > .005))
            counts['invalid_non_event_price_day'] += int((~finite).sum())
            ordinary_ix = ordinary_ix[finite]
            yesterday = ordinary_ix - 1
            expected_daily = raw_close[ordinary_ix] / raw_close[yesterday]
            observed_daily = adj_close[ordinary_ix] / adj_close[yesterday]
            errors = np.abs(expected_daily - observed_daily)
            bounds = .005 * (1 + np.abs(expected_daily)) / (adj_close[yesterday] - .005) + 1e-9
            counts['audited_non_event_day'] += len(ordinary_ix)
            recent_mask = dates[ordinary_ix] >= np.datetime64('2021-01-01')
            counts['non_event_day_since_2021'] += int(recent_mask.sum())
            if recent_mask.any():
                non_event_max_since_2021 = max(non_event_max_since_2021, float(errors[recent_mask].max()))
                counts['non_event_prev_below_one_yuan_since_2021'] += int(((adj_close[yesterday] < 1) & recent_mask).sum())
                counts['non_event_delta_above_one_pct_since_2021'] += int(((errors > .01) & recent_mask).sum())
                recent_positions = np.flatnonzero(recent_mask)
                recent_worst = int(recent_positions[np.argmax(errors[recent_mask])])
                if errors[recent_worst] >= .005:
                    day_ix = ordinary_ix[recent_worst]
                    largest_non_event_errors_since_2021.append({
                        'code': code, 'date': str(frame.date.iat[day_ix].date()),
                        'error': float(errors[recent_worst]),
                        'rounding_bound': float(bounds[recent_worst]),
                        'adjusted_prev': float(adj_close[day_ix - 1]),
                        'expected': float(expected_daily[recent_worst]),
                        'observed': float(observed_daily[recent_worst])})
            top_pos = int(np.argmax(errors))
            if errors[top_pos] >= .005:
                day_ix = ordinary_ix[top_pos]
                largest_non_event_rounding_errors.append({'code': code, 'date': str(frame.date.iat[day_ix].date()),
                                                    'error': float(errors[top_pos]), 'rounding_bound': float(bounds[top_pos]),
                                                    'adjusted_prev': float(adj_close[day_ix - 1]),
                                                    'expected': float(expected_daily[top_pos]), 'observed': float(observed_daily[top_pos])})
            mismatch = np.flatnonzero(errors > bounds)
            counts['non_event_out_of_rounding_bound'] += len(mismatch)
            for pos in mismatch[:max(0, 40 - len(largest_non_event_errors))]:
                ix = ordinary_ix[pos]
                largest_non_event_errors.append({'code': code, 'date': str(frame.date.iat[ix].date()),
                                                 'expected': float(expected_daily[pos]),
                                                 'observed': float(observed_daily[pos]),
                                                 'error': float(errors[pos]),
                                                 'rounding_bound': float(bounds[pos])})
        for ix, (shares, cash, mult) in mapped.items():
            counts['audited_event_day'] += 1
            if not np.isfinite([raw_close[ix - 1], raw_close[ix], adj_close[ix - 1], adj_close[ix], shares, cash]).all() or min(raw_close[ix - 1], adj_close[ix - 1]) <= 0:
                counts['invalid_price_or_action'] += 1
                exceptions.append({'code': code, 'date': str(frame.date.iat[ix].date()), 'reason': 'invalid_price_or_action'})
                continue
            expected = (mult * raw_close[ix] if mult is not None else
                        (1 + shares) * raw_close[ix] + cash) / raw_close[ix - 1]
            observed = adj_close[ix] / adj_close[ix - 1]
            delta = abs(expected - observed)
            # Both adjusted closes were rounded to cents, bounding the ratio error.
            rounding_bound = .005 * (1 + abs(expected)) / max(adj_close[ix - 1] - .005, 1e-12) + 1e-9
            deltas.append(delta)
            recent = frame.date.iat[ix] >= pd.Timestamp('2021-01-01')
            if recent:
                recent_deltas.append(delta)
                counts['event_day_since_2021'] += 1
                if delta > .01:
                    counts['event_delta_above_one_pct_since_2021'] += 1
                if adj_close[ix - 1] < 1.0:
                    counts['event_adjusted_prev_below_one_yuan_since_2021'] += 1
            if adj_close[ix - 1] < 1.0:
                counts['event_adjusted_prev_below_one_yuan'] += 1
            if delta > .01:
                counts['event_delta_above_one_pct'] += 1
            if delta >= .005:
                largest_rounding_errors.append({'code': code, 'date': str(frame.date.iat[ix].date()),
                                                'error': delta, 'rounding_bound': rounding_bound,
                                                'adjusted_prev': adj_close[ix - 1],
                                                'expected': expected, 'observed': observed})
            if delta > rounding_bound:
                counts['out_of_rounding_bound'] += 1
                max_over_bound = max(max_over_bound, delta / rounding_bound)
                exceptions.append({'code': code, 'date': str(frame.date.iat[ix].date()), 'reason': 'event_return_mismatch', 'expected': expected, 'observed': observed, 'error': delta, 'rounding_bound': rounding_bound, 's': shares, 'D': cash, 'mult': mult})
    result = {'counts': dict(counts), 'delta_quantiles': {str(q): float(np.quantile(deltas, q)) for q in (.5, .9, .95, .99, 1.)} if deltas else {}, 'missing_raw_codes': missing_raw_ids, 'shifted_event_examples': shifted_event_examples, 'non_event_max_error_since_2021': non_event_max_since_2021, 'non_event_mismatch_examples': largest_non_event_errors, 'largest_non_event_rounding_errors': sorted(largest_non_event_rounding_errors, key=lambda row: row['error'], reverse=True)[:25], 'largest_non_event_errors_since_2021': sorted(largest_non_event_errors_since_2021, key=lambda row: row['error'], reverse=True)[:25], 'recent_delta_quantiles': {str(q): float(np.quantile(recent_deltas, q)) for q in (.5, .9, .95, .99, 1.)} if recent_deltas else {}, 'max_error_over_rounding_bound': max_over_bound, 'largest_rounding_errors': sorted(largest_rounding_errors, key=lambda row: row['error'], reverse=True)[:25], 'exception_count': len(exceptions), 'exceptions': exceptions[:200]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--codes', help='comma-separated symbols; by default audit all stocks')
    args = parser.parse_args()
    result = audit(args.data, args.codes.split(',') if args.codes else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'counts': result['counts'], 'delta_quantiles': result['delta_quantiles'], 'recent_delta_quantiles': result['recent_delta_quantiles'], 'exception_count': result['exception_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
