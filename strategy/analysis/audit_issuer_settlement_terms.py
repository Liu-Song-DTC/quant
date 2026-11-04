#!/usr/bin/env python3
"""Cross-check issuer-notice settlement terms against frozen held events (read-only).

Cash payment *date* is not proof that funds are available before opening orders.
This audit does not replay trades or calculate a corrected NAV.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = {'code', 'ex_date', 'record_date', 'shareholder_bonus',
          'shareholder_cash', 'cash_payment_date', 'bonus_trading_date', 'source_url'}
OPTIONAL_FIELDS = {'price_reference_cash'}


def load_issuer_terms(issuer_file):
    source = pd.read_csv(issuer_file, dtype=str, keep_default_na=False)
    if not FIELDS.issubset(source.columns) or set(source.columns) - FIELDS - OPTIONAL_FIELDS:
        raise ValueError(f'issuer term columns must be {sorted(FIELDS)} '
                         'with optional price_reference_cash')
    # Differential distributions can pay a holder more than the diluted cash
    # amount used in the exchange's ex-dividend reference price.  D belongs to
    # the price series; shareholder_cash belongs to the account ledger.
    if 'price_reference_cash' not in source:
        source['price_reference_cash'] = source['shareholder_cash']
    source['ex_date'] = pd.to_datetime(source.ex_date, format='%Y-%m-%d', errors='raise')
    if source.duplicated(['code', 'ex_date']).any():
        raise ValueError('duplicate issuer event')
    if not source.code.str.fullmatch(r'\d{6}').all():
        raise ValueError('issuer code must be six digits')
    if not source.source_url.str.startswith('https://').all():
        raise ValueError('issuer terms require an HTTPS source notice')
    for field in ('record_date', 'cash_payment_date', 'bonus_trading_date'):
        supplied = source[field].ne('')
        source[field] = pd.to_datetime(source[field], format='%Y-%m-%d', errors='coerce')
        if (supplied & source[field].isna()).any():
            raise ValueError(f'invalid issuer {field}')
    if (source.record_date.isna().any() or source.cash_payment_date.isna().any() or
            (source.record_date >= source.ex_date).any() or
            (source.cash_payment_date < source.ex_date).any()):
        raise ValueError('invalid issuer entitlement or payment date')
    for field in ('shareholder_bonus', 'shareholder_cash', 'price_reference_cash'):
        source[field] = pd.to_numeric(source[field], errors='coerce')
    amount_fields = ['shareholder_bonus', 'shareholder_cash', 'price_reference_cash']
    if (source[amount_fields].isna().any().any() or
            (source[amount_fields] < 0).any().any() or
            (source.shareholder_bonus.gt(0) != source.bonus_trading_date.notna()).any() or
            (source.bonus_trading_date.dropna() < source.loc[source.bonus_trading_date.notna(), 'ex_date']).any()):
        raise ValueError('invalid issuer share/cash or bonus trading date')
    return source


def audit(held_file, issuer_file):
    held = pd.read_csv(held_file, dtype={'code': str}, keep_default_na=True)
    held['code'] = held.code.str.zfill(6)
    held['ex_date'] = pd.to_datetime(held.ex_date, format='%Y-%m-%d', errors='raise')
    if held.duplicated(['code', 'ex_date']).any():
        raise ValueError('held events contain duplicate code/ex-date; consolidate holdings first')
    source = load_issuer_terms(issuer_file)
    matched = source.merge(held, on=['code', 'ex_date'], how='left', validate='one_to_one',
                           indicator=True, suffixes=('_issuer', '_event'))
    if (matched._merge != 'both').any():
        raise ValueError('issuer evidence not in held frozen events')
    recorded = pd.to_datetime(matched.record_date_event, errors='coerce')
    if (recorded.isna().any() or (recorded != matched.record_date_issuer).any() or
            not np.isclose(matched.shareholder_bonus, matched.s.astype(float), rtol=0, atol=1e-6).all() or
            not np.isclose(matched.price_reference_cash, matched.D.astype(float), rtol=0, atol=1e-6).all()):
        raise ValueError('issuer terms conflict with frozen event or record-date audit')
    summary = {'held_event_rows': int(len(held)),
               'events_with_issuer_verified_terms': int(len(matched)),
               'events_without_issuer_verified_terms': int(len(held) - len(matched)),
               'issuer_verified_bonus_events': int(matched.shareholder_bonus.gt(0).sum())}
    summary['differential_cash_events'] = int((~np.isclose(
        matched.shareholder_cash, matched.price_reference_cash, rtol=0, atol=1e-6)).sum())
    return matched, summary


def guard_current_events(events_file, issuer_file, record_parts=None):
    """Fail before the price-directory swap when known issuer terms drift."""
    source = load_issuer_terms(issuer_file)
    events = pd.read_pickle(events_file).copy()
    events['code'] = events.code.astype(str).str.zfill(6)
    events['ex_date'] = pd.to_datetime(events.ex_date, errors='raise')
    keys = source[['code', 'ex_date']]
    selected = events.merge(keys, on=['code', 'ex_date'], how='inner')
    if selected.duplicated(['code', 'ex_date']).any():
        raise ValueError('duplicate issuer-verified event in current event table')
    matched = source.merge(selected, on=['code', 'ex_date'], how='left',
                           validate='one_to_one', indicator=True)
    if (matched._merge != 'both').any():
        raise ValueError('issuer-verified event missing from current event table')
    if (not np.isclose(matched.shareholder_bonus, pd.to_numeric(matched.s, errors='coerce'),
                       rtol=0, atol=1e-6).all() or
            not np.isclose(matched.price_reference_cash, pd.to_numeric(matched.D, errors='coerce'),
                           rtol=0, atol=1e-6).all()):
        raise ValueError('current event terms conflict with issuer notice; stop price rebuild')
    provider_dates_checked = 0
    if record_parts is not None:
        for row in source.itertuples(index=False):
            path = Path(record_parts) / f'{row.code}.pkl'
            if not path.exists():
                continue  # Old caches and provider timeouts have no sidecar yet.
            with path.open('rb') as file:
                part = pickle.load(file)
            entries = [pd.to_datetime(reg, errors='coerce')
                       for code, ex, reg in part
                       if code == row.code and pd.Timestamp(ex) == row.ex_date]
            if len(entries) != 1 or pd.isna(entries[0]) or entries[0] != row.record_date:
                raise ValueError(f'provider record date conflicts with issuer notice: '
                                 f'{row.code}/{row.ex_date.date()}')
            provider_dates_checked += 1
    return {'issuer_verified_events_checked': int(len(source)),
            'issuer_term_mismatches': 0,
            'provider_dates_cross_checked': provider_dates_checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--held', type=Path)
    choice.add_argument('--events', type=Path,
                        help='Check the current event table before rebuilding production prices')
    parser.add_argument('--record-parts', type=Path,
                        help='Optional provider registration-date sidecars; check those that exist')
    parser.add_argument('--issuer', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.events is not None:
        summary = guard_current_events(args.events, args.issuer, args.record_parts)
        rows = None
    else:
        rows, summary = audit(args.held, args.issuer)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        if rows is not None:
            rows.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
