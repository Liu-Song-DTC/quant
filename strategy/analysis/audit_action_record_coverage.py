#!/usr/bin/env python3
"""Read-only coverage of entitlement dates for actually held stock events."""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def audit(trades_file, events_file, record_parts, manual_records=None):
    trades = pd.read_csv(trades_file, dtype={'code': str},
                         parse_dates=['entry_date', 'exit_date'])
    events = pd.read_pickle(events_file)
    events = events.copy()
    events['code'] = events.code.astype(str).str.zfill(6)
    events['ex_date'] = pd.to_datetime(events.ex_date)
    held = trades.merge(events, on='code', how='inner')
    held = held.loc[(held.entry_date < held.ex_date) &
                    (held.ex_date <= held.exit_date)].copy()
    event_keys = set(zip(events.code, events.ex_date))
    records = {}
    for code in held.code.unique():
        part = Path(record_parts) / f'{code}.pkl'
        if not part.exists():
            continue
        with part.open('rb') as file:
            rows = pickle.load(file)
        for row in rows:
            if len(row) != 3 or row[0] != code:
                raise ValueError(f'invalid record-date part: {part}')
            key = (code, pd.Timestamp(row[1]))
            val = pd.to_datetime(row[2], errors='coerce')
            if key in records and not ((pd.isna(val) and pd.isna(records[key])) or
                                       val == records[key]):
                raise ValueError(f'conflicting record date: {key}')
            records[key] = val
    manual = {}
    if manual_records is not None:
        source = pd.read_csv(manual_records, dtype=str, keep_default_na=False)
        expected = {'code', 'ex_date', 'record_date', 'source_url'}
        if set(source.columns) != expected:
            raise ValueError(f'issuer record columns must be {sorted(expected)}')
        for row in source.itertuples(index=False):
            code, ex_date, record_date, url = row
            ex = pd.to_datetime(ex_date, format='%Y-%m-%d', errors='coerce')
            reg = pd.to_datetime(record_date, format='%Y-%m-%d', errors='coerce')
            key = (code, ex)
            if (len(code) != 6 or not code.isdigit() or key not in event_keys or
                    pd.isna(reg) or not reg < ex or not url.startswith('https://')):
                raise ValueError(f'invalid issuer record for {code}/{ex_date}')
            if key in manual and manual[key] != (reg, url):
                raise ValueError(f'conflicting issuer records: {key}')
            if key in records and pd.notna(records[key]) and records[key] != reg:
                raise ValueError(f'provider/issuer record conflict: {key}')
            manual[key] = (reg, url)
    held['record_date'] = [records.get((r.code, r.ex_date), pd.NaT)
                           for r in held.itertuples()]
    held['record_date_source'] = ['provider' if pd.notna(value) else ''
                                  for value in held.record_date]
    held['issuer_notice_url'] = ''
    for idx, row in held.iterrows():
        key = (row.code, row.ex_date)
        if key in manual:
            held.at[idx, 'record_date'] = manual[key][0]
            held.at[idx, 'record_date_source'] = 'provider+issuer' if pd.notna(row.record_date) else 'issuer'
            held.at[idx, 'issuer_notice_url'] = manual[key][1]
    held['record_date_known'] = held.record_date.notna()
    held['event_decomposed'] = held.s.notna() & held.D.notna()
    # A vendor multiplier or a value replaced to match vendor QFQ prices is
    # insufficient evidence of the shareholder's cash/share entitlement.
    # Even 'em' is a source label, not issuer-level confirmation of the terms.
    held['terms_from_unmodified_provider'] = held.event_decomposed & (
        held.src.eq('em') if 'src' in held.columns else False)
    summary = {
        'closed_trades': int(len(trades)),
        'held_event_rows': int(len(held)),
        'affected_trades': int(held[['code', 'entry_date', 'exit_date']].drop_duplicates().shape[0]),
        'events_with_record_date': int(held.record_date_known.sum()),
        'events_from_provider': int(held.record_date_source.isin(['provider', 'provider+issuer']).sum()),
        'events_from_issuer_only': int((held.record_date_source == 'issuer').sum()),
        'events_missing_record_date': int((~held.record_date_known).sum()),
        'events_without_cash_share_decomposition': int((~held.event_decomposed).sum()),
        'events_without_unmodified_provider_terms': int((~held.terms_from_unmodified_provider).sum()),
    }
    return held, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trades', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--record-parts', type=Path, required=True)
    parser.add_argument('--manual-records', type=Path,
                        help='Optional issuer-verified records; kept separate from provider sidecars')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    detail, summary = audit(args.trades, args.events, args.record_parts, args.manual_records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    detail.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
