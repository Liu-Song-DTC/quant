#!/usr/bin/env python3
"""Fetch only corporate-action record dates for held events, preserving QFQ data."""
import argparse
from contextlib import nullcontext
import json
import sys
from unittest.mock import patch

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.fetch_actions_table import _record_date_rows, _save_record_dates
from strategy.analysis.audit_action_record_coverage import audit


def backfill(trades, events, record_parts, codes=None, workers=2, fetch=None):
    held, _ = audit(trades, events, record_parts)
    targets = (held.loc[~held.record_date_known]
               .groupby('code')['ex_date'].agg(lambda x: set(x)).to_dict())
    if codes is not None:
        targets = {code: dates for code, dates in targets.items() if code in codes}
    use_default_fetch = fetch is None
    fetch = fetch or (lambda code: ak.stock_fhps_detail_em(symbol=code))

    def one(code, dates):
        try:
            frame = fetch(code)
            if frame is None or frame.empty:
                raise ValueError('empty source response')
            extracted = _record_date_rows(code, frame)
            observed = {row[1]: row[2] for row in extracted}
            missing = sorted(dates - observed.keys())
            if missing:
                raise ValueError(f'missing held ex-dates: {missing}')
            if any(pd.isna(observed[date]) for date in dates):
                raise ValueError('held ex-date has no source record date')
            _save_record_dates(code, frame, record_parts)
            return {'code': code, 'status': 'ok', 'held_events': len(dates)}
        except Exception as exc:
            return {'code': code, 'status': 'failed',
                    'error': f'{type(exc).__name__}: {str(exc)[-240:]}'}

    result = []
    # AKShare's implementation calls requests.get without a timeout. Bound both
    # connection and read waits so one stalled stock cannot hang the whole audit.
    original_get = requests.get

    def bounded_get(*args, **kwargs):
        kwargs.setdefault('timeout', (6, 18))
        return original_get(*args, **kwargs)

    with patch.object(requests, 'get', bounded_get) if use_default_fetch else nullcontext():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(one, code, dates): code for code, dates in targets.items()}
            for task in as_completed(tasks):
                item = task.result()
                print(f"[record-date] {item['code']}: {item['status']}", flush=True)
                result.append(item)
    result.sort(key=lambda item: item['code'])
    return result, audit(trades, events, record_parts)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trades', required=True, type=Path)
    parser.add_argument('--events', required=True, type=Path)
    parser.add_argument('--record-parts', required=True, type=Path)
    parser.add_argument('--codes', help='comma-separated held symbols; default all held symbols')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    selected = set(args.codes.split(',')) if args.codes else None
    result, summary = backfill(args.trades, args.events, args.record_parts,
                               selected, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'fetches': result, 'coverage': summary},
                                   ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    if any(row['status'] != 'ok' for row in result):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
