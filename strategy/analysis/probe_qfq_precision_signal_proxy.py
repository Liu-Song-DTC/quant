#!/usr/bin/env python3
"""Read-only short-momentum precision probe on a frozen run's selections.

This conditions on selections from the old strategy; it cannot demonstrate
counterfactual selections, trading profits, or a full-universe rank change.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.rebuild_table_qfq import build_one


def probe(run):
    run = Path(run)
    prices = run / 'data/stock_data'
    chosen = pd.read_csv(run / 'corrected/portfolio_selections.csv',
                         dtype={'code': str}, parse_dates=['date'])
    if chosen.empty or chosen[['date', 'code']].isna().any().any():
        raise ValueError('missing or malformed frozen selection records')
    events = pd.read_pickle(run / 'data/alternative_data/events_table.pkl').copy()
    events['code'] = events.code.astype(str).str.zfill(6)
    with (prices / 'qfq_anchor.json').open() as file:
        anchors = json.load(file)
    rows = []
    for code, selections in chosen.groupby('code', sort=True):
        if code not in anchors:
            raise ValueError(f'frozen anchor absent: {code}')
        raw = pd.read_csv(prices / 'raw_data' / code / 'none.csv', encoding='utf-8-sig')
        raw['日期'] = pd.to_datetime(raw['日期'], errors='raise')
        raw = raw.sort_values('日期').reset_index(drop=True)
        q = pd.read_csv(prices / 'backtrader_data' / f'{code}_qfq.csv',
                        parse_dates=['datetime']).set_index('datetime')
        if raw['日期'].duplicated().any() or q.index.has_duplicates:
            raise ValueError(f'duplicate price date: {code}')
        acts = [(e.ex_date, e.s, e.D, e.mult)
                for e in events.loc[events.code == code].itertuples()]
        calculated = build_one(code, acts, raw, None, anchor='frozen',
                               kfix=anchors)
        if calculated is None:
            raise ValueError(f'unable to reconstruct: {code}')
        precise = pd.Series(calculated[0], index=raw['日期']).reindex(q.index)
        if precise.isna().any() or not np.allclose(np.round(precise, 2), q.close,
                                                   rtol=0, atol=1e-6):
            raise ValueError(f'frozen QFQ does not match cents reconstruction: {code}')
        for period in (3, 5, 10, 20):
            old = q.close / q.close.shift(period) - 1
            exact = precise / precise.shift(period) - 1
            for day in selections.date:
                if day not in old.index:
                    raise ValueError(f'selected date absent from price bars: {code}/{day.date()}')
                x, y = float(old.loc[day]), float(exact.loc[day])
                if not np.isfinite(x) or not np.isfinite(y):
                    raise ValueError(f'insufficient momentum lookback: {code}/{day.date()}/{period}')
                rows.append({'code': code, 'date': str(day.date()), 'period': period,
                             'rounded_momentum': x, 'precise_momentum': y,
                             'difference_bp': (x - y) * 10000,
                             'sign_flip': bool(np.sign(x) != np.sign(y))})
    detail = pd.DataFrame(rows)
    summary = {'selected_rows': int(len(chosen)), 'selected_symbols': int(chosen.code.nunique()),
               'by_period': {}}
    for period, group in detail.groupby('period'):
        deviations = group.difference_bp.abs()
        summary['by_period'][str(period)] = {
            'observations': len(group), 'median_abs_bp': float(deviations.median()),
            'max_abs_bp': float(deviations.max()), 'sign_flips': int(group.sign_flip.sum())}
    return detail, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='JSON summary plus paired detailed CSV')
    args = parser.parse_args()
    detail, summary = probe(args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        detail.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
