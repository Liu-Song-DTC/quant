#!/usr/bin/env python3
"""Read-only five-day momentum rank sensitivity among frozen buy candidates.

The candidates and decision dates are held fixed. Their original strategy
scores are not recomputed, so this is not a counterfactual strategy backtest.
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
    selected = pd.read_csv(run / 'corrected/portfolio_selections.csv',
                           usecols=['date'], parse_dates=['date'])
    decision_days = set(selected.date.dt.strftime('%Y-%m-%d'))
    if not decision_days:
        raise ValueError('frozen selection decisions are empty')
    csv = run / 'strategy/rolling_validation_results/backtest_signals.csv'
    batches = []
    for batch in pd.read_csv(csv, usecols=['code', 'date', 'buy'], dtype={'code': str},
                             chunksize=250_000):
        batches.append(batch.loc[batch.buy.eq(True) & batch.date.isin(decision_days),
                                 ['code', 'date']])
    candidates = pd.concat(batches, ignore_index=True)
    candidates.code = candidates.code.str.zfill(6)
    if candidates.duplicated(['code', 'date']).any():
        raise ValueError('duplicate frozen buy signal for a decision date')
    events = pd.read_pickle(run / 'data/alternative_data/events_table.pkl').copy()
    events['code'] = events.code.astype(str).str.zfill(6)
    with (prices / 'qfq_anchor.json').open() as file:
        anchors = json.load(file)
    rows = []; unavailable = []
    for code, group in candidates.groupby('code', sort=True):
        source = prices / 'raw_data' / code / 'none.csv'
        adjusted = prices / 'backtrader_data' / f'{code}_qfq.csv'
        if not source.exists() or not adjusted.exists() or code not in anchors:
            unavailable.append(code)
            continue
        raw = pd.read_csv(source, encoding='utf-8-sig')
        raw['日期'] = pd.to_datetime(raw['日期'], errors='raise')
        raw = raw.sort_values('日期').reset_index(drop=True)
        q = pd.read_csv(adjusted, usecols=['datetime', 'close'],
                        parse_dates=['datetime']).set_index('datetime')
        if raw['日期'].duplicated().any() or q.index.has_duplicates:
            raise ValueError(f'duplicate price date: {code}')
        acts = [(e.ex_date, e.s, e.D, e.mult)
                for e in events.loc[events.code == code].itertuples()]
        built = build_one(code, acts, raw, None, anchor='frozen', kfix=anchors)
        if built is None:
            raise ValueError(f'cannot reconstruct frozen QFQ: {code}')
        precise = pd.Series(built[0], index=raw['日期']).reindex(q.index)
        if precise.isna().any() or not np.allclose(np.round(precise, 2), q.close,
                                                   rtol=0, atol=1e-6):
            raise ValueError(f'frozen QFQ price differs beyond cents: {code}')
        rounded_mom = q.close / q.close.shift(5) - 1
        precise_mom = precise / precise.shift(5) - 1
        for day in group.date:
            day = pd.Timestamp(day)
            if day not in q.index:
                raise ValueError(f'candidate day has no price: {code}/{day.date()}')
            x, y = float(rounded_mom.loc[day]), float(precise_mom.loc[day])
            if np.isfinite(x) and np.isfinite(y):
                rows.append((str(day.date()), code, x, y))
    detail = pd.DataFrame(rows, columns=['date', 'code', 'rounded_mom5', 'precise_mom5'])
    if detail.empty:
        raise ValueError('no candidate momentum observations')
    rank_changes = []
    for day, group in detail.groupby('date', sort=True):
        # Stable code tie-break prevents CSV insertion order from changing top K.
        old = group.sort_values(['rounded_mom5', 'code'], ascending=[False, True])
        new = group.sort_values(['precise_mom5', 'code'], ascending=[False, True])
        for k in (10, 20):
            if len(group) >= k:
                old_top = set(old.code.head(k)); new_top = set(new.code.head(k))
                rank_changes.append({'date': day, 'candidates': len(group), 'top_k': k,
                                     'members_changed': len(old_top - new_top)})
    rank = pd.DataFrame(rank_changes)
    if rank.empty:
        raise ValueError('no decision day has at least ten comparable candidates')
    summary = {'buy_candidates_on_decision_days': len(candidates),
               'comparable_candidates': len(detail), 'unique_candidates': detail.code.nunique(),
               'unavailable_raw_or_anchor_codes': len(unavailable),
               'days_compared': detail.date.nunique(),
               'by_top_k': {str(k): {'days': len(group),
                                    'days_with_membership_change': int(group.members_changed.gt(0).sum()),
                                    'total_members_displaced': int(group.members_changed.sum())}
                            for k, group in rank.groupby('top_k')}}
    return summary, rank, unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='JSON summary and daily ranking CSV')
    args = parser.parse_args()
    summary, rank, unavailable = probe(args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, 'unavailable_codes': unavailable},
                                       ensure_ascii=False, indent=2) + '\n')
        rank.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
