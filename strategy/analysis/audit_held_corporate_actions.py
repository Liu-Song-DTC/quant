#!/usr/bin/env python3
"""Locate corporate actions encountered by actual simulated positions.

The QFQ position share count is synthetic, so this is an exposure audit, not a
raw-price cash/dividend replay. Use the events table from the *same sealed run*
as the fills; do not mix run snapshots with current production events.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def audit(fills: pd.DataFrame, events: pd.DataFrame, through: str):
    required_fills = {'date', 'code', 'side', 'shares'}
    required_events = {'code', 'ex_date', 's', 'D', 'mult', 'src'}
    if required_fills - set(fills) or required_events - set(events):
        raise ValueError('required fill/event columns missing')
    f, e = fills.copy(), events.copy()
    f['date'] = pd.to_datetime(f['date'], errors='raise').dt.normalize()
    e['ex_date'] = pd.to_datetime(e['ex_date'], errors='raise').dt.normalize()
    f['code'] = f['code'].astype(str).str.zfill(6)
    e['code'] = e['code'].astype(str).str.zfill(6)
    if f['date'].isna().any() or e['ex_date'].isna().any():
        raise ValueError('missing fill/event date')
    if not f['date'].is_monotonic_increasing:
        raise ValueError('fills must be chronological')
    if e.duplicated(['code', 'ex_date']).any():
        raise ValueError('multiple corporate actions for a symbol/date; aggregate first')
    if (pd.to_numeric(f['shares'], errors='raise') <= 0).any():
        raise ValueError('fill shares must be positive')
    last_date = pd.Timestamp(through).normalize()
    if len(f) and f['date'].max() > last_date:
        raise ValueError('through predates a fill')
    # At a shared date, the corporate action belongs to holders from the
    # previous close; purchases and sales at the opening follow the action.
    steps = [(r.date, 1, n, 'fill', r) for n, r in enumerate(f.itertuples(index=False))]
    steps.extend((r.ex_date, 0, n, 'event', r)
                 for n, r in enumerate(e[e['ex_date'] <= last_date].itertuples(index=False)))
    steps.sort(key=lambda item: (item[0], item[1], item[2]))
    positions = defaultdict(int)
    rows = []
    for date, _, _, kind, record in steps:
        code = record.code
        if kind == 'fill':
            if record.side == 'BUY':
                positions[code] += int(record.shares)
            elif record.side == 'SELL':
                positions[code] -= int(record.shares)
                if positions[code] < 0:
                    raise ValueError(f'negative simulated holding: {code} {date.date()}')
            else:
                raise ValueError(f'unknown fill side: {record.side}')
            continue
        held = positions[code]
        if held <= 0:
            continue
        # Synthetic shares are only a marker of position exposure; no cash
        # entitlement is inferred from them here.
        rows.append({'ex_date': date.date().isoformat(), 'code': code,
                     'synthetic_qfq_shares_before_open': held,
                     's': record.s, 'D': record.D, 'mult': record.mult,
                     'src': record.src,
                     'cash_or_share_terms_missing': bool(pd.isna(record.s) or pd.isna(record.D))})
    detail = pd.DataFrame(rows, columns=['ex_date', 'code', 'synthetic_qfq_shares_before_open',
                                          's', 'D', 'mult', 'src', 'cash_or_share_terms_missing'])
    by_source = Counter(detail['src'])
    summary = {'through': last_date.date().isoformat(),
               'held_action_events': len(detail),
               'held_symbols': int(detail['code'].nunique()),
               'missing_cash_or_share_terms': int(detail['cash_or_share_terms_missing'].sum()),
               'by_source': dict(sorted(by_source.items()))}
    return detail, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True, help='sealed-run events_table.pkl')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    fills = pd.read_csv(a.results / 'execution_fills.csv', dtype={'code': str})
    nav = pd.read_csv(a.results / 'equity_curve.csv', usecols=['date'])
    events = pd.read_pickle(a.events)
    detail, summary = audit(fills, events, str(nav.date.iloc[-1]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out.with_suffix('.csv'), index=False)
    a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
