#!/usr/bin/env python3
"""Audit price quantization on frozen event-free closed trades.

With no corporate action between entry and exit, a multiplicatively adjusted
close should have the same return as the unadjusted close before serialization.
The published QFQ CSV is rounded to cents, which can break that identity.
This measures price-return discrepancies, not full trade P&L or a corrected NAV.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def audit(run):
    root = Path(run)
    trades = pd.read_csv(root / 'corrected/trade_realized.csv', dtype={'code': str},
                         parse_dates=['entry_date', 'exit_date'])
    events = pd.read_pickle(root / 'data/alternative_data/events_table.pkl').copy()
    events['code'] = events.code.astype(str).str.zfill(6)
    events['ex_date'] = pd.to_datetime(events.ex_date, errors='raise')
    price_root = root / 'data/stock_data'
    rows = []
    for code, group in trades.groupby('code', sort=True):
        q = pd.read_csv(price_root / 'backtrader_data' / f'{code}_qfq.csv',
                        usecols=['datetime', 'close'], parse_dates=['datetime'])
        r = pd.read_csv(price_root / 'raw_data' / code / 'none.csv',
                        usecols=['日期', '收盘'], encoding='utf-8-sig')
        if q.datetime.duplicated().any() or r['日期'].duplicated().any():
            raise ValueError(f'duplicate price date for {code}')
        q = pd.Series(pd.to_numeric(q['close'], errors='raise').values,
                      index=q.datetime)
        r = pd.Series(pd.to_numeric(r['收盘'], errors='raise').values,
                      index=pd.to_datetime(r['日期'], errors='raise'))
        action_dates = events.loc[events.code == code, 'ex_date']
        for trade in group.itertuples(index=False):
            if ((action_dates > trade.entry_date) &
                    (action_dates <= trade.exit_date)).any():
                continue
            try:
                q_return = float(q.loc[trade.exit_date] / q.loc[trade.entry_date] - 1)
                raw_return = float(r.loc[trade.exit_date] / r.loc[trade.entry_date] - 1)
            except KeyError as error:
                raise ValueError(f'missing entry/exit price for {code}: {error}') from error
            rows.append({'code': code, 'entry_date': trade.entry_date.date().isoformat(),
                         'exit_date': trade.exit_date.date().isoformat(),
                         'qfq_close_return': q_return, 'raw_close_return': raw_return,
                         'difference_bp': (q_return - raw_return) * 10000})
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError('no event-free trades to audit')
    absolute = detail.difference_bp.abs()
    return detail, {'event_free_trades': int(len(detail)),
                    'over_1bp': int((absolute > 1).sum()),
                    'max_absolute_bp': float(absolute.max()),
                    'p95_absolute_bp': float(absolute.quantile(.95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='JSON summary; paired detail CSV')
    args = parser.parse_args()
    detail, report = audit(args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        detail.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
