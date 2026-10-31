#!/usr/bin/env python3
"""Read-only raw-close coverage of actual held sessions in a frozen run.

This audits the old fills' held intervals, not newly selected names in a
counterfactual raw-price backtest. A QFQ traded bar without a raw close is a
missing execution/valuation source, while a missing raw bar with no trade is
not an executable stock session.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strategy.core.raw_execution_prices import load_raw_ohlc


def probe(run):
    run = Path(run)
    trades = pd.read_csv(run / 'corrected/trade_realized.csv',
                         dtype={'code': str}, parse_dates=['entry_date', 'exit_date'])
    if trades.empty or trades[['entry_date', 'exit_date', 'code']].isna().any().any():
        raise ValueError('missing or malformed frozen realized trades')
    root = run / 'data/stock_data'
    problems = []
    counts = {'symbols': 0, 'active_held_bars': 0, 'suspended_held_bars': 0}
    for code, portions in trades.groupby('code', sort=True):
        qfile = root / 'backtrader_data' / f'{code}_qfq.csv'
        try:
            q = pd.read_csv(qfile, usecols=['datetime', 'volume'],
                            parse_dates=['datetime']).set_index('datetime')
            if q.index.has_duplicates:
                raise ValueError('duplicate QFQ session')
            q = q.loc[q.index >= portions.entry_date.min()]
            raw = load_raw_ohlc(root / 'raw_data' / code / 'none.csv', q.index)
        except (OSError, ValueError, KeyError) as exc:
            problems.append({'code': code, 'issue': 'source', 'detail': str(exc)})
            continue
        counts['symbols'] += 1
        held = pd.Series(False, index=q.index)
        for trade in portions.itertuples(index=False):
            held |= (q.index >= trade.entry_date) & (q.index <= trade.exit_date)
        active = held & (q.volume > 0)
        counts['active_held_bars'] += int(active.sum())
        counts['suspended_held_bars'] += int((held & (q.volume <= 0)).sum())
        for day in q.index[active & raw['close'].isna()]:
            problems.append({'code': code, 'issue': 'active_close_missing',
                             'detail': str(day.date())})
    return {**counts, 'symbols_in_trades': int(trades.code.nunique()),
            'problems': problems, 'problem_count': len(problems)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True,
                        help='Frozen review run with corrected/trade_realized.csv and copied prices')
    parser.add_argument('--out', type=Path, help='Optional JSON audit artifact')
    args = parser.parse_args()
    report = probe(args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'problems'}, ensure_ascii=False))
    if report['problems']:
        raise SystemExit('raw held price coverage has gaps; inspect report or stdout')


if __name__ == '__main__':
    main()
