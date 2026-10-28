#!/usr/bin/env python3
"""Explain one NAV drawdown with executed cash flows and QFQ daily marks.

Both endpoints are *close-of-day* NAV observations. Use the backtest's own frozen
price snapshot; a mismatch against the saved NAV raises instead of attributing
an updated QFQ history to the old run. This is research accounting, not a
replay of raw-price dividends or a prediction of future drawdowns.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def attribute(results, data, start, end, tolerance=.05):
    results, data = Path(results), Path(data)
    curve = pd.read_csv(results / 'equity_curve.csv', dtype={'date': str})
    if curve['date'].duplicated().any() or not curve['date'].is_monotonic_increasing:
        raise ValueError('NAV dates must be unique and increasing')
    days = curve['date'].tolist()
    if start not in days or end not in days or start >= end:
        raise ValueError('drawdown endpoints must be ordered backtest dates')
    fills = pd.read_csv(results / 'execution_fills.csv', dtype={'date': str, 'code': str})
    rows = []
    for code, group in fills.groupby('code', sort=True):
        before = group[group.date <= start]
        within = group[(group.date > start) & (group.date <= end)]
        signed_before = before.shares.where(before.side == 'BUY', -before.shares)
        signed_within = within.shares.where(within.side == 'BUY', -within.shares)
        if not group.side.isin(('BUY', 'SELL')).all():
            raise ValueError(f'invalid fill side: {code}')
        start_shares = int(signed_before.sum())
        end_shares = start_shares + int(signed_within.sum())
        if start_shares < 0 or end_shares < 0:
            raise ValueError(f'invalid fill position: {code}')
        if start_shares == 0 and within.empty:
            continue
        bars = pd.read_csv(data / f'{code}_qfq.csv', usecols=['datetime', 'close'])
        prices = bars.drop_duplicates('datetime', keep='last').set_index('datetime')['close']
        # Execution engine stores float32 daily close and carries suspended marks.
        prices = prices.reindex(days[:days.index(end) + 1]).ffill().astype('float32')
        p0 = float(prices.loc[start])
        p1 = float(prices.loc[end])
        if not np.isfinite([p0, p1]).all() or min(p0, p1) <= 0:
            raise ValueError(f'missing price for held stock: {code}')
        net_cash = ((within.gross_notional.where(within.side == 'SELL', -within.gross_notional)
                     - within.commission - within.stamp_tax - within.impact_cost).sum())
        contribution = end_shares * p1 - start_shares * p0 + net_cash
        rows.append({'code': code, 'start_shares': start_shares,
                     'end_shares': end_shares, 'fills': len(within),
                     'pnl': float(contribution)})
    detail = pd.DataFrame(rows, columns=['code', 'start_shares', 'end_shares', 'fills', 'pnl'])
    detail = detail.sort_values('pnl').reset_index(drop=True)
    nav = curve.set_index('date')['nav']
    delta = float(nav.loc[end] - nav.loc[start])
    error = float(detail.pnl.sum() - delta)
    if abs(error) > tolerance:
        raise ValueError(f'QFQ snapshot/fills do not reconcile: residual={error:.2f}, '
                         f'NAV delta={delta:.2f}; use the backtest frozen data')
    return detail, {'start': start, 'end': end, 'start_nav': float(nav.loc[start]),
                    'end_nav': float(nav.loc[end]), 'nav_delta': delta,
                    'stock_pnl_sum': float(detail.pnl.sum()), 'residual': error,
                    'stocks': len(detail)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True,
                        help='backtrader_data folder from the exact backtest input snapshot')
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    detail, report = attribute(args.results, args.data, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    detail.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
