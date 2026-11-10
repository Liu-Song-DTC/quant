#!/usr/bin/env python3
"""Audit real 100-share lot sizes implied by synthetic QFQ BUY fills.

This read-only probe compares one executed QFQ order's *pre-fee notional* with
what the same notional could buy in raw 100-share lots. It neither replays a
portfolio nor claims corrected returns: raw-price execution also requires
cash dividends, stock splits, rights issues, and their effective dates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(results, raw_root, qfq_root):
    fills = pd.read_csv(Path(results) / 'execution_fills.csv', dtype={'date': str, 'code': str})
    if not fills.date.is_monotonic_increasing:
        raise ValueError('execution fills must be ordered by date')
    positions = {}
    new_entries = {}
    for fill in fills.itertuples(index=False):
        held = positions.get(fill.code, 0)
        if fill.side == 'BUY':
            new_entries[(fill.date, fill.code)] = held == 0
            positions[fill.code] = held + fill.shares
        elif fill.side == 'SELL':
            positions[fill.code] = held - fill.shares
            if positions[fill.code] < 0:
                raise ValueError(f'negative position: {fill.code}')
        else:
            raise ValueError(f'unknown fill side: {fill.side}')
    buys = fills.loc[fills.side == 'BUY']
    if (buys.shares <= 0).any() or buys.duplicated(['date', 'code']).any():
        raise ValueError('invalid or duplicated BUY fills; aggregate before auditing')
    rows = []
    for code, group in buys.groupby('code', sort=False):
        raw_path = Path(raw_root) / code / 'none.csv'
        qfq_path = Path(qfq_root) / f'{code}_qfq.csv'
        raw = pd.read_csv(raw_path, usecols=['日期', '开盘'], dtype={'日期': str})
        qfq = pd.read_csv(qfq_path, usecols=['datetime', 'open'], dtype={'datetime': str})
        if raw['日期'].duplicated().any() or qfq['datetime'].duplicated().any():
            raise ValueError(f'duplicate price dates: {code}')
        raw_open = raw.set_index('日期')['开盘']
        qfq_open = qfq.set_index('datetime')['open']
        for fill in group.itertuples(index=False):
            try:
                rpx = float(raw_open.loc[fill.date])
                apx = float(qfq_open.loc[fill.date])
            except KeyError as exc:
                raise ValueError(f'missing price for executed BUY {code} {fill.date}') from exc
            # Filled price embeds slippage and any gap penalty; the recorded
            # adjustment backs out exactly to the reference opening price.
            recorded_open = float(fill.price - fill.price_adjustment_cost / fill.shares)
            if not np.isfinite([rpx, apx, recorded_open]).all() or min(rpx, apx) <= 0:
                raise ValueError(f'invalid BUY price: {code} {fill.date}')
            if not np.isclose(recorded_open, apx, rtol=1e-5, atol=1e-4):
                raise ValueError(f'QFQ snapshot disagrees with recorded fill: {code} {fill.date}')
            target = fill.shares * apx
            raw_lots = int(np.floor(target / (rpx * 100) + 1e-10))
            actual_notional = raw_lots * 100 * rpx
            rows.append({'date': fill.date, 'code': code, 'qfq_shares': int(fill.shares),
                         'raw_shares_for_same_notional': raw_lots * 100,
                         'new_entry': bool(new_entries[(fill.date, fill.code)]),
                         'raw_over_qfq_price': rpx / apx,
                         'qfq_pre_fee_notional': float(target),
                         'raw_pre_fee_notional': float(actual_notional),
                         'lot_rounding_cash': float(target - actual_notional)})
    detail = pd.DataFrame(rows, columns=['date', 'code', 'qfq_shares',
                          'raw_shares_for_same_notional', 'new_entry', 'raw_over_qfq_price',
                          'qfq_pre_fee_notional', 'raw_pre_fee_notional', 'lot_rounding_cash'])
    report = {'buy_fills': len(detail),
              'raw_price_over_qfq_10pct': int((detail.raw_over_qfq_price > 1.1).sum()),
              'raw_price_over_qfq_2x': int((detail.raw_over_qfq_price > 2).sum()),
              'not_enough_for_one_raw_lot': int((detail.raw_shares_for_same_notional == 0).sum()),
              'new_entries': int(detail.new_entry.sum()),
              'new_entries_without_raw_lot': int((detail.new_entry &
                    (detail.raw_shares_for_same_notional == 0)).sum()),
              'topups_without_raw_lot': int((~detail.new_entry &
                    (detail.raw_shares_for_same_notional == 0)).sum()),
              'qfq_vs_raw_share_count_mismatch': int((detail.raw_shares_for_same_notional != detail.qfq_shares).sum()),
              'total_raw_lot_rounding_cash': float(detail.lot_rounding_cash.sum()),
              'median_raw_over_qfq_price': float(detail.raw_over_qfq_price.median()) if len(detail) else None}
    return detail, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--qfq', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    detail, report = audit(a.results, a.raw, a.qfq)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    detail.to_csv(a.out.with_suffix('.csv'), index=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
