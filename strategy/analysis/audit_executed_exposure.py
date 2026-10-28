"""Read-only replay of cash, synthetic shares and daily exposure from fills."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def replay(fills_dir: Path, data_dir: Path, initial_cash: float):
    equity = pd.read_csv(fills_dir / 'equity_curve.csv')
    fills = pd.read_csv(fills_dir / 'execution_fills.csv', dtype={'code': str})
    states = pd.read_csv(fills_dir / 'regime_state.csv')
    days = pd.to_datetime(equity['date'])
    if not days.is_monotonic_increasing or days.duplicated().any():
        raise ValueError('daily equity calendar is not strictly increasing')
    calendar = {day: i for i, day in enumerate(equity['date'])}
    close = {}
    for code in fills['code'].unique():
        bars = pd.read_csv(data_dir / 'stock_data/backtrader_data' / f'{code}_qfq.csv',
                           usecols=['datetime', 'close'])
        bars['datetime'] = pd.to_datetime(bars['datetime'])
        close[code] = bars.set_index('datetime')['close'].reindex(days).ffill().to_numpy(dtype='float32')
    by_day = defaultdict(list)
    for row in fills.itertuples(index=False):
        by_day[calendar[row.date]].append(row)
    cash = float(initial_cash)
    positions = defaultdict(int)
    rows = []
    for i, day in enumerate(equity['date']):
        for fill in by_day[i]:
            if fill.side not in ('BUY', 'SELL'):
                raise ValueError(f'unknown fill side: {fill.side}')
            positions[fill.code] += fill.shares if fill.side == 'BUY' else -fill.shares
            if positions[fill.code] < 0:
                raise ValueError(f'sold more shares than held: {fill.code}')
            cash += ((-fill.gross_notional if fill.side == 'BUY' else fill.gross_notional)
                     - fill.commission - fill.stamp_tax - fill.impact_cost)
        position_value = sum(float(np.float32(shares) * np.float32(close[code][i]))
                             for code, shares in positions.items() if shares)
        nav = float(equity['nav'].iat[i])
        rows.append((day, cash, position_value, nav, position_value / nav,
                     abs(cash + position_value - nav), sum(n > 0 for n in positions.values())))
    frame = pd.DataFrame(rows, columns=['date', 'cash', 'position_value', 'nav',
                                        'actual_exposure', 'nav_abs_error', 'actual_n_pos'])
    frame = frame.merge(states, on='date', how='left', validate='one_to_one')
    target_exposure = frame['target_exposure'] if 'target_exposure' in frame else frame['exposure']
    report = {'days': len(frame), 'fills': len(fills), 'max_nav_abs_error': float(frame.nav_abs_error.max()),
              'min_cash': float(frame.cash.min()), 'max_actual_exposure': float(frame.actual_exposure.max()),
              'days_actual_exposure_over_one': int((frame.actual_exposure > 1).sum()),
              'max_reported_target_exposure': float(target_exposure.max()),
              'days_target_exposure_over_one': int((target_exposure > 1).sum())}
    if 'target_exposure' in frame:
        report['max_reported_actual_exposure_diff'] = float((frame['exposure'] - frame.actual_exposure).abs().max())
    if report['max_nav_abs_error'] > .05 or report['min_cash'] < -1e-6:
        raise ValueError(f'fill replay disagrees with broker ledger: {report}')
    return frame, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--fills', type=Path, required=True)
    parser.add_argument('--initial-cash', type=float, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frame, report = replay(args.fills, args.data, args.initial_cash)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        frame.to_csv(args.output.with_name('daily.csv'), index=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
