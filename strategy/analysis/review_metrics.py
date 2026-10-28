#!/usr/bin/env python3
"""Comparable net short-horizon labels and account metrics for isolated runs.

Forward labels are marked-to-market at horizons, not claims of executable exits.
Blocked exits remain in the sample and are separately counted (no loser removal).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def account_metrics(directory, initial_cash=250000):
    directory = Path(directory)
    curve = pd.read_csv(directory / 'equity_curve.csv', parse_dates=['date'])
    if curve['date'].duplicated().any() or not curve['date'].is_monotonic_increasing:
        raise ValueError('NAV dates must be unique and increasing')
    nav = curve['nav'].to_numpy(dtype=float)
    if not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError('invalid account NAV')
    returns = pd.Series(nav).pct_change().dropna()
    out = {'start': str(curve['date'].min().date()), 'end': str(curve['date'].max().date()),
           'days': len(nav), 'final_nav': float(nav[-1]),
           'total_return': float(nav[-1] / initial_cash - 1),
           'annual_return': float((nav[-1] / initial_cash) ** (252 / max(len(nav) - 1, 1)) - 1),
           'max_drawdown': float((nav / np.maximum.accumulate(nav) - 1).min()),
           'sharpe': float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))
                     if returns.std(ddof=0) > 0 else 0.0}
    trade_path = directory / 'trade_realized.csv'
    if trade_path.exists():
        trades = pd.read_csv(trade_path)
        out.update(closed_trades=len(trades), trade_win_rate=float((trades['ret'] > 0).mean()) if len(trades) else None,
                   trade_mean_return=float(trades['ret'].mean()) if len(trades) else None,
                   trade_return_basis='net_lifecycle_cashflow' if 'pnl' in trades else 'legacy_price_return')
        if 'pnl' in trades:
            gains = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
            losses = -trades.loc[trades['pnl'] < 0, 'pnl'].sum()
            out['profit_factor'] = float(gains / losses) if losses > 0 else None
            if len(trades) > 1:
                remaining = trades.drop(trades['pnl'].idxmax())
                out['realized_pnl_ex_best_trade'] = float(remaining['pnl'].sum())
                out['net_trade_win_rate_ex_best_trade'] = float((remaining['ret'] > 0).mean())
                out['net_trade_mean_return_ex_best_trade'] = float(remaining['ret'].mean())
    return out


def forward_labels(directory, data_path, commission=.0005, slippage=.001, horizons=(1, 3, 5, 10),
                   entry_basis='simulated'):
    if entry_basis not in ('simulated', 'fills'):
        raise ValueError(f'unknown entry basis: {entry_basis}')
    fills = {}
    if entry_basis == 'fills':
        records = pd.read_csv(Path(directory) / 'execution_fills.csv', dtype={'code': str})
        buys = records[records['side'] == 'BUY'].copy()
        if not buys.empty:
            buys['entry_cost'] = (buys['gross_notional'] + buys['commission']
                                  + buys['impact_cost'])
            grouped = buys.groupby(['date', 'code'], sort=False)[['shares', 'entry_cost']].sum()
            if (grouped['shares'] <= 0).any() or (grouped['entry_cost'] <= 0).any():
                raise ValueError('invalid executed BUY fill shares or cost')
            fills = (grouped['entry_cost'] / grouped['shares']).to_dict()
    selections = pd.read_csv(Path(directory) / 'portfolio_selections.csv', dtype={'code': str}, parse_dates=['date'])
    selections = selections.drop_duplicates(['date', 'code'])
    if selections.empty:
        return pd.DataFrame(columns=['date', 'code', 'horizon', 'entry_available',
                                     'exit_available', 'net_mtm_return'])
    index = pd.read_csv(Path(data_path) / 'sh000001_qfq.csv', parse_dates=['datetime'])
    calendar = pd.DatetimeIndex(index['datetime']).sort_values().unique()
    rows = []
    for code, group in selections.groupby('code'):
        frame = pd.read_csv(Path(data_path) / f'{code}_qfq.csv', parse_dates=['datetime'])
        frame = frame.drop_duplicates('datetime', keep='last').set_index('datetime').reindex(calendar)
        op = frame['open'].to_numpy(dtype=float)
        close = frame['close'].ffill().to_numpy(dtype=float)
        vol = frame['volume'].fillna(0).to_numpy(dtype=float)
        rate = .20 if code.startswith(('30', '68')) else .30 if code[0] in '48' else .10
        for day in group['date']:
            decision = int(calendar.searchsorted(day))
            if decision >= len(calendar) or calendar[decision] != day:
                raise ValueError(f'selection date absent from calendar: {day}')
            entry = decision + 1
            if entry >= len(calendar):
                continue
            if entry_basis == 'fills':
                entry_price = fills.get((calendar[entry].strftime('%Y-%m-%d'), code))
                entry_ok = entry_price is not None
            else:
                entry_ok = bool(np.isfinite(op[entry]) and op[entry] > 0 and vol[entry] > 0
                                and op[entry] / close[decision] - 1 < rate - .003)
                entry_price = op[entry] * (1 + slippage) * (1 + commission)
            for horizon in horizons:
                exit_day = entry + horizon - 1
                if exit_day >= len(calendar):
                    continue
                stamp = .001 if calendar[exit_day] < pd.Timestamp('2023-08-28') else .0005
                net = (close[exit_day] * (1 - slippage) * (1 - commission - stamp)
                       / entry_price - 1) if entry_ok else np.nan
                prev = close[exit_day - 1]
                exit_ok = bool(vol[exit_day] > 0 and close[exit_day] / prev - 1 > -rate + .003)
                rows.append({'date': day, 'code': code, 'horizon': horizon,
                             'entry_available': entry_ok, 'exit_available': exit_ok,
                             'entry_basis': entry_basis, 'net_mtm_return': net})
    return pd.DataFrame(rows)


def summarize_labels(labels, bootstrap_samples=1000):
    summary = {}
    rng = np.random.default_rng(20261024)
    for horizon, all_rows in labels.groupby('horizon'):
        rows = all_rows.dropna(subset=['net_mtm_return']).copy()
        rows['win'] = rows['net_mtm_return'] > 0
        daily = rows.groupby('date').agg(wins=('win', 'sum'), n=('win', 'size'))
        wins, counts = daily['wins'].to_numpy(), daily['n'].to_numpy()
        distribution = []
        # 20-decision-day moving blocks preserve nearby picks' dependence.
        block = min(20, len(daily))
        if block:
            for _ in range(bootstrap_samples):
                starts = rng.integers(0, len(daily), size=int(np.ceil(len(daily) / block)))
                idx = ((starts[:, None] + np.arange(block)) % len(daily)).ravel()[:len(daily)]
                distribution.append(wins[idx].sum() / counts[idx].sum())
        annual = {}
        for year, group in rows.groupby(rows['date'].dt.year):
            annual[str(year)] = {'samples': len(group), 'win_rate': float(group['win'].mean()),
                                 'mean_net_mtm_return': float(group['net_mtm_return'].mean())}
        summary[str(horizon)] = {'selected_labels': len(all_rows), 'entry_available_samples': len(rows),
                                 'entry_unavailable': int((~all_rows['entry_available']).sum()),
                                 'exit_unavailable': int((~rows['exit_available']).sum()),
                                 'net_mtm_win_rate': float(rows['win'].mean()) if len(rows) else None,
                                 'mean_net_mtm_return': float(rows['net_mtm_return'].mean()) if len(rows) else None,
                                 'fixed_slot_net_return': float(all_rows['net_mtm_return'].fillna(0).mean())
                                     if len(all_rows) else None,
                                 'fixed_slot_win_rate': float((all_rows['net_mtm_return'] > 0).mean())
                                     if len(all_rows) else None,
                                 'win_rate_block_bootstrap_ci95': list(np.quantile(distribution, [.025, .975]))
                                     if distribution else None, 'annual': annual}
        if len(rows) > 1:
            trimmed = rows.sort_values('net_mtm_return').iloc[:-max(1, int(np.ceil(len(rows) * .05)))]
            summary[str(horizon)]['exclude_top_5pct_sensitivity'] = {
                'samples': len(trimmed), 'net_mtm_win_rate': float(trimmed['win'].mean()),
                'mean_net_mtm_return': float(trimmed['net_mtm_return'].mean())}
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--initial-cash', type=float, default=250000)
    p.add_argument('--entry-basis', choices=('simulated', 'fills'), default='fills',
                   help='fills uses executed BUY cash cost; simulated reproduces legacy next-open marks')
    args = p.parse_args()
    labels = forward_labels(args.results, args.data, entry_basis=args.entry_basis)
    result = {'account': account_metrics(args.results, args.initial_cash),
              'short_horizon': summarize_labels(labels),
              'label_basis': args.entry_basis + '_next_open_to_horizon_close_net_mark_to_market',
              'limitations': ['Blocked exits retained as marks; not executable profit.',
                              'Historical periods were already used for research, not untouched OOS.']}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    labels.to_csv(out.with_suffix('.labels.csv'), index=False)
    print(out)
