#!/usr/bin/env python3
"""Annual and short-horizon decomposition of an immutable paired experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def annual(run, labels_path, initial_cash=250000.):
    nav = pd.read_csv(run / 'corrected/equity_curve.csv', parse_dates=['date']).set_index('date')['nav']
    trades = pd.read_csv(run / 'corrected/trade_realized.csv', parse_dates=['entry_date', 'exit_date'], dtype={'code': str})
    labels = pd.read_csv(labels_path, parse_dates=['date'], dtype={'code': str})
    results = []
    previous = initial_cash
    for year, values in nav.groupby(nav.index.year):
        # Include the last previous-year close as this year's starting equity.
        levels = np.r_[previous, values.to_numpy()]
        exits = trades[trades['exit_date'].dt.year == year]
        result = {'year': int(year), 'return': float(levels[-1]/previous-1),
                  'max_drawdown': float((levels/np.maximum.accumulate(levels)-1).min()),
                  'closed_trades': len(exits), 'winning_trades': int((exits['ret'] > 0).sum()),
                  'trade_win_rate': float((exits['ret'] > 0).mean()) if len(exits) else None,
                  'mean_trade_return': float(exits['ret'].mean()) if len(exits) else None}
        for horizon in [3, 5]:
            cohort = labels[(labels['date'].dt.year == year) & (labels['horizon'] == horizon)
                            & labels['entry_available']]
            result[f'win_{horizon}'] = float((cohort['net_mtm_return'] > 0).mean()) if len(cohort) else None
            result[f'win_count_{horizon}'] = int((cohort['net_mtm_return'] > 0).sum())
            result[f'samples_{horizon}'] = len(cohort)
        results.append(result)
        previous = levels[-1]
    drawdown = nav / nav.cummax() - 1
    trough = drawdown.idxmin()
    peak = nav.loc[:trough].idxmax()
    return results, {'peak': str(peak.date()), 'trough': str(trough.date()),
                     'peak_nav': float(nav.loc[peak]), 'trough_nav': float(nav.loc[trough]),
                     'max_drawdown': float(drawdown.loc[trough])}


def drawdown_contributions(run, episode):
    peak, trough = pd.Timestamp(episode['peak']), pd.Timestamp(episode['trough'])
    fills = pd.read_csv(run / 'corrected/execution_fills.csv', parse_dates=['date'], dtype={'code': str})
    fills = fills[fills['date'] <= trough].copy()
    fills['code'] = fills['code'].str.zfill(6)
    result = []
    for code, group in fills.groupby('code'):
        sign = np.where(group['side'] == 'BUY', 1, -1)
        shares = group['shares'] * sign
        start_shares = int(shares[group['date'] <= peak].sum())
        end_shares = int(shares.sum())
        within = group[(group['date'] > peak) & (group['date'] <= trough)]
        if within.empty and start_shares == 0:
            continue
        prices = pd.read_csv(run / f'data/stock_data/backtrader_data/{code}_qfq.csv',
                             parse_dates=['datetime'], usecols=['datetime', 'close']).set_index('datetime')['close'].sort_index()
        start = float(prices.loc[:peak].iloc[-1]) if start_shares else 0.
        end = float(prices.loc[:trough].iloc[-1]) if end_shares else 0.
        cash = float((within['gross_notional'] * np.where(within['side'] == 'SELL', 1, -1)).sum())
        costs = float(within[['commission', 'stamp_tax', 'impact_cost']].sum().sum())
        pnl = end_shares*end - start_shares*start + cash - costs
        result.append({'code': code, 'pnl': pnl, 'drawdown_contribution': pnl/episode['peak_nav'],
                       'start_shares': start_shares, 'end_shares': end_shares})
    result.sort(key=lambda row: row['pnl'])
    residual = sum(row['pnl'] for row in result) - (episode['trough_nav'] - episode['peak_nav'])
    return {'contributions': result, 'nav_change_reconciliation_error': residual,
            'basis': 'peak-close holdings to trough-close holdings plus intervening net fill cashflows; execution price includes price adjustment'}


def run(baseline, candidate, labels, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = {}
    for name, path in [('baseline', baseline), ('candidate', candidate)]:
        rows, episode = annual(path, labels / f'{name}_labels.csv')
        pd.DataFrame(rows).to_csv(output / f'{name}_annual.csv', index=False)
        result[name] = {'annual': rows, 'maximum_drawdown_episode': episode,
                        'drawdown_attribution': drawdown_contributions(path, episode)}
    result['definitions'] = {'annual_trade_win_rate': 'net closed lifecycle returns grouped by EXIT year',
                             'short_horizon_win_rate': 'net mark-to-market labels grouped by selection year, not executable exit guarantee',
                             'annual_drawdown': 'year-local peak including previous year-end equity; full-run peak can differ',
                             'causality': 'holdings attribution explains portfolio loss, not why a predictor changed'}
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'candidate', 'labels', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    run(args.baseline, args.candidate, args.labels, args.output)
