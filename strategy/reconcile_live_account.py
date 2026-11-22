#!/usr/bin/env python3
"""Import broker execution reports without running signals or issuing orders."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from core.execution_ledger import apply_execution_reports, unresolved_orders
from core.live_account import load_account, ACCOUNT_FILE
from core.live_state import commit_json_files, live_run_lock
from core.config_loader import load_config
from core.deployment_profile import activate_profile, accepted465_active
from core.live_accepted465 import STATE_KEY, create_live_bridge, _day


def reconcile(fills, root, initial_capital=500000, *, as_of=None):
    """Book actual reports; caller holds live_run_lock (the CLI does so).

    An existing accepted465 account selects its persisted policy even when the
    environment variable was omitted. Missing data/profile validation fails
    before committing; there is no fallback to the unbridged ledger.
    """
    root = Path(root).resolve()
    positions_file = root / 'current_positions.json'
    positions = json.loads(positions_file.read_text()) if positions_file.exists() else {}
    account_file = root / Path(ACCOUNT_FILE).name
    account = load_account(initial_capital, positions, account_file=account_file)
    reports = pd.read_csv(fills, dtype=str, keep_default_na=False).to_dict('records')
    if STATE_KEY in account and not os.environ.get('QUANT_TRADING_PROFILE'):
        os.environ['QUANT_TRADING_PROFILE'] = 'accepted465'
    activate_profile()
    cfg = load_config()
    costs = dict(commission=float(cfg.get('backtest.commission', .0005)),
                 stamp_tax=float(cfg.get('backtest.stamp_tax', .0005)))
    if accepted465_active() or STATE_KEY in account:
        # Require coverage through both persisted observations and actual new
        # fills. Never guess a fill date from its planned execution/import date.
        dates = [_day(value) for value in
                 (as_of, account.get('last_date'),
                  account.get(STATE_KEY, {}).get('observed_date')) if value]
        for report in reports:
            key = f"{report['run_id']}:{int(report['seq'])}"
            pending = account.get('pending_orders', {}).get(key)
            if pending is not None:
                dates.append(_day(pending['decision_date']))
                if float(report.get('volume', 0) or 0) > 0:
                    dates.append(_day(report.get('ts', '')))
        if not dates:
            raise ValueError('accepted465 reconciliation requires --as-of for an undated account')
        bridge = create_live_bridge(account, positions, max(dates), config=cfg, root=root)
        count = bridge.apply_reports(reports, **costs)
    else:
        count = apply_execution_reports(account, positions, reports, **costs)
    commit_json_files({positions_file: positions, account_file: account},
                      root / '.live_state_journal.json')
    return {'reports_applied': count, 'cash': account['cash'],
            'positions': len(positions), 'unresolved': len(unresolved_orders(account))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fills', required=True)
    parser.add_argument('--initial-capital', type=float, default=500000)
    parser.add_argument('--as-of', help='Required local session coverage date; actual fill dates also checked')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    with live_run_lock(project):
        print(json.dumps(reconcile(args.fills, project, args.initial_capital,
                                   as_of=args.as_of), ensure_ascii=False))
