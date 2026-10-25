#!/usr/bin/env python3
"""Import broker execution reports without running signals or issuing orders."""
import argparse
import json
from pathlib import Path

import pandas as pd

from core.execution_ledger import apply_execution_reports, unresolved_orders
from core.live_account import load_account, ACCOUNT_FILE
from core.live_state import commit_json_files, live_run_lock
from core.config_loader import load_config


def reconcile(fills, root, initial_capital=500000):
    root = Path(root).resolve()
    positions_file = root / 'current_positions.json'
    positions = json.loads(positions_file.read_text()) if positions_file.exists() else {}
    account_file = root / Path(ACCOUNT_FILE).name
    account = load_account(initial_capital, positions, account_file=account_file)
    reports = pd.read_csv(fills, dtype=str, keep_default_na=False).to_dict('records')
    cfg = load_config()
    count = apply_execution_reports(account, positions, reports,
                                   commission=float(cfg.get('backtest.commission', .0005)),
                                   stamp_tax=float(cfg.get('backtest.stamp_tax', .0005)))
    commit_json_files({positions_file: positions, account_file: account},
                      root / '.live_state_journal.json')
    return {'reports_applied': count, 'cash': account['cash'],
            'positions': len(positions), 'unresolved': len(unresolved_orders(account))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fills', required=True)
    parser.add_argument('--initial-capital', type=float, default=500000)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    with live_run_lock(project):
        print(json.dumps(reconcile(args.fills, project, args.initial_capital), ensure_ascii=False))
