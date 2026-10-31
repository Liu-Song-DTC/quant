#!/usr/bin/env python3
"""Read-only record-close to ex-close wealth bridge for issuer-verified events.

This tests price/bonus/gross-dividend arithmetic on a synthetic 100-share
holding. It does not model broker cash availability, tax, orders, or account NAV.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strategy.analysis.audit_issuer_settlement_terms import guard_current_events, load_issuer_terms
from strategy.core.raw_corporate_action_ledger import IssuerAction, RawCorporateActionLedger
from strategy.core.raw_execution_prices import load_raw_ohlc


def probe(issuer, events, raw_dir, qfq_dir):
    guard_current_events(events, issuer)
    issuer_rows = load_issuer_terms(issuer)
    rows = []
    for row in issuer_rows.itertuples(index=False):
        code = row.code
        calendar = pd.DatetimeIndex([row.record_date, row.ex_date])
        raw = load_raw_ohlc(Path(raw_dir) / code / 'none.csv', calendar)
        qfq = pd.read_csv(Path(qfq_dir) / f'{code}_qfq.csv',
                          usecols=['datetime', 'close'], parse_dates=['datetime'])
        if qfq.datetime.duplicated().any():
            raise ValueError(f'duplicate QFQ bar: {code}')
        qfq = qfq.set_index('datetime').reindex(calendar)
        before = float(raw.loc[row.record_date, 'close'])
        after = float(raw.loc[row.ex_date, 'close'])
        q_before = float(qfq.loc[row.record_date, 'close'])
        q_after = float(qfq.loc[row.ex_date, 'close'])
        if not all(pd.notna(v) and v > 0 for v in (before, after, q_before, q_after)):
            raise ValueError(f'missing or invalid price around issuer event: {code}')
        # Deliberately never spend the gross receivable in this two-date probe.
        # 2100-01-01 is a probe-only sentinel, not a claim about broker settlement.
        action = IssuerAction(code, row.record_date, row.ex_date,
                              float(row.shareholder_bonus), float(row.shareholder_cash),
                              row.cash_payment_date, date(2100, 1, 1),
                              None if pd.isna(row.bonus_trading_date) else row.bonus_trading_date)
        ledger = RawCorporateActionLedger(100 * before)
        ledger.advance(row.record_date)
        ledger.fill(code, 'BUY', 100, before)
        ledger.capture_at_record_close(action)
        ledger.advance(row.ex_date)
        raw_return = ledger.equity({code: after}) / (100 * before) - 1
        qfq_return = q_after / q_before - 1
        rows.append({'code': code, 'record_date': row.record_date.date().isoformat(),
                     'ex_date': row.ex_date.date().isoformat(),
                     'raw_record_close': before, 'raw_ex_close': after,
                     'bonus_tradable_on_ex': ledger.shares.get(code, 0) - 100,
                     'bonus_receivable_shares': ledger.restricted_bonus.get(code, 0),
                     'gross_cash_receivable': ledger.cash_receivable,
                     'raw_wealth_return': raw_return, 'qfq_close_return': qfq_return,
                     'difference_pp': (raw_return - qfq_return) * 100})
    result = pd.DataFrame(rows)
    if result.empty or result['difference_pp'].abs().max() > .5:
        raise ValueError('raw-wealth to QFQ bridge diverges beyond 0.5 percentage points')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issuer', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--qfq', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.issuer, args.events, args.raw, args.qfq)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(json.dumps({'events': len(result),
                      'max_abs_difference_pp': float(result.difference_pp.abs().max()),
                      'out': str(args.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
