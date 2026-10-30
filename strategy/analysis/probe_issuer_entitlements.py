#!/usr/bin/env python3
"""Read-only settlement probe for *hypothetical raw shares* on record-date close.

No QFQ fill quantity is interpreted as real shares. Gross dividends are receivables,
not spendable cash; broker credit time and eventual dividend taxes remain unknown.
"""
import argparse
import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

if __package__:
    from .audit_issuer_settlement_terms import audit
else:
    from audit_issuer_settlement_terms import audit


def derive_entitlement(shares_on_record, bonus_per_share, cash_per_share):
    """Return new raw shares and gross dividend, without rounding away fractional rights."""
    if isinstance(shares_on_record, bool) or not isinstance(shares_on_record, int) or shares_on_record < 0:
        raise ValueError('record-date shares must be a nonnegative integer')
    try:
        bonus, cash = Decimal(str(bonus_per_share)), Decimal(str(cash_per_share))
    except InvalidOperation as exc:
        raise ValueError('invalid issuer terms') from exc
    if not bonus.is_finite() or not cash.is_finite() or bonus < 0 or cash < 0:
        raise ValueError('issuer terms must be finite and nonnegative')
    additional = shares_on_record * bonus
    if additional != additional.to_integral_value():
        raise ValueError('fractional bonus shares require issuer/broker cash-in-lieu rules')
    return int(additional), shares_on_record * cash


def make_probe(held, issuer, shares_on_record):
    matched, summary = audit(held, issuer)
    rows = []
    for row in matched.itertuples(index=False):
        new_shares, gross_receivable = derive_entitlement(
            shares_on_record, row.shareholder_bonus, row.shareholder_cash)
        rows.append({'code': row.code, 'ex_date': row.ex_date.date().isoformat(),
                     'record_date': row.record_date_issuer.date().isoformat(),
                     'hypothetical_raw_shares_on_record': shares_on_record,
                     'bonus_shares': new_shares,
                     'raw_shares_after_bonus': shares_on_record + new_shares,
                     'gross_cash_receivable': str(gross_receivable),
                     'announced_cash_payment_date': row.cash_payment_date.date().isoformat(),
                     'announced_bonus_trading_date': (
                         row.bonus_trading_date.date().isoformat()
                         if not pd.isna(row.bonus_trading_date) else ''),
                     'cash_available_before_open': 'UNKNOWN',
                     'cash_dividend_tax': 'UNKNOWN'})
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--issuer', type=Path, required=True)
    parser.add_argument('--shares-on-record', type=int, default=100,
                        help='Illustrative integer raw shares per event; NOT simulated historical holdings')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, summary = make_probe(args.held, args.issuer, args.shares_on_record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({**summary, 'probe_rows': len(rows),
                      'hypothetical_raw_shares_per_event': args.shares_on_record}, ensure_ascii=False))


if __name__ == '__main__':
    main()
