"""A holder's cash and the exchange's virtual ex-dividend cash are distinct."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.audit_issuer_settlement_terms import audit, guard_current_events


class IssuerVirtualDividendTests(unittest.TestCase):
    def test_differential_cash_uses_reference_amount_for_price_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / 'events.pkl'
            held = root / 'held.csv'
            issuer = root / 'issuer.csv'
            pd.DataFrame({'code': ['600989'], 'ex_date': ['2026-04-28'],
                          's': [0.0], 'D': [0.42]}).to_pickle(events)
            pd.DataFrame({'code': ['600989'], 'ex_date': ['2026-04-28'],
                          'record_date': ['2026-04-27'], 's': [0.0],
                          'D': [0.42]}).to_csv(held, index=False)
            row = pd.DataFrame({'code': ['600989'], 'ex_date': ['2026-04-28'],
                                'record_date': ['2026-04-27'],
                                'shareholder_bonus': ['0'],
                                'shareholder_cash': ['0.4921'],
                                'price_reference_cash': ['0.42'],
                                'cash_payment_date': ['2026-04-28'],
                                'bonus_trading_date': [''],
                                'source_url': ['https://static.cninfo.com.cn/finalpage/2026-04-22/1225136155.PDF']})
            row.to_csv(issuer, index=False)
            _, report = audit(held, issuer)
            self.assertEqual(report['differential_cash_events'], 1)
            self.assertEqual(guard_current_events(events, issuer)['issuer_term_mismatches'], 0)
            row.loc[0, 'price_reference_cash'] = '0.4921'
            row.to_csv(issuer, index=False)
            with self.assertRaisesRegex(ValueError, 'conflict'):
                audit(held, issuer)
            with self.assertRaisesRegex(ValueError, 'stop price rebuild'):
                guard_current_events(events, issuer)


if __name__ == '__main__':
    unittest.main()
