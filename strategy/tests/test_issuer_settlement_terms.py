"""Issuer evidence must not silently rewrite accounting for frozen events."""
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.audit_issuer_settlement_terms import audit, guard_current_events


class IssuerSettlementTests(unittest.TestCase):
    def test_cross_check_and_reject_changed_dividend_or_payment_day(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held, source = root / 'held.csv', root / 'issuer.csv'
            pd.DataFrame({'code': ['300580', '300953'],
                          'ex_date': ['2023-06-21', '2026-05-28'],
                          'record_date': ['2023-06-20', '2026-05-27'],
                          's': [0.5, 0.4], 'D': [0.4, 0.3]}).to_csv(held, index=False)
            rows = pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                                 'record_date': ['2023-06-20'],
                                 'shareholder_bonus': ['0.5'], 'shareholder_cash': ['0.4'],
                                 'cash_payment_date': ['2023-06-21'],
                                 'bonus_trading_date': ['2023-06-21'],
                                 'source_url': ['https://static.cninfo.com.cn/notice.pdf']})
            rows.to_csv(source, index=False)
            checked, report = audit(held, source)
            self.assertEqual((report['events_with_issuer_verified_terms'],
                              report['events_without_issuer_verified_terms'],
                              report['issuer_verified_bonus_events']), (1, 1, 1))
            self.assertEqual(checked.shareholder_cash.tolist(), [0.4])
            rows.loc[0, 'shareholder_cash'] = '0.5'
            rows.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'conflict'):
                audit(held, source)
            rows.loc[0, 'shareholder_cash'] = '0.4'
            rows.loc[0, 'cash_payment_date'] = '2023-06-20'
            rows.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'payment date'):
                audit(held, source)

    def test_bonus_requires_issuer_trading_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held, source = root / 'held.csv', root / 'issuer.csv'
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          'record_date': ['2023-06-20'], 's': [0.5], 'D': [0.4]}).to_csv(held, index=False)
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          'record_date': ['2023-06-20'], 'shareholder_bonus': ['0.5'],
                          'shareholder_cash': ['0.4'], 'cash_payment_date': ['2023-06-21'],
                          'bonus_trading_date': [''],
                          'source_url': ['https://static.cninfo.com.cn/notice.pdf']}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'bonus trading date'):
                audit(held, source)

    def test_refresh_guard_stops_missing_or_revised_issuer_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, issuer = root / 'events.pkl', root / 'issuer.csv'
            terms = pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                                  'record_date': ['2023-06-20'], 'shareholder_bonus': ['0.5'],
                                  'shareholder_cash': ['0.4'], 'cash_payment_date': ['2023-06-21'],
                                  'bonus_trading_date': ['2023-06-21'],
                                  'source_url': ['https://static.cninfo.com.cn/notice.pdf']})
            terms.to_csv(issuer, index=False)
            frame = pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                                  's': [0.5], 'D': [0.4]})
            frame.to_pickle(events)
            self.assertEqual(guard_current_events(events, issuer)['issuer_verified_events_checked'], 1)
            frame.loc[0, 'D'] = 0.5
            frame.to_pickle(events)
            with self.assertRaisesRegex(ValueError, 'stop price rebuild'):
                guard_current_events(events, issuer)
            frame.iloc[0:0].to_pickle(events)
            with self.assertRaisesRegex(ValueError, 'missing'):
                guard_current_events(events, issuer)

    def test_refresh_guard_checks_provider_date_when_sidecar_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, issuer, parts = root / 'events.pkl', root / 'issuer.csv', root / 'parts'
            parts.mkdir()
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          's': [0.5], 'D': [0.4]}).to_pickle(events)
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          'record_date': ['2023-06-20'], 'shareholder_bonus': ['0.5'],
                          'shareholder_cash': ['0.4'], 'cash_payment_date': ['2023-06-21'],
                          'bonus_trading_date': ['2023-06-21'],
                          'source_url': ['https://static.cninfo.com.cn/notice.pdf']}).to_csv(issuer, index=False)
            self.assertEqual(guard_current_events(events, issuer, parts)['provider_dates_cross_checked'], 0)
            with (parts / '300580.pkl').open('wb') as file:
                pickle.dump([('300580', pd.Timestamp('2023-06-21'),
                              pd.Timestamp('2023-06-19'))], file)
            with self.assertRaisesRegex(ValueError, 'provider record date conflicts'):
                guard_current_events(events, issuer, parts)
            with (parts / '300580.pkl').open('wb') as file:
                pickle.dump([('300580', pd.Timestamp('2023-06-21'),
                              pd.Timestamp('2023-06-20'))], file)
            self.assertEqual(guard_current_events(events, issuer, parts)['provider_dates_cross_checked'], 1)


if __name__ == '__main__':
    unittest.main()
