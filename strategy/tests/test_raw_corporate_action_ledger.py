"""Hand-checkable synthetic fixtures, not issuer or broker timing evidence."""
import unittest

from strategy.core.raw_corporate_action_ledger import IssuerAction, RawCorporateActionLedger


class RawCorporateActionLedgerTests(unittest.TestCase):
    def test_record_holder_receives_bonus_and_dividend_without_early_spendable_cash(self):
        action = IssuerAction('999901', '2023-06-20', '2023-06-21',
                              0.5, 0.4, '2023-06-21', '2023-06-22', '2023-06-22')
        book = RawCorporateActionLedger(2000)
        book.advance('2023-06-20')
        book.fill('999901', 'BUY', 100, 10, fees=5)
        book.capture_at_record_close(action)
        self.assertEqual(book.equity({'999901': 10}), 1995)
        book.advance('2023-06-21')
        self.assertEqual(book.shares['999901'], 100)
        self.assertEqual(book.restricted_bonus['999901'], 50)
        self.assertEqual(book.cash_receivable, 40)
        self.assertEqual(book.cash, 995)
        self.assertAlmostEqual(book.equity({'999901': 6.4}), 1995)
        with self.assertRaisesRegex(ValueError, 'untradable'):
            book.fill('999901', 'SELL', 150, 6.4)
        with self.assertRaisesRegex(ValueError, 'insufficient cash'):
            book.fill('000001', 'BUY', 100, 10)
        book.advance('2023-06-22')
        self.assertEqual(book.shares['999901'], 150)
        self.assertEqual(book.restricted_bonus['999901'], 0)
        self.assertEqual(book.cash_receivable, 0)
        self.assertEqual(book.cash, 1035)
        self.assertAlmostEqual(book.equity({'999901': 6.4}), 1995)
        book.fill('999901', 'SELL', 150, 6.4, fees=5)
        self.assertEqual(book.cash, 1990)

    def test_entitlement_fixed_at_record_close_even_if_original_shares_sold(self):
        action = IssuerAction('999902', '2026-05-20', '2026-05-21',
                              0, 7, '2026-05-21', '2026-05-22', None)
        book = RawCorporateActionLedger(3000)
        book.advance('2026-05-20')
        book.fill('999902', 'BUY', 100, 20)
        book.capture_at_record_close(action)
        book.advance('2026-05-21')
        book.fill('999902', 'SELL', 100, 13)
        self.assertEqual(book.cash_receivable, 700)
        self.assertEqual(book.cash, 2300)
        book.advance('2026-05-22')
        self.assertEqual(book.cash, 3000)
        self.assertEqual(book.cash_receivable, 0)

    def test_rejects_fractional_bonus_and_unknown_availability(self):
        book = RawCorporateActionLedger(1000)
        book.advance('2026-05-20')
        with self.assertRaises(ValueError):
            IssuerAction('999903', '2026-05-20', '2026-05-21',
                         .4, .3, '2026-05-21', '2026-05-20', '2026-05-21')
        fractional = IssuerAction('999903', '2026-05-20', '2026-05-21',
                                  .345, .3, '2026-05-21', '2026-05-22', '2026-05-21')
        book.fill('999903', 'BUY', 100, 5)
        with self.assertRaisesRegex(ValueError, 'fractional bonus'):
            book.capture_at_record_close(fractional)


if __name__ == '__main__':
    unittest.main()
