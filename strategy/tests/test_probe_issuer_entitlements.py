"""Hypothetical entitlement accounting must not confuse issuer terms and reference prices."""
import unittest
from decimal import Decimal

from strategy.analysis.probe_issuer_entitlements import derive_entitlement


class ProbeEntitlementsTests(unittest.TestCase):
    def test_stock_and_cash_are_separate_entitlements(self):
        self.assertEqual(derive_entitlement(100, '0.4', '0.3'), (40, Decimal('30.0')))
        self.assertEqual(derive_entitlement(100, '0', '7.0'), (0, Decimal('700.0')))
        self.assertEqual(derive_entitlement(0, '0.4', '0.3'), (0, Decimal('0.0')))

    def test_fractional_shares_and_invalid_inputs_fail_closed(self):
        for shares, bonus, cash in [(1, '0.3', '0.15'), (-1, '0', '0'),
                                    (100, '-0.3', '0.15'), (100, 'nan', '0')]:
            with self.subTest(shares=shares, bonus=bonus, cash=cash):
                with self.assertRaises(ValueError):
                    derive_entitlement(shares, bonus, cash)


if __name__ == '__main__':
    unittest.main()
