"""Verify executed cash flows, carried positions, and snapshot drift detection."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.attribute_drawdown import attribute


class DrawdownAttributionTests(unittest.TestCase):
    def test_flat_account_has_zero_contributions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'bars'
            data.mkdir()
            pd.DataFrame({'date': ['2025-06-02', '2025-06-03'], 'nav': [100., 100.]}).to_csv(
                root / 'equity_curve.csv', index=False)
            pd.DataFrame(columns=['date', 'code', 'side', 'shares', 'gross_notional',
                                  'commission', 'stamp_tax', 'impact_cost']).to_csv(
                                      root / 'execution_fills.csv', index=False)
            detail, report = attribute(root, data, '2025-06-02', '2025-06-03')
            self.assertTrue(detail.empty)
            self.assertEqual(report['nav_delta'], 0.)

    def test_reconciles_holding_mark_and_sale_cost_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, data = root / 'result', root / 'bars'
            result.mkdir()
            data.mkdir()
            pd.DataFrame({'date': ['2025-06-02', '2025-06-03', '2025-06-04'],
                          'nav': [1000., 1099., 1197.]}).to_csv(result / 'equity_curve.csv', index=False)
            pd.DataFrame({'date': ['2025-06-02', '2025-06-04'], 'code': ['000001'] * 2,
                          'side': ['BUY', 'SELL'], 'shares': [100, 100],
                          'gross_notional': [1000., 1200.], 'commission': [1., 1.],
                          'stamp_tax': [0., 1.], 'impact_cost': [0., 0.]}).to_csv(
                              result / 'execution_fills.csv', index=False)
            bars = pd.DataFrame({'datetime': ['2025-06-02', '2025-06-03', '2025-06-04'],
                                 'close': [10., 11., 12.]})
            bars.to_csv(data / '000001_qfq.csv', index=False)
            detail, report = attribute(result, data, '2025-06-03', '2025-06-04')
            self.assertAlmostEqual(detail.pnl.iat[0], 98.)
            self.assertAlmostEqual(report['residual'], 0.)
            bars.loc[1, 'close'] = 10.
            bars.to_csv(data / '000001_qfq.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'do not reconcile'):
                attribute(result, data, '2025-06-03', '2025-06-04')


if __name__ == '__main__':
    unittest.main()
