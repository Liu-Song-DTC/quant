"""QFQ signal shares must not be assumed to equal executable raw shares."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.audit_qfq_lot_sizes import audit


class QFQLotAuditTests(unittest.TestCase):
    def test_addon_below_real_lot_does_not_count_as_new_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, raw, qfq = root / 'result', root / 'raw', root / 'qfq'
            result.mkdir()
            (raw / '000001').mkdir(parents=True)
            qfq.mkdir()
            pd.DataFrame({'date': ['2025-01-06', '2025-01-07'],
                          'code': ['000001', '000001'], 'side': ['BUY', 'BUY'],
                          'shares': [300, 100], 'price': [10., 10.],
                          'price_adjustment_cost': [0., 0.]}).to_csv(
                              result / 'execution_fills.csv', index=False)
            pd.DataFrame({'日期': ['2025-01-06', '2025-01-07'],
                          '开盘': [20., 20.]}).to_csv(raw / '000001/none.csv', index=False)
            pd.DataFrame({'datetime': ['2025-01-06', '2025-01-07'],
                          'open': [10., 10.]}).to_csv(qfq / '000001_qfq.csv', index=False)
            detail, report = audit(result, raw, qfq)
            self.assertEqual(report['new_entries'], 1)
            self.assertEqual(report['new_entries_without_raw_lot'], 0)
            self.assertEqual(report['topups_without_raw_lot'], 1)
            self.assertEqual(detail.new_entry.tolist(), [True, False])

    def test_raw_lot_rounding_and_snapshot_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, raw, qfq = root / 'result', root / 'raw', root / 'qfq'
            result.mkdir()
            (raw / '000001').mkdir(parents=True)
            qfq.mkdir()
            pd.DataFrame({'date': ['2025-01-06'], 'code': ['000001'],
                          'side': ['BUY'], 'shares': [300], 'price': [10.01],
                          'price_adjustment_cost': [3.0]}).to_csv(
                              result / 'execution_fills.csv', index=False)
            pd.DataFrame({'日期': ['2025-01-06'], '开盘': [20.]}).to_csv(
                raw / '000001/none.csv', index=False)
            bars = pd.DataFrame({'datetime': ['2025-01-06'], 'open': [10.]})
            bars.to_csv(qfq / '000001_qfq.csv', index=False)
            detail, report = audit(result, raw, qfq)
            self.assertEqual(report['raw_price_over_qfq_2x'], 0)
            self.assertEqual(report['qfq_vs_raw_share_count_mismatch'], 1)
            self.assertEqual(detail.raw_shares_for_same_notional.iat[0], 100)
            self.assertAlmostEqual(detail.lot_rounding_cash.iat[0], 1000.)
            bars['open'] = 9.
            bars.to_csv(qfq / '000001_qfq.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'snapshot disagrees'):
                audit(result, raw, qfq)


if __name__ == '__main__':
    unittest.main()
