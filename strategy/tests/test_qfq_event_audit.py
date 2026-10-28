"""The event audit must flag mismatches beyond two-cent rounding tolerance."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.audit_qfq_events import audit


class QFQEventAuditTests(unittest.TestCase):
    def test_fused_multiplier_takes_precedence_over_cash_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = data / 'stock_data/raw_data/000001'
            bt = data / 'stock_data/backtrader_data'
            actions = data / 'alternative_data'
            for path in (raw, bt, actions):
                path.mkdir(parents=True)
            pd.DataFrame({'code': ['000001'], 'ex_date': ['2025-01-06'],
                          's': [0.], 'D': [0.], 'mult': [2.04]}).to_pickle(actions / 'events_table.pkl')
            pd.DataFrame({'日期': ['2025-01-03', '2025-01-06'],
                          '收盘': [10., 5.]}).to_csv(raw / 'none.csv', index=False)
            pd.DataFrame({'datetime': ['2025-01-03', '2025-01-06'],
                          'close': [10., 10.20]}).to_csv(bt / '000001_qfq.csv', index=False)
            self.assertEqual(audit(data)['counts'].get('out_of_rounding_bound', 0), 0)

    def test_detects_bad_adjustment_but_accepts_correct_split_and_dividend(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = data / 'stock_data/raw_data/000001'
            bt = data / 'stock_data/backtrader_data'
            actions = data / 'alternative_data'
            for path in (raw, bt, actions):
                path.mkdir(parents=True)
            pd.DataFrame({'code': ['000001'], 'ex_date': ['2025-01-06'],
                          's': [1.], 'D': [.2]}).to_pickle(actions / 'events_table.pkl')
            pd.DataFrame({'日期': ['2025-01-03', '2025-01-06', '2025-01-07'],
                          '收盘': [10., 5., 5.1]}).to_csv(raw / 'none.csv', index=False)
            price = bt / '000001_qfq.csv'
            pd.DataFrame({'datetime': ['2025-01-03', '2025-01-06', '2025-01-07'],
                          'close': [10., 10.20, 10.40]}).to_csv(price, index=False)
            good = audit(data)
            self.assertEqual(good['counts']['audited_event_day'], 1)
            self.assertEqual(good['counts'].get('out_of_rounding_bound', 0), 0)
            pd.DataFrame({'datetime': ['2025-01-03', '2025-01-06', '2025-01-07'],
                          'close': [10., 9.90, 10.10]}).to_csv(price, index=False)
            bad = audit(data)
            self.assertEqual(bad['counts']['out_of_rounding_bound'], 1)
            self.assertEqual(bad['exceptions'][0]['reason'], 'event_return_mismatch')


if __name__ == '__main__':
    unittest.main()
