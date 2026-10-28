"""Execution exposure must be shown separately from a cash-scaled target book."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.core.regime_reporting import regime_state_frame
from strategy.analysis.audit_executed_exposure import replay


class RegimeReportingTests(unittest.TestCase):
    def test_fill_replay_rejects_broken_nav_and_recovers_cash_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / 'report'
            report_dir.mkdir()
            bars = root / 'stock_data/backtrader_data'
            bars.mkdir(parents=True)
            pd.DataFrame({'datetime': ['2025-06-04', '2025-06-05'], 'close': [5., 6.]}).to_csv(
                bars / '000001_qfq.csv', index=False)
            pd.DataFrame({'date': ['2025-06-04', '2025-06-05'],
                          'nav': [999.75, 1099.75]}).to_csv(report_dir / 'equity_curve.csv', index=False)
            pd.DataFrame({'date': ['2025-06-04'], 'code': ['000001'], 'side': ['BUY'],
                          'shares': [100], 'gross_notional': [500.], 'commission': [.25],
                          'stamp_tax': [0.], 'impact_cost': [0.]}).to_csv(
                              report_dir / 'execution_fills.csv', index=False)
            pd.DataFrame({'date': ['2025-06-04', '2025-06-05'], 'regime': ['NORM', 'NORM'],
                          'exposure': [.5, 1.5]}).to_csv(report_dir / 'regime_state.csv', index=False)
            frame, report = replay(report_dir, root, 1000.)
            self.assertAlmostEqual(frame['cash'].iat[-1], 499.75)
            self.assertAlmostEqual(report['max_actual_exposure'], 600 / 1099.75)
            self.assertEqual(report['days_target_exposure_over_one'], 1)
            pd.DataFrame({'date': ['2025-06-04', '2025-06-05'],
                          'nav': [999.75, 1199.75]}).to_csv(report_dir / 'equity_curve.csv', index=False)
            with self.assertRaises(ValueError):
                replay(report_dir, root, 1000.)

    def test_actual_positions_are_not_mislabeled_as_targets(self):
        calendar = pd.to_datetime(['2025-06-04', '2025-06-05'])
        frame = regime_state_frame(calendar, [('NORM', 3, .82), ('NORM', 8, 1.497344)],
                                   np.array([.77, .897345]), np.array([3, 6]))
        self.assertEqual(frame['date'].tolist(), ['2025-06-04', '2025-06-05'])
        self.assertEqual(frame['n_pos'].tolist(), [3, 6])
        self.assertEqual(frame['target_n_pos'].tolist(), [3, 8])
        self.assertAlmostEqual(frame['exposure'].iat[1], .897345)
        self.assertAlmostEqual(frame['target_exposure'].iat[1], 1.497344)

    def test_rejects_leverage_and_misaligned_dates(self):
        date = pd.to_datetime(['2025-06-05'])
        with self.assertRaises(ValueError):
            regime_state_frame(date, [('NORM', 1, 1.2)], [1.1], [1])
        with self.assertRaises(ValueError):
            regime_state_frame(date, [], [.8], [1])


if __name__ == '__main__':
    unittest.main()
