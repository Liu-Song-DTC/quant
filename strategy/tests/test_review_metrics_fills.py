"""Short-horizon research metrics must include missed slots and booked entry costs."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.review_metrics import forward_labels, summarize_labels


class ExecutedShortHorizonTests(unittest.TestCase):
    def test_missed_buy_remains_cash_slot_and_executed_cost_is_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / 'results'
            data = root / 'bars'
            results.mkdir()
            data.mkdir()
            days = pd.date_range('2025-06-02', periods=4, freq='B')
            pd.DataFrame({'datetime': days}).to_csv(data / 'sh000001_qfq.csv', index=False)
            for code in ('000001', '000002'):
                pd.DataFrame({'datetime': days, 'open': [10., 10., 10., 10.],
                              'close': [10., 10., 11., 11.],
                              'volume': [1000] * 4}).to_csv(data / f'{code}_qfq.csv', index=False)
            pd.DataFrame({'date': [days[0]] * 2, 'code': ['000001', '000002']}).to_csv(
                results / 'portfolio_selections.csv', index=False)
            pd.DataFrame({'date': [days[1]], 'code': ['000001'], 'side': ['BUY'],
                          'shares': [100], 'gross_notional': [1000.], 'commission': [1.],
                          'impact_cost': [2.]}).to_csv(results / 'execution_fills.csv', index=False)
            marks = forward_labels(results, data, horizons=(2,), entry_basis='fills')
            summary = summarize_labels(marks, bootstrap_samples=0)['2']
            self.assertEqual(len(marks), 2)
            self.assertEqual(summary['entry_unavailable'], 1)
            self.assertEqual(summary['entry_available_samples'], 1)
            self.assertAlmostEqual(marks.loc[marks.code == '000001', 'net_mtm_return'].iat[0],
                                   11 * (1 - .001) * (1 - .0005 - .0005) / (1003 / 100) - 1)
            self.assertEqual(summary['net_mtm_win_rate'], 1.)
            self.assertEqual(summary['fixed_slot_win_rate'], .5)
            self.assertAlmostEqual(summary['fixed_slot_net_return'],
                                   marks['net_mtm_return'].dropna().iat[0] / 2)
            simulated = forward_labels(results, data, horizons=(2,))
            self.assertEqual(simulated['entry_available'].sum(), 2)
            self.assertNotEqual(simulated.net_mtm_return.iat[0], marks.net_mtm_return.iat[0])

    def test_invalid_entry_basis_fails_closed(self):
        with self.assertRaises(ValueError):
            forward_labels('/missing', '/missing', entry_basis='unknown')


if __name__ == '__main__':
    unittest.main()
