import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

path = Path(__file__).resolve().parents[1] / 'analysis/review_annual_pair.py'
spec = importlib.util.spec_from_file_location('annual_pair', path)
annual_pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annual_pair)


class AnnualPairTests(unittest.TestCase):
    def test_year_boundary_equity_and_trade_exit_year_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'corrected').mkdir()
            pd.DataFrame({'date': ['2025-12-31', '2026-01-02', '2026-01-05'],
                          'nav': [100., 90., 95.]}).to_csv(root/'corrected/equity_curve.csv', index=False)
            pd.DataFrame({'entry_date': ['2025-12-30'], 'exit_date': ['2026-01-05'],
                          'code': ['000001'], 'ret': [-.05]}).to_csv(root/'corrected/trade_realized.csv', index=False)
            pd.DataFrame({'date': ['2026-01-02'], 'code': ['000001'], 'horizon': [3],
                          'entry_available': [True], 'net_mtm_return': [.02]}).to_csv(root/'labels.csv', index=False)
            years, _ = annual_pair.annual(root, root/'labels.csv', initial_cash=100.)
            self.assertAlmostEqual(years[-1]['max_drawdown'], -.10)
            self.assertAlmostEqual(years[-1]['return'], -.05)
            self.assertEqual(years[-1]['closed_trades'], 1)
            self.assertEqual(years[0]['closed_trades'], 0)
            self.assertEqual(years[-1]['win_3'], 1.)

    def test_drawdown_bridge_counts_existing_positions_new_trades_and_costs_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'corrected').mkdir()
            price_dir = root/'data/stock_data/backtrader_data'
            price_dir.mkdir(parents=True)
            pd.DataFrame({'datetime': ['2026-01-02', '2026-01-05'], 'close': [10., 9.]}).to_csv(
                price_dir/'000001_qfq.csv', index=False)
            pd.DataFrame([
                dict(date='2026-01-02', code='000001', side='BUY', shares=10, gross_notional=100.,
                     commission=1., stamp_tax=0., impact_cost=0., price_adjustment_cost=2.),
                dict(date='2026-01-05', code='000001', side='SELL', shares=4, gross_notional=36.,
                     commission=1., stamp_tax=.1, impact_cost=.2, price_adjustment_cost=3.),
            ]).to_csv(root/'corrected/execution_fills.csv', index=False)
            result = annual_pair.drawdown_contributions(root, dict(
                peak='2026-01-02', trough='2026-01-05', peak_nav=100., trough_nav=88.7))
            self.assertAlmostEqual(result['contributions'][0]['pnl'], -11.3)
            self.assertAlmostEqual(result['nav_change_reconciliation_error'], 0.)


if __name__ == '__main__':
    unittest.main()
