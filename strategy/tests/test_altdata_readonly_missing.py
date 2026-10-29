"""Read-only backtest/order execution must never fetch missing alternative data."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy.core import alternative_data
from strategy.core.portfolio import PortfolioConstructor


class AlternativeDataReadOnlyTests(unittest.TestCase):
    def test_missing_sources_fail_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = alternative_data.AlternativeDataProvider(directory)
            with patch.object(alternative_data, '_AUTOREFRESH_OFF', True):
                for loader in (provider.load_dragon_tiger, provider.load_northbound,
                               provider.load_margin, provider.load_reduction,
                               provider.load_unlock, provider.load_yjyg,
                               provider.load_reduction_plans,
                               lambda: provider.get_dragon_tiger_dates('000001', '2026-10-28')):
                    with self.subTest(loader=loader):
                        with self.assertRaisesRegex(FileNotFoundError, 'read-only alternative data'):
                            loader()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_reduction_data_never_turns_off_enabled_portfolio_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            portfolio = PortfolioConstructor.__new__(PortfolioConstructor)
            portfolio._alt_provider = alternative_data.AlternativeDataProvider(directory)
            portfolio.exclude_reduction = True
            portfolio.exclude_reduction_plans = True
            portfolio._reduction_codes_cache_date = None
            portfolio._plan_codes_cache_date = None
            portfolio.reduction_recent_days = 30
            with patch.object(alternative_data, '_AUTOREFRESH_OFF', True):
                with self.assertRaisesRegex(RuntimeError, '减持过滤数据不可读'):
                    portfolio._reduction_codes('2026-10-28')
                with self.assertRaisesRegex(RuntimeError, '减持计划数据不可读'):
                    portfolio._reduction_plan_codes('2026-10-28')

    def test_stale_published_file_is_read_without_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reduction_records.pkl'
            pd.DataFrame({'code': ['000001'], 'eitime': [pd.Timestamp('2021-01-01')]}).to_pickle(path)
            provider = alternative_data.AlternativeDataProvider(directory)
            with (patch.object(alternative_data, '_AUTOREFRESH_OFF', True),
                  patch.object(provider, '_fetch_reduction_em', side_effect=AssertionError('network'))):
                self.assertEqual(provider.load_reduction()['code'].tolist(), ['000001'])


if __name__ == '__main__':
    unittest.main()
