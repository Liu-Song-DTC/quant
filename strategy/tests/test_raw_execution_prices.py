"""Missing bars must never be converted into executable orders."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.core.raw_execution_prices import load_raw_ohlc


class RawExecutionPricesTests(unittest.TestCase):
    def test_unadjusted_bar_alignment_and_suspension_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'none.csv'
            pd.DataFrame({'股票代码': ['999901', '999901'],
                          '日期': ['2026-05-20', '2026-05-22'],
                          '开盘': [20.0, 19.4], '最高': [20.5, 19.8],
                          '最低': [19.5, 19.1], '收盘': [20.0, 19.3]}).to_csv(source, index=False)
            bars = load_raw_ohlc(source, pd.date_range('2026-05-20', periods=3))
            self.assertEqual(bars.loc['2026-05-20', 'open'], 20)
            self.assertTrue(bars.loc['2026-05-21', ['open', 'low', 'close']].isna().all())
            self.assertEqual(bars.loc['2026-05-22', 'close'], 19.3)

    def test_duplicate_or_broken_bar_fails_before_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'none.csv'
            days = pd.DatetimeIndex(['2026-05-20', '2026-05-21'])
            pd.DataFrame({'股票代码': ['999901', '999901'],
                          '日期': ['2026-05-20', '2026-05-20'],
                          '开盘': [10, 10], '最高': [11, 11], '最低': [9, 9],
                          '收盘': [10, 10]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_raw_ohlc(source, days)
            pd.DataFrame({'股票代码': ['999901'], '日期': ['2026-05-20'],
                          '开盘': [12], '最高': [11], '最低': [9],
                          '收盘': [10]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'invalid raw OHLC'):
                load_raw_ohlc(source, days)
            pd.DataFrame({'股票代码': ['999901'], '日期': ['2026-05-20'],
                          '开盘': [None], '最高': [None], '最低': [None],
                          '收盘': [None]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'invalid raw OHLC'):
                load_raw_ohlc(source, days)
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                load_raw_ohlc(source, days)

    def test_rejects_a_different_security_under_the_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / '999901'
            folder.mkdir()
            source = folder / 'none.csv'
            pd.DataFrame({'股票代码': ['999902'], '日期': ['2026-05-20'],
                          '开盘': [10], '最高': [11], '最低': [9],
                          '收盘': [10]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_raw_ohlc(source, pd.DatetimeIndex(['2026-05-20']))


if __name__ == '__main__':
    unittest.main()
