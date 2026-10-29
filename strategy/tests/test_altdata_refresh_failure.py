"""The dedicated daily refresh must not report stale reduction data as fresh."""
import tempfile
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy.core.alternative_data import AlternativeDataProvider


class AlternativeRefreshFailureTests(unittest.TestCase):
    def test_reduction_download_error_keeps_old_cache_but_fails_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'reduction_records.pkl'
            old = pd.DataFrame({'code': ['000001'], 'eitime': [pd.Timestamp('2026-01-01')]})
            old.to_pickle(source)
            before = source.read_bytes()
            yesterday = time.time() - 72 * 3600
            os.utime(source, (yesterday, yesterday))
            provider = AlternativeDataProvider(directory, strict_refresh=True)
            with patch.object(AlternativeDataProvider, '_fetch_reduction_em',
                              side_effect=OSError('offline')):
                with self.assertRaisesRegex(OSError, 'offline'):
                    provider.load_reduction()
                self.assertEqual(len(AlternativeDataProvider(directory).load_reduction()), 1)
            self.assertEqual(source.read_bytes(), before)

    def test_reduction_plan_month_failure_stops_strict_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'reduction_plans.pkl'
            old = pd.DataFrame({'ann_date': ['2026-01-01'], 'code': ['000001'],
                                'title': ['拟减持'], 'kind': ['start']})
            old.to_pickle(source)
            before = source.read_bytes()
            provider = AlternativeDataProvider(directory, strict_refresh=True)
            with patch('akshare.stock_fundamental.stock_notice._stock_notice_report',
                       side_effect=OSError('offline')):
                with self.assertRaisesRegex(RuntimeError, '月份 .* 拉取失败'):
                    provider.load_reduction_plans()
                self.assertEqual(len(AlternativeDataProvider(directory).load_reduction_plans()), 1)
            self.assertEqual(source.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
