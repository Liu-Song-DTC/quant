"""Live incremental bars must append while retaining published historical bytes."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from data import akshare_kline_downloader as downloader


CODE = '000001'


def bars(amount_old='123.4567', adjusted=False, include_new=True):
    close = [10.0, 10.5, 11.0, 11.5][:4 if include_new else 3]
    dates = ['2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29'][:len(close)]
    return pd.DataFrame({'股票名称': ['平安银行'] * len(close),
                         '股票代码': [CODE] * len(close), '日期': dates,
                         '收盘': close, '成交额': [amount_old] * len(close)})


class IncrementalRawTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / CODE).mkdir()
        self.patch_raw = patch.object(downloader, 'RAW', str(self.root))
        self.patch_raw.start()
        self.addCleanup(self.patch_raw.stop)
        self.original = {}
        for kind in ('none', 'qfq', 'hfq'):
            path = self.root / CODE / f'{kind}.csv'
            path.write_text('股票名称,股票代码,日期,收盘,成交额\n'
                            '平安银行,000001,2026-10-26,10.0,123.4567\n'
                            '平安银行,000001,2026-10-27,10.5,123.4567\n'
                            '平安银行,000001,2026-10-28,11.0,123.4567\n', encoding='utf-8')
            self.original[kind] = path.read_bytes()
        (self.root / CODE / 'fq_factors.csv').write_text(
            '日期,symbol,qfq_factor,hfq_factor\n'
            '2026-10-26,000001,1.0,1.0\n', encoding='utf-8')

    def test_normal_increment_preserves_all_historical_bytes_and_backfills_factors(self):
        old_factor = (self.root / CODE / 'fq_factors.csv').read_bytes()
        def fetch(code, beg, end, fqt):
            out = bars(amount_old='123.0')
            out.loc[len(out)-1, '成交额'] = '124.0'
            return out
        with patch.object(downloader, 'fetch_with_retry', side_effect=fetch):
            self.assertEqual(downloader.incremental(CODE, '20261029'), 'incr')
        for kind in ('none', 'qfq', 'hfq'):
            content = (self.root / CODE / f'{kind}.csv').read_bytes()
            self.assertTrue(content.startswith(self.original[kind]), kind)
            self.assertIn(b'000001,2026-10-29', content)
        factor = self.root / CODE / 'fq_factors.csv'
        self.assertTrue(factor.read_bytes().startswith(old_factor))
        with factor.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['日期'] for row in rows],
                         ['2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29'])
        for kind in ('none', 'qfq', 'hfq'):
            content = (self.root / CODE / f'{kind}.csv').read_bytes()
            self.assertEqual(downloader.incremental(CODE, '20261029'), 'skip')
            self.assertEqual((self.root / CODE / f'{kind}.csv').read_bytes(), content)

    def test_provider_failure_does_not_write_partial_price_files(self):
        def fetch(code, beg, end, fqt):
            if fqt == 2:
                raise RuntimeError('provider offline')
            return bars()
        with patch.object(downloader, 'fetch_with_retry', side_effect=fetch):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                downloader.incremental(CODE, '20261029')
        for kind in ('none', 'qfq', 'hfq'):
            self.assertEqual((self.root / CODE / f'{kind}.csv').read_bytes(), self.original[kind])

    def test_reanchor_full_reload_preserves_none_history_only(self):
        old_none = (self.root / CODE / 'none.csv').read_bytes()
        with patch.object(downloader, 'fetch_with_retry', return_value=bars(amount_old='999.0')):
            downloader.full_download(CODE, '20160101', '20261029',
                                     preserve_unadjusted_history=True)
        self.assertTrue((self.root / CODE / 'none.csv').read_bytes().startswith(old_none))
        with (self.root / CODE / 'qfq.csv').open() as f:
            values = list(csv.DictReader(f))
        self.assertEqual(values[0]['成交额'], '999.0')
        self.assertEqual(values[-1]['日期'], '2026-10-29')
        self.assertEqual((self.root / CODE / 'fq_factors.csv').read_text().splitlines()[-1].split(',')[0],
                         '2026-10-29')


if __name__ == '__main__':
    unittest.main()
