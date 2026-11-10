"""A partial/failed/test K-line fetch cannot masquerade as a completed refresh."""
import json
import sys
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import akshare_kline_downloader as downloader


def listed_codes(*, return_snapshot=False):
    return ({'000001'}, None) if return_snapshot else {'000001'}


class KlineStampTests(unittest.TestCase):
    def test_completed_stamps_only(self):
        stamp = {'roster_sha256': 'hash', 'synced_through': '2026-10-28',
                 'listed': 2, 'status_counts': {'skip': 2}}
        matches = downloader._sync_stamp_matches
        self.assertTrue(matches(stamp, 'hash', '2026-10-28'))
        self.assertFalse(matches(stamp, 'hash', '2026-10-27'))
        self.assertFalse(matches(dict(stamp, listed=3), 'hash', '2026-10-28'))
        self.assertFalse(matches(dict(stamp, status_counts={'skip': 1, 'FAIL:timeout': 1}),
                                 'hash', '2026-10-28'))
        self.assertFalse(matches(dict(stamp, status_counts={}), 'hash', '2026-10-28'))
        self.assertFalse(matches(dict(stamp, status_counts={'skip': 1}), 'hash', '2026-10-28'))

    def test_failed_refresh_cannot_write_success_marker_or_skip_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = Path(tmp) / 'raw', Path(tmp) / 'meta'
            raw.mkdir()
            meta.mkdir()
            with patch.object(downloader, 'RAW', str(raw)), \
                 patch.object(downloader, 'META', str(meta)), \
                 patch.object(downloader, 'last_complete_trading_day', return_value='2026-10-28'), \
                 patch.object(downloader, 'install_patch'), \
                 patch.object(downloader, 'download_index'), \
                 patch.object(downloader, 'get_listed_codes', side_effect=listed_codes), \
                 patch.dict(sys.modules, {'record_st_status': SimpleNamespace(record_listing_status=lambda _: 0)}), \
                 patch.object(downloader, 'incremental', side_effect=RuntimeError('offline')), \
                 patch.object(downloader, 'full_download', side_effect=RuntimeError('offline')):
                with self.assertRaises(SystemExit):
                    downloader.run()
                stamps = list(meta.glob('kline_sync_*.stamp'))
                self.assertFalse(stamps)
                with self.assertRaises(SystemExit):
                    downloader.run()
                self.assertFalse(stamps)

    def test_forced_refresh_failure_revokes_old_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = Path(tmp) / 'raw', Path(tmp) / 'meta'
            raw.mkdir()
            meta.mkdir()
            with patch.object(downloader, 'RAW', str(raw)), \
                 patch.object(downloader, 'META', str(meta)), \
                 patch.object(downloader, 'last_complete_trading_day', return_value='2026-10-28'), \
                 patch.object(downloader, 'install_patch'), \
                 patch.object(downloader, 'download_index'), \
                 patch.object(downloader, 'get_listed_codes', side_effect=listed_codes), \
                 patch.dict(sys.modules, {'record_st_status': SimpleNamespace(record_listing_status=lambda _: 0)}), \
                 patch.object(downloader, 'incremental', return_value='skip') as incremental, \
                 patch.object(downloader, 'full_download', side_effect=RuntimeError('offline')):
                downloader.run()
                stamp = next(meta.glob('kline_sync_*.stamp'))
                self.assertTrue(stamp.exists())
                incremental.side_effect = RuntimeError('offline')
                with patch.dict('os.environ', {'QUANT_FORCE_FETCH': '1'}):
                    with self.assertRaises(SystemExit):
                        downloader.run()
                self.assertFalse(stamp.exists())
                with self.assertRaises(SystemExit):
                    downloader.run()

    def test_test_mode_does_not_consume_or_overwrite_complete_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = Path(tmp) / 'raw', Path(tmp) / 'meta'
            raw.mkdir()
            meta.mkdir()
            with patch.object(downloader, 'RAW', str(raw)), \
                 patch.object(downloader, 'META', str(meta)), \
                 patch.object(downloader, 'last_complete_trading_day', return_value='2026-10-28'), \
                 patch.object(downloader, 'install_patch'), \
                 patch.object(downloader, 'download_index'), \
                 patch.object(downloader, 'get_listed_codes', side_effect=listed_codes), \
                 patch.dict(sys.modules, {'record_st_status': SimpleNamespace(record_listing_status=lambda _: 0)}), \
                 patch.object(downloader, 'incremental', return_value='skip'):
                downloader.run()
                stamp = next(meta.glob('kline_sync_*.stamp'))
                before = stamp.read_bytes()
                self.assertEqual(json.loads(before)['listed'], 1)
                downloader.run(test_n=1)
                self.assertEqual(stamp.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
