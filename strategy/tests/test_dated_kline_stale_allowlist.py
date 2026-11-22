"""A verified halt must not grant a permanent stale-price exemption."""
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_data_freshness as fresh


class DatedStaleAllowlistTests(unittest.TestCase):
    def test_halt_cannot_excuse_missing_pre_halt_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);allow=folder/'allow.txt'
            allow.write_text('002122 halt until=2026-11-20 last_trade=2026-11-19\n')
            price=folder/'002122_qfq.csv'
            price.write_text('datetime,open,high,low,close\n2026-11-13,1,1,1,1\n')
            with patch.object(fresh,'KLINE_ALLOWLIST',allow),patch.object(fresh,'BT_DIR',folder):
                self.assertEqual(fresh._load_kline_allowlist(date(2026,11,20)),set())
                price.write_text('datetime,open,high,low,close\n2026-11-19,1,1,1,1\n')
                self.assertEqual(fresh._load_kline_allowlist(date(2026,11,20)),{'002122'})
                self.assertEqual(fresh._load_kline_allowlist(date(2026,11,23)),set())

    def test_dated_exemption_expires_and_bad_dates_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'allow.txt'
            file.write_text('688121 halt until=2026-10-30\n'
                            '688122 halt until=not-a-date\n'
                            '688123 permanent legacy entry\n', encoding='utf-8')
            with patch.object(fresh, 'KLINE_ALLOWLIST', file):
                self.assertEqual(fresh._load_kline_allowlist(date(2026, 10, 30)),
                                 {'688121', '688123'})
                self.assertEqual(fresh._load_kline_allowlist(date(2026, 11, 2)),
                                 {'688123'})


if __name__ == '__main__':
    unittest.main()
