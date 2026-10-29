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
