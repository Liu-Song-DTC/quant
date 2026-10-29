"""Date-level validation of immutable before/after refresh inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.compare_refresh_snapshots import compare


class RefreshComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.before = self.root / 'before'
        self.after = self.root / 'after'
        self.before.mkdir()
        self.after.mkdir()

    def seal(self, root, files):
        records = {}
        for name, content in files.items():
            path = root / 'data' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            records[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (root / 'data_snapshot.json').write_text(json.dumps({'sha256': 'fixture', 'files': records}))

    def test_new_rows_preserve_history_and_vendor_restatement_is_reported(self):
        raw = '股票代码,日期,收盘\n1,2026-10-28,10\n'
        new_raw = raw + '1,2026-10-29,11\n'
        bt = 'datetime,close\n2026-10-28,10\n'
        new_bt = bt + '2026-10-29,11\n'
        vendor = '日期,收盘\n2026-10-28,10\n'
        vendor_changed = '日期,收盘\n2026-10-28,9\n2026-10-29,11\n'
        keys = ['stock_data/raw_data/000001/none.csv',
                'stock_data/backtrader_data/000001_qfq.csv',
                'stock_data/raw_data/000001/qfq.csv']
        self.seal(self.before, dict(zip(keys, [raw, bt, vendor])))
        self.seal(self.after, dict(zip(keys, [new_raw, new_bt, vendor_changed])))
        report = compare(self.before, self.after)
        self.assertEqual(report['problem_count'], 0)
        self.assertEqual(report['totals']['raw_none']['added_dates'], 1)
        self.assertEqual(report['totals']['bt_qfq']['added_dates'], 1)
        self.assertEqual(report['totals']['vendor_qfq']['changed_dates'], 1)

    def test_old_price_revision_and_seal_tampering_fail(self):
        name = 'stock_data/raw_data/000001/none.csv'
        self.seal(self.before, {name: '日期,收盘\n2026-10-28,10\n'})
        self.seal(self.after, {name: '日期,收盘\n2026-10-28,9\n'})
        report = compare(self.before, self.after)
        self.assertEqual(report['problem_count'], 1)
        self.assertEqual(report['totals']['raw_none']['field:收盘'], 1)
        self.assertEqual(report['examples'][0]['examples'][0]['date'], '2026-10-28')
        (self.after / 'data' / name).write_text('日期,收盘\n2026-10-28,8\n')
        report = compare(self.before, self.after)
        self.assertEqual(report['problems'][0]['reason'], 'unreadable_or_corrupt')

    def test_duplicate_date_is_rejected(self):
        name = 'stock_data/backtrader_data/000001_qfq.csv'
        self.seal(self.before, {name: 'datetime,close\n2026-10-28,10\n'})
        self.seal(self.after, {name: 'datetime,close\n2026-10-28,10\n2026-10-28,11\n'})
        self.assertEqual(compare(self.before, self.after)['problems'][0]['reason'],
                         'unreadable_or_corrupt')


if __name__ == '__main__':
    unittest.main()
