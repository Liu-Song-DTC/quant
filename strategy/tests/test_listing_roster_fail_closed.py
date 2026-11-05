"""Production listing gate must refuse stale or unavailable market rosters."""
import os
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'strategy'))
sys.path.insert(0, str(ROOT / 'data'))
import check_data_freshness as gate
from check_listing_reconciliation import load_roster


class FailClosedRosterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        sd = root / 'data' / 'stock_data'
        self.meta = sd / 'stock_metadata'
        self.meta.mkdir(parents=True)
        self.bt = sd / 'backtrader_data'
        self.bt.mkdir()
        self.today = self.meta / f'listed_codes_{date.today():%m%d}.csv'
        self.project = patch.object(gate, 'PROJECT', root)
        self.bt_patch = patch.object(gate, 'BT_DIR', self.bt)
        self.project.start()
        self.bt_patch.start()
        self.addCleanup(self.bt_patch.stop)
        self.addCleanup(self.project.stop)

    def test_missing_today_roster_refuses_even_if_old_roster_exists(self):
        (self.meta / 'listed_codes_1231.csv').write_text('000001\n')
        problem = gate._check_listing_reconciliation()
        self.assertIn('缺少今日名册', problem)

    def test_same_name_from_previous_year_cannot_pass(self):
        self.today.write_text('000001\n')
        old = time.time() - 370 * 86400
        os.utime(self.today, (old, old))
        self.assertIn('未在今天生成', gate._check_listing_reconciliation())

    def test_import_failure_refuses(self):
        self.today.write_text('000001\n')
        with patch.dict(sys.modules, {'check_listing_reconciliation': None}):
            self.assertIn('无法完成对账', gate._check_listing_reconciliation())

    def test_roster_shrinkage_refuses_before_missing_files(self):
        self.today.write_text(''.join(f'{i:06d}\n' for i in range(3500)))
        prior = self.meta / 'listed_codes_older.csv'
        prior.write_text(''.join(f'{i:06d}\n' for i in range(4000)))
        yesterday = time.time() - 86400
        os.utime(prior, (yesterday, yesterday))
        self.assertIn('名册不可信', gate._check_listing_reconciliation())

    def test_empty_roster_refuses(self):
        self.today.write_text('')
        self.assertIn('名册不可信', gate._check_listing_reconciliation())

    def test_reference_selection_uses_mtime_across_year_boundary(self):
        current = self.meta / 'listed_codes_0102.csv'
        current.write_text('000001\n')
        newer = self.meta / 'listed_codes_1231.csv'
        older = self.meta / 'listed_codes_1101.csv'
        newer.write_text('000001\n000002\n')
        older.write_text('000001\n000002\n000003\n')
        os.utime(newer, (time.time() - 86400,) * 2)
        os.utime(older, (time.time() - 2 * 86400,) * 2)
        _, reference = load_roster(current)
        self.assertEqual(reference, 2)
