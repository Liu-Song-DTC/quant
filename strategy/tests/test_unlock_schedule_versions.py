"""Vendor unlock revisions must not change previous decision-day sets."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.core.unlock_schedule_versions import record_version, resolve_asof, verify_current
from strategy.core.alternative_data import AlternativeDataProvider
from strategy.core.portfolio import PortfolioConstructor


class UnlockScheduleVersionsTests(unittest.TestCase):
    def test_missing_manifest_is_not_a_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'unlock_schedule.pkl'
            pd.DataFrame({'code': ['000001']}).to_pickle(src)
            with self.assertRaisesRegex(FileNotFoundError, 'manifest missing'):
                resolve_asof(src, '2026-10-28')

    def test_snapshot_selection_and_pre_history_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'unlock_schedule.pkl'
            def write(rows):
                pd.DataFrame(rows, columns=['code', 'unlock_date', 'ratio']).assign(
                    unlock_date=lambda d: pd.to_datetime(d['unlock_date'])).to_pickle(src)
            write([('000001', '2026-11-01', .10)])
            self.assertEqual(record_version(src, '2026-10-28'), 'added')
            frozen = resolve_asof(src, '2021-01-01')
            self.assertEqual(record_version(src, '2026-10-28'), 'same')
            write([('000002', '2026-11-01', .10)])
            self.assertEqual(record_version(src, '2026-10-29'), 'added')
            self.assertEqual(resolve_asof(src, '2026-10-28'), frozen)
            self.assertNotEqual(resolve_asof(src, '2026-10-29'), frozen)
            provider = AlternativeDataProvider(directory)
            self.assertEqual(provider.get_unlock_codes('2026-10-28'), {'000001'})
            self.assertEqual(verify_current(src), '2026-10-29')
            self.assertEqual(provider.get_unlock_codes('2026-10-29'), {'000002'})
            self.assertEqual(provider.get_unlock_codes('2026-10-28'), {'000001'})

    def test_missing_snapshot_fails_instead_of_falling_back_to_current(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'unlock_schedule.pkl'
            pd.DataFrame({'code': ['000001']}).to_pickle(src)
            record_version(src, '2026-10-28')
            next((Path(directory) / 'unlock_versions').glob('*.pkl')).unlink()
            with self.assertRaisesRegex(ValueError, 'missing/corrupt'):
                resolve_asof(src, '2026-10-28')
            with self.assertRaisesRegex(ValueError, 'missing/corrupt'):
                verify_current(src)
            portfolio = PortfolioConstructor.__new__(PortfolioConstructor)
            portfolio.exclude_unlock = True
            portfolio._alt_provider = AlternativeDataProvider(directory)
            portfolio._unlock_codes_cache_date = None
            portfolio.unlock_ahead_days = 30
            portfolio.unlock_min_ratio = .05
            with self.assertRaisesRegex(RuntimeError, '解禁数据版本不可读'):
                portfolio._unlock_codes('2026-10-28')


if __name__ == '__main__':
    unittest.main()
