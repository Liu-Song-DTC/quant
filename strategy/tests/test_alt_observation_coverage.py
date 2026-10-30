"""An archived event date must not be presented as evidence of supplier observation."""
import json
import tempfile
import unittest
from pathlib import Path

from strategy.analysis.audit_alt_observation_coverage import audit
from strategy.core.alt_data_versions import TABLES


class AltObservationCoverageTests(unittest.TestCase):
    def test_flags_versions_first_observed_after_backtest_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'alt_versions').mkdir()
            for filename in TABLES:
                stem = Path(filename).stem
                day = '2026-10-29' if stem != 'margin_daily' else '2020-01-01'
                (root / 'alt_versions' / f'{stem}.json').write_text(
                    json.dumps({'schema': 1, 'versions': [
                        {'observed_on': day, 'sha256': 'a' * 64}]}) + '\n')
            result = audit(root, '2021-01-01', '2026-10-28')
            self.assertEqual(result['tables_without_any_observed_version_by_end'], len(TABLES) - 1)
            coverage = {entry['table']: entry for entry in result['tables']}
            self.assertFalse(coverage['margin_daily.pkl']['entire_window_pre_observation'])
            self.assertTrue(coverage['northbound_daily.pkl']['entire_window_pre_observation'])
            with self.assertRaisesRegex(ValueError, 'end precedes'):
                audit(root, '2026-10-29', '2026-10-28')
            (root / 'alt_versions' / 'dragon_tiger.json').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'manifest'):
                audit(root, '2021-01-01', '2026-10-28')


if __name__ == '__main__':
    unittest.main()
