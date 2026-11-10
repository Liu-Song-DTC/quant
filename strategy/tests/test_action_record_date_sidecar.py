"""Record-date enrichment must leave the frozen QFQ event format untouched."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data import fetch_actions_table as actions


class ActionRecordDateTests(unittest.TestCase):
    def detail(self, record='2025-06-12'):
        return pd.DataFrame({'除权除息日': ['2025-06-13'],
                             '股权登记日': [record],
                             '最新公告日期': ['2025-06-01'],
                             '送转股份-送转总比例': [4.0],
                             '现金分红-现金分红比例': [3.0]})

    def test_source_record_date_kept_separate_from_price_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            part, side = Path(tmp) / 'events', Path(tmp) / 'record_dates'
            part.mkdir()
            with patch.object(actions, 'PART', str(part)), \
                 patch.object(actions, 'RECORD_PART', str(side)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=self.detail()):
                self.assertEqual(actions.fetch_one('300953'), 1)
            with (part / '300953.pkl').open('rb') as file:
                original = pickle.load(file)
            with (side / '300953.pkl').open('rb') as file:
                record = pickle.load(file)
            self.assertEqual(original, [('300953', pd.Timestamp('2025-06-13'), .4, .3)])
            self.assertEqual(record, [('300953', pd.Timestamp('2025-06-13'),
                                       pd.Timestamp('2025-06-12'))])

    def test_partial_source_withdrawal_preserves_old_ex_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            part, side = Path(tmp) / 'events', Path(tmp) / 'record_dates'
            part.mkdir()
            old = [('300760', pd.Timestamp('2024-11-07'), 0.0, 1.65),
                   ('300760', pd.Timestamp('2025-06-13'), .4, .3)]
            with (part / '300760.pkl').open('wb') as file:
                pickle.dump(old, file)
            with patch.object(actions, 'PART', str(part)), \
                 patch.object(actions, 'RECORD_PART', str(side)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=self.detail()), \
                 patch('builtins.print') as log:
                self.assertEqual(actions.fetch_one('300760'), 2)
            with (part / '300760.pkl').open('rb') as file:
                restored = pickle.load(file)
            self.assertCountEqual(restored, old)
            self.assertIn('2024-11-07', str(log.call_args_list))

    def test_invalid_sidecar_does_not_disrupt_existing_price_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            part, side = Path(tmp) / 'events', Path(tmp) / 'record_dates'
            part.mkdir()
            broken = self.detail().drop(columns='股权登记日')
            with patch.object(actions, 'PART', str(part)), \
                 patch.object(actions, 'RECORD_PART', str(side)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=broken):
                self.assertEqual(actions.fetch_one('300953'), 1)
            self.assertTrue((part / '300953.pkl').exists())
            self.assertFalse((side / '300953.pkl').exists())

    def test_same_date_source_revision_preserves_first_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            part, side = Path(tmp) / 'events', Path(tmp) / 'record_dates'
            part.mkdir()
            old = [('300953', pd.Timestamp('2025-06-13'), .4, .3)]
            with (part / '300953.pkl').open('wb') as file:
                pickle.dump(old, file)
            revised = self.detail().copy()
            revised['现金分红-现金分红比例'] = 4.0
            with patch.object(actions, 'PART', str(part)), \
                 patch.object(actions, 'RECORD_PART', str(side)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=revised):
                self.assertEqual(actions.fetch_one('300953'), 1)
            with (part / '300953.pkl').open('rb') as file:
                self.assertEqual(pickle.load(file), old)

    def test_provider_revision_is_logged_but_does_not_rewrite_record_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            side = Path(tmp)
            with patch.object(actions, 'RECORD_PART', str(side)):
                actions._save_record_dates('300953', self.detail())
                with patch('builtins.print') as log:
                    actions._save_record_dates('300953', self.detail('2025-06-11'))
                self.assertIn('上游登记日改变', str(log.call_args_list))
                with (side / '300953.pkl').open('rb') as file:
                    self.assertEqual(pickle.load(file)[0][2], pd.Timestamp('2025-06-12'))

    def test_bad_record_date_is_rejected_from_sidecar(self):
        with self.assertRaisesRegex(ValueError, 'must precede ex-date'):
            actions._record_date_rows('300953', self.detail('2025-06-13'))

    def test_missing_source_column_fails_sidecar_only(self):
        with self.assertRaisesRegex(ValueError, 'record-date columns missing'):
            actions._record_date_rows('300953', self.detail().drop(columns='股权登记日'))


if __name__ == '__main__':
    unittest.main()
