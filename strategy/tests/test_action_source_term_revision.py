"""Historical event terms stay stable when the provider revises one date."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data import fetch_actions_table as actions


class ActionSourceRevisionTests(unittest.TestCase):
    def test_multiple_distributions_same_day_survive_actual_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            day = pd.Timestamp('2026-06-01')
            prior = [('002043', day, 0., .132), ('002043', day, 0., .18)]
            with (parts / '002043.pkl').open('wb') as handle:
                pickle.dump(prior, handle)
            response = pd.DataFrame({
                '除权除息日': ['2026-06-01', '2026-06-01', '2026-11-09'],
                '送转股份-送转总比例': [0., 0., 0.],
                '现金分红-现金分红比例': [1.8, 1.32, 1.2],
            })
            with patch.object(actions, 'PART', str(parts)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions, '_save_record_dates', create=True), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=response):
                self.assertEqual(actions.fetch_one('002043'), 3)
            with (parts / '002043.pkl').open('rb') as handle:
                saved = pickle.load(handle)
            self.assertEqual(saved[:2], prior)
            self.assertAlmostEqual(sum(r[3] for r in saved[:2]), .312)
            self.assertEqual(saved[-1], ('002043', pd.Timestamp('2026-11-09'), 0., .12))

    def test_partial_withdrawal_or_revision_preserves_entire_group(self):
        day = pd.Timestamp('2026-05-28')
        prior = [('300760', day, 0., .31), ('300760', day, 0., 1.25)]
        for incoming in [prior[:1], prior[1:], [prior[0], ('300760', day, 0., 2.)]]:
            with self.subTest(incoming=incoming), patch('builtins.print'):
                self.assertEqual(actions._merge_saved_events('300760', incoming, prior), prior)

    def test_identical_distributions_preserve_multiplicity(self):
        row = ('000001', pd.Timestamp('2025-06-13'), 0., .2)
        with patch('builtins.print'):
            self.assertEqual(actions._merge_saved_events('000001', [row], [row, row]), [row, row])

    def test_same_ex_date_revision_keeps_previous_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            prior = [('300953', pd.Timestamp('2025-06-13'), .4, .3)]
            with (parts / '300953.pkl').open('wb') as handle:
                pickle.dump(prior, handle)
            response = pd.DataFrame({
                '除权除息日': ['2025-06-13'],
                '送转股份-送转总比例': [4.0],
                '现金分红-现金分红比例': [4.0],
            })
            with patch.object(actions, 'PART', str(parts)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=response), \
                 patch('builtins.print') as output:
                self.assertEqual(actions.fetch_one('300953'), 1)
            with (parts / '300953.pkl').open('rb') as handle:
                self.assertEqual(pickle.load(handle), prior)
            self.assertIn('来源改值守卫', str(output.call_args_list))

    def test_nonempty_response_keeps_missing_old_event(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            prior = [('300953', pd.Timestamp('2024-05-28'), 0.0, .2),
                     ('300953', pd.Timestamp('2025-06-13'), .4, .3)]
            with (parts / '300953.pkl').open('wb') as handle:
                pickle.dump(prior, handle)
            response = pd.DataFrame({
                '除权除息日': ['2025-06-13'],
                '送转股份-送转总比例': [4.0],
                '现金分红-现金分红比例': [3.0],
            })
            with patch.object(actions, 'PART', str(parts)), \
                 patch.object(actions, 'MAX_AGE', 0), \
                 patch.object(actions.ak, 'stock_fhps_detail_em', return_value=response), \
                 patch('builtins.print') as output:
                self.assertEqual(actions.fetch_one('300953'), 2)
            with (parts / '300953.pkl').open('rb') as handle:
                self.assertEqual(set(pickle.load(handle)), set(prior))
            self.assertIn('来源撤回守卫', str(output.call_args_list))


if __name__ == '__main__':
    unittest.main()
