"""Historical event terms stay stable when the provider revises one date."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data import fetch_actions_table as actions


class ActionSourceRevisionTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
