"""The history-cell detector hashes row content, not provider ordering."""
import argparse
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from strategy.tools import histcells


class HistoryCellRowOrderTests(unittest.TestCase):
    def test_same_date_and_code_with_several_holders_is_order_independent(self):
        rows = pd.DataFrame([
            {'end_date': '2026-10-28', 'code': '000001', 'holder': 'a', 'change_num': 1},
            {'end_date': '2026-10-28', 'code': '000001', 'holder': 'b', 'change_num': 2},
            {'end_date': '2026-10-29', 'code': '000001', 'holder': 'a', 'change_num': 3},
        ])
        first, err = histcells._dates_from_df(rows, ['end_date'], ['code'])
        reordered, err2 = histcells._dates_from_df(rows.iloc[[1, 0, 2]], ['end_date'], ['code'])
        self.assertIsNone(err)
        self.assertIsNone(err2)
        self.assertEqual(first, reordered)
        changed, _ = histcells._dates_from_df(rows.assign(change_num=[1, 9, 3]),
                                               ['end_date'], ['code'])
        self.assertNotEqual(first['2026-10-28'], changed['2026-10-28'])
        duplicate, _ = histcells._dates_from_df(pd.concat([rows, rows.iloc[:1]]),
                                                 ['end_date'], ['code'])
        self.assertNotEqual(first['2026-10-28'], duplicate['2026-10-28'])

    def test_old_hash_versions_are_rejected_before_scanning(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'legacy.pkl').write_bytes(pickle.dumps(
                {'ts': 'yesterday', 'layers': {}, 'git': ''}))
            with patch.object(histcells, 'SNAP_DIR', directory), \
                    patch.object(histcells, '_scan_all') as scan:
                self.assertEqual(histcells.cmd_diff(argparse.Namespace(label='legacy')), 3)
                scan.assert_not_called()


if __name__ == '__main__':
    unittest.main()
