"""The optimized reader retains exact dated membership and CSV float64 values."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.signal_store import SignalStore


class SignalStoreMembershipLookupTests(unittest.TestCase):
    def test_boundary_membership_and_index_exemptions_preserve_exact_csv_values(self):
        dates = ['2026-01-01', '2026-01-02', '2026-01-04', '2026-01-05', '2026-01-06']
        codes = ['600000', '600001', '600002', '000001', 'sh000001', 'sh000852', '399006']
        rows = pd.DataFrame([{'code': code, 'date': date, 'buy': True, 'sell': False,
                              'score': .123456789012345 + i * .000000000000001}
                             for i, (date, code) in enumerate((d, c) for d in dates for c in codes)])
        mapping = {pd.Timestamp('2026-01-02'): {'600000'}, pd.Timestamp('2026-01-05'): {'600001'}}
        exemptions = {'000001', 'sh000001', 'sh000852', '399006'}
        def permitted(row):
            if row.code in exemptions:
                return True
            if row.date < '2026-01-02':
                return False
            return row.code == ('600000' if row.date < '2026-01-05' else '600001')
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'signals.csv'
            rows.to_csv(csv, index=False)
            parsed = pd.read_csv(csv, dtype={'code': str, 'date': str})
            keep = parsed.apply(permitted, axis=1)
            expected = parsed.loc[keep].set_index(['code', 'date']).sort_index()
            initial, replay = SignalStore(), SignalStore()
            initial.finalize(csv, membership_map=mapping)
            replay.finalize(csv, membership_map=mapping, sealed_replay=True)
            for store in (initial, replay):
                pd.testing.assert_frame_equal(store._df, expected, check_exact=True)
                self.assertEqual(store._df.score.dtype, np.dtype('float64'))
                self.assertEqual(len(store._df), 24)

    def test_absent_membership_preserves_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'signals.csv'
            csv.write_text('code,date,buy,sell,score\n600000,2026-01-01,True,False,0.123456789012345\n')
            store = SignalStore()
            store.finalize(csv)
            self.assertEqual(len(store._df), 1)
            self.assertTrue(store.get('600000', '2026-01-01').buy)


if __name__ == '__main__':
    unittest.main()
