"""Future appends must not change a previously observed pool membership."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import stock_pool


class PoolClassifierPrefixTests(unittest.TestCase):
    def test_future_rows_cannot_change_old_membership(self):
        for past, future in ((1., .01), (.01, 1.)):
            with self.subTest(past=past, future=future), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                dates = pd.date_range('2020-01-01', periods=160)
                ratios = np.r_[np.full(60, past), np.full(100, future)]
                frame = pd.DataFrame(dict(datetime=dates, close=10., amount=200_000_000.,
                    volume=200_000_000. / (1000 * ratios)))
                path = Path(directory) / '600123_qfq.csv'
                frame.iloc[:60].to_csv(path, index=False)
                before = stock_pool.get_pool_membership_map([dates[59]], data_dir=directory)[dates[59]]
                frame.to_csv(path, index=False)
                after = stock_pool.get_pool_membership_map([dates[59]], data_dir=directory)[dates[59]]
                single = stock_pool.get_stock_pool(data_dir=directory, todate=str(dates[59].date()))
                self.assertEqual(before, after)
                self.assertEqual('600123' in after, '600123' in single)

    def test_prefix_helper_matches_scalar_with_missing_and_extreme_values(self):
        for kind in ('finite', 'missing', 'infinite', 'no_amount'):
            with self.subTest(kind=kind), np.errstate(over='ignore', invalid='ignore'):
                frame = pd.DataFrame(dict(close=np.full(70, 10.), volume=np.full(70, 200_000.),
                    amount=np.r_[np.full(30, 50_000_000.), np.full(40, 200_000_000.)]))
                if kind == 'missing': frame.loc[10:25, 'amount'] = np.nan
                elif kind == 'infinite': frame.loc[10:25, 'amount'] = np.inf
                elif kind == 'no_amount': frame = frame.drop(columns='amount')
                expected = [stock_pool._is_non_stock_series(frame.iloc[:i+1]) for i in range(len(frame))]
                np.testing.assert_array_equal(stock_pool._prefix_nonstock_flags(frame), expected)


if __name__ == '__main__':
    unittest.main()
