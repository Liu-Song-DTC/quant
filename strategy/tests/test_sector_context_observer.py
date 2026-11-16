"""Information boundaries of the independent sector observation layer."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from observe_sector_before_entry import calendar_sector_features


class SectorContextObserverTests(unittest.TestCase):
    def fixtures(self, n=30):
        dates = pd.bdate_range('2026-06-01', periods=n)
        market = pd.DataFrame(dict(date=dates, benchmark_return=0., equal_weight_return=.01))
        sector = pd.DataFrame(dict(date=dates, sector='A', ret_1d=.02, amount_share=.1))
        return dates, market, sector

    def test_missing_session_is_unknown_and_not_skipped(self):
        dates, market, sectors = self.fixtures(12)
        result = calendar_sector_features(sectors.drop(index=3), market).set_index('date')
        self.assertTrue(pd.isna(result.loc[dates[3], 'ret_1d']))
        self.assertTrue(pd.isna(result.loc[dates[7], 'relative_compound_5d']))
        self.assertTrue(pd.isna(result.loc[dates[7], 'positive_relative_days_5']))
        self.assertAlmostEqual(result.loc[dates[8], 'relative_compound_5d'], 1.02**5 - 1.)
        self.assertEqual(result.loc[dates[8], 'positive_relative_days_5'], 5.)

    def test_future_returns_and_future_sectors_cannot_rewrite_history(self):
        dates, market, sectors = self.fixtures()
        cut = dates[24]
        prefix = calendar_sector_features(sectors[sectors.date <= cut], market[market.date <= cut])
        sectors.loc[sectors.date > cut, 'ret_1d'] = .9
        future = pd.DataFrame(dict(date=dates[25:], sector='FUTURE', ret_1d=.5, amount_share=.2))
        full = calendar_sector_features(pd.concat([sectors, future], ignore_index=True), market)
        historical = full[full.date <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), historical, check_exact=True)
        self.assertFalse((historical.sector == 'FUTURE').any())

    def test_market_relative_strength_separates_common_style_move(self):
        dates, market, sectors = self.fixtures()
        result = calendar_sector_features(sectors, market).set_index('date')
        row = result.loc[dates[19]]
        self.assertAlmostEqual(row.relative_compound_20d, 1.02**20 - 1., places=12)
        self.assertAlmostEqual(row.relative_to_market_20d, (1.02 / 1.01)**20 - 1., places=12)
        self.assertLess(row.relative_to_market_20d, row.relative_compound_20d)


if __name__ == '__main__':
    unittest.main()
