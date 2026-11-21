"""Winning-branch provenance must stay categorical and decision-time only."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.chan_theory import detect_buy_sell_points
from core.buy_point_origin import BuyPointOrigin as Origin, origin_label


class BuyPointOriginTests(unittest.TestCase):
    def detect(self, close, volume):
        return detect_buy_sell_points([], [], [], len(close), close, volume,
                                      close * 1.01, close * .99)

    def test_flat_no_point_has_no_invented_origin(self):
        result = self.detect(np.full(100, 10.), np.full(100, 100.))
        self.assertFalse(result['buy_point'].any())
        self.assertFalse(result['buy_point_origin'].any())

    def test_close_rebound_is_not_labeled_as_an_opening_gap(self):
        close = np.r_[np.full(74, 100.), [100., 98., 96., 94., 93., 96.]]
        volume = np.full(80, 100.); volume[-1] = 120.
        result = self.detect(close, volume)
        self.assertEqual(result['buy_point'][-1], 1)
        self.assertEqual(result['buy_point_origin'][-1], Origin.B1_CLOSE_REBOUND)

    def test_ma20_and_ma60_routes_use_the_actual_support_condition(self):
        close = np.r_[np.linspace(10, 15, 80), [16, 17, 18, 19.5, 19, 18.7, 18.4,
            18.1, 17.8, 17.6, 17.5, 17.4, 17.3, 17.2, 17.15, 17.1, 17.05, 17., 16.9, 17.2]]
        volume = np.full(100, 100.); volume[-3:] = 90.
        for prices, expected in [(close, Origin.B6_MA20_PULLBACK),
                (np.r_[close[:-5], [15.1, 15.1, 15.1, 15., 15.1]], Origin.B6_MA60_PULLBACK)]:
            result = self.detect(prices, volume)
            self.assertEqual(result['buy_point'][-1], 6)
            self.assertEqual(result['buy_point_origin'][-1], expected)

    def test_appending_future_bars_does_not_relabel_observed_rebounds(self):
        close = np.r_[np.full(74, 100.), [100., 98., 96., 94., 93., 96.]]
        volume = np.full(80, 100.); volume[-1] = 120.
        prefix = self.detect(close, volume)
        full = self.detect(np.r_[close, np.linspace(200., 1., 40)], np.r_[volume, np.full(40, 100.)])
        np.testing.assert_array_equal(prefix['buy_point_origin'], full['buy_point_origin'][:80])

    def test_legacy_or_unrecognized_origins_remain_unknown(self):
        for value in (0, None, -1, 9999, float('nan')):
            self.assertEqual(origin_label(value), 'UNKNOWN')
        self.assertEqual(origin_label(Origin.B2_IMPLICIT_FRACTAL_MACD_ZONE), 'B2_IMPLICIT_FRACTAL_MACD_ZONE')


if __name__ == '__main__':
    unittest.main()
