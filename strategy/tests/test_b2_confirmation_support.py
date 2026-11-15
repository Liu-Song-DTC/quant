"""B2's existing support premise must hold when the event becomes observable.

Fixed primitive fixtures isolate delayed support from historical geometry,
quality revisions, implicit-B1 semantics and multi-level confirmation policy.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import chan_theory as ct


class B2ConfirmationSupportTests(unittest.TestCase):
    def setUp(self):
        n = 110
        self.close = np.full(n, 100.)
        self.close[81:86] = [103., 106., 108., 105., 103.]
        self.high = self.close + 1.
        self.low = self.close - 1.
        self.fractals = np.zeros(n, dtype=bool)
        self.fractals[[80, 85]] = True
        self.quality = np.where(self.fractals, .5, 0.)
        self.macd = np.linspace(-2., 2., n)
        self.buy = np.zeros(n, dtype=int)
        self.direction = np.zeros(n)
        self.visibility = np.arange(n)
        self.visibility[85] = 88
        self.stroke = ct.Stroke(81, 83, 1, 103., 108., 109., 102., 84)

    def calculate(self, low=None, stop=110):
        arrays = (self.close, self.high, self.low if low is None else low,
                  self.fractals, self.quality, self.macd, self.buy, self.direction)
        return ct.detect_second_buy_point(*(x[:stop] for x in arrays),
                                          [self.stroke], self.visibility[:stop])

    def test_intact_support_emits_on_confirmation_with_original_confidence(self):
        result = self.calculate()
        self.assertEqual(np.flatnonzero(result['second_buy_point']).tolist(), [88])
        self.assertEqual(result['second_buy_b1_ref'][88], 80)
        expected = .30 + .5*.25 + .20 + .10 + .05 + (102.-99.)/(99.+1e-10)*.5
        self.assertEqual(result['second_buy_confidence'][88], expected)

    def test_support_breach_during_confirmation_delay_invalidates_event(self):
        for bar in [86, 87, 88]:
            with self.subTest(bar=bar):
                low = self.low.copy(); low[bar] = 95.
                result = self.calculate(low)
                self.assertFalse(result['second_buy_point'][88])
                self.assertEqual(result['second_buy_b1_ref'][88], -1)
                self.assertEqual(result['second_buy_confidence'][88], 0.)

    def test_existing_strict_support_boundary_is_unchanged(self):
        threshold = self.low[80] * .98
        for minimum, expected in [(threshold, False),
                                  (np.nextafter(threshold, -np.inf), False),
                                  (np.nextafter(threshold, np.inf), True)]:
            with self.subTest(minimum=minimum):
                low = self.low.copy(); low[86] = minimum
                self.assertEqual(bool(self.calculate(low)['second_buy_point'][88]), expected)

    def test_future_prices_cannot_invalidate_an_already_observed_event(self):
        full = self.calculate()
        low = self.low.copy(); low[89:] = 1.
        changed = self.calculate(low)
        prefix = self.calculate(stop=89)
        for field in full:
            np.testing.assert_array_equal(full[field][:89], changed[field][:89])
            np.testing.assert_array_equal(full[field][:89], prefix[field])
        self.assertFalse(self.calculate(stop=88)['second_buy_point'].any())

    def test_pre_candidate_episode_support_policy_is_a_separate_change(self):
        low = self.low.copy(); low[82] = 95.
        reference = self.calculate()
        earlier = self.calculate(low)
        for field in reference:
            np.testing.assert_array_equal(reference[field], earlier[field])


if __name__ == '__main__':
    unittest.main()
