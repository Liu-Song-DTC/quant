"""Regression contract for a later explicit B1 erasing an already known B2.

Primitive geometry fixture isolates zone suppression from quality calculation.
The production observation boundary, not an unbounded historical snapshot,
must preserve the value available at the original daily close.
"""
import unittest
import numpy as np
from unittest.mock import patch

import test_b2_confirmation_support as support_fixture
from core.b2_observation import observe_second_buy_at


class B2FutureZoneContract(unittest.TestCase):
    def setUp(self):
        self.fixture=support_fixture.B2ConfirmationSupportTests()
        self.fixture.setUp()

    def observed(self, index):
        f=self.fixture
        def geometry(high,low,close):
            n=len(close)
            return dict(bottom_fractals=f.fractals[:n],stroke_direction=f.direction[:n],
                        strokes=[f.stroke] if f.stroke.vis_idx<n else [])
        def clock(high,low):return dict(fractal_vis=f.visibility[:len(high)])
        def quality(close,*args):return dict(bottom_fractal_quality=f.quality[:len(close)])
        with patch.object(support_fixture.ct,'detect_strokes',side_effect=geometry), \
             patch.object(support_fixture.ct,'detect_fractals',side_effect=clock), \
             patch.object(support_fixture.ct,'analyze_bottom_fractal',side_effect=quality):
            return observe_second_buy_at(f.close,f.high,f.low,np.ones(len(f.close)),
                                         f.close,f.close,f.macd,f.buy,index)

    def test_future_explicit_b1_cannot_erase_past_implicit_b1_and_b2(self):
        f=self.fixture
        before=self.observed(88)
        self.assertTrue(before['second_buy_point'])
        f.buy[89]=1  # A later explicit B1 is only nine bars from implicit B1 80.
        self.assertFalse(f.calculate()['second_buy_point'][88])  # Reproduce old unbounded defect.
        self.assertEqual(before,self.observed(88))

    def test_distant_future_b1_is_an_unchanged_control(self):
        f=self.fixture
        before=self.observed(88)
        f.buy[100]=1
        self.assertEqual(before,self.observed(88))

    def test_no_event_before_confirmation(self):
        self.assertFalse(self.observed(87)['second_buy_point'])

    def test_later_price_breach_cannot_change_past_observation(self):
        before=self.observed(88)
        self.fixture.low[89:]=1.
        self.assertEqual(before,self.observed(88))


if __name__=='__main__':unittest.main()
