"""Decision-time quality must equal the information actually available then."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from core import chan_theory as ct


class BottomFractalAsOfTests(unittest.TestCase):
    @staticmethod
    def inputs(n=150, seed=20261115):
        rng = np.random.default_rng(seed)
        close = 15 + np.cumsum(rng.normal(0, .28, n))
        high = close + rng.uniform(.02, .45, n)
        low = close - rng.uniform(.02, .45, n)
        volume = rng.uniform(100, 2000, n)
        return close, high, low, volume, ct._calc_ema(close, 20), ct._calc_ema(close, 60), rng.normal(0, .15, n)

    def test_matches_actual_prefix_snapshot_without_new_confirmation_delay(self):
        args = self.inputs()
        result = ct.analyze_bottom_fractal_asof(*args)
        observed_positive = False
        for stop in range(3, len(args[0])+1):
            prefix = tuple(value[:stop] for value in args)
            geometry = ct.detect_strokes(prefix[1], prefix[2], prefix[0])
            expected = ct.analyze_bottom_fractal(*prefix, geometry['bottom_fractals'], geometry['stroke_direction'])
            for key in result:
                self.assertEqual(result[key][stop-1], expected[key][-1], (stop, key))
            observed_positive |= expected['bottom_fractal_quality'][-1] > 0
        self.assertTrue(observed_positive)

    def test_appending_or_replacing_future_bars_never_rewrites_history(self):
        args = self.inputs()
        full = ct.analyze_bottom_fractal_asof(*args)
        for stop in [2, 21, 60, 99, 149]:
            short = ct.analyze_bottom_fractal_asof(*(x[:stop] for x in args))
            for key in full:
                np.testing.assert_array_equal(full[key][:stop], short[key])
        changed = [x.copy() for x in args]
        for x in changed:
            x[90:] *= 3
        other = ct.analyze_bottom_fractal_asof(*changed)
        for key in full:
            np.testing.assert_array_equal(full[key][:90], other[key][:90])

    def test_empty_and_initial_history_have_no_invented_fractal(self):
        for n in [0, 1, 2]:
            args = tuple(np.ones(n, dtype=float) for _ in range(7))
            result = ct.analyze_bottom_fractal_asof(*args)
            self.assertEqual(len(result), 5)
            for values in result.values():
                np.testing.assert_array_equal(values, np.zeros(n))

    def test_public_outputs_use_decision_snapshots(self):
        args = self.inputs()
        def calculate(values):
            c, h, l, v, e20, e60, macd = values
            return ct.compute_enhanced_chan_output(c, h, l, ema20=e20, ema60=e60,
                macd_hist=macd, volume=v)
        full = calculate(args)
        fields = [name for name in full if name.startswith('bottom_fractal_')]
        self.assertEqual(len(fields), 5)
        for stop in [61, 77, 100, 121, 149]:
            prefix = calculate(tuple(x[:stop] for x in args))
            for name in fields:
                self.assertEqual(full[name][stop-1], prefix[name][-1], (stop, name))

    def test_finite_dispatch_and_missing_value_fallback_are_exact(self):
        args = tuple(np.nan_to_num(x, nan=0.) for x in self.inputs())
        expected = ct._reference_bottom_fractal_asof(*args)
        with patch.object(ct, '_reference_bottom_fractal_asof', side_effect=AssertionError('slow path used')):
            actual = ct.analyze_bottom_fractal_asof(*args)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        missing = [x.copy() for x in args]; missing[3][30] = np.nan
        expected = ct._reference_bottom_fractal_asof(*missing)
        with patch('core.fractal_observation_kernel.quality_prefix_arrays', side_effect=AssertionError('unsupported fast path')):
            actual = ct.analyze_bottom_fractal_asof(*missing)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


if __name__ == '__main__':
    unittest.main()
