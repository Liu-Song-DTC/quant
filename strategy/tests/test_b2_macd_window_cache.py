"""Exact B2 reductions and causal observation boundaries for cached windows."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

import test_b2_confirmation_support as fixture
from core.b2_observation import _finite_macd_windows, observe_second_buy_at


class B2MacdCacheTests(unittest.TestCase):
    def test_exact_scalar_windows_and_future_invariance(self):
        rng = np.random.default_rng(20261115)
        for values in (np.empty(0), rng.normal(size=251), rng.normal(size=502)[::2],
                       np.tile([1e20, 1., -1e20, .001, -1., 1e-30], 30)):
            cached = _finite_macd_windows(values)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reference = np.array([
                    [np.nanmean(values[max(0, i-5):i+1]),
                     np.nanmean(values[max(0, i-10):max(0, i-5)]),
                     np.nanmean(values[max(0, i-3):i+1])]
                    for i in range(len(values))]).reshape(-1, 3)
            np.testing.assert_array_equal(cached, reference)
            for stop in (0, 1, 4, 5, 6, 7, 11, len(values)//2, len(values)):
                if stop > len(values):
                    continue
                np.testing.assert_array_equal(cached[:stop], _finite_macd_windows(values[:stop]))
                changed = values.copy(); changed[stop:] = 97.
                np.testing.assert_array_equal(cached[:stop], _finite_macd_windows(changed)[:stop])

    def test_unsupported_inputs_retain_scalar_path(self):
        for values in (np.array([np.nan]), np.array([np.inf]),
                       np.array([1.], dtype=np.float32), np.array([1]),
                       np.ones((2, 2)), [1.]):
            self.assertIsNone(_finite_macd_windows(values))

    def test_detector_equal_at_support_boundary_and_future_zone(self):
        f = fixture.B2ConfirmationSupportTests(); f.setUp()
        cache = _finite_macd_windows(f.macd)
        for explicit_future_zone in (False, True):
            f.buy[89] = int(explicit_future_zone)
            for stop in (88, 89, 90, 110):
                for low in (f.low[80]*.98, np.nextafter(f.low[80]*.98, np.inf), 99.):
                    f.low[86] = low
                    reference = f.calculate(stop=stop)
                    arrays = (f.close, f.high, f.low, f.fractals, f.quality, f.macd,
                              f.buy, f.direction)
                    result = fixture.ct.detect_second_buy_point(
                        *(x[:stop] for x in arrays), [f.stroke], f.visibility[:stop],
                        _macd_windows=cache[:stop])
                    for field in reference:
                        np.testing.assert_array_equal(result[field], reference[field])

    def test_observer_limits_cached_data_to_decision(self):
        f = fixture.B2ConfirmationSupportTests(); f.setUp()
        cache = _finite_macd_windows(f.macd)
        original = fixture.ct.detect_second_buy_point
        calls = []

        def checked(*args, **kwargs):
            self.assertEqual(len(args[0]), 89)
            self.assertEqual(kwargs['_macd_windows'].shape, (89, 3))
            calls.append(1)
            return original(*args, **kwargs)

        args = (f.close, f.high, f.low, np.ones(110), f.close, f.close,
                f.macd, f.buy, 88)
        reference = observe_second_buy_at(*args)
        with patch.object(fixture.ct, 'detect_second_buy_point', side_effect=checked):
            self.assertEqual(observe_second_buy_at(*args, _macd_windows=cache), reference)
            cache[89:] = 1e10
            self.assertEqual(observe_second_buy_at(*args, _macd_windows=cache), reference)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
