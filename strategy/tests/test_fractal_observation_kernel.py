"""Equivalence checks for the unconnected numeric prefix kernel."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from core import chan_theory as ct
from core.fractal_observation_kernel import geometry_arrays, quality_prefix_arrays


class FractalObservationKernelTests(unittest.TestCase):
    def test_geometry_matches_canonical_with_inclusions_and_equal_prices(self):
        rng = np.random.default_rng(7331)
        for n in [0, 1, 2, 3, 17, 60, 121, 300]:
            for rounded in [False, True]:
                close = 15 + np.cumsum(rng.normal(0, .4, n))
                high = close+rng.uniform(0, .8, n)
                low = close-rng.uniform(0, .8, n)
                if rounded:
                    high = np.round(high, 0); low = np.round(low, 0)
                for stop in sorted(set([n, n//2])):
                    for minimum in [1, 4, 7]:
                        original = ct.detect_strokes(high[:stop], low[:stop], close[:stop], minimum)
                        bottom, direction = geometry_arrays(high[:stop], low[:stop], minimum)
                        np.testing.assert_array_equal(bottom, original['bottom_fractals'])
                        np.testing.assert_array_equal(direction, original['stroke_direction'])

    def test_quality_exact_against_prefix_reference(self):
        for seed in [0, 49, 7331]:
            rng = np.random.default_rng(seed); n = 180
            close = 15+np.cumsum(rng.normal(0, .4, n))
            high = close+rng.uniform(0, .8, n); low = close-rng.uniform(0, .8, n)
            volume = rng.uniform(0, 1000, n)
            args = close, high, low, volume, ct._calc_ema(close, 20), ct._calc_ema(close, 60), rng.normal(size=n)
            expected = ct._reference_bottom_fractal_asof(*args)
            actual = quality_prefix_arrays(*args, ct._sma(volume, 20))
            for i, name in enumerate(expected):
                np.testing.assert_array_equal(actual[i], expected[name], err_msg=f'{seed}:{name}')
            for stop in [19, 60, 179]:
                prefix = quality_prefix_arrays(*(x[:stop] for x in args), ct._sma(volume[:stop], 20))
                np.testing.assert_array_equal(actual[:, :stop], prefix)


if __name__ == '__main__':
    unittest.main()
