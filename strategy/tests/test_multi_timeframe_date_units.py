"""The same instants must never expose a future higher-timeframe bar."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multi_timeframe import MultiTimeframeAnalyzer


class MultiTimeframeDateUnitTests(unittest.TestCase):
    def test_latest_completed_bar_is_independent_of_datetime_unit(self):
        dates = pd.DatetimeIndex(['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-10'])
        ends = pd.DatetimeIndex(['2025-01-03', '2025-01-10'])
        expected = np.array([-1, 0, 0, 1])
        for daily_unit in ('s', 'ms', 'us', 'ns'):
            for bar_unit in ('s', 'ms', 'us', 'ns'):
                with self.subTest(daily=daily_unit, bar=bar_unit):
                    actual = MultiTimeframeAnalyzer._build_daily_map_fast(
                        dates.as_unit(daily_unit), ends.as_unit(bar_unit).to_numpy(), len(dates))
                    np.testing.assert_array_equal(actual, expected)

    def test_full_analysis_and_future_perturbation_preserve_closed_history(self):
        dates = pd.bdate_range('2021-01-01', periods=430)
        phase = np.arange(len(dates), dtype=float)
        close = 100. + .03 * phase + 8. * np.sin(phase / 13.)
        args = (close, close + 2., close - 2., close - .3, np.full(len(close), 10000.))
        analyzer = MultiTimeframeAnalyzer()
        full = analyzer.analyze(*args, dates.as_unit('us').to_numpy())
        other = analyzer.analyze(*args, dates.as_unit('ns').to_numpy())
        for name in vars(full):
            np.testing.assert_array_equal(getattr(full, name), getattr(other, name), err_msg=name)
        for count in (180, 247, 322):
            prefix = analyzer.analyze(*(values[:count] for values in args), dates[:count].as_unit('ns').to_numpy())
            changed = tuple(values.copy() for values in args)
            for values in changed:
                values[count:] *= 3.
            perturbed = analyzer.analyze(*changed, dates.as_unit('ns').to_numpy())
            for name in vars(full):
                # Trading consumers discard the first 60 daily warmup rows.
                # Early pattern initialization has a separate two-week policy.
                np.testing.assert_array_equal(getattr(prefix, name)[60:], getattr(full, name)[60:count], err_msg=name)
                np.testing.assert_array_equal(getattr(perturbed, name)[60:count], getattr(full, name)[60:count], err_msg=name)


if __name__ == '__main__':
    unittest.main()
