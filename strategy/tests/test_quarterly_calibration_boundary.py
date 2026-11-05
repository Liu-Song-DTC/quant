"""The quarter's factor selection must never train on that quarter's returns."""

import unittest

import pandas as pd

from strategy.analysis.quarterly_calibrate import select_calibration_window


class QuarterlyCalibrationBoundaryTest(unittest.TestCase):
    def test_purges_labels_crossing_quarter_start(self):
        rows = pd.DataFrame({
            'date': ['2020-11-20', '2020-12-10', '2020-12-10', '2020-12-31'],
            'label_end_date': ['2020-12-18', '2020-12-31', '2021-01-05', None],
            'code': ['early', 'safe', 'suspended', 'missing'],
        })
        window = select_calibration_window(
            rows, pd.Timestamp('2020-10-01'), pd.Timestamp('2020-12-31'),
            pd.Timestamp('2021-01-01'))
        self.assertEqual(window['code'].tolist(), ['early', 'safe'])

    def test_old_cache_fails_closed(self):
        rows = pd.DataFrame({'date': ['2020-12-01']})
        with self.assertRaisesRegex(RuntimeError, 'label_end_date'):
            select_calibration_window(
                rows, pd.Timestamp('2020-10-01'), pd.Timestamp('2020-12-31'),
                pd.Timestamp('2021-01-01'))


if __name__ == '__main__':
    unittest.main()
