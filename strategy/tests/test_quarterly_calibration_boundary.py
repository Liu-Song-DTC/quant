"""The quarter's factor selection must never train on that quarter's returns."""

import unittest

import pandas as pd

from strategy.analysis.quarterly_calibrate import get_calibration_window, select_calibration_window


class QuarterlyCalibrationBoundaryTest(unittest.TestCase):
    def test_window_uses_requested_completed_quarters_including_q1(self):
        cases = [
            ('2021Q1', 20, '2016-01-01', '2020-12-31'),
            ('2021Q1', 1, '2020-10-01', '2020-12-31'),
            ('2026Q2', 4, '2025-04-01', '2026-03-31'),
            ('2024Q1', 20, '2019-01-01', '2023-12-31'),
            ('2021Q4', 24, '2015-10-01', '2021-09-30'),
        ]
        for quarter, count, expected_start, expected_end in cases:
            with self.subTest(quarter=quarter, count=count):
                start, end = get_calibration_window(quarter, count)
                self.assertEqual(start, pd.Timestamp(expected_start))
                self.assertEqual(end, pd.Timestamp(expected_end))
                quarters = pd.period_range(start, end, freq='Q')
                self.assertEqual(len(quarters), count)
                self.assertTrue((quarters < pd.Period(quarter, freq='Q')).all())

    def test_invalid_window_lengths_are_rejected(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'positive integer'):
                get_calibration_window('2021Q1', count)

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
