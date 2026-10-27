import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.macro_asof import known_history, macro_warning_features


def record(observed, released, value=4., series='us10y', basis='verified_release'):
    return dict(series=series, observation_at=observed, available_at=released,
                value=value, source='test-source', availability_basis=basis)


class MacroAsOfTests(unittest.TestCase):
    def test_us_close_is_unavailable_during_same_chinese_trading_day(self):
        rows = pd.DataFrame([record('2026-10-26T16:00:00-04:00', '2026-10-26T16:15:00-04:00')])
        self.assertTrue(known_history(rows, '2026-10-26T15:00:00+08:00').empty)
        self.assertEqual(len(known_history(rows, '2026-10-27T09:00:00+08:00')), 1)

    def test_weekly_fx_release_and_revisions_not_available_at_observation_date(self):
        rows = pd.DataFrame([
            record('2026-10-20T12:00:00-04:00', '2026-10-26T16:15:00-04:00', 7., 'usdcny'),
            record('2026-10-20T12:00:00-04:00', '2026-11-02T16:15:00-05:00', 7.1, 'usdcny'),
        ])
        self.assertTrue(known_history(rows, '2026-10-23T09:00:00+08:00').empty)
        self.assertEqual(known_history(rows, '2026-10-27T09:00:00+08:00')['value'].tolist(), [7.])
        self.assertEqual(known_history(rows, '2026-11-03T09:00:00+08:00')['value'].tolist(), [7.1])

    def test_proxy_exclusion_conflicts_and_timezone_validation(self):
        row = record('2026-10-20T16:00:00-04:00', '2026-10-20T17:00:00-04:00', basis='proxy')
        rows = pd.DataFrame([row])
        self.assertTrue(known_history(rows, '2026-10-27T09:00:00+08:00').empty)
        self.assertEqual(len(known_history(rows, '2026-10-27T09:00:00+08:00', True)), 1)
        with self.assertRaises(ValueError):
            known_history(rows, '2026-10-27')
        row['availability_basis'] = 'verified_release'
        with self.assertRaises(ValueError):
            known_history(pd.DataFrame([row, dict(row, value=5.)]), '2026-10-27T09:00:00+08:00')

    def test_staleness_and_yield_changes_are_not_return_percentages(self):
        rows = pd.DataFrame([record(f'2026-10-{day:02d}T16:00:00-04:00',
                                    f'2026-10-{day:02d}T17:00:00-04:00', 4 + (day-20)*.01)
                             for day in range(20, 26)])
        limits = dict.fromkeys(['us10y', 'usdcny', 'vix', 'a_realized_vol20'], 7.)
        result = macro_warning_features(rows, '2026-10-27T09:00:00+08:00', limits)
        self.assertAlmostEqual(result['features']['us10y_change_5_bp'], 5.)
        self.assertEqual(result['coverage']['usdcny']['state'], 'missing')
        result = macro_warning_features(rows, '2026-11-15T09:00:00+08:00', limits)
        self.assertEqual(result['coverage']['us10y']['state'], 'stale')
        self.assertEqual(result['features'], {})


if __name__ == '__main__':
    unittest.main()
