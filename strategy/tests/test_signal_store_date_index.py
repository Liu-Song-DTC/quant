"""Date lookup must retain observed-date and duplicate/alias ordering semantics."""
from datetime import date, datetime
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.signal_store import SignalStore


class SignalStoreDateIndexTests(unittest.TestCase):
    def test_aliases_preserve_group_encounter_order_and_first_full_key(self):
        # A late alias is encountered first because code is the leading index.
        # Keep this order, including the later early-alias overwrite for z.
        frame = pd.DataFrame([
            ('a', '2026-11-20 15:00:00', True, 'a-late'),
            ('b', '2026-11-20 09:30:00', False, 'b-early'),
            ('b', '2026-11-20 15:00:00', True, 'b-late'),
            ('z', '2026-11-20 09:30:00', True, 'z-early'),
            ('z', '2026-11-20 09:30:00', False, 'duplicate'),
            ('z', '2026-11-20 15:00:00', True, 'z-late'),
            ('a', '2026-11-23', True, 'next-day'),
        ], columns=['code', 'date', 'buy', 'industry']).set_index(['code', 'date']).sort_index()
        store = SignalStore()
        store._df = frame
        universe = ['z', 'b', 'a', 'z', 'missing']
        expected = [('z', 'z-early'), ('b', 'b-late'), ('a', 'a-late'), ('z', 'z-early')]
        self.assertEqual(store.buy_signals_for_date(date(2026, 11, 20), universe), expected)
        self.assertEqual(store.buy_signals_for_date('2026-11-23', universe), [('a', 'next-day')])
        self.assertEqual(store.buy_signals_for_date(datetime(2026, 11, 20, 8), universe), expected)
        self.assertEqual(store.buy_signals_for_date('2026-11-21', universe), [])

    def test_has_date_ignores_unused_levels_left_by_membership_filter(self):
        index = pd.MultiIndex.from_product([['a'], ['2026-11-20', '2026-11-23']], names=['code', 'date'])
        store = SignalStore()
        store._df = pd.DataFrame({'buy': [True, True]}, index=index).iloc[:1]
        self.assertEqual(len(store._df.index.levels[1]), 2)
        self.assertTrue(store.has_date('2026-11-20'))
        self.assertFalse(store.has_date('2026-11-23'))

    def test_has_date_unique_normalization_matches_expanded_index(self):
        for dates in [
            ['2026-11-20', '2026-11-20 15:00:00', None, '2026-11-23'],
            [1, 1, 2, None],
            pd.to_datetime(['2026-11-20', '2026-11-20', None, '2026-11-23']),
            pd.to_datetime(['2026-11-20', '2026-11-20', None, '2026-11-23'], utc=True),
        ]:
            with self.subTest(dates=dates):
                store = SignalStore()
                store._df = pd.DataFrame({'buy': [True] * 4}, index=pd.MultiIndex.from_arrays(
                    [['a', 'b', 'c', 'd'], dates], names=['code', 'date']))
                expected = set(store._df.index.get_level_values('date').astype(str).str[:10])
                for query in [str(value) for value in expected] + ['missing', '2026-11-19']:
                    self.assertEqual(store.has_date(query), query[:10] in expected)

    def test_missing_buy_and_missing_industry_keep_existing_outputs(self):
        frame = pd.DataFrame({'code': ['a', 'b'], 'date': ['2026-11-20'] * 2,
                              'buy': [True, False]}).set_index(['code', 'date'])
        for columns, expected in [(['buy'], [('a', '')]), ([], [])]:
            with self.subTest(columns=columns):
                store = SignalStore()
                store._df = frame[columns]
                self.assertEqual(store.buy_signals_for_date('2026-11-20', ['b', 'a']), expected)

    def test_empty_store_has_no_date_or_candidates(self):
        store = SignalStore()
        store._df = pd.DataFrame({'buy': pd.Series(dtype=bool)}, index=pd.MultiIndex.from_arrays(
            [[], []], names=['code', 'date']))
        self.assertFalse(store.has_date('2026-11-20'))
        self.assertEqual(store.buy_signals_for_date('2026-11-20', ['a']), [])


if __name__ == '__main__':
    unittest.main()
