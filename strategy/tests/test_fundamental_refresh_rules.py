"""Partial source coverage must not delete an existing stock's other quarters."""
import importlib.util
from pathlib import Path
import unittest
import sys
from unittest.mock import patch

import pandas as pd


spec = importlib.util.spec_from_file_location(
    'fundamental_refresh_rules', Path(__file__).resolve().parents[2] / 'data/fundamental_refresh_rules.py')
rules = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[2] / 'data'), *sys.path]):
    spec.loader.exec_module(rules)


class FundamentalRefreshRulesTests(unittest.TestCase):
    def test_default_quarters_roll_forward_and_handle_year_boundary(self):
        self.assertEqual(rules.recent_completed_quarters('2026-10-27'),
                         ['20260331', '20260630', '20260930'])
        self.assertEqual(rules.recent_completed_quarters('2026-01-01'),
                         ['20250630', '20250930', '20251231'])
        self.assertEqual(rules.recent_completed_quarters('2026-09-29', 1), ['20260630'])
        self.assertEqual(rules.recent_completed_quarters('2026-09-30', 1), ['20260930'])

    def test_partial_refresh_keeps_unreceived_target_periods(self):
        old = pd.DataFrame({'报告期': ['20251231', '20260331', '20260630'], 'roe': [10., 11., 12.]})
        incoming = pd.DataFrame({'报告期': ['20260630'], 'roe': [13.]})
        result = rules.merge_report_rows(old, incoming)
        self.assertEqual(result['报告期'].tolist(), ['20251231', '20260331', '20260630'])
        self.assertEqual(result['roe'].tolist(), [10., 11., 13.])
        self.assertEqual(old['roe'].tolist(), [10., 11., 12.])

    def test_refresh_periods_follow_existing_deadline_proxy(self):
        data = str(Path(__file__).resolve().parents[2] / 'data')
        with patch.object(sys, 'path', [data, *sys.path]):
            self.assertEqual(rules.recent_refresh_quarters('2026-10-27'),
                             ['20251231', '20260331', '20260630'])
            self.assertEqual(rules.recent_refresh_quarters('2026-11-01'),
                             ['20260331', '20260630', '20260930'])
            self.assertEqual(rules.recent_refresh_quarters('2026-04-30'),
                             ['20250331', '20250630', '20250930'])
            self.assertEqual(rules.recent_refresh_quarters('2026-05-01'),
                             ['20250930', '20251231', '20260331'])

    def test_new_period_and_new_source_fields_are_preserved(self):
        old = pd.DataFrame({'报告期': ['20260331'], 'roe': [10.], 'old_field': [1.]})
        incoming = pd.DataFrame({'报告期': ['20260630'], 'roe': [11.], 'new_field': [2.]})
        result = rules.merge_report_rows(old, incoming)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[1, 'new_field'], 2.)
        self.assertEqual(result.loc[0, 'old_field'], 1.)

    def test_empty_source_is_no_change_and_invalid_or_duplicate_periods_fail(self):
        old = pd.DataFrame({'报告期': ['20260331'], 'roe': [10.]})
        pd.testing.assert_frame_equal(rules.merge_report_rows(old, pd.DataFrame()), old)
        for periods in [['bad'], ['20260231'], ['20260331', '20260331']]:
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                rules.merge_report_rows(old, pd.DataFrame({'报告期': periods}))

    def test_published_report_revision_detected_but_new_quarter_is_allowed(self):
        old = pd.DataFrame({'报告期': ['20260331'], '净利润': ['2.0'],
                            '数据可用日期': ['20260430']})
        incoming = pd.DataFrame({'报告期': ['20260331', '20260630'],
                                 '净利润': [2.1, 3.0],
                                 '数据可用日期': ['20260430', '20260831']})
        changes = rules.published_report_changes(old, incoming)
        self.assertEqual(changes, [{'period': '20260331', 'column': '净利润',
                                    'old': '2.0', 'new': '2.1'}])
        incoming.loc[0, '净利润'] = 2.0
        self.assertEqual(rules.published_report_changes(old, incoming), [])

    def test_published_report_revision_fails_on_ambiguous_quarter(self):
        old = pd.DataFrame({'报告期': ['20260331'], '净利润': ['2.0']})
        incoming = pd.DataFrame({'报告期': ['20260331', '20260331'], '净利润': [2, 3]})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            rules.published_report_changes(old, incoming)


if __name__ == '__main__':
    unittest.main()
