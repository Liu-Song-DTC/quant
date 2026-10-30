"""A late alternative-data backfill must not change an earlier decision."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy.core.alt_data_versions import record_snapshot, resolve_asof, verify_current
from strategy.core.alternative_data import AlternativeDataProvider
from strategy.core import alternative_data


class AlternativeObservationVersionsTests(unittest.TestCase):
    def test_late_reduction_event_does_not_rewrite_prior_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reduction_records.pkl'
            cols = ['code', 'start_date', 'end_date', 'eitime']
            old = pd.DataFrame([('000001', '2026-10-01', '2026-10-30', '2026-10-27')], columns=cols)
            old[['start_date', 'end_date', 'eitime']] = old[['start_date', 'end_date', 'eitime']].apply(pd.to_datetime)
            old.to_pickle(path)
            record_snapshot(path, '2026-10-29')
            new = pd.DataFrame([('000002', '2026-10-01', '2026-10-30', '2026-10-28')], columns=cols)
            new[['start_date', 'end_date', 'eitime']] = new[['start_date', 'end_date', 'eitime']].apply(pd.to_datetime)
            pd.concat([old, new], ignore_index=True).to_pickle(path)
            record_snapshot(path, '2026-10-30')
            with patch.object(alternative_data, '_AUTOREFRESH_OFF', True):
                provider = AlternativeDataProvider(directory)
                self.assertEqual(provider.get_reduction_codes('2026-10-28'), {'000001'})
                self.assertEqual(provider.get_reduction_codes('2026-10-30'), {'000001', '000002'})
                self.assertEqual(provider.get_reduction_codes('2026-10-28'), {'000001'})

    def test_history_is_immutable_and_missing_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'northbound_daily.pkl'
            path.write_bytes(b'old')
            record_snapshot(path, '2026-10-29')
            first = resolve_asof(path, '2026-10-28')
            path.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'differs'):
                verify_current(path)
            record_snapshot(path, '2026-10-30')
            self.assertEqual(resolve_asof(path, '2026-10-28'), first)
            self.assertEqual(first.read_bytes(), b'old')
            self.assertEqual(resolve_asof(path, '2026-10-30').read_bytes(), b'new')
            with self.assertRaisesRegex(ValueError, 'backdated'):
                record_snapshot(path, '2026-10-27')
            first.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                resolve_asof(path, '2026-10-28')

    def test_lagged_market_value_uses_decision_day_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'northbound_daily.pkl'
            old = pd.DataFrame({'date': pd.to_datetime(['2026-10-28']),
                                'net_flow': [1.0]})
            old.to_pickle(path)
            record_snapshot(path, '2026-10-29')
            pd.concat([old, pd.DataFrame({'date': pd.to_datetime(['2026-10-29']),
                                          'net_flow': [2.0]})]).to_pickle(path)
            record_snapshot(path, '2026-10-30')
            with (patch.object(alternative_data, '_AUTOREFRESH_OFF', True),
                  patch.object(AlternativeDataProvider, '_nb_signal_impl',
                               side_effect=lambda df, date: float(len(df)))):
                provider = AlternativeDataProvider(directory)
                self.assertEqual(provider.get_northbound_signal(
                    '2026-10-29', observed_by='2026-10-29'), 1.0)
                self.assertEqual(provider.get_northbound_signal(
                    '2026-10-29', observed_by='2026-10-30'), 2.0)
                self.assertEqual(provider.get_northbound_signal(
                    '2026-10-29', observed_by='2026-10-29'), 1.0)

    def test_aggregate_dragon_tiger_is_not_proxied_before_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dragon_tiger.pkl'
            pd.DataFrame({'代码': ['000001'], '最近上榜日': ['2026-10-27'],
                          '买入额': [10.0], '卖出额': [1.0]}).to_pickle(path)
            record_snapshot(path, '2026-10-29')
            with patch.object(alternative_data, '_AUTOREFRESH_OFF', True):
                provider = AlternativeDataProvider(directory)
                self.assertEqual(provider.get_dragon_tiger_signal('000001', '2026-10-28'), 0.0)
                self.assertGreater(provider.get_dragon_tiger_signal('000001', '2026-10-29'), 0.0)

    def test_required_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reduction_records.pkl'
            pd.DataFrame({'code': ['000001'], 'eitime': [pd.Timestamp('2026-10-01')]}).to_pickle(path)
            with (patch.dict(os.environ, {'QUANT_ALT_VERSIONS_REQUIRED': '1'}),
                  patch.object(alternative_data, '_AUTOREFRESH_OFF', True)):
                provider = AlternativeDataProvider(directory)
                with self.assertRaises(FileNotFoundError):
                    provider.get_reduction_codes('2026-10-28')


if __name__ == '__main__':
    unittest.main()
