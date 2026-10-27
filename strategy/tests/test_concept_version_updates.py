"""Daily refreshes preserve old concept versions and stop on archive failures."""
import contextlib
import importlib.util
import io
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / 'data' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


versions = load_module('concept_versions_under_test', 'concept_map_version.py')
refresh = load_module('concept_refresh_under_test', 'refresh_concept_map_0903.py')


class ConceptVersionUpdateTests(unittest.TestCase):
    def write_version(self, path, today, mapping, date=None):
        with patch.object(versions.time, 'strftime', side_effect=lambda fmt:
                          today if fmt == '%Y-%m-%d' else today + ' 18:00:00'):
            return versions.record_version(mapping, hist=str(path), date=date)

    def test_new_day_append_preserves_fixed_first_and_all_prior_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.pkl'
            self.write_version(path, '2026-05-31', {'a': ['first']})
            self.write_version(path, '2026-06-30', {'a': ['second']})
            before = pickle.loads(path.read_bytes())['versions']
            self.write_version(path, '2026-07-01', {'b': ['third']})
            after = pickle.loads(path.read_bytes())['versions']
            self.assertEqual(after[:2], before)
            self.assertEqual(after[0]['date'], '2026-05-31')
            self.assertEqual(len(after), 3)

    def test_backfill_and_historical_overwrite_cannot_rewrite_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.pkl'
            self.write_version(path, '2026-05-31', {'a': ['first']})
            self.write_version(path, '2026-06-30', {'a': ['second']})
            original = path.read_bytes()
            for date in ['2026-05-01', '2026-06-15', '2026-07-01']:
                with self.subTest(date=date), self.assertRaises(ValueError):
                    self.write_version(path, '2026-07-02', {'b': ['changed']}, date=date)
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self.write_version(path, '2026-07-02', {'b': ['changed']},
                                                date='2026-05-31'), 'exists')
            self.assertEqual(path.read_bytes(), original)

    def test_today_refresh_can_change_today_but_not_first_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.pkl'
            self.write_version(path, '2026-05-31', {'a': ['first']})
            first = pickle.loads(path.read_bytes())['versions'][0]
            self.write_version(path, '2026-07-01', {'b': ['today']})
            self.assertEqual(self.write_version(path, '2026-07-01', {'c': ['updated']}), 'updated')
            self.assertEqual(self.write_version(path, '2026-07-01', {'c': ['updated']}), 'same')
            self.assertEqual(pickle.loads(path.read_bytes())['versions'][0], first)

    def test_invalid_or_future_date_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.pkl'
            for date in ['bad', '20260701', '2026-07-03']:
                with self.subTest(date=date), self.assertRaises(ValueError):
                    self.write_version(path, '2026-07-02', {'a': ['first']}, date=date)
                self.assertFalse(path.exists())

    def test_refresh_failure_blocks_success(self):
        boards = pd.DataFrame({'f12': [f'BK{i}' for i in range(400)],
                               'f14': [f'concept{i}' for i in range(400)]})
        members = pd.DataFrame({'f12': ['000001']})
        for archive_error, consistency in [(RuntimeError('archive failed'), None),
                                            (None, types.SimpleNamespace(returncode=1, stdout='mismatch', stderr='')),
                                            (None, RuntimeError('guard failed')),
                                            (None, types.SimpleNamespace(returncode=0, stdout='', stderr=''))]:
            with self.subTest(archive_error=archive_error, consistency=consistency), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'map.pkl'
                path.write_bytes(pickle.dumps({'000001': ['old']}))
                record = Mock(side_effect=archive_error, return_value='added')
                fake_module = types.SimpleNamespace(record_version=record)
                checker = Mock(side_effect=consistency if isinstance(consistency, Exception) else None,
                               return_value=consistency)
                with patch.object(refresh, 'MAP', str(path)), \
                        patch.object(refresh, 'clist', side_effect=lambda fs, fields:
                                     boards if fs.startswith('m:') else members), \
                        patch.object(refresh.time, 'sleep'), \
                        patch.dict(sys.modules, {'concept_map_version': fake_module}), \
                        patch.object(sys, 'path', sys.path.copy()), \
                        patch('subprocess.run', checker), contextlib.redirect_stdout(io.StringIO()):
                    result = refresh.main()
                expected_success = archive_error is None and not isinstance(consistency, Exception) \
                    and consistency.returncode == 0
                self.assertEqual(result, 0 if expected_success else 1)
                if archive_error:
                    checker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
