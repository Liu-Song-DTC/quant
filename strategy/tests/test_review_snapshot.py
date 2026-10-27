"""Frozen-input and explicit concept prehistory policies; temporary fixtures only."""
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import review_validation as review
from core.concept_heat import ConceptHeatCalculator
from core import cmap_hist


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.price = self.data / 'stock_data/backtrader_data/000001_qfq.csv'
        self.price.parent.mkdir(parents=True)
        self.price.write_text('close\n10\n')
        self.root_patch = patch.object(review, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_content_rewrite_with_same_stat_is_detected(self):
        before, _ = review.input_stamp({})
        stat = self.price.stat()
        self.price.write_text('close\n20\n')
        os.utime(self.price, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        after, _ = review.input_stamp({})
        self.assertNotEqual(before['sha256'], after['sha256'])

    def test_snapshot_is_independent_and_tampering_rejected(self):
        nested = self.data / 'alternative_data/cb_daily/a.csv'
        nested.parent.mkdir(parents=True)
        nested.write_text('close\n20\n')
        target = self.root / 'run'
        target.mkdir()
        seal = review.freeze_data(target, {})
        frozen = target / 'data/stock_data/backtrader_data/000001_qfq.csv'
        self.assertNotEqual(self.price.stat().st_ino, frozen.stat().st_ino)
        self.price.write_text('close\n30\n')
        self.assertEqual(review.verify_snapshot(target)[0]['sha256'], seal['sha256'])
        frozen.chmod(0o644)
        frozen.write_text('close\n40\n')
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            review.verify_snapshot(target)

    def test_hfq_and_external_override_rejected(self):
        target = self.root / 'run'
        target.mkdir()
        with self.assertRaises(ValueError):
            review.freeze_data(target, {'QUANT_DATA_PATH': '/elsewhere'})
        self.price.with_name('000001_hfq.csv').write_text('close\n20\n')
        with self.assertRaisesRegex(ValueError, 'qfq-only'):
            review.freeze_data(target, {})

    def test_legacy_run_cannot_be_replayed(self):
        with self.assertRaisesRegex(RuntimeError, 'no sealed'):
            review.verify_snapshot(self.root)

    def run_fixture(self):
        from types import SimpleNamespace
        target = self.root / 'run'
        target.mkdir()
        seal = review.freeze_data(target, {})
        (target / 'manifest.json').write_text(json.dumps({'source_sha256': {}, 'data_sha256': seal['sha256']}))
        (target / 'strategy/rolling_validation_results').mkdir(parents=True)
        return target, SimpleNamespace(returncode=0)

    def test_run_scans_twice_and_rejects_content_change_during_execution(self):
        target, success = self.run_fixture()
        price = target / 'data/stock_data/backtrader_data/000001_qfq.csv'
        def mutate(*args, **kwargs):
            old = price.stat()
            price.chmod(0o644)
            price.write_text('close\n20\n')
            os.utime(price, ns=(old.st_atime_ns, old.st_mtime_ns))
            return success
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(review, 'input_stamp', wraps=review.input_stamp) as scan, \
                patch.object(review.subprocess, 'run', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'input data changed'):
                review.run(target)
            self.assertEqual(scan.call_count, 2)
        self.assertFalse(json.loads((target / 'corrected_input_stamp.json').read_text())['unchanged'])
        self.assertFalse((target / 'corrected').exists())

    def test_run_rejects_tampering_before_launch_and_accepts_unchanged_inputs(self):
        target, success = self.run_fixture()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(review, 'input_stamp', wraps=review.input_stamp) as scan, \
                patch.object(review.subprocess, 'run', return_value=success) as launch:
            review.run(target)
            self.assertEqual(scan.call_count, 2)
            self.assertEqual(launch.call_count, 1)
        self.assertTrue(json.loads((target / 'corrected_input_stamp.json').read_text())['unchanged'])
        price = target / 'data/stock_data/backtrader_data/000001_qfq.csv'
        price.chmod(0o644)
        price.write_text('close\n30\n')
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(review.subprocess, 'run', return_value=success) as launch:
            with self.assertRaisesRegex(RuntimeError, 'differs from seal'):
                review.run(target)
            launch.assert_not_called()

    def test_mtime_only_change_preserves_content_identity(self):
        before, _ = review.input_stamp({})
        stat = self.price.stat()
        os.utime(self.price, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        after, _ = review.input_stamp({})
        self.assertEqual(before, after)

    def test_first_proxy_default_and_empty_control_use_identical_data(self):
        import ast
        from types import SimpleNamespace
        from typing import Optional
        import pandas as pd
        source_root = Path(__file__).resolve().parents[1]
        core = self.root / 'strategy/core'
        core.mkdir(parents=True)
        config = self.root / 'strategy/config'
        config.mkdir()
        (config / 'factor_config.yaml').write_text('factor_mode: fixed\n')
        for name in ('signal_engine.py', 'concept_heat.py'):
            (core / name).write_bytes((source_root / 'core' / name).read_bytes())
        proxy = review.prepare('proxy')
        empty = review.prepare('empty', snapshot_from=proxy, strict_concept_boundary=True)
        manifests = [json.loads((p / 'manifest.json').read_text()) for p in (proxy, empty)]
        self.assertEqual(manifests[0]['data_sha256'], manifests[1]['data_sha256'])
        self.assertEqual(manifests[0]['concept_boundary'], 'first_snapshot_proxy_before_first_version')
        self.assertEqual(manifests[1]['concept_boundary'], 'empty_before_first_version')
        changed = {k for k, v in manifests[0]['source_sha256'].items()
                   if manifests[1]['source_sha256'].get(k) != v}
        self.assertEqual(changed, {'strategy/core/concept_heat.py', 'strategy/core/signal_engine.py'})
        versions = [('2026-05-31', {'a': ['first']}), ('2026-06-30', {'b': ['second']})]
        for target, expected_map, expected_version in (
                (proxy, {'a': ['first']}, '2026-05-31'), (empty, {}, '')):
            signal_tree = ast.parse((target / 'strategy/core/signal_engine.py').read_text())
            node = next(n for n in signal_tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_concept_map_asof')
            ns = {'pd': pd, 'CONCEPT_MAP_HIST': versions, '_CONCEPT_MAP_HIST_CACHE': {}}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<signal-lookup>', 'exec'), ns)
            self.assertEqual(ns['_concept_map_asof']('2021-01-01'), (expected_map, expected_version))
            self.assertEqual(ns['_concept_map_asof']('2026-07-01'), ({'b': ['second']}, '2026-06-30'))
            heat_tree = ast.parse((target / 'strategy/core/concept_heat.py').read_text())
            cls = next(n for n in heat_tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'ConceptHeatCalculator')
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_version_for')
            ns = {'pd': pd, 'Optional': Optional}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<heat-lookup>', 'exec'), ns)
            instance = SimpleNamespace(_cmap_versions=versions, _cmap_ver_cache={})
            self.assertEqual(ns['_version_for'](instance, '2021-01-01'), expected_version)
            self.assertEqual(ns['_version_for'](instance, '2026-07-01'), '2026-06-30')


class ConceptPITTests(unittest.TestCase):
    def test_heat_before_first_version_and_backward_queries(self):
        calc = ConceptHeatCalculator('unused')
        calc._stock_concepts = {'future': ['future']}
        calc._concept_stocks = {'future': ['future']}
        calc._cmap_versions = [('2026-05-31', {'a': ['first']}),
                               ('2026-06-30', {'b': ['second']})]
        self.assertEqual(calc.stock_concepts_asof('2026-05-30')[0], {'a': ['first']})
        self.assertEqual(calc.stock_concepts_asof('2026-05-31')[0], {'a': ['first']})
        calc._apply_map_asof('2026-07-01')
        self.assertEqual(calc._stock_concepts, {'b': ['second']})
        calc._apply_map_asof('2021-01-01')
        self.assertEqual(calc._stock_concepts, {'a': ['first']})
        self.assertEqual(calc.stock_concepts_asof('2026-06-29')[0], {'a': ['first']})

    def test_signal_consumer_has_same_boundary(self):
        # Exercise the real pure lookup without importing the engine's production data.
        import ast
        import pandas as pd
        source = Path(__file__).resolve().parents[1] / 'core/signal_engine.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_concept_map_asof')
        ns = {'pd': pd, 'CONCEPT_MAP_HIST': [('2026-05-31', {'a': ['first']})],
              '_CONCEPT_MAP_HIST_CACHE': {}, 'STOCK_CONCEPT_MAP': {'future': ['future']}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        lookup = ns['_concept_map_asof']
        self.assertEqual(lookup('2026-05-30'), ({'a': ['first']}, '2026-05-31'))
        self.assertEqual(lookup('2026-05-31'), ({'a': ['first']}, '2026-05-31'))

    def test_missing_or_corrupt_history_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp) / 'history.pkl'
            mapping = Path(tmp) / 'map.pkl'
            mapping.write_bytes(pickle.dumps({'future': ['future']}))
            with patch.dict(os.environ, {'QUANT_CONCEPT_MAP_HIST': str(hist)}):
                with self.assertRaises(FileNotFoundError):
                    cmap_hist.resolve_hist_path()
                hist.write_bytes(pickle.dumps({'versions': []}))
                with self.assertRaisesRegex(RuntimeError, 'invalid concept history'):
                    ConceptHeatCalculator(str(mapping)).load()


if __name__ == '__main__':
    unittest.main()
