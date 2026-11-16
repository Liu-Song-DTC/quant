"""Signal-only dependencies must invalidate signals without invalidating raw values."""
import ast
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

STRATEGY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY))
from core import factor_preparer as fp


class SignalDependencyFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='signal_dependency_test_')
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        for rel in set(fp._SIGNAL_CODE_FILES) | {'core/divergence_detector.py'}:
            dst = self.project/'strategy'/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(STRATEGY/rel, dst)
        quarter = self.project/'strategy/config/quarterly_factors'
        quarter.mkdir(parents=True, exist_ok=True)
        (quarter/'2021Q1.yaml').write_text('test: 1\n')
        source = (STRATEGY/'bt_execution.py').read_text()
        fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == '_signal_code_fingerprint')
        ns = dict(os=os, __file__=str(self.project/'strategy/bt_execution.py'),
                  _PROJECT_DIR=str(self.project), DATA_PATH=str(self.project/'data/prices'),
                  _resolve_cmap_hist=lambda: None, _FACTOR_CODE_FILES=fp._FACTOR_CODE_FILES,
                  _yaml_stripped_digest=fp._yaml_stripped_digest)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(STRATEGY/'bt_execution.py'), 'exec'), ns)
        self.signal = ns['_signal_code_fingerprint']
        fn = next(node for node in ast.parse((STRATEGY/'core/factor_preparer.py').read_text()).body if isinstance(node, ast.FunctionDef) and node.name == '_raw_code_fingerprint')
        ns = dict(fp.__dict__); ns['__file__'] = str(self.project/'strategy/core/factor_preparer.py')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(STRATEGY/'core/factor_preparer.py'), 'exec'), ns)
        self.raw = ns['_raw_code_fingerprint']

    def test_divergence_change_invalidates_signal_but_preserves_raw_identity(self):
        signal, raw = self.signal(), self.raw()
        path = self.project/'strategy/core/divergence_detector.py'
        path.write_text(path.read_text()+'\n# changed detector content\n')
        self.assertNotEqual(self.signal(), signal)
        self.assertEqual(self.raw(), raw)

    def test_content_identity_ignores_touch_and_tracks_signal_engine_change(self):
        path = self.project/'strategy/core/signal_engine.py'
        signal, raw = self.signal(), self.raw()
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000000))
        self.assertEqual(self.signal(), signal)
        path.write_text(path.read_text()+'\n# changed signal source content\n')
        self.assertNotEqual(self.signal(), signal)
        self.assertEqual(self.raw(), raw)

    def test_b2_observation_source_invalidates_factor_and_signal_caches(self):
        signal, raw = self.signal(), self.raw()
        path = self.project/'strategy/core/b2_observation.py'
        path.write_text(path.read_text()+'\n# changed observation-time implementation\n')
        self.assertNotEqual(self.signal(), signal)
        self.assertNotEqual(self.raw(), raw)

    def test_industry_clock_policy_invalidates_signals_without_changing_raw(self):
        # The exported industry changes portfolio grouping, so it cannot share
        # a signal identity with the midpoint-clock policy. Raw factors do not
        # consume this switch and must retain their own identity.
        with patch.dict(os.environ, {'QUANT_INDUSTRY_PIT': '0'}):
            signal, raw = self.signal(), self.raw()
        with patch.dict(os.environ, {'QUANT_INDUSTRY_PIT': '1'}):
            self.assertNotEqual(self.signal(), signal)
            self.assertEqual(self.raw(), raw)


if __name__ == '__main__':
    unittest.main()
