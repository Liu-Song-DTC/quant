"""A rejected model file must not turn a working predictor into zero forecasts."""
from pathlib import Path
import contextlib
import io
import pickle
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class MLModelLoadStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(351)
        dates = pd.bdate_range('2025-01-01', periods=100)
        frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), 100),
            'a': rng.normal(size=1200), 'b': rng.normal(size=1200), 'c': rng.normal(size=1200),
            'future_ret': rng.normal(size=1200)})
        frame['label_end_date'] = frame.date + pd.offsets.BDay(5)
        cls.config = {'ml': {'feature_whitelist': ['a', 'b', 'c'], 'log_context': 'LOAD_STATE_RANDOM_DATA'}}
        trained = MLFactorPredictor(cls.config)
        trained.xgb_params.update(n_jobs=1, n_estimators=4, early_stopping_rounds=2)
        with contextlib.redirect_stdout(io.StringIO()):
            assert trained.train(frame, {'regime': 0}) is not None
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = str(Path(cls.directory.name) / 'model.json')
        trained.save_model(cls.path)
        cls.metadata = Path(cls.path + '.meta').read_bytes()
        cls.request = frame[frame.date == dates[-1]]
        cls.expected = trained.predict(cls.request, {'regime': 0})

    def setUp(self):
        Path(self.path + '.meta').write_bytes(self.metadata)
        self.loaded = MLFactorPredictor(self.config)
        self.loaded.load_model(self.path)

    def test_late_metadata_failure_preserves_all_previous_state_and_predictions(self):
        before = self.loaded.__dict__.copy()
        bad = pickle.loads(self.metadata)
        bad['feature_cols'] = ['broken']
        bad['trained_features'] = ['broken']
        bad.pop('cross_feature_pairs')
        Path(self.path + '.meta').write_bytes(pickle.dumps(bad))
        with self.assertRaises(ValueError):
            self.loaded.load_model(self.path)
        self.assertEqual(self.loaded.__dict__.keys(), before.keys())
        for key, value in before.items():
            self.assertIs(self.loaded.__dict__[key], value, key)
        np.testing.assert_array_equal(self.loaded.predict(self.request, {'regime': 0}), self.expected)

    def test_valid_load_is_identical_and_inconsistent_feature_contract_is_rejected(self):
        np.testing.assert_array_equal(self.loaded.predict(self.request, {'regime': 0}), self.expected)
        before = self.loaded.model
        bad = pickle.loads(self.metadata)
        bad['trained_features'] = bad['trained_features'][:-1]
        Path(self.path + '.meta').write_bytes(pickle.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'feature schema'):
            self.loaded.load_model(self.path)
        self.assertIs(self.loaded.model, before)
        np.testing.assert_array_equal(self.loaded.predict(self.request, {'regime': 0}), self.expected)


if __name__ == '__main__':
    unittest.main()
