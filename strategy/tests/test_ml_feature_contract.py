import contextlib
import io
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class MLFeatureContractTests(unittest.TestCase):
    def test_missing_configured_features_are_visible_and_inputs_are_unchanged(self):
        frame = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=3),
                              'code': ['1', '2', '3'], 'a': [1., 2., 3.],
                              'b': ['bad', 'input', 'column'], 'unused': [2., 3., 4.]})
        original = frame.copy(deep=True)
        report = MLFactorPredictor.describe_feature_contract(frame, ['a', 'b', 'missing'])
        self.assertEqual(report['available_base'], ['a'])
        self.assertEqual(report['absent'], ['missing'])
        self.assertEqual(report['non_numeric'], ['b'])
        self.assertEqual(report['generated_but_not_requested'], ['unused'])
        pd.testing.assert_frame_equal(frame, original)

    def test_training_diagnostics_and_validation_survive_model_save_without_changing_predictions(self):
        rng = np.random.default_rng(17)
        dates = pd.bdate_range('2025-01-01', periods=100)
        frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), 100),
                              'a': rng.normal(size=1200), 'b': rng.normal(size=1200),
                              'c': rng.normal(size=1200), 'future_ret': rng.normal(size=1200)})
        frame['label_end_date'] = frame['date'] + pd.offsets.BDay(5)
        config = {'ml': {'feature_whitelist': ['a', 'b', 'c', 'missing'],
                         'log_context': 'REGRESSION_RANDOM_DATA'}}
        predictor = MLFactorPredictor(config)
        predictor.xgb_params.update(n_jobs=1, n_estimators=4, early_stopping_rounds=2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            predictor.train(frame)
        self.assertIn('[ML][FEATURE_CONTRACT]', output.getvalue())
        self.assertEqual(predictor.feature_contract['absent'], ['missing'])
        last = frame[frame['date'] == dates[-1]]
        expected = predictor.predict(last)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.json')
            predictor.save_model(path)
            restored = MLFactorPredictor(config)
            restored.load_model(path)
            self.assertEqual(restored.feature_contract, predictor.feature_contract)
            self.assertEqual(restored.validation_summary, predictor.validation_summary)
            np.testing.assert_array_equal(expected, restored.predict(last))
            # Historical metadata has no schema or validation evidence. Loading
            # must remain compatible, while marking that evidence unavailable.
            meta_path = Path(path + '.meta')
            meta = pickle.loads(meta_path.read_bytes())
            meta.pop('feature_contract'); meta.pop('validation_summary')
            meta_path.write_bytes(pickle.dumps(meta))
            restored.load_model(path)
            self.assertEqual(restored.feature_contract['status'], 'legacy_metadata_unavailable')
            self.assertIsNone(restored.validation_summary)
            np.testing.assert_array_equal(expected, restored.predict(last))


if __name__ == '__main__':
    unittest.main()
