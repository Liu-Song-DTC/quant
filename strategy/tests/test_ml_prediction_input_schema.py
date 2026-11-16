"""A trained input disappearing is a feed error, not an ordinary NaN cell."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class RecordingMember:
    def __init__(self):
        self.calls = 0

    def predict(self, values):
        self.calls += 1
        return np.asarray(values).sum(axis=1)


class PredictionInputSchemaTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'date': pd.to_datetime(['2026-01-05'] * 3),
                                   'code': ['1', '2', '3'], 'a': [1., 2., 3.], 'b': [4., 2., 3.]})
        self.predictor = MLFactorPredictor({'ml': {'feature_whitelist': ['a', 'b', 'optional']}})
        self.member = RecordingMember()
        self.predictor.model = self.member
        self.predictor._ensemble_models = [self.member]
        self.predictor.feature_cols = ['a', 'b']
        self.predictor._trained_features = ['a', 'b']
        self.predictor._cross_feature_pairs = [('a', 'b')]
        self.predictor._regime_feature_sources = []

    def test_missing_trained_input_rejected_before_state_or_model_changes(self):
        before = self.predictor.__dict__.copy()
        with self.assertRaisesRegex(ValueError, "missing=\\['a'\\]"):
            self.predictor.predict(self.frame.drop(columns='a'))
        self.assertEqual(self.member.calls, 0)
        for key, value in before.items():
            self.assertIs(self.predictor.__dict__[key], value, key)

    def test_wrong_dtype_and_duplicate_names_are_rejected(self):
        bad = self.frame.copy()
        bad['a'] = ['1', '2', '3']
        with self.assertRaisesRegex(ValueError, 'unsupported_dtype'):
            self.predictor.predict(bad)
        duplicate = pd.concat([self.frame, self.frame[['a']]], axis=1)
        with self.assertRaisesRegex(ValueError, 'duplicate column'):
            self.predictor.predict(duplicate)
        self.assertEqual(self.member.calls, 0)

    def test_optional_untrained_name_and_nan_cells_remain_supported(self):
        request = self.frame.copy()
        request.loc[0, 'a'] = np.nan
        original = request.copy(deep=True)
        result = self.predictor.predict(request)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.isfinite(result).all())
        pd.testing.assert_frame_equal(request, original, check_exact=True)

    def test_cross_and_nested_regime_cross_require_their_inputs(self):
        for name, sources in [('cross_a_b', []), ('regime_cross_a_b', ['cross_a_b'])]:
            with self.subTest(name=name):
                self.predictor._trained_features = [name]
                self.predictor._regime_feature_sources = sources
                self.assertEqual(len(self.predictor.predict(self.frame, {'regime': 1})), 3)
                count = self.member.calls
                with self.assertRaisesRegex(ValueError, "missing=\\['a'\\]"):
                    self.predictor.predict(self.frame.drop(columns='a'), {'regime': 1})
                self.assertEqual(self.member.calls, count)

    def test_precomputed_cross_input_is_still_supported(self):
        self.predictor._trained_features = ['cross_a_b']
        request = self.frame[['code', 'date']].copy()
        request['cross_a_b'] = [.2, .3, .4]
        result = self.predictor.predict(request)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.isfinite(result).all())

    def test_no_model_keeps_existing_zero_prediction_behavior(self):
        np.testing.assert_array_equal(MLFactorPredictor().predict(self.frame), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
