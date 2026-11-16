"""Label-maturity metadata is never a factor or an implicit epoch timestamp."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class MLMetadataGuardTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'code': ['A', 'B', 'A', 'B'],
            'date': pd.to_datetime(['2025-01-02', '2025-01-02', '2025-01-03', '2025-01-03']),
            'a': [1., 2., 3., 4.], 'b': [4., 3., 2., 1.],
            'future_ret': [.1, -.1, .2, -.2],
            'label_end_date': [20250106, 20250106, 20250107, 20250107]})
        self.model = MLFactorPredictor({'ml': {'use_cross_features': False,
                                               'use_regime_features': False}})

    def test_maturity_is_excluded_even_when_numeric(self):
        self.model.prepare_features(self.frame.copy(), is_train=True)
        self.assertEqual(self.model.feature_cols, ['a', 'b'])
        contract = self.model.describe_feature_contract(self.frame)
        self.assertEqual(contract['available_base'], ['a', 'b'])

    def test_ambiguous_maturity_rejected_before_fit_and_old_model_is_preserved(self):
        previous = object()
        self.model.model = previous
        for dtype in ('int64', 'float64', 'Int64'):
            with self.subTest(dtype=dtype):
                frame = self.frame.copy()
                frame['label_end_date'] = frame.label_end_date.astype(dtype)
                with patch('xgboost.XGBRegressor.fit') as fit:
                    with self.assertRaisesRegex(ValueError, 'label_end_date.*explicit datetime'):
                        self.model.train(frame)
                    fit.assert_not_called()
                self.assertIs(self.model.model, previous)
                with self.assertRaisesRegex(ValueError, 'label_end_date.*explicit datetime'):
                    self.model.split_masks(frame, pd.Timestamp('2025-01-03'))


if __name__ == '__main__':
    unittest.main()
