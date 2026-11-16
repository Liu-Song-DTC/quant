"""Market state cannot be averaged from unrelated rows or lost during sorting."""
from pathlib import Path
import contextlib
import io
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class MLRegimeAlignmentTests(unittest.TestCase):
    def test_mismatched_or_nonvector_regimes_are_rejected_instead_of_averaged(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2025-01-02'] * 2),
                              'a': [1., 2.], 'b': [2., 1.]})
        for values in ([1., 1., 1.], [1., 1., 1., -1.], pd.Series([1., 1., 1.]),
                       np.ones((2, 1)), np.array(1.)):
            with self.subTest(values=str(values)):
                model = MLFactorPredictor({'ml': {'use_cross_features': False}})
                with self.assertRaisesRegex(ValueError, 'one-dimensional.*match feature rows'):
                    model.prepare_features(frame.copy(), {'regime': values}, is_train=True)
        model = MLFactorPredictor({'ml': {'use_cross_features': False}})
        before = model.prepare_features(frame.copy(), {'by_date': {'2025-01-02': 1}}, is_train=True)
        after = model.prepare_features(frame.copy(), {'by_date': {'2025-01-02': 1, '2099-01-01': -1}}, is_train=True)
        pd.testing.assert_frame_equal(before, after, check_exact=True)

    def test_training_row_states_survive_filtering_and_sorting(self):
        dates = pd.bdate_range('2025-01-01', periods=100)
        rng = np.random.default_rng(905)
        frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), 100),
            'a': rng.normal(size=1200), 'b': rng.normal(size=1200), 'c': rng.normal(size=1200),
            'future_ret': rng.normal(size=1200)})
        frame['label_end_date'] = frame.date + pd.offsets.BDay(5)
        frame = frame.sample(frac=1, random_state=31).copy()
        frame.loc[frame.index[:7], ['a', 'b', 'c']] = np.nan
        states = {str(day.date()): float((i % 3) - 1) for i, day in enumerate(dates)}
        observations = frame.date.dt.strftime('%Y-%m-%d').map(states).to_numpy()
        config = {'ml': {'feature_whitelist': ['a', 'b', 'c'], 'use_regime_features': True,
                         'log_context': 'REGIME_ALIGNMENT_RANDOM_DATA'}}
        models = []
        for regime in ({'by_date': states}, {'regime': observations}):
            model = MLFactorPredictor(config)
            model.xgb_params.update(n_jobs=1, n_estimators=4, early_stopping_rounds=2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNotNone(model.train(frame, regime))
            models.append(model)
        left, right = models
        self.assertEqual(left._trained_features, right._trained_features)
        for a, b in zip(left._ensemble_models, right._ensemble_models):
            self.assertEqual(a.get_booster().save_raw(raw_format='json'), b.get_booster().save_raw(raw_format='json'))
        last = frame[frame.date == dates[-1]]
        np.testing.assert_array_equal(left.predict(last, {'by_date': states}), right.predict(last, {'by_date': states}))


if __name__ == '__main__':
    unittest.main()
