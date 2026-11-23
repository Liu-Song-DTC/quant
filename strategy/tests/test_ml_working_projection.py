"""Unused inputs preserve model behavior and market-input validation."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


def population():
    rng = np.random.default_rng(711)
    dates = pd.bdate_range('2024-01-02', periods=100)
    size = len(dates) * 12
    frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), len(dates)),
                          'a': rng.normal(size=size).astype('float32'),
                          'b': rng.normal(size=size), 'c': rng.normal(size=size)})
    frame['future_ret'] = .01 * frame.a - .005 * frame.b + rng.normal(0, .01, size)
    frame['label_end_date'] = frame.date + pd.offsets.BDay(5)
    frame.loc[[0, 15], ['a', 'b', 'c']] = np.nan
    return frame


def predictor(whitelist):
    model = MLFactorPredictor({'ml': {'feature_whitelist': whitelist,
                                     'use_cross_features': True, 'use_regime_features': True,
                                     'log_context': 'WORKING_PROJECTION_TEST'}})
    model.xgb_params.update(n_estimators=4, max_depth=2, n_jobs=1, early_stopping_rounds=2)
    return model


def test_unused_columns_do_not_change_model_or_hide_input_contract():
    data = population()
    before = data.copy(deep=True)
    extended = data.assign(unused_label_echo=data.future_ret, unused_infinite=np.inf)
    narrow, wide = predictor(['c', 'a', 'b']), predictor(['c', 'a', 'b'])
    regime = {'by_date': {str(day.date()): i % 3 - 1 for i, day in enumerate(data.date.unique())}}
    assert narrow.train(data, regime) == wide.train(extended, regime)
    assert wide._trained_features == narrow._trained_features
    assert wide._trained_features[:3] == ['a', 'b', 'c']
    assert wide.validation_summary == narrow.validation_summary
    assert wide._cross_feature_pairs == narrow._cross_feature_pairs
    np.testing.assert_array_equal(wide.predict(data, regime), narrow.predict(data, regime))
    assert set(wide.feature_contract['generated_but_not_requested']) == {'unused_label_echo', 'unused_infinite'}
    pd.testing.assert_frame_equal(data, before, check_exact=True)


@pytest.mark.parametrize('whitelist', [None, []])
def test_no_whitelist_keeps_all_available_features(whitelist):
    data = population().assign(extra_numeric=np.tile(np.arange(12), 100).astype('float32'))
    model = predictor(whitelist)
    assert model.train(data) is not None
    assert 'extra_numeric' in model._trained_features
    assert 'extra_numeric' in model.feature_contract['available_base']


def test_unused_market_feature_still_requires_one_observation_per_date():
    data = population().assign(unused_market=np.tile(np.arange(12), 100).astype('float32'))
    model = predictor(['a', 'b', 'c'])
    model = MLFactorPredictor({'ml': {**deepcopy(model.config['ml']),
                                     'market_feature_cols': ['unused_market']}})
    with pytest.raises(ValueError, match='market feature differs across stocks'):
        model.train(data)
    assert model.model is None
