import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class MarketFeatureScopeTests(unittest.TestCase):
    def test_shared_market_values_survive_while_stock_values_are_standardized(self):
        frame=pd.DataFrame({'date':['2026-10-20']*3+['2026-10-21']*3,
                            'stock':[1.,2.,3.,3.,4.,5.], 'market':[20.]*3+[30.]*3})
        transformed=MLFactorPredictor._cross_sectional_zscore(frame.copy(),['stock','market'],['market'])
        np.testing.assert_array_equal(transformed['market'],frame['market'])
        np.testing.assert_allclose(transformed['stock'],[-1.,0.,1.,-1.,0.,1.])
        legacy=MLFactorPredictor._cross_sectional_zscore(frame.copy(),['stock','market'])
        np.testing.assert_array_equal(legacy['market'],np.zeros(6))

    def test_conflicting_shared_inputs_and_label_scope_are_rejected(self):
        frame=pd.DataFrame({'date':['2026-10-20']*3,'market':[20.,21.,20.]})
        with self.assertRaises(ValueError):
            MLFactorPredictor._cross_sectional_zscore(frame,['market'],['market'])
        for names in ['market',['future_ret'],['label_end_date']]:
            with self.assertRaises(ValueError):
                MLFactorPredictor({'ml':{'market_feature_cols':names}})

    def test_model_retains_its_fitted_scope_after_loading_under_different_config(self):
        rng=np.random.default_rng(14);dates=pd.bdate_range('2025-01-01',periods=100)
        frame=pd.DataFrame({'date':np.repeat(dates,12),'code':np.tile(np.arange(12),100),
                            'a':rng.normal(size=1200),'b':rng.normal(size=1200),
                            'market':np.repeat(np.sin(np.arange(100)/10),12),
                            'future_ret':rng.normal(size=1200)})
        frame['label_end_date']=frame['date']+pd.offsets.BDay(5)
        config={'ml':{'feature_whitelist':['a','b','market'],'market_feature_cols':['market'],
                      'log_context':'REGRESSION_RANDOM_DATA'}}
        model=MLFactorPredictor(config);model.xgb_params.update(n_estimators=4,n_jobs=1,early_stopping_rounds=2)
        with contextlib.redirect_stdout(io.StringIO()):model.train(frame)
        self.assertTrue(all('market' not in pair for pair in model._cross_feature_pairs))
        last=frame[frame['date']==dates[-1]];prediction=model.predict(last)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'model.json');model.save_model(path)
            restored=MLFactorPredictor({'ml':{'feature_whitelist':['a','b','market']}})
            restored.load_model(path)
            self.assertEqual(restored._market_feature_cols,['market'])
            np.testing.assert_array_equal(restored.predict(last),prediction)


if __name__=='__main__':unittest.main()
