import contextlib
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class ModelPreprocessingContractTests(unittest.TestCase):
    def test_saved_model_owns_feature_scope_order_flags_and_fill_threshold(self):
        rng=np.random.default_rng(31);dates=pd.bdate_range('2025-01-01',periods=100)
        frame=pd.DataFrame({'date':np.repeat(dates,12),'code':np.tile(np.arange(12),100),
                            'a':rng.normal(size=1200),'b':rng.normal(size=1200),
                            'c':rng.normal(size=1200),'future_ret':rng.normal(size=1200)})
        frame['label_end_date']=frame['date']+pd.offsets.BDay(5)
        for day in dates:
            mask=frame.date==day
            for column in ['a','b','c']:
                frame.loc[mask&(frame.code==11),column]=frame.loc[mask&(frame.code!=11),column].mean()
        config={'ml':{'feature_whitelist':['a','b','c'],'use_fill_indicator':True,
                      'fill_tol':1e-7,'use_cross_features':True,'use_regime_features':True,
                      'log_context':'REGRESSION_RANDOM_DATA'}}
        model=MLFactorPredictor(config);model.xgb_params.update(n_estimators=4,n_jobs=1,early_stopping_rounds=2)
        with contextlib.redirect_stdout(io.StringIO()):model.train(frame,{'by_date':{}})
        last=frame[frame.date==dates[-1]]
        expected=model.predict(last,{'regime':1})
        config['ml']['feature_whitelist']=['a']
        np.testing.assert_array_equal(expected,model.predict(last,{'regime':1}))
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'model.json');model.save_model(path)
            loaded=MLFactorPredictor({'ml':{'feature_whitelist':['a'],
                                           'use_cross_features':False,'use_regime_features':False,
                                           'use_fill_indicator':False,'fill_tol':.1}})
            loaded.load_model(path)
            np.testing.assert_array_equal(expected,loaded.predict(last[last.columns[::-1]],{'regime':1}))
            self.assertTrue(loaded.use_fill_indicator)
            self.assertEqual(loaded.fill_tol,1e-7)
            self.assertTrue(loaded._use_cross_features);self.assertTrue(loaded._use_regime_features)
            # Invalid preprocessing evidence must fail before replacing a good model.
            before=loaded.model;meta_path=Path(path+'.meta')
            meta=pickle.loads(meta_path.read_bytes());meta['preprocessing']['version']=999
            meta_path.write_bytes(pickle.dumps(meta))
            with self.assertRaises(ValueError):loaded.load_model(path)
            self.assertIs(loaded.model,before)
        original_model=model.model
        original_members=list(model._ensemble_models)
        original_contract=model.feature_contract
        config['ml']['feature_whitelist']=['a','b','c']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(model.train(frame.iloc[:10],{'by_date':{}}))
        self.assertIs(model.feature_contract,original_contract)
        with patch('xgboost.XGBRegressor.fit',side_effect=RuntimeError('synthetic fit failure')):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                model.train(frame,{'by_date':{}})
        self.assertIs(model.model,original_model)
        self.assertEqual(model._ensemble_models,original_members)
        self.assertIs(model.feature_contract,original_contract)
        np.testing.assert_array_equal(model.predict(last,{'regime':1}),expected)


if __name__=='__main__':unittest.main()
