"""ML prediction coverage must not overwrite signal/factor eligibility."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


def assembly_kernel():
        path=Path(__file__).resolve().parents[1]/'core/signal_engine.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SignalEngine')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_vectorized_score_assembly')
        def assigns(node,name):
            return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        start=next(i for i,n in enumerate(method.body) if assigns(n,'ml_normalized'))
        end=next(i for i,n in enumerate(method.body) if i>start and assigns(n,'dt_signal'))
        kernel=compile(ast.Module(body=deepcopy(method.body[start:end]),type_ignores=[]),str(path),'exec')
        return kernel


class MLSignalValidityTests(unittest.TestCase):
    def test_cached_ml_lookup_keeps_original_signal_eligibility(self):
        kernel=assembly_kernel()
        dates=pd.date_range('2026-10-20',periods=4)
        eligibility=np.array([True,False,True,False])
        engine=SimpleNamespace(ml_enabled=True,ml_predictor=None,ml_blend_weight=.4,
            _ml_predictions={('000001',dates[1]):.1},
            _ml_cache={'000001':(np.array(dates[1:]),np.array([.1,.2,.3]))})
        ns=dict(np=np,pd=pd,self=engine,n=4,code='000001',dates=dates,
                valid=eligibility.copy(),adjusted_score=np.ones(4))
        exec(kernel,ns)
        np.testing.assert_array_equal(ns['valid'],eligibility)
        np.testing.assert_allclose(ns['ml_normalized'],np.tanh(np.array([0.,.1,.2,.3])*3))

    def execute(self,engine,dates,valid):
        ns=dict(np=np,pd=pd,self=engine,n=len(dates),code='000001',dates=dates,
                valid=valid.copy(),adjusted_score=np.ones(len(dates)))
        exec(assembly_kernel(),ns)
        return ns

    def test_required_predictions_intersect_factor_validity_and_zero_is_available(self):
        dates=pd.date_range('2026-10-20',periods=3)
        engine=SimpleNamespace(ml_enabled=True,ml_predictor=None,ml_blend_weight=.4,
            ml_coverage_policy='require_prediction',_ml_predictions={('000001',dates[0]):0.,
                                                                    ('000001',dates[1]):.2})
        result=self.execute(engine,dates,np.array([True,False,True]))
        np.testing.assert_array_equal(result['ml_prediction_available'],[True,True,False])
        np.testing.assert_array_equal(result['valid'],[True,False,False])

    def test_future_cache_appearance_does_not_rewrite_past_eligibility(self):
        dates=pd.date_range('2026-10-20',periods=2)
        future=pd.Timestamp('2026-10-25')
        engine=SimpleNamespace(ml_enabled=True,ml_predictor=None,ml_blend_weight=.4,
            ml_coverage_policy='require_prediction',_ml_predictions={('other',future):.2})
        before=self.execute(engine,dates,np.ones(2,dtype=bool))
        engine._ml_cache={'000001':(np.array([future]),np.array([.2]))}
        after=self.execute(engine,dates,np.ones(2,dtype=bool))
        np.testing.assert_array_equal(before['valid'],after['valid'])
        np.testing.assert_array_equal(after['valid'],[False,False])


if __name__=='__main__':
    unittest.main()
