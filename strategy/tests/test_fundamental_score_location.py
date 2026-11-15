"""Score calibration preserves quality differences and the pre-calibration time boundary."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
ENGINE=Path(__file__).resolve().parents[1]/'core/signal_engine.py'
LOCATION_OFFSET=0.3641060497319204

def score_prefix(grades,dates,technical=None):
    tree=ast.parse(ENGINE.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_vectorized_score_assembly')
    nodes=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='limit_pct' for t in node.targets):
            break
        nodes.append(deepcopy(node))
    grades=np.asarray(grades,dtype=float)
    technical=np.zeros(len(grades)) if technical is None else np.asarray(technical,dtype=float)
    inputs={'valid':np.ones(len(grades),dtype=bool),'factor_value':technical,'fundamental_score':grades}
    before={k:v.copy() for k,v in inputs.items()}
    ns={'np':np,'pd':pd,'s':inputs,'dates':dates,'self':SimpleNamespace(fundamental_weight=.3)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ENGINE),'exec'),ns)
    return ns,inputs,before

def test_positive_quality_gaps_remain_in_the_trading_score():
    ns,_,_=score_prefix([.1,.5,.9],pd.to_datetime(['2021-01-04']*3))
    np.testing.assert_allclose(np.diff(ns['score']),[.12,.12],rtol=0,atol=1e-15)
    np.testing.assert_allclose(ns['score'],.3*(np.array([.1,.5,.9])+LOCATION_OFFSET),rtol=0,atol=1e-15)

def test_zero_negative_and_missing_quality_receive_only_the_technical_score():
    ns,_,_=score_prefix([0,-.1,np.nan],pd.to_datetime(['2021-01-04']*3),[.3,-.1,.8])
    np.testing.assert_array_equal(ns['score'],[.3,-.1,.8])

def test_mixed_dates_apply_the_offset_after_the_training_window_only():
    ns,_,_=score_prefix([.5,.5,.5],pd.to_datetime(['2020-12-31','2021-01-04',None]))
    np.testing.assert_allclose(ns['score'],[.15,.15+.3*LOCATION_OFFSET,.15],rtol=0,atol=1e-15)

def test_absent_dates_keep_the_uncalibrated_score():
    ns,_,_=score_prefix([.1,.5,.9],None)
    np.testing.assert_allclose(ns['score'],[.03,.15,.27],rtol=0,atol=1e-15)

def test_calibration_keeps_raw_quality_and_technical_inputs_unchanged():
    ns,inputs,before=score_prefix([.1,.5,.9],pd.to_datetime(['2026-10-28']*3),[.2,-.3,.4])
    for key in before:
        np.testing.assert_array_equal(inputs[key],before[key])
    np.testing.assert_array_equal(ns['pre_discount_score'],[.2,-.3,.4])
