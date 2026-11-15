"""Existing independent strategy adjustments must affect the slot decision."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

PORTFOLIO=Path(__file__).resolve().parents[1] / 'core/portfolio.py'

def apply_before_limit(rows,weights,enabled=True,reserved=()):
    tree=ast.parse(PORTFOLIO.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_desired_value')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='compute_weights']
    assert len(calls)==1
    pre=next(n for n in fn.body if isinstance(n,ast.If) and any(c is calls[0] for c in ast.walk(n)))
    sort=max((n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='qualified' and n.value.func.attr=='sort'),key=lambda n:n.lineno)
    appends=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='selected' and n.func.attr=='append']
    assert calls[0].lineno<sort.lineno<min(n.lineno for n in appends)
    count=[]
    def compute(candidates,*args):
        count.append([c['code'] for c in candidates])
        return dict(weights)
    ms=SimpleNamespace(enabled=enabled,compute_weights=compute)
    qualified=deepcopy(rows)
    ns=dict(self=SimpleNamespace(_multi_strategy=ms),qualified=qualified,_bp2_reserved=[c for c in qualified if c['code'] in reserved],signal_store=None,market_regime=1,drawdown=0.)
    exec(compile(ast.Module(body=[deepcopy(pre),deepcopy(sort)],type_ignores=[]),str(PORTFOLIO),'exec'),ns)
    return ns['qualified'],ns['_bp2_reserved'],count

def row(code,score,locked=False):
    return dict(code=code,effective_score=score,_locked=locked)

def test_discounted_old_pick_loses_slot_to_stronger_adjusted_candidate():
    result,_,calls=apply_before_limit([row('old_pick',.8268),row('fresh',.7602)],{'old_pick':.8,'fresh':1.})
    assert result[0]['code']=='fresh'
    assert abs(result[1]['effective_score']-.66144)<1e-12
    assert len(calls)==1 and set(calls[0])=={'old_pick','fresh'}

def test_disabled_strategy_adjustment_keeps_existing_selection():
    result,_,calls=apply_before_limit([row('old_pick',.8268),row('fresh',.7602)],{},enabled=False)
    assert result[0]['code']=='old_pick' and not calls

def test_hard_reject_cannot_return_through_a_reserved_channel():
    result,reserved,_=apply_before_limit([row('rejected',1.2,True),row('valid',.4)],{'rejected':0.,'valid':1.},reserved=['rejected'])
    assert [c['code'] for c in result]==['valid'] and not reserved

def test_valid_minimum_holding_lock_is_preserved():
    result,_,_=apply_before_limit([row('locked',.2,True),row('fresh',.9)],{'locked':.8,'fresh':1.})
    assert result[0]['code']=='locked'
