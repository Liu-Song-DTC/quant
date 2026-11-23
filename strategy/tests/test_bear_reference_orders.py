from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.strategy import Strategy
from analysis.bear_reference_orders import reference_plan
from analysis.bear_reference_orders import attach_reference_report, annotate_reference_levels
import json


class Portfolio:
    def __init__(self):
        self.last_selection=[]
        self.calls=[]
    def build(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs['bear_risk'] is False
        assert kwargs['market_regime']==-1
        assert kwargs['severe_bear'] is True
        self.last_selection=[dict(code='000001',score=.9,weight=.1)]
        return {'000001':1000.}


class Sector:
    def __init__(self):self.calls=[]
    def compute_signal_density(self,*args):self.calls.append(args)


def strategy():
    s=Strategy.__new__(Strategy)
    s.portfolio=Portfolio();s.sector_rotation=Sector();s.sentiment_orchestrator=None
    s.signal_store=SimpleNamespace(buy_signals_for_date=lambda *args:[])
    s.industry_stock_counts={}
    s.index_data=pd.DataFrame([dict(datetime=pd.Timestamp('2026-11-23'),
                                   regime=-1,momentum_score=0.,bear_risk=True,severe_bear=True)])
    return s


def test_shared_decision_method_on_private_copy_does_not_alter_actual_market_or_state():
    s=strategy();index=s.index_data.copy(deep=True)
    actual_orders=[dict(stock_code='000002.SZ',action='close')]
    def factory(targets,selections):
        assert targets=={'000001':1000.} and selections[0]['weight']==.1
        return [dict(stock_code='000001.SZ',action='open',order_amount=1000,order_shares=100,price=10.)]
    report=reference_plan(s,date(2026,11,23),['000001'],{},10000.,{'000001':10.},{},
                          {'effective_bear':True,'status':'BEAR'},factory)
    pd.testing.assert_frame_equal(s.index_data,index,check_exact=True)
    assert not s.portfolio.calls and not s.portfolio.last_selection and not s.sector_rotation.calls
    assert actual_orders==[dict(stock_code='000002.SZ',action='close')]
    assert report['orders']==[] and report['buy_enabled'] is False and report['executable'] is False
    assert report['reference_orders'][0]['order_shares']==100
    assert report['reference_orders'][0]['executable'] is False
    assert report['market_status']['status']=='BEAR'


@pytest.mark.parametrize('amount',[10001.,float('nan'),-1.])
def test_invalid_or_unfunded_reference_never_becomes_an_executable_plan(amount):
    s=strategy()
    with pytest.raises(ValueError):
        reference_plan(s,date(2026,11,23),[],{},10000.,{},{},
            {'effective_bear':True},lambda *args:[dict(action='open',order_amount=amount)])
    assert not s.portfolio.calls


def test_existing_inventory_is_unknown_rather_than_an_invented_no_add_merge():
    s=strategy()
    report=reference_plan(s,date(2026,11,23),[],{'000001':100.},900.,{},{},
                          {'effective_bear':True},lambda *args:pytest.fail('must not format'))
    assert report['status']=='unavailable_existing_inventory'
    assert report['reference_orders']==[] and not s.portfolio.calls


def test_non_bear_does_not_run_an_extra_portfolio():
    s=strategy()
    report=reference_plan(s,date(2026,11,23),[],{},10000.,{},{},
                          {'effective_bear':False},lambda *args:pytest.fail('must not format'))
    assert report['status']=='not_effective_bear'
    assert not s.portfolio.calls


def test_attachment_preserves_actual_orders_and_never_creates_or_changes_account(tmp_path):
    actual=dict(trading_profile='accepted465',signal_identity={'hash':'x'},
                signal_date='2026-11-23',execution_date='2026-11-24',
                market_status={'effective_bear':True},orders=[],run_id='20261124',buy_enabled=False)
    path=tmp_path/'trade_orders.json';path.write_text(json.dumps(actual))
    account=tmp_path/'live_account.json';account.write_text('{"cash":500000}')
    before=account.read_bytes()
    report=dict(status='complete',signal_identity=actual['signal_identity'],
                signal_date=actual['signal_date'],execution_date=actual['execution_date'],
                orders=[],observation_only=True,executable=False,buy_enabled=False,reference_orders=[dict(
                    stock_code='000001.SZ',order_shares=100,observation_only=True,
                    executable=False,buy_enabled=False)])
    out=attach_reference_report(tmp_path,report)
    assert {k:v for k,v in out.items() if k!='bear_reference_plan'}==actual
    assert account.read_bytes()==before


@pytest.mark.parametrize('defect',['identity','executable','row_executable'])
def test_stale_or_executable_reference_cannot_be_attached(tmp_path,defect):
    actual=dict(trading_profile='accepted465',signal_identity={'hash':'x'},
                signal_date='2026-11-23',execution_date='2026-11-24',
                market_status={'effective_bear':True},orders=[])
    path=tmp_path/'trade_orders.json';path.write_text(json.dumps(actual));before=path.read_bytes()
    report=dict(status='complete',signal_identity=actual['signal_identity'],
                signal_date=actual['signal_date'],execution_date=actual['execution_date'],
                orders=[],observation_only=True,executable=False,buy_enabled=False,reference_orders=[])
    if defect=='identity':report['signal_identity']={'hash':'other'}
    elif defect=='executable':report['executable']=True
    else:report['reference_orders']=[dict(executable=True,buy_enabled=True)]
    with pytest.raises(ValueError):attach_reference_report(tmp_path,report)
    assert path.read_bytes()==before


def test_wrong_side_stop_is_retained_as_invalid_observation_never_an_active_stop():
    row=annotate_reference_levels(dict(price=82.55,stop_loss_price=94.53,take_profit_price=103.19))
    assert row['stop_loss_price'] is None
    assert row['observed_stop_loss_price_reference']==94.53
    assert row['stop_loss_validity']=='invalid_reference_wrong_side_or_nonfinite'
    assert row['take_profit_price']==103.19


def test_reference_retains_the_shared_98_percent_target_ceiling():
    class FullBudget(Portfolio):
        def build(self,**kwargs):
            super().build(**kwargs)
            return {'000001':20000.}
    s=strategy();s.portfolio=FullBudget()
    def factory(targets,selections):
        assert targets=={'000001':9800.}
        return [dict(stock_code='000001.SZ',action='open',order_amount=9800.,
                     order_shares=100,price=98.)]
    result=reference_plan(s,date(2026,11,23),[],{},10000.,{},{},
                          {'effective_bear':True},factory)
    assert result['reference_spend']==9800.
    assert not s.portfolio.calls
