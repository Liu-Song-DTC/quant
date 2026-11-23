from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.strategy import Strategy
from core.portfolio import PortfolioConstructor
from analysis.live_order_planning import generate_order_positions,validate_empty_bear_replacement,validate_order_price_references


class Portfolio:
    advance_market_state=PortfolioConstructor.advance_market_state
    def __init__(self):
        self._market_state_date=date(2026,11,20);self._bear_streak=5
        self.calls=[];self.last_selection=[]
    def build(self,**kwargs):
        self.advance_market_state(kwargs['date'],kwargs['bear_risk'])
        self.calls.append(kwargs)
        self.last_selection=[{'code':'301603','weight':.45}]
        return {'301603':180000.}


class Sector:
    def compute_signal_density(self,*args):pass


def setup(bear=True):
    s=Strategy.__new__(Strategy);s.portfolio=Portfolio();s.sector_rotation=Sector()
    s.sentiment_orchestrator=None;s.industry_stock_counts={}
    s.signal_store=SimpleNamespace(buy_signals_for_date=lambda *args:[])
    s.index_data=pd.DataFrame([dict(datetime=pd.Timestamp('2026-11-23'),
        momentum_score=-.1,trend_score=-.1,regime=-1 if bear else 0,
        bear_risk=bear,bear_risk_fast=False,severe_bear=False)])
    return s


def test_bear_orders_use_the_real_portfolio_once_and_preserve_actual_market_clock():
    s=setup();frame=s.index_data.copy(deep=True)
    result=generate_order_positions(s,date(2026,11,23),['301603'],{},500000.,{'301603':82.55},{})
    assert result=={'301603':180000.}
    assert len(s.portfolio.calls)==1 and s.portfolio.calls[0]['bear_risk'] is False
    assert s.portfolio.calls[0]['market_regime']==-1
    assert s.portfolio.calls[0]['trend_score']==-.1
    assert s.portfolio._bear_streak==6
    assert s.portfolio.last_selection[0]['code']=='301603'
    pd.testing.assert_frame_equal(frame,s.index_data,check_exact=True)


def test_same_day_repeat_does_not_double_count_bear_streak():
    s=setup()
    for _ in range(2):generate_order_positions(s,date(2026,11,23),[],{},500000.,{},{})
    assert s.portfolio._bear_streak==6


def test_non_bear_retains_original_inputs():
    s=setup(False)
    generate_order_positions(s,date(2026,11,23),[],{},500000.,{},{})
    assert len(s.portfolio.calls)==1 and s.portfolio.calls[0]['bear_risk'] is False
    assert s.portfolio.calls[0]['market_regime']==0 and s.portfolio._bear_streak==0


@pytest.mark.parametrize('bad',['orders','holdings','pending','nonbear','session'])
def test_same_day_replacement_cannot_replace_live_inventory_or_registered_orders(bad):
    old=dict(trading_profile='accepted465',orders=[],market_status={'effective_bear':True},execution_date='2026-11-24')
    account={'pending_orders':{}};positions={}
    if bad=='orders':old['orders']=[{'stock_code':'301603.SZ'}]
    elif bad=='holdings':positions={'301603':{'shares':100}}
    elif bad=='pending':account['pending_orders']={'20261124:1':{}}
    elif bad=='nonbear':old['market_status']['effective_bear']=False
    else:old['execution_date']='2026-11-25'
    with pytest.raises(ValueError):validate_empty_bear_replacement(old,account,positions,'2026-11-24')


def test_empty_bear_replacement_does_not_mutate_ledger():
    old=dict(trading_profile='accepted465',orders=[],market_status={'effective_bear':True},execution_date='2026-11-24')
    account={'pending_orders':{},'cash':500000.};before=deepcopy(account)
    validate_empty_bear_replacement(old,account,{},'2026-11-24')
    assert account==before


def test_wrong_side_stop_cannot_be_published_as_an_active_stop():
    orders=[dict(action='open',price=82.55,stop_loss_price=94.53)]
    validate_order_price_references(orders)
    assert orders[0]['stop_loss_price']==0.
    assert orders[0]['observed_stop_loss_reference']==94.53
