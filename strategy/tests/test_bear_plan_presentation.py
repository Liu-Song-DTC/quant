from datetime import date
from types import SimpleNamespace
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.bear_plan_presentation import present_bear_plan


def test_bear_label_preserves_normal_orders_without_candidates_or_execution_veto():
    day=date(2026,11,23)
    p=SimpleNamespace(_market_state_date=day,_bear_streak=6,advance_market_state=lambda *args:True)
    store=SimpleNamespace(get=lambda *args:pytest.fail('presentation must not select candidates'))
    orders=[dict(stock_code='301603.SZ',action='open',shares=2100,amount=173355)]
    result=present_bear_plan(day,p,True,[],store,orders,{},3,market_regime=-1)
    assert result['orders'] is orders
    assert orders[0]['shares']==2100 and orders[0]['market_status']=='BEAR'
    assert result['buy_enabled'] is True and result['watchlist']==[]
    assert result['market_status']['status']=='BEAR'
    assert not any(k in orders[0] for k in ['executable','observation_only'])


def test_normal_market_does_not_change_order_values():
    day=date(2026,11,23)
    p=SimpleNamespace(_market_state_date=day,_bear_streak=0,advance_market_state=lambda *args:False)
    orders=[dict(stock_code='000001.SZ',action='open',shares=100)]
    before=[dict(row) for row in orders]
    result=present_bear_plan(day,p,False,[],None,orders,{},3,market_regime=0)
    assert result['orders']==before
    assert result['buy_enabled'] is True


def test_stale_market_state_is_rejected():
    p=SimpleNamespace(_market_state_date=date(2026,11,20))
    with pytest.raises(ValueError):present_bear_plan(date(2026,11,23),p,True,[],None,[],{},3)
