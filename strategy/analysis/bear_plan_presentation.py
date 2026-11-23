"""Label the normal live order plan with the actual market classification.

No candidate selection, position allocation, order filtering or execution veto
is implemented here. Orders have already passed the normal live pipeline.
"""
import numpy as np
import pandas as pd


def present_bear_plan(date, portfolio, raw_bear_risk, universe, signal_store,
                      orders, actual_positions, top_n, market_regime=None):
    day=pd.Timestamp(date).date()
    if pd.isna(day) or getattr(portfolio,'_market_state_date',None)!=day:
        raise ValueError('same-day order calculation must advance market state first')
    if not isinstance(raw_bear_risk,(bool,np.bool_)):
        raise ValueError('raw_bear_risk must be a known boolean')
    effective=bool(portfolio.advance_market_state(day,bool(raw_bear_risk)))
    is_bear=bool(raw_bear_risk or market_regime==-1)
    market=dict(status='BEAR' if is_bear else 'NOT_BEAR',
                effective_bear=effective,raw_bear_risk=bool(raw_bear_risk),
                bear_streak=portfolio._bear_streak,market_state_date=day.isoformat(),
                bear_label_only=True)
    if is_bear:
        for order in orders:
            order['market_status']='BEAR'
    return dict(orders=orders,watchlist=[],market_status=market,buy_enabled=True,
                market_note=('BEAR：正常出单，熊市仅作市场标记，是否执行由用户决定。'
                             if is_bear else '按正常选股、仓位和现金约束生成订单。'))
