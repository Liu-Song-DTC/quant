"""Normal live order calculation with the actual bear classification retained.

Order creation is distinct from a broker fill. A BEAR label does not suppress
the requested daily plan. Historical account simulations keep their original
market gates; both entry points use Strategy.generate_positions and the same
portfolio implementation. No copied portfolio, reference orders or fills.
"""
from copy import copy
import math
import pandas as pd


def generate_order_positions(strategy,date,universe,current_positions,cash,prices,cost):
    day=pd.Timestamp(date).date()
    mask=strategy.index_data.datetime.dt.date.eq(day)
    if int(mask.sum())!=1:
        raise ValueError('Live planning requires exactly one decision-day market row')
    raw_bear=strategy.index_data.loc[mask,'bear_risk'].iloc[0]
    if not pd.api.types.is_bool(raw_bear):
        raise ValueError('Live planning requires a known boolean bear classification')
    if not bool(raw_bear):
        return strategy.generate_positions(date,universe,current_positions,cash,prices,cost)
    # Record the truthful market clock once. The normal-order call below is
    # same-day, so its False entry gate cannot reset the real bear streak.
    strategy.portfolio.advance_market_state(day,True)
    planning=copy(strategy)
    planning.index_data=strategy.index_data.copy(deep=True)
    planning.index_data.loc[mask,'bear_risk']=False
    # The real portfolio/selection/state is used exactly once, without a second
    # reference portfolio. Regime, trend, volatility, signals and risk limits
    # are unchanged; only the live plan's unconditional bear veto is absent.
    return planning.generate_positions(date,universe,current_positions,cash,prices,cost)


def validate_empty_bear_replacement(committed,account,positions,execution_date):
    """Only an empty, unqueued bear plan can be replaced for this same session."""
    if (committed.get('trading_profile')!='accepted465' or committed.get('orders') or
            not committed.get('market_status',{}).get('effective_bear')):
        raise ValueError('Replacement is limited to a previously empty bear plan')
    if positions or account.get('pending_orders'):
        raise ValueError('Cannot replace a bear plan with actual inventory or registered orders')
    if committed.get('execution_date')!=str(execution_date):
        raise ValueError('Replacement execution session differs')


def validate_order_price_references(orders):
    """Never publish a long stop above its own buy price as an active stop."""
    for order in orders:
        if order.get('action') not in ('open','adjust'):continue
        price=float(order.get('price',0))
        stop=float(order.get('stop_loss_price',0) or 0)
        if price>0 and (not math.isfinite(stop) or stop>=price):
            order['observed_stop_loss_reference']=stop if math.isfinite(stop) else None
            order['stop_loss_price']=0.0
            order['stop_loss_validity']='invalid_reference_not_below_price; no_valid_stop_available'
    return orders
