"""Non-executable bear reference plan through the maintained decision path.

Only the copied decision-day bear_risk input is relaxed. Actual market labels,
source scores, other entry/risk rules and actual account state are retained.
This is explicitly a hypothetical reference, never accepted465's actual target.
"""
from copy import copy, deepcopy
import json
import math
from pathlib import Path
import pandas as pd


class _ReferenceJournal:
    """Tag every nested decision event; redirect logging only, not decisions."""
    def __init__(self, original):self.original=original
    def __getattr__(self, name):return getattr(self.original,name)
    def event(self,event,*args,**kwargs):
        kwargs.update(observation_only=True,decision_lane='bear_reference',
                      market_input_override='decision_day_bear_risk=False')
        return self.original.event(event,*args,**kwargs)


def _scalar(value):
    if hasattr(value,'isoformat'):return value.isoformat()
    if hasattr(value,'item'):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def annotate_reference_levels(order):
    """A generic pivot above price is an observation, not a valid long stop."""
    price=float(order['price'])
    for key,validity,valid in (
            ('stop_loss_price','stop_loss_validity',lambda v:0<v<price),
            ('take_profit_price','take_profit_validity',lambda v:v>price)):
        raw=order.get(key)
        if raw is None:
            order[validity]='unknown_missing_reference'
            continue
        value=float(raw)
        if not math.isfinite(value) or not valid(value):
            order['observed_'+key+'_reference']=_scalar(value)
            order[key]=None
            order[validity]='invalid_reference_wrong_side_or_nonfinite'
        else:
            order[validity]='price_side_consistent_only_not_native_thesis_validation'
    return order


def reference_plan(strategy, day, universe, positions, cash, prices, cost,
                   market_status, order_factory):
    report = dict(schema=1, observation_only=True, executable=False,
                  buy_enabled=False, market_status=deepcopy(market_status),
                  orders=[], reference_orders=[],
                  assumption='Only the copied decision-day bear_risk=False; all other shared strategy inputs and rules unchanged',
                  process_contract='Dedicated dry-run subprocess only; module diagnostics/caches are not a second actual account decision',
                  note='熊市参考买单，禁止执行；参考权重不是465方案当日实际仓位。其他准入仍可拒绝全部候选。')
    if not market_status.get('effective_bear'):
        report['status']='not_effective_bear'
        return report
    # Existing inventory needs the full no-add/reservation merge before a
    # reference can claim complete sizing. Never invent that missing merge.
    if positions:
        report.update(status='unavailable_existing_inventory',
                      reason='Reference with existing inventory requires original no-add/pending merge; no invented quantities')
        return report
    preview=copy(strategy)
    preview.portfolio=deepcopy(strategy.portfolio)
    preview.sector_rotation=deepcopy(strategy.sector_rotation)
    preview.sentiment_orchestrator=deepcopy(strategy.sentiment_orchestrator)
    preview.index_data=strategy.index_data.copy(deep=True)
    mask=preview.index_data.datetime.dt.date.eq(pd.Timestamp(day).date())
    if int(mask.sum())!=1 or 'bear_risk' not in preview.index_data:
        raise ValueError('Bear reference requires one exact decision-day market row')
    if not bool(preview.index_data.loc[mask,'bear_risk'].iloc[0]):
        raise ValueError('Reported bear state conflicts with raw decision-day context')
    preview.index_data.loc[mask,'bear_risk']=False
    from core.pipeline_logger import plog
    journal=plog.journal
    try:
        plog.journal=_ReferenceJournal(journal)
        targets=preview.generate_positions(day,list(universe),deepcopy(positions),
                                           float(cash),dict(prices),deepcopy(cost))
    finally:
        plog.journal=journal
    # Same final target projection as generate_trade_orders; the supported
    # account is flat, so NAV equals actual cash and no sale proceeds exist.
    target_total=float(sum(v for v in targets.values() if v>0))
    if target_total>float(cash)*0.98:
        scale=(float(cash)*0.98)/max(target_total,1.0)
        targets={code:value*scale for code,value in targets.items()}
    selections=deepcopy(getattr(preview.portfolio,'last_selection',[]))
    orders=order_factory(targets,selections)
    spend=0.0
    for order in orders:
        if order.get('action')!='open':
            raise ValueError('Flat-account reference cannot include inventory adjustments')
        amount=float(order['order_amount'])
        if not math.isfinite(amount) or amount<=0:
            raise ValueError('Reference amount must be finite and positive')
        raw_price=float(order.get('reference_raw_price',order.get('price',float('nan'))))
        shares=float(order.get('order_shares',float('nan')))
        if not math.isfinite(raw_price) or raw_price<=0 or not math.isfinite(shares) or shares<=0:
            raise ValueError('Reference requires known positive raw price and incremental shares')
        spend+=shares*raw_price
    if spend>float(cash)+1e-8:
        raise ValueError('Reference exceeds actual cash; no assumed future sale proceeds')
    for order in orders:
        order.update(observation_only=True,executable=False,buy_enabled=False,
                     market_status='BEAR',quantity_basis='reference_only_not_submitted')
        annotate_reference_levels(order)
    report.update(status='complete' if orders else 'no_eligible_reference_orders',
                  reference_orders=orders,reference_targets={k:float(v) for k,v in targets.items()},
                  reference_selection=[{k:_scalar(v) for k,v in row.items()} for row in selections],reference_spend=spend,
                  reference_cash_remaining_before_fees=float(cash)-spend,
                  spend_basis='Unrounded raw price times reference shares; excludes commission/minimum fees; not executable funding approval',
                  actual_target_weights_remain_zero=True)
    return report


def attach_reference_report(root, report):
    """Add a disabled reference section only; never queue or change the ledger."""
    from core.live_state import live_run_lock, atomic_bytes
    root=Path(root)
    with live_run_lock(root):
        path=root/'trade_orders.json'
        original=path.read_bytes();actual=json.loads(original)
        if (actual.get('trading_profile')!='accepted465' or
                report.get('signal_identity')!=actual.get('signal_identity') or
                report.get('signal_date')!=actual.get('signal_date') or
                report.get('execution_date')!=actual.get('execution_date')):
            raise ValueError('Reference must bind the exact committed signal bundle and execution session')
        if not actual.get('market_status',{}).get('effective_bear'):
            raise ValueError('Cannot attach a bear reference to a non-bear actual plan')
        if report.get('status') not in ('complete','no_eligible_reference_orders'):
            raise ValueError('Incomplete reference cannot be published as a complete plan')
        if report.get('orders') or any(report.get(k) is not False for k in ('executable','buy_enabled')):
            raise ValueError('Reference cannot contain executable orders')
        if report.get('observation_only') is not True:
            raise ValueError('Reference must be explicitly observation-only')
        for order in report.get('reference_orders',[]):
            if order.get('executable') is not False or order.get('buy_enabled') is not False or order.get('observation_only') is not True:
                raise ValueError('Every reference order must be explicitly disabled')
        updated=deepcopy(actual);updated['bear_reference_plan']=deepcopy(report)
        assert {k:v for k,v in updated.items() if k!='bear_reference_plan'}=={k:v for k,v in actual.items() if k!='bear_reference_plan'}
        atomic_bytes(path,json.dumps(updated,ensure_ascii=False,indent=2,allow_nan=False).encode())
        return updated
