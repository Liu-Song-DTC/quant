"""Entry-plan lineage for diagnostics only; never a portfolio/exit input.

Only pending plans, live order references and held-entry snapshots are retained.
Existing journals provide the detailed score stages via (decision date, code);
we do not retain an all-candidate trace cache or reconstruct missing native bases.
"""
from functools import wraps
from dataclasses import MISSING, fields as dataclass_fields, is_dataclass
from datetime import date as calendar_date
import math
import os
import sys
from uuid import uuid4


_SIGNAL_FIELDS = (
    'buy', 'sell', 'score', 'factor_name', 'factor_value', 'industry',
    'pre_discount_score', 'adjusted_score', 'factor_score', 'ml_score',
    'gate_quality', 'bom_quality_score', 'risk_vol', 'risk_extreme',
    'chan_buy_point', 'bi_buy_point', 'signal_level', 'buy_point_origin',
    'second_buy_point', 'second_buy_origin', 'second_buy_b1_ref',
    'chan_sell_point', 'chan_buy_strength', 'chan_sell_strength',
    'chan_divergence_type', 'chan_divergence_strength', 'chan_structure_score',
    'trend_type', 'ma_trend_up', 'b3_trend_confirmed', 'exhaustion_risk',
    'chan_pivot_zd', 'chan_pivot_zg', 'structure_stop_price',
    'nearest_support_pct', 'nearest_resistance_pct', 'stroke_phase',
)
_SELECTION_FIELDS = (
    'score', 'effective_score', 'rank_pct', 'weight', 'is_held',
    'chan_buy_point', 'signal_level', 'trend_type', 'confidence',
)


def _value(value):
    """Detach JSON-safe scalars only; NaN/NA never mean a known False/zero."""
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _day(value):
    return str(value)[:10]


def _order_bar_date(order, fallback):
    """Notifications precede next(); its last_date can still be yesterday."""
    try:
        current = order.data.datetime.date(0)
        if isinstance(current, calendar_date):
            return current, 'order.data.datetime.date(0)'
    except Exception:
        pass  # Missing/unready mock or broker feed must not affect execution.
    return fallback, 'supplied_date_fallback; current_bar_date_unknown'


def _observe(method):
    """Observer failures cannot interrupt selection, orders or accounting."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if not self.enabled or self.closed:
            return
        try:
            method(self, *args, **kwargs)
        except Exception as error:
            self.errors += 1
            if self.errors == 1:
                print(f'[ENTRY_OBSERVATION_ERROR] {type(error).__name__}: {error}',
                      file=sys.stderr)
    return guarded


class EntryObservation:
    """Write to the existing decision journal without returning decision data.

    A broker order ref identifies an order; a vectorized execution uses an
    explicitly synthetic fill ID. Both link back to the frozen entry plan ID.
    Adds/reductions never replace the first-fill entry basis. No validity rule
    for native support is implemented here: even finite pivot/stop references
    have validity='unknown'. No bars, prices, labels or future dates are read.
    """

    def __init__(self, logger, namespace='execution'):
        self.logger = logger
        self.enabled = bool(getattr(getattr(logger, 'journal', None), 'enabled', False))
        self.namespace = f'{namespace}:{uuid4().hex}'
        self.pending = {}
        self.held = {}
        self.orders = {}
        self.sequence = 0
        self.errors = 0
        self.closed = False

    def _emit(self, event, date, code=None, **payload):
        try:
            self.logger.detail('entry_observation_' + event, date=date, code=code,
                               observation_schema=1, observation_only=True, **payload)
        except Exception as error:
            self.errors += 1
            if self.errors == 1:
                print(f'[ENTRY_OBSERVATION_ERROR] writer: {error}', file=sys.stderr)

    def _snapshot(self, store, code, date):
        fields = dict.fromkeys(_SIGNAL_FIELDS)
        missing_reasons = dict.fromkeys(_SIGNAL_FIELDS, 'source_attribute_absent')
        frame = getattr(store, '_df', None)
        duplicate_rows = 0
        if frame is not None:
            # Read persisted values, not Signal's compatibility defaults for
            # absent/null columns. This does not populate its candidate LRU.
            try:
                row = frame.loc[(code, _day(date))]
            except KeyError:
                row = None
            if row is not None and getattr(row, 'ndim', 1) == 2:
                duplicate_rows = len(row)
                row = row.iloc[0]  # Same row used by SignalStore, explicitly flagged.
            if row is not None:
                for name in fields:
                    fields[name] = _value(row.get(name))
                    missing_reasons[name] = ('column_absent_from_store_schema' if name not in row.index
                                             else 'source_null_nonfinite_or_unsupported')
            else:
                missing_reasons = dict.fromkeys(_SIGNAL_FIELDS, 'no_exact_decision_source_row')
            source = 'signal_store_persisted_row'
        else:
            sig = store.get(code, date) if store is not None else None
            defaults = {field.name: field.default for field in dataclass_fields(sig)} if is_dataclass(sig) else {}
            for name in fields:
                attr = '_gate_quality' if name == 'gate_quality' else name
                raw = getattr(sig, attr, None)
                fields[name] = _value(raw)
                missing_reasons[name] = ('source_attribute_absent' if not hasattr(sig, attr)
                                         else 'source_null_nonfinite_or_unsupported')
                default = defaults.get(attr, MISSING)
                if default is not MISSING and fields[name] is not None and raw == default:
                    # Object-only sources do not reveal whether a constructor
                    # explicitly supplied a value equal to its dataclass default.
                    fields[name] = None
                    missing_reasons[name] = 'dataclass_default_assignment_provenance_unknown'
            source = 'signal_object' if sig is not None else 'unknown'
        return dict(source=source, asof=_day(date), values=fields,
                    known={name: value is not None for name, value in fields.items()},
                    field_sources={name: (f'{source}._gate_quality' if source == 'signal_object'
                                          and name == 'gate_quality' else f'{source}.{name}')
                                   if value is not None else None
                                   for name, value in fields.items()},
                    missing_reasons={name: missing_reasons[name] for name, value in fields.items()
                                     if value is None},
                    missing=[name for name, value in fields.items() if value is None],
                    duplicate_source_rows=duplicate_rows,
                    validity='unknown',
                    buy_point={name: fields[name] for name in
                               ('chan_buy_point', 'bi_buy_point', 'signal_level',
                                'buy_point_origin', 'second_buy_origin')},
                    score_components={name: fields[name] for name in
                                      ('score', 'pre_discount_score', 'adjusted_score',
                                       'factor_score', 'ml_score', 'gate_quality',
                                       'bom_quality_score')},
                    invalidation=dict(validity='unknown', source='no_frozen_native_exit_basis',
                                      reason='Current sell/structure flags are observations, not entry validity'),
                    support=dict(validity='unknown', source='observed_references_only',
                                 references={name: fields[name] for name in
                                             ('chan_pivot_zd', 'chan_pivot_zg',
                                              'structure_stop_price', 'nearest_support_pct')},
                                 reason='No validated entry-specific support/invalidator provenance'),
                    trace_join=dict(decision_date=_day(date), code=code,
                                    events=['signal_decision', 'signal_neutral', 'common_gate',
                                            'candidate_score_components', 'multi_strategy_adjustment'],
                                    scope='Join existing same-run/source trace; events may be absent on replay'))

    def _plan(self, code, date, snapshot, selection=None, reason='new_position_target'):
        self.sequence += 1
        plan = dict(plan_id=f'{self.namespace}:{os.getpid()}:{self.sequence}',
                    decision_date=_day(date) if date is not None else None,
                    entry=snapshot, entry_order_id=None, latest_decision=None)
        self._emit('plan', date, code, plan_id=plan['plan_id'], reason=reason,
                   decision_date=plan['decision_date'], entry=snapshot, selection=selection,
                   completeness='partial; missing native provenance/intermediate score stages remain unknown')
        return plan

    @_observe
    def prepare(self, date, targets, held_values, strategy, held_codes=None):
        """Call after final target sizing; only target-new/actually-held codes.

        A replaced pending decision is not a cancellation of a broker order.
        Outstanding order records retain their own original plan until terminal.
        """
        for code, plan in self.pending.items():
            self._emit('plan_released', date, code, plan_id=plan['plan_id'],
                       reason='decision_replaced_or_unfilled; outstanding orders retain original plan')
        self.pending.clear()
        actual_held = set(held_codes) if held_codes is not None else set(held_values)
        store = getattr(strategy, 'signal_store', None)
        selections = {row['code']: row for row in
                      getattr(getattr(strategy, 'portfolio', None), 'last_selection', ())}
        targets = targets or {}
        for code in sorted(actual_held | {c for c, value in targets.items() if value > 0}):
            snapshot = self._snapshot(store, code, date)
            if code not in actual_held:
                selected = selections.get(code, {})
                selection_date = selected.get('date')
                current_selection = selection_date is not None and _day(selection_date) == _day(date)
                selection = {name: _value(selected.get(name)) if current_selection else None
                             for name in _SELECTION_FIELDS}
                selection['source'] = 'portfolio.last_selection'
                selection['source_date'] = _day(selection_date) if selection_date is not None else None
                selection['known'] = {name: selection[name] is not None for name in _SELECTION_FIELDS}
                selection['missing_reason'] = None if current_selection else 'last_selection_not_same_decision_day'
                selection['final_target_value'] = _value(targets[code])
                self.pending[code] = self._plan(code, date, snapshot, selection)
            elif code in self.held:
                triggers = [dict(row) for row in
                            getattr(self.logger, '_exit_trigger_evidence', {}).get(code, ())
                            if _day(row.get('date')) == _day(date)][-8:]
                decision = dict(date=_day(date), signal=snapshot,
                                held_value=_value(held_values.get(code)),
                                target_value=_value(targets.get(code, 0)),
                                recorded_exit_triggers=triggers,
                                trigger_scope='Same-day recorded requests, not sole cause of later fills')
                self.held[code]['latest_decision'] = decision
                # One observation per held code/day, never all universe signals.
                self._emit('held_decision', date, code, plan_id=self.held[code]['plan_id'],
                           entry_order_id=self.held[code]['entry_order_id'], decision=decision)

    @_observe
    def order_created(self, order, date):
        if order is None:
            return
        code = order.data._name
        plan = self.held.get(code) or self.pending.get(code)
        ref = str(order.ref)
        self.orders[ref] = dict(code=code, plan=plan, executed=0,
                                side='BUY' if order.isbuy() else 'SELL')
        self._emit('order', date, code, order_id=ref, order_id_kind='broker_ref',
                   side=self.orders[ref]['side'], plan_id=plan['plan_id'] if plan else None,
                   entry_order_id=plan['entry_order_id'] if plan else None)

    @_observe
    def notify_order(self, order, date, remaining_shares):
        ref = str(order.ref)
        record = self.orders.get(ref)
        if record is None:
            return  # Repeated terminal notifications do not duplicate fills.
        status_date, status_date_source = _order_bar_date(order, date)
        total = abs(int(order.executed.size))
        delta = total - record['executed']
        status = order.getstatusname()
        try:
            if delta > 0:
                remaining = remaining_shares() if callable(remaining_shares) else remaining_shares
                fill_date = status_date
                try:
                    dt = getattr(order.executed, 'dt', 0)
                    if dt and dt >= 1:
                        fill_date = calendar_date.fromordinal(int(dt))
                except (TypeError, ValueError, OverflowError):
                    pass
                self._fill(record['code'], fill_date, record['side'], delta, order.executed.price,
                           remaining, ref, 'broker_ref', record['plan'],
                           'broker_cumulative_average; not incremental_fill_price')
                record['executed'] = total
        finally:
            self._emit('order_status', status_date, record['code'], order_id=ref, status=status,
                       date_source=status_date_source, cumulative_executed_shares=total,
                       fill_observation_recorded=delta <= 0 or record['executed'] == total,
                       plan_id=record['plan']['plan_id'] if record['plan'] else None)
            # A failed observation read must not retain completed orders forever.
            if status in ('Completed', 'Canceled', 'Cancelled', 'Expired', 'Margin', 'Rejected'):
                self.orders.pop(ref, None)
                plan = record['plan']
                code = record['code']
                if (plan is not None and code not in self.held and self.pending.get(code) is plan
                        and not any(row['plan'] is plan for row in self.orders.values())):
                    self.pending.pop(code, None)

    def _fill(self, code, date, side, shares, price, remaining_shares, order_id,
              order_id_kind, originating_plan=None, price_basis='actual_fill_price'):
        plan = self.held.get(code)
        if plan is None and side == 'BUY':
            plan = originating_plan or self.pending.get(code)
            if plan is None:
                plan = self._plan(code, None, None, reason='entry_basis_not_observed')
            self.held[code] = plan
            plan['entry_order_id'] = order_id
            self.pending.pop(code, None)
        self._emit('fill', date, code, plan_id=plan['plan_id'] if plan else None,
                   entry_decision_date=plan['decision_date'] if plan else None,
                   entry_order_id=plan['entry_order_id'] if plan else None,
                   order_id=order_id, order_id_kind=order_id_kind,
                   originating_plan_id=originating_plan['plan_id'] if originating_plan else None,
                   side=side, shares=int(shares), price=_value(price), price_basis=price_basis,
                   remaining_shares=int(remaining_shares),
                   latest_decision=plan['latest_decision'] if plan else None,
                   basis_status=('observed_partial' if plan and plan['entry']
                                 and any(value is not None for value in plan['entry']['values'].values())
                                 else 'unknown'))
        if side == 'SELL' and remaining_shares == 0:
            self.held.pop(code, None)

    @_observe
    def vectorized_fill(self, code, date, side, shares, price, shares_at_hook, fill_id,
                        position_already_applied=False):
        """Raw-account shares already reflect the fill; QFQ positions may not."""
        remaining = (shares_at_hook if position_already_applied else
                     shares_at_hook + shares if side == 'BUY' else shares_at_hook - shares)
        self._fill(code, date, side, shares, price, remaining,
                   str(fill_id), 'synthetic_vectorized_fill; not broker_order_ref')

    def close(self, date=None):
        if self.closed:
            return
        if self.enabled:
            self._emit('end', date, pending=len(self.pending), held=len(self.held),
                       active_orders=len(self.orders), errors=self.errors,
                       scope='Run ended; held positions are not asserted sold')
        self.pending.clear()
        self.held.clear()
        self.orders.clear()
        self.closed = True
