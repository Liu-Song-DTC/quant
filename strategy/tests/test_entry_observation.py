"""Observation lifecycle and execution parity, using only mock/synthetic inputs."""
import ast
from collections import defaultdict
from copy import deepcopy
from datetime import date
import datetime
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.entry_observation import EntryObservation


class Logger:
    def __init__(self, enabled=True):
        self.journal = SimpleNamespace(enabled=enabled)
        self.events = []
        self._exit_trigger_evidence = {}

    def detail(self, event, **payload):
        self.events.append(deepcopy(dict(event=event, **payload)))


def strategy(signal=None, frame=None):
    signal = signal or SimpleNamespace(
        buy=True, sell=False, score=.8, ml_score=.25, pre_discount_score=.4,
        adjusted_score=.5, _gate_quality=.7, bom_quality_score=.6,
        chan_buy_point=2, bi_buy_point=1, signal_level=2, chan_pivot_zd=9.,
        structure_stop_price=8., ma_trend_up=True)
    store = SimpleNamespace(_df=frame, get=Mock(return_value=signal))
    return SimpleNamespace(signal_store=store, portfolio=SimpleNamespace(
        _entry_reasons={}, last_selection=[dict(code='000001', date='2026-11-20', score=.8,
                                                effective_score=1.7, weight=.1)]))


def entries(logger, kind):
    return [row for row in logger.events if row['event'] == 'entry_observation_' + kind]


def order(ref=1, buy=True):
    return SimpleNamespace(ref=ref, data=SimpleNamespace(_name='000001'),
                           isbuy=lambda: buy, executed=SimpleNamespace(size=0, price=0., dt=0),
                           status_name='Submitted')


def test_plan_freezes_known_fields_and_never_promotes_pivot_or_stop_to_support():
    log = Logger(); obs = EntryObservation(log); s = strategy()
    old_signal = deepcopy(vars(s.signal_store.get.return_value))
    old_selection = deepcopy(s.portfolio.last_selection)
    assert obs.prepare('2026-11-20', {'000001': 10000}, {}, s) is None
    event = entries(log, 'plan')[0]; snapshot = event['entry']
    assert snapshot['buy_point']['chan_buy_point'] == 2
    assert snapshot['buy_point']['buy_point_origin'] is None
    assert not snapshot['known']['buy_point_origin']
    assert snapshot['known']['chan_buy_point']
    assert snapshot['score_components']['ml_score'] == .25
    assert event['selection']['effective_score'] == 1.7
    assert snapshot['support']['references']['chan_pivot_zd'] == 9.
    assert snapshot['support']['references']['structure_stop_price'] == 8.
    assert snapshot['support']['validity'] == snapshot['invalidation']['validity'] == 'unknown'
    assert 'candidate_score_components' in snapshot['trace_join']['events']
    assert vars(s.signal_store.get.return_value) == old_signal
    s.signal_store.get.return_value.score = -9
    assert obs.pending['000001']['entry']['values']['score'] == .8
    assert s.portfolio._entry_reasons == {}
    assert s.portfolio.last_selection == old_selection
    assert old_signal['score'] == .8
    json.dumps(event, allow_nan=False)


def test_persisted_null_absent_columns_do_not_become_signal_defaults():
    index = pd.MultiIndex.from_tuples([('000001', '2026-11-20')])
    frame = pd.DataFrame(dict(score=[.8], ml_score=[np.nan], gate_quality=[pd.NA],
                              chan_buy_point=[2], chan_pivot_zd=[float('inf')]), index=index)
    s = strategy(frame=frame); obs = EntryObservation(Logger())
    obs.prepare('2026-11-20', {'000001': 10000}, {}, s)
    snap = obs.pending['000001']['entry']
    assert snap['values']['score'] == .8
    assert snap['values']['ml_score'] is snap['values']['gate_quality'] is None
    assert snap['values']['ma_trend_up'] is None
    assert not snap['known']['ma_trend_up'] and not snap['known']['ml_score']
    assert snap['known']['score']
    assert snap['field_sources']['score'] == 'signal_store_persisted_row.score'
    assert snap['missing_reasons']['ma_trend_up'] == 'column_absent_from_store_schema'
    assert snap['missing_reasons']['ml_score'] == 'source_null_nonfinite_or_unsupported'
    assert snap['support']['references']['chan_pivot_zd'] is None
    assert s.signal_store.get.call_count == 0
    json.dumps(snap, allow_nan=False)
    pd.testing.assert_frame_equal(frame, s.signal_store._df)


def test_stale_last_selection_is_not_current_decision_score_composition():
    log = Logger(); obs = EntryObservation(log)
    obs.prepare('2026-11-23', {'000001': 100}, {}, strategy())
    selected = entries(log, 'plan')[0]['selection']
    assert selected['source_date'] == '2026-11-20'
    assert selected['effective_score'] is None and not selected['known']['effective_score']
    assert selected['missing_reason'] == 'last_selection_not_same_decision_day'


def test_dataclass_object_defaults_without_assignment_provenance_remain_unknown():
    from core.signal import Signal
    s = strategy(signal=Signal(score=.8, ml_score=.25))
    obs = EntryObservation(Logger()); obs.prepare('2026-11-20', {'000001': 100}, {}, s)
    snap = obs.pending['000001']['entry']
    assert snap['known']['score'] and snap['known']['ml_score']
    for name in ('nearest_support_pct', 'chan_buy_strength', 'b3_trend_confirmed'):
        assert not snap['known'][name]
        assert snap['values'][name] is None
        assert snap['missing_reasons'][name] == 'dataclass_default_assignment_provenance_unknown'
    assert not snap['known']['buy_point_origin'] and not snap['known']['structure_stop_price']


def test_cached_signal_default_fields_are_unknown_but_real_zero_false_stay_known():
    # A reconstructed Signal could contain default zero/False for all these;
    # the observer must consult the persisted row rather than that object.
    index = pd.MultiIndex.from_tuples([('000001', '2026-11-20')])
    source = pd.DataFrame(dict(score=[.8], ma_trend_up=[False], ml_score=[0.]), index=index)
    s = strategy(frame=source); obs = EntryObservation(Logger())
    obs.prepare('2026-11-20', {'000001': 100}, {}, s)
    snap = obs.pending['000001']['entry']
    for name in ('nearest_support_pct', 'chan_buy_strength', 'buy_point_origin', 'structure_stop_price'):
        assert not snap['known'][name] and snap['values'][name] is None
        assert snap['missing_reasons'][name] == 'column_absent_from_store_schema'
    assert snap['known']['ma_trend_up'] and snap['values']['ma_trend_up'] is False
    assert snap['known']['ml_score'] and snap['values']['ml_score'] == 0.
    assert s.signal_store.get.call_count == 0


def test_duplicate_persisted_rows_flagged_same_first_row_as_store():
    index = pd.MultiIndex.from_tuples([('000001', '2026-11-20')] * 2)
    s = strategy(frame=pd.DataFrame(dict(score=[.3, .9]), index=index))
    obs = EntryObservation(Logger()); obs.prepare('2026-11-20', {'000001': 100}, {}, s)
    snap = obs.pending['000001']['entry']
    assert snap['values']['score'] == .3
    assert snap['duplicate_source_rows'] == 2


def test_only_exact_decision_row_read_and_future_changes_cannot_change_entry():
    index = pd.MultiIndex.from_tuples([('000001', '2026-11-20'),
                                       ('000001', '2026-11-23')])
    frame = pd.DataFrame(dict(score=[.5, 999.], chan_buy_point=[2, 9],
                              future_ret=[-100., 100.]), index=index)
    snapshots = []
    for changed in (False, True):
        source = frame.copy()
        if changed:
            source.loc[('000001', '2026-11-23'), :] = -999
            source['future_ret'] = 777
        obs = EntryObservation(Logger()); s = strategy(frame=source)
        obs.prepare('2026-11-20', {'000001': 100}, {}, s)
        snapshots.append(obs.pending['000001']['entry'])
    assert snapshots[0] == snapshots[1]
    assert 'future_ret' not in snapshots[0]['values']


def test_fill_add_reduce_exit_same_plan_and_original_basis_without_mutation():
    log = Logger(); obs = EntryObservation(log); s = strategy()
    obs.prepare('2026-11-20', {'000001': 10000}, {}, s)
    plan = obs.pending['000001']; pid = plan['plan_id']
    obs.vectorized_fill('000001', '2026-11-23', 'BUY', 100, 10., 0, 'fill:1')
    assert not obs.pending and obs.held['000001'] is plan
    log._exit_trigger_evidence['000001'] = [
        dict(date='2026-11-20', reason='stale'),
        dict(date='2026-11-23', reason='recorded_rule')]
    s.signal_store.get.return_value.score = -.2
    obs.prepare('2026-11-23', {'000001': 500}, {'000001': 1000}, s)
    obs.vectorized_fill('000001', '2026-11-24', 'BUY', 100, 11., 100, 'fill:2')
    obs.vectorized_fill('000001', '2026-11-24', 'SELL', 100, 12., 200, 'fill:3')
    assert obs.held['000001']['entry']['values']['score'] == .8
    assert obs.held['000001']['entry_order_id'] == 'fill:1'
    obs.vectorized_fill('000001', '2026-11-25', 'SELL', 100, 12., 100, 'fill:4')
    fills = entries(log, 'fill')
    assert [row['remaining_shares'] for row in fills] == [100, 200, 100, 0]
    assert {row['plan_id'] for row in fills} == {pid}
    assert {row['entry_order_id'] for row in fills} == {'fill:1'}
    last = fills[-1]['latest_decision']
    assert last['date'] == '2026-11-23'
    assert [row['reason'] for row in last['recorded_exit_triggers']] == ['recorded_rule']
    assert last['signal']['values']['score'] == -.2
    assert not obs.pending and not obs.held and not obs.orders
    assert s.portfolio._entry_reasons == {}


@pytest.mark.parametrize('position_already_applied', [False, True])
def test_before_and_after_position_hooks_match_through_buy_add_reduce_full_exit(position_already_applied):
    log = Logger(); obs = EntryObservation(log)
    obs.prepare('2026-11-20', {'000001': 10000}, {}, strategy())
    plan_id = obs.pending['000001']['plan_id']
    steps = [('BUY', 100, 0, 100), ('BUY', 200, 100, 300),
             ('SELL', 100, 300, 200), ('SELL', 200, 200, 0)]
    for index, (side, shares, before, after) in enumerate(steps):
        obs.vectorized_fill('000001', '2026-11-23', side, shares, 10.,
                            after if position_already_applied else before, f'fill:{index}',
                            position_already_applied=position_already_applied)
        assert ('000001' in obs.held) == (after > 0)
    fills = entries(log, 'fill')
    assert [row['remaining_shares'] for row in fills] == [100, 300, 200, 0]
    assert {row['plan_id'] for row in fills} == {plan_id}
    assert {row['entry_order_id'] for row in fills} == {'fill:0'}
    assert not obs.pending and not obs.held and not obs.orders and obs.errors == 0


@pytest.mark.parametrize('raw_arm', [False, True])
def test_actual_record_fill_hook_uses_post_fill_raw_account_and_pre_fill_qfq_positions(raw_arm):
    path = Path(__file__).resolve().parents[1] / 'bt_execution.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_record_fill')
    log = Logger(); obs = EntryObservation(log)
    obs.prepare('2026-11-20', {'000001': 10000}, {}, strategy())
    raw_account = SimpleNamespace(shares={})
    positions = np.array([0])
    scope = dict(COMMISSION=.0001, _execution_fills=[], plog=log,
                 _entry_observation=obs, positions=positions, code_to_idx={'000001': 0},
                 raw_execution_arm=raw_arm, _raw_account=raw_account)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    for side, shares, before, after in [('BUY', 300, 0, 300),
                                         ('SELL', 100, 300, 200), ('SELL', 200, 200, 0)]:
        # Keep positions deliberately unsynchronized in raw mode, matching the
        # current engine; the authoritative raw account has already been filled.
        positions[0] = before
        raw_account.shares['000001'] = after
        scope['_record_fill']('000001', '2026-11-23', side, shares, 10., 0., 0., 10.)
        assert positions[0] == before and raw_account.shares['000001'] == after
    assert [row['remaining_shares'] for row in entries(log, 'fill')] == [300, 200, 0]
    assert len(scope['_execution_fills']) == 3
    assert not obs.held and not obs.pending and obs.errors == 0


def test_actual_held_codes_prevent_false_new_plan_when_valuation_missing():
    log = Logger(); obs = EntryObservation(log); s = strategy()
    obs.prepare('2026-11-20', {'000001': 100}, {}, s, held_codes=['000001'])
    assert not obs.pending and not entries(log, 'plan')


@pytest.mark.parametrize('terminal', ['Canceled', 'Expired', 'Margin', 'Rejected'])
def test_broker_unfilled_terminal_drops_order_and_plan(terminal):
    obs = EntryObservation(Logger()); s = strategy(); o = order()
    o.getstatusname = lambda: terminal
    obs.prepare('2026-11-20', {'000001': 100}, {}, s)
    obs.order_created(o, '2026-11-20')
    obs.notify_order(o, '2026-11-20', 0)
    assert not obs.pending and not obs.orders and not obs.held


def test_partial_callbacks_dedupe_cumulative_fill_and_keep_original_order_link():
    log = Logger(); obs = EntryObservation(log); s = strategy(); o = order()
    o.getstatusname = lambda: o.status_name
    obs.prepare('2026-11-20', {'000001': 100}, {}, s)
    obs.order_created(o, '2026-11-20')
    pid = obs.pending['000001']['plan_id']
    obs.notify_order(o, '2026-11-20', 0)  # Submitted, no fill
    o.status_name = 'Partial'
    o.executed.size = 100; o.executed.price = 10.; o.executed.dt = date(2026, 11, 23).toordinal()
    obs.notify_order(o, '2026-11-20', 100)
    obs.notify_order(o, '2026-11-20', 100)  # Same cumulative execution, no duplicate
    o.executed.size = 300; o.status_name = 'Completed'
    obs.notify_order(o, '2026-11-20', 300)
    obs.notify_order(o, '2026-11-20', 300)
    sell = order(ref=2, buy=False); sell.getstatusname = lambda: 'Completed'
    sell.executed.size = -300; sell.executed.price = 9.; sell.executed.dt = date(2026, 11, 24).toordinal()
    obs.order_created(sell, '2026-11-23')
    obs.notify_order(sell, '2026-11-23', 0)
    fills = entries(log, 'fill')
    assert [r['shares'] for r in fills] == [100, 200, 300]
    assert [r['date'] for r in fills] == [date(2026, 11, 23)] * 2 + [date(2026, 11, 24)]
    assert {r['plan_id'] for r in fills} == {pid}
    assert [r['order_id'] for r in fills] == ['1', '1', '2']
    assert {r['entry_order_id'] for r in fills} == {'1'}
    assert 'cumulative_average' in fills[1]['price_basis']
    assert not obs.orders and not obs.pending and not obs.held


def test_notify_before_next_uses_current_bar_for_status_and_execution_date_for_fill():
    log = Logger(); obs = EntryObservation(log); o = order()
    obs.prepare('2026-11-20', {'000001': 100}, {}, strategy())
    obs.order_created(o, '2026-11-20')
    o.data.datetime = SimpleNamespace(date=lambda index: date(2026, 11, 24))
    o.getstatusname = lambda: 'Partial'
    o.executed.size = 100; o.executed.price = 10.
    o.executed.dt = date(2026, 11, 23).toordinal()
    obs.notify_order(o, '2026-11-20', 100)
    fill = entries(log, 'fill')[0]; status = entries(log, 'order_status')[0]
    assert fill['date'] == date(2026, 11, 23)
    assert status['date'] == date(2026, 11, 24)
    assert status['date_source'] == 'order.data.datetime.date(0)'
    assert fill['price'] == 10. and fill['shares'] == 100
    assert fill['price_basis'] == 'broker_cumulative_average; not incremental_fill_price'
    assert 'gross_notional' not in fill and obs.errors == 0


@pytest.mark.parametrize('feed_state', ['missing', 'raises', 'invalid'])
def test_status_date_feed_unavailable_safely_falls_back_and_marks_source(feed_state):
    log = Logger(); obs = EntryObservation(log); o = order()
    obs.prepare('2026-11-20', {'000001': 100}, {}, strategy())
    obs.order_created(o, '2026-11-20'); o.getstatusname = lambda: 'Canceled'
    if feed_state == 'raises':
        o.data.datetime = SimpleNamespace(date=Mock(side_effect=IndexError('no bar')))
    elif feed_state == 'invalid':
        o.data.datetime = SimpleNamespace(date=lambda index: None)
    obs.notify_order(o, '2026-11-20', 0)
    status = entries(log, 'order_status')[0]
    assert status['date'] == '2026-11-20' and 'fallback' in status['date_source']
    assert not entries(log, 'fill') and not obs.pending and not obs.orders
    assert obs.errors == 0


def test_no_executed_timestamp_uses_current_bar_for_fill_as_well_as_status():
    log = Logger(); obs = EntryObservation(log); o = order()
    obs.prepare('2026-11-20', {'000001': 100}, {}, strategy())
    obs.order_created(o, '2026-11-20')
    o.data.datetime = SimpleNamespace(date=lambda index: date(2026, 11, 23))
    o.getstatusname = lambda: 'Completed'
    o.executed.size = 100; o.executed.price = 10.
    obs.notify_order(o, '2026-11-20', 100)
    assert entries(log, 'fill')[0]['date'] == entries(log, 'order_status')[0]['date'] == date(2026, 11, 23)
    assert obs.errors == 0 and not obs.orders


@pytest.mark.parametrize('mode', ['disabled', 'unknown_order', 'submitted'])
def test_position_getter_not_called_without_observed_fill_delta(mode):
    obs = EntryObservation(Logger(enabled=mode != 'disabled')); o = order()
    o.getstatusname = lambda: 'Submitted'
    getter = Mock(side_effect=AssertionError('Broker should not be read'))
    if mode != 'unknown_order':
        obs.prepare('2026-11-20', {'000001': 100}, {}, strategy())
        obs.order_created(o, '2026-11-20')
    obs.notify_order(o, '2026-11-20', getter)
    getter.assert_not_called()
    assert obs.errors == 0


def test_callable_position_read_only_for_new_fill_delta():
    obs = EntryObservation(Logger()); o = order()
    o.getstatusname = lambda: 'Partial'
    obs.prepare('2026-11-20', {'000001': 100}, {}, strategy())
    obs.order_created(o, '2026-11-20')
    o.executed.size = 100; o.executed.price = 10.
    getter = Mock(return_value=100)
    obs.notify_order(o, '2026-11-20', getter)
    obs.notify_order(o, '2026-11-20', getter)
    getter.assert_called_once_with()
    assert obs.errors == 0 and '000001' in obs.held


def test_replaced_decision_does_not_rebind_outstanding_order_to_future_plan():
    log = Logger(); obs = EntryObservation(log); s = strategy(); o = order()
    o.getstatusname = lambda: 'Completed'
    obs.prepare('2026-11-20', {'000001': 100}, {}, s)
    obs.order_created(o, '2026-11-20')
    original = obs.pending['000001']
    s.signal_store.get.return_value.score = 999
    obs.prepare('2026-11-23', {'000001': 100}, {}, s)
    assert obs.pending['000001']['plan_id'] != original['plan_id']
    o.executed.size = 100; o.executed.price = 10
    obs.notify_order(o, '2026-11-23', 100)
    assert obs.held['000001'] is original
    assert original['entry']['values']['score'] == .8
    assert not obs.pending and not obs.orders


def test_disabled_does_not_read_signal_store_or_iterate_holdings():
    log = Logger(enabled=False); obs = EntryObservation(log); s = strategy()
    def forbidden():
        raise AssertionError('Must not consume disabled observation inputs')
        yield
    obs.prepare('2026-11-20', {'000001': 100}, {}, s, forbidden())
    obs.order_created(order(), '2026-11-20')
    obs.vectorized_fill('000001', '2026-11-23', 'BUY', 100, 10., 0, 1)
    obs.close()
    assert s.signal_store.get.call_count == 0
    assert not log.events and not obs.pending and not obs.held and not obs.orders


def test_bounded_state_and_run_end_does_not_assert_inventory_sold():
    log = Logger(); obs = EntryObservation(log); s = strategy()
    for i in range(100):
        obs.prepare('2026-11-20', {f'{i:06d}': 100}, {}, s)
        assert len(obs.pending) == 1 and not obs.held and not obs.orders
    assert s.signal_store.get.call_count == 100
    obs.vectorized_fill('000099', '2026-11-23', 'BUY', 100, 10., 0, 'last')
    obs.close('2026-11-23'); obs.close('2026-11-23')
    assert not obs.pending and not obs.held and not obs.orders
    assert len(entries(log, 'end')) == 1
    assert entries(log, 'end')[0]['held'] == 1


def test_no_entry_basis_observed_stays_unknown_instead_of_using_fill_day_signal():
    log = Logger(); obs = EntryObservation(log)
    obs.vectorized_fill('000001', '2026-11-23', 'BUY', 100, 10., 0, 1)
    assert obs.held['000001']['entry'] is None
    assert obs.held['000001']['decision_date'] is None
    assert entries(log, 'fill')[0]['basis_status'] == 'unknown'


def test_read_and_write_errors_do_not_raise_into_execution(capsys):
    log = Logger(); obs = EntryObservation(log); s = strategy()
    s.signal_store.get.side_effect = RuntimeError('read failure')
    assert obs.prepare('2026-11-20', {'000001': 100}, {}, s) is None
    assert obs.errors == 1
    log.detail = Mock(side_effect=OSError('disk full'))
    obs.vectorized_fill('000001', '2026-11-23', 'BUY', 100, 10., 0, 1)
    obs.close()
    assert obs.errors >= 2
    assert obs.closed and not obs.held and not obs.pending
    assert 'ENTRY_OBSERVATION_ERROR' in capsys.readouterr().err


def test_execution_hooks_are_side_effect_calls_and_do_not_write_exit_state():
    path = Path(__file__).resolve().parents[1] / 'bt_execution.py'
    tree = ast.parse(path.read_text())
    hooks = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and ast.unparse(node.func.value) in ('_entry_observation', 'self._entry_observation')]
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    assert len(hooks) >= 9
    assert all(isinstance(parents[call], ast.Expr) for call in hooks)
    helper = ast.parse((path.parent / 'core/entry_observation.py').read_text())
    assert not any(isinstance(node, ast.Attribute) and node.attr == '_entry_reasons'
                   for node in ast.walk(helper))


def test_real_broker_notify_costs_and_stats_equal_without_observation(capsys):
    path = Path(__file__).resolve().parents[1] / 'bt_execution.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BacktraderExecution')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'notify_order')
    reference = deepcopy(method)
    # Compare the trading callback with its observation call removed; the real
    # cost/stat logic runs on mock completed buy and sell orders in both cases.
    reference.body.pop(0)
    results = []
    for source in (reference, method):
        ns = dict(datetime=datetime)
        exec(compile(ast.Module(body=[source], type_ignores=[]), str(path), 'exec'), ns)
        log = Logger(); obs = EntryObservation(log)
        obs.prepare('2026-11-20', {'000001': 1000}, {}, strategy())
        state = SimpleNamespace(_entry_observation=obs, last_date='2026-11-20',
                                cost=defaultdict(list), _fill_stats=dict(buy_filled=0),
                                getposition=lambda data: SimpleNamespace(size=state.remaining))
        for ref, buy, size, remaining in [(1, True, 100, 100), (2, False, -100, 0)]:
            o = order(ref, buy); o.getstatusname = lambda: 'Completed'
            o.Submitted = 1; o.Accepted = 2; o.Completed = 4; o.Canceled = 5; o.Margin = 7
            o.status = o.Completed; o.executed.size = size; o.executed.price = 10.
            o.executed.value = 1000.; o.executed.comm = .1
            o.executed.dt = date(2026, 11, 23).toordinal()
            state.remaining = remaining
            obs.order_created(o, '2026-11-20')
            ns['notify_order'](state, o)
        results.append((dict(state.cost), state._fill_stats))
    assert results[0] == results[1] == ({}, dict(buy_filled=1))


def test_broker_position_getter_failure_does_not_stop_cost_processing_or_leak_order(capsys):
    path = Path(__file__).resolve().parents[1] / 'bt_execution.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BacktraderExecution')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'notify_order')
    ns = dict(datetime=datetime)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    log = Logger(); obs = EntryObservation(log)
    obs.prepare('2026-11-20', {'000001': 1000}, {}, strategy())
    o = order(); o.getstatusname = lambda: 'Completed'
    o.Submitted = 1; o.Accepted = 2; o.Completed = 4; o.Canceled = 5; o.Margin = 7
    o.status = o.Completed; o.executed.size = 100; o.executed.price = 10.
    o.executed.value = 1000.; o.executed.comm = .1
    o.executed.dt = date(2026, 11, 23).toordinal()
    obs.order_created(o, '2026-11-20')
    state = SimpleNamespace(_entry_observation=obs, last_date='2026-11-20',
                            cost=defaultdict(list), _fill_stats=dict(buy_filled=0),
                            getposition=Mock(side_effect=RuntimeError('broker unavailable')))
    ns['notify_order'](state, o)
    assert state.cost['000001'] == [100, 10.]
    assert state._fill_stats['buy_filled'] == 1
    assert obs.errors == 1 and not obs.orders and not obs.pending and not obs.held
    assert not entries(log, 'fill')
    assert entries(log, 'order_status')[-1]['fill_observation_recorded'] is False
    assert 'ENTRY_OBSERVATION_ERROR' in capsys.readouterr().err


@pytest.mark.parametrize('raw_arm', [False, True])
def test_tiny_actual_execution_observation_on_off_bitwise_parity(monkeypatch, tmp_path, raw_arm):
    # The existing harness runs the real vectorized function on ten synthetic
    # bars in a temporary data/output tree, without importing the production CLI.
    from tests.test_review_regressions import ExecutionTests
    from core.decision_journal import DecisionJournal
    from core.pipeline_logger import plog
    from core import entry_observation
    results = []; observers = []
    original = entry_observation.EntryObservation
    def recording(*args, **kwargs):
        observer = original(*args, **kwargs); observers.append(observer)
        return observer
    monkeypatch.setattr(entry_observation, 'EntryObservation', recording)
    for key in list(__import__('os').environ):
        if key.startswith(('QUANT_', 'POOL_')) or key == 'FACTOR_CONFIG_PATH':
            monkeypatch.delenv(key)
    for enabled in (False, True):
        journal = DecisionJournal(directory=tmp_path / str(enabled), enabled=enabled)
        monkeypatch.setattr(plog, 'journal', journal)
        result, seen = ExecutionTests().replay([10., 10., 10., 10., 10., 10., 10., 9., 9., 9.],
                                              raw_arm=raw_arm,
                                              goals={5: {'600000': 10000}, 6: {'600000': 5000}, 7: {}})
        journal.close()
        assert journal.error is None and journal.failed_events == 0
        if enabled:
            with gzip.open(journal.path, 'rt', encoding='utf-8') as stream:
                events = [json.loads(line) for line in stream]
            plans = [e for e in events if e['event'] == 'entry_observation_plan']
            fills = [e for e in events if e['event'] == 'entry_observation_fill']
            assert len(plans) == 1 and len(fills) == 3
            assert {e['plan_id'] for e in fills} == {plans[0]['plan_id']}
            assert {e['entry_decision_date'] for e in fills} == {plans[0]['decision_date']}
            assert all(e['basis_status'] == 'unknown' for e in fills)
        results.append((result, seen))
    a, b = results[0][0], results[1][0]
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], (np.ndarray, pd.Series, pd.DataFrame)):
            if isinstance(a[key], pd.Series):
                pd.testing.assert_series_equal(a[key], b[key], check_exact=True)
            elif isinstance(a[key], pd.DataFrame):
                pd.testing.assert_frame_equal(a[key], b[key], check_exact=True)
            else:
                np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key], key
    assert [fill['side'] for fill in a['execution_fills']] == ['BUY', 'SELL', 'SELL']
    assert len(observers) == 2 and all(obs.closed for obs in observers)
    assert all(not obs.pending and not obs.held and not obs.orders for obs in observers)
    assert observers[1].errors == 0
