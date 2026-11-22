"""Focused presentation checks using the real source market-state method."""

import ast
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.bear_plan_presentation import present_bear_plan
from core.execution_ledger import queue_orders
from core.portfolio import PortfolioConstructor
from core.signal import Signal
from core.signal_store import SignalStore


DAY = date(2026, 11, 20)


def portfolio(streak=1, raw=True):
    # Avoid configuration/data loading and all account/build side effects.
    owner = PortfolioConstructor.__new__(PortfolioConstructor)
    owner.v_recovery_enabled = False
    owner._market_state_date = DAY - timedelta(days=1)
    owner._bear_streak = streak - 1
    owner._dbg = {'calls': 7, 'bear_risk_days': 3}
    owner._v_recovery_active = False
    owner._v_nonbear_streak = 2
    owner._v_recovery_dates = set()
    owner._entry_dates = {'000001': DAY - timedelta(days=4)}
    owner._peak_prices = {'000001': 12.5}
    owner.advance_market_state(DAY, raw)
    return owner


def present(owner=None, store=None, universe=('000001',), orders=None,
            positions=None, raw=True, top_n=10, day=DAY):
    return present_bear_plan(
        day, owner if owner is not None else portfolio(), raw, universe,
        store if store is not None else SignalStore(),
        orders if orders is not None else [],
        positions if positions is not None else {}, top_n,
    )


def test_empty_account_bear_has_observations_without_executable_buys():
    store = SignalStore()
    store.set('000001', DAY, Signal(buy=True, score=.75, industry='银行', factor_name='SRC'))
    original = [dict(stock_code='000001.SZ', action='open', shares=100)]
    result = present(store=store, orders=original)
    assert set(result) == {'orders', 'watchlist', 'market_status', 'buy_enabled', 'watchlist_note'}
    assert result['orders'] == [] and not result['buy_enabled']
    assert result['market_status'] == dict(
        status='BEAR', effective_bear=True, raw_bear_risk=True, bear_streak=1,
        market_state_date=str(DAY), v_recovery_enabled=False, strategy_exception=None,
    )
    assert result['watchlist'] == [dict(
        code='000001', score=.75, industry='银行', factor_name='SRC',
        signal_date=str(DAY), observation_only=True, buy_enabled=False,
    )]
    assert '不买入' in result['watchlist_note']
    assert '不承诺通过全部组合准入' in result['watchlist_note']
    assert '权重' in result['watchlist_note']
    assert original == [dict(stock_code='000001.SZ', action='open', shares=100)]


def test_bear_preserves_close_reduction_and_order_but_removes_actual_increases():
    orders = [
        dict(stock_code='000001.SZ', action='adjust', shares=200, prev_shares=999),
        dict(stock_code='000002.SZ', action='close', shares=500),
        dict(stock_code='000001.SZ', action='adjust', shares=50, prev_shares=0),
        dict(stock_code='000003.SZ', action='open', shares=100),
        dict(stock_code='000001.SZ', action='keep', shares=100),
        dict(stock_code='000001.SZ', action='adjust', shares=100),
        dict(stock_code='000004.SZ', action='close'),
    ]
    positions = {'000001': {'shares': 100}, '000002': {'shares': 500}}
    before = deepcopy((orders, positions))
    result = present(orders=orders, positions=positions)
    assert result['orders'] == [orders[i] for i in (1, 2, 4, 5, 6)]
    assert (orders, positions) == before


@pytest.mark.parametrize('state_day', [None, DAY - timedelta(days=1), DAY + timedelta(days=1)])
def test_missing_same_day_market_transition_is_rejected_without_advancing(state_day):
    owner = portfolio()
    owner._market_state_date = state_day
    before = deepcopy(vars(owner))
    with pytest.raises(ValueError, match='_market_state_date'):
        present(owner=owner)
    assert vars(owner) == before


def test_repeated_presentation_preserves_entire_portfolio_and_source_scores():
    owner, store = portfolio(), SignalStore()
    signal = Signal(buy=True, score=.8137, adjusted_score=.2)
    store.set('000001', DAY, signal)
    before, source_before = deepcopy(vars(owner)), deepcopy(vars(signal))
    first = present(owner=owner, store=store)
    for _ in range(4):
        assert present(owner=owner, store=store) == first
        assert vars(owner) == before
        assert vars(signal) == source_before
    first['watchlist'][0]['score'] = -99
    assert vars(signal) == source_before


def test_exact_day_finite_source_scores_flags_ties_and_top_n():
    store = SignalStore()
    rows = {
        '000003': Signal(buy=True, score=.8),
        '000001': Signal(buy=True, score=.8),
        '000002': Signal(buy=True, score=.9),
        'conflict': Signal(buy=True, sell=True, score=99),
        'hold': Signal(buy=False, score=99),
        'unknown_buy': Signal(buy='True', score=99),
        'unknown_sell': Signal(buy=True, sell=None, score=99),
        'numpy': Signal(buy=np.bool_(True), sell=np.bool_(False), score=np.float64(.7)),
    }
    for i, score in enumerate((None, 'unknown', '.99', float('nan'), float('inf'), -float('inf'), True, object())):
        rows[f'invalid{i}'] = Signal(buy=True, score=score)
    for code, signal in rows.items():
        store.set(code, DAY, signal)
    store.set('stale', DAY - timedelta(days=1), Signal(buy=True, score=100))
    store.set('future', DAY + timedelta(days=1), Signal(buy=True, score=100))
    universe = list(reversed(rows)) + ['stale', 'future', 'missing', '000001']
    result = present(store=store, universe=universe, top_n=3)
    assert [row['code'] for row in result['watchlist']] == ['000002', '000001', '000003']
    assert [row['score'] for row in result['watchlist']] == [.9, .8, .8]
    assert [row['code'] for row in present(store=store, universe=universe)['watchlist']] == [
        '000002', '000001', '000003', 'numpy',
    ]
    assert all(row['signal_date'] == str(DAY) for row in result['watchlist'])


def test_no_matching_today_candidate_has_explicit_empty_reason():
    store = SignalStore()
    store.set('000001', DAY - timedelta(days=1), Signal(buy=True, score=1))
    result = present(store=store)
    assert result['watchlist'] == []
    assert '无当日' in result['watchlist_note']


def test_zero_top_n_explains_empty_watchlist():
    store = SignalStore()
    store.set('000001', DAY, Signal(buy=True, score=1))
    result = present(store=store, top_n=0)
    assert result['watchlist'] == [] and 'top_n=0' in result['watchlist_note']


@pytest.mark.parametrize('raw,streak,exception', [(False, 1, None), (True, 61, 'FAST')])
def test_non_bear_and_source_day61_fast_preserve_orders_exactly(raw, streak, exception):
    owner = portfolio(streak=streak, raw=raw)
    orders = [dict(stock_code='000002.SZ', action='open', shares=100),
              dict(stock_code='000001.SZ', action='adjust', shares=200)]
    before = deepcopy(vars(owner))
    result = present(owner=owner, orders=orders, raw=raw)
    assert result['orders'] is orders
    assert result['buy_enabled'] and result['watchlist'] == []
    assert result['market_status']['status'] == 'NOT_BEAR'
    assert result['market_status']['strategy_exception'] == exception
    assert vars(owner) == before


def test_effective_decision_is_delegated_to_source_not_local_threshold():
    owner = portfolio(streak=1)
    owner.advance_market_state = lambda day, raw: False
    result = present(owner=owner)
    assert result['market_status']['status'] == 'NOT_BEAR'
    assert result['buy_enabled']


@pytest.mark.parametrize('configuration', [True, None, 0, 'False'])
def test_unsupported_or_unknown_v_recovery_configuration_is_not_guessed(configuration):
    owner = portfolio()
    owner.v_recovery_enabled = configuration
    before = deepcopy(vars(owner))
    with pytest.raises(ValueError, match='v_recovery_enabled=False'):
        present(owner=owner)
    assert vars(owner) == before


@pytest.mark.parametrize('value', [None, -1, 1.5, '3', True])
def test_invalid_top_n_is_rejected_without_state_changes(value):
    owner = portfolio()
    before = deepcopy(vars(owner))
    with pytest.raises(ValueError, match='top_n'):
        present(owner=owner, top_n=value)
    assert vars(owner) == before


@pytest.mark.parametrize('quantity', [None, 'unknown', float('nan'), -1, 1.5])
def test_unknown_order_quantity_is_not_assumed_to_be_a_reduction(quantity):
    with pytest.raises(ValueError, match='shares'):
        present(orders=[dict(stock_code='000001.SZ', action='adjust', shares=quantity)])


def test_timestamp_api_date_reads_exact_daily_store_keys():
    store = SignalStore()
    store.set('000001', DAY, Signal(buy=True, score=.5))
    assert present(store=store, day=pd.Timestamp(DAY))['watchlist'][0]['signal_date'] == str(DAY)


@pytest.fixture(scope='module')
def entry_cash_gate():
    """Compile the real cash gate without importing/running the live entry."""
    source = Path(__file__).resolve().parents[1] / 'generate_trade_orders.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    lot_shares = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == '_lot_shares')
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == '_sell_proceeds'
                         for target in node.targets))
    end = next(i for i in range(start, len(main.body))
               if isinstance(main.body[i], ast.If)
               and isinstance(main.body[i].test, ast.Compare)
               and isinstance(main.body[i].test.left, ast.Name)
               and main.body[i].test.left.id == '_buy_total')
    return compile(ast.Module(body=[lot_shares, *main.body[start:end + 1]],
                              type_ignores=[]), str(source), 'exec')


@pytest.mark.parametrize(
    'action,held,cash,expected_target,expected_amount,expected_quantity',
    [('open', 0, 5000, 500, 5000, 500),
     ('adjust', 500, 2000, 700, 2000, 200)],
    ids=['open_cash_5000', 'held_500_add_cash_2000'],
)
def test_entry_cash_scaling_syncs_order_schema_and_ledger(
        entry_cash_gate, action, held, cash, expected_target,
        expected_amount, expected_quantity):
    code, price, original_target = '000001', 10., 1000
    positions = {code: dict(shares=held, amount=held * price)} if held else {}
    positions_before = deepcopy(positions)
    order = dict(stock_code=code + '.SZ', action=action, price=price,
                 amount=int(original_target * price), shares=original_target,
                 order_amount=int((original_target - held) * price),
                 order_shares=original_target - held)
    if action == 'adjust':
        order['prev_shares'] = held
    account = dict(cash=cash)
    scope = dict(
        orders=[order], prev_positions=positions, account=account,
        prices={code: price}, total_equity=held * price + cash,
        new_positions={code: dict(shares=original_target,
                                 amount=int(original_target * price))},
    )
    exec(entry_cash_gate, scope)
    assert scope['orders'] == [order]
    assert order['shares'] == expected_target
    assert order['amount'] == int(expected_target * price)
    assert order['order_shares'] == expected_quantity
    assert order['order_amount'] == expected_amount
    assert order['weight_pct'] == round(order['amount'] / scope['total_equity'], 4)
    assert scope['new_positions'][code] == dict(
        shares=expected_target, amount=int(expected_target * price))

    queue_orders(account, scope['orders'], positions, DAY,
                 execution_date='2026-11-23')
    pending = account['pending_orders'][order['order_id']]
    assert pending['side'] == 'BUY'
    assert pending['quantity'] == expected_quantity
    assert pending['planned_amount'] == expected_amount
    assert order['order_quantity'] == expected_quantity
    assert order['order_planned_amount'] == expected_amount
    assert positions == positions_before and account['cash'] == cash
