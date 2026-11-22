"""Live bridge contracts with actual ledger reports; no market data or backtest."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.execution_ledger import queue_orders
from core.live_accepted465 import LiveAccepted465, STATE_KEY


CODE = '000001'


class Context:
    def __init__(self):
        self.close = 10.
        self.peer = True

    def prepare(self, code, day):
        return dict(known=True, ATR20=1.)

    def observe(self, code, day):
        return dict(quote_known=True, close=self.close, price_two_below=True,
                    peer_known=self.peer, peer_ebb=self.peer)


def setup_bridge(scale=lambda code, day: 2.):
    account = dict(cash=100000., pending_orders={}, processed_reports={}, ledger_version=2)
    positions = {}
    context = Context()
    bridge = LiveAccepted465(account, positions, context, scale, context_id='frozen-test-context')
    return account, positions, context, bridge


def plan(bridge, day='2026-11-02', point=2, target=100, side='open'):
    store = SimpleNamespace(get=lambda code, date: SimpleNamespace(chan_buy_point=point, bi_buy_point=2))
    entries = bridge.prepare_entries(day, {CODE: target * 20.}, store) if target else {}
    orders = [dict(stock_code=CODE + '.SZ', action=side, shares=target,
                   price=20., order_amount=abs(target - bridge.positions.get(CODE, {}).get('shares', 0)) * 20.)]
    queue_orders(bridge.account, orders, bridge.positions, day, execution_date=day[:8] + f'{int(day[8:])+1:02}')
    bridge.bind_orders(orders, entries)
    return orders[0]


def report(order, *, qty=100, price=20., fees=2., status='FILLED_ALL',
           stamp='2026-11-03 09:25:00', fill_id='fill-1'):
    return dict(run_id=order['run_id'], seq=order['seq'], stock_code=CODE + '.SZ',
                order_type=order['order_side'], volume=qty, price=price, fees=fees,
                order_status=status, ts=stamp, fill_id=fill_id)


def restart(account, positions, context, scale=lambda code, day: 2.):
    # Exact JSON serialization round trip, as in the live account transaction.
    account = json.loads(json.dumps(account, allow_nan=False))
    positions = json.loads(json.dumps(positions, allow_nan=False))
    return LiveAccepted465(account, positions, context, scale, context_id='frozen-test-context')


def test_pending_survives_restart_without_fake_holdings_then_partial_fill_commits():
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    assert positions == {} and account['cash'] == 100000.
    assert bridge.guard.active == {}
    bridge = restart(account, positions, context)
    fill = report(order, qty=40, fees=.8, status='PARTIAL')
    assert bridge.apply_reports([fill]) == 1
    assert bridge.positions[CODE]['shares'] == 40
    assert bridge.account['cash'] == pytest.approx(99199.2)
    basis = bridge.guard.active[CODE]
    assert basis['first_fill_unit_cost'] == pytest.approx(10.01)
    assert basis['first_fill_date'] == '2026-11-03'
    assert basis['ATR20'] == 1.
    assert basis['first_fill_raw_per_qfq'] == 2.
    assert not basis['first_fill_fees_estimated']
    before = deepcopy(bridge.account)
    assert bridge.apply_reports([fill]) == 0
    assert bridge.account == before
    bridge.apply_reports([report(order, qty=60, fees=1.2, fill_id='fill-2')])
    assert bridge.positions[CODE]['shares'] == 100
    assert bridge.guard.active[CODE] == basis


@pytest.mark.parametrize('status', ['REJECTED', 'UNFILLED_CANCEL', 'SKIPPED_NOT_ARMED'])
def test_unfilled_terminal_report_never_commits(status):
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    bridge.apply_reports([report(order, qty=0, fees=0, status=status, stamp='')])
    assert bridge.guard.active == {} and positions == {} and account['cash'] == 100000.


@pytest.mark.parametrize('point', [1, 4, 8])
def test_BI2_is_not_primary_B2(point):
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge, point=point)
    bridge.apply_reports([report(order)])
    assert positions[CODE]['shares'] == 100
    assert bridge.guard.active == {}


def test_observation_units_idempotence_exit_persistence_and_actual_full_close():
    account, positions, context, bridge = setup_bridge()
    bridge.apply_reports([report(plan(bridge))])
    context.close = 12.
    rows = bridge.observe('2026-11-03', {CODE: 24.})
    assert rows[0]['peak_close'] == 12.
    saved = deepcopy(account)
    bridge = restart(account, positions, context)
    assert bridge.observe('2026-11-03', {CODE: 24.}) == rows
    assert bridge.account == saved
    with pytest.raises(ValueError, match='close changed'):
        bridge.observe('2026-11-03', {CODE: 22.})
    assert bridge.account == saved
    context.close = 10.8
    bridge.observe('2026-11-04', {CODE: 21.6})
    assert CODE in bridge.guard.exits_for('2026-11-04')
    sell = plan(bridge, day='2026-11-04', target=0, side='close')
    bridge = restart(bridge.account, bridge.positions, context)
    bridge.observe('2026-11-05', {})  # suspension/blocked sell cannot undo exit
    assert CODE in bridge.guard.exits_for('2026-11-05')
    bridge.apply_reports([report(sell, qty=40, status='PARTIAL', stamp='2026-11-05 09:25:00')])
    assert bridge.positions[CODE]['shares'] == 60 and CODE in bridge.guard.active
    bridge.apply_reports([report(sell, qty=60, stamp='2026-11-05 09:25:01', fill_id='sell-2')])
    assert bridge.positions == {} and bridge.guard.active == {}
    assert bridge.guard.exits_for('2026-11-05') == {}


def test_additional_buy_cannot_reset_frozen_basis():
    account, positions, context, bridge = setup_bridge()
    bridge.apply_reports([report(plan(bridge))])
    basis = deepcopy(bridge.guard.active[CODE])
    add = plan(bridge, day='2026-11-03', target=200, side='adjust')
    bridge.apply_reports([report(add, stamp='2026-11-04 09:25:00', fill_id='add')])
    assert positions[CODE]['shares'] == 200
    assert bridge.guard.active[CODE] == basis


@pytest.mark.parametrize('scale', [0., -1., float('nan')])
def test_bad_scale_rolls_back_both_ledger_and_lifecycle(scale):
    account, positions, context, bridge = setup_bridge(lambda code, day: scale)
    order = plan(bridge)
    before = deepcopy(account)
    with pytest.raises(ValueError, match='scale'):
        bridge.apply_reports([report(order)])
    assert account == before and positions == {} and bridge.guard.active == {}


@pytest.mark.parametrize('stamp', ['', '09:25:00', '2026-11-02 09:25:00'])
def test_invalid_or_same_decision_fill_date_rolls_back(stamp):
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    before = deepcopy(account)
    with pytest.raises(ValueError):
        bridge.apply_reports([report(order, stamp=stamp)])
    assert account == before and positions == {}


def test_actual_timestamp_uses_shanghai_day_and_fill_day_scale():
    seen = []

    def scale(code, day):
        seen.append(day)
        return 4. if day == '2026-11-03' else 2.

    account, positions, context, bridge = setup_bridge(scale)
    bridge.apply_reports([report(plan(bridge), stamp='2026-11-02T17:25:00+00:00')])
    assert set(seen) == {'2026-11-03'}
    assert bridge.guard.active[CODE]['first_fill_unit_cost'] == pytest.approx(5.005)


def test_pending_constraint_cannot_change_and_missing_evidence_rolls_back():
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    before = deepcopy(account)
    bad = {CODE: dict(decision_date='2026-11-02', source_chan_buy_point=1, constraint=None)}
    with pytest.raises(ValueError, match='evidence changed'):
        bridge.bind_orders([order], bad)
    assert account == before
    account[STATE_KEY]['pending_entries'].clear()
    before = deepcopy(account)
    with pytest.raises(ValueError, match='lacks frozen'):
        bridge.apply_reports([report(order)])
    assert account == before and positions == {}


def test_existing_inventory_requires_explicit_unclassified_bootstrap():
    account, positions, context, bridge = setup_bridge()
    account.pop(STATE_KEY)
    positions[CODE] = dict(shares=100, entry_price=20.)
    with pytest.raises(ValueError, match='bootstrap'):
        LiveAccepted465(account, positions, context, lambda c, d: 2., context_id='frozen-test-context')
    bridge = LiveAccepted465(account, positions, context, lambda c, d: 2.,
                             context_id='frozen-test-context', bootstrap_existing=True)
    assert account[STATE_KEY]['unclassified_initial_positions'] == [CODE]
    assert bridge.guard.active == {}


def test_same_day_new_fill_can_be_observed_after_pre_fill_empty_observation():
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    bridge.observe('2026-11-03', {})
    bridge.apply_reports([report(order)])
    rows = bridge.observe('2026-11-03', {CODE: 20.})
    assert rows[0]['peak_close'] == 10.
    assert bridge.observe('2026-11-03', {CODE: 20.}) == rows


def test_late_fill_and_external_reconciliation_are_detected():
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    bridge.observe('2026-11-04', {})
    before = deepcopy(account)
    with pytest.raises(ValueError, match='Late fill'):
        bridge.apply_reports([report(order)])
    assert account == before and positions == {}
    account.setdefault('execution_history', []).append({})
    with pytest.raises(ValueError, match='bypassed'):
        restart(account, positions, context)


def test_same_day_retry_freezes_peer_evidence_across_restart():
    account, positions, context, bridge = setup_bridge()
    bridge.apply_reports([report(plan(bridge))])
    context.close = 12.
    bridge.observe('2026-11-03', {CODE: 24.})
    context.close = 10.8
    context.peer = False
    before = bridge.observe('2026-11-04', {CODE: 21.6})
    assert not before[0]['exit_committed']
    context.peer = True
    bridge = restart(account, positions, context)
    assert bridge.observe('2026-11-04', {CODE: 21.6}) == before
    assert bridge.guard.exits_for('2026-11-04') == {}
    bridge.observe('2026-11-05', {CODE: 21.6})
    assert CODE in bridge.guard.exits_for('2026-11-05')


def test_entire_report_batch_rolls_back_if_later_first_fill_is_invalid():
    account, positions, context, bridge = setup_bridge()
    order = plan(bridge)
    before = deepcopy(account)
    # Ledger accepts both fills, but actual first-fill date violates the bridge.
    with pytest.raises(ValueError):
        bridge.apply_reports([
            report(order, qty=0, fees=0, status='PARTIAL', fill_id='zero'),
            report(order, stamp='2026-11-02 09:25:00')])
    assert account == before and positions == {} and bridge.guard.active == {}


def test_full_close_then_real_reentry_has_new_decision_basis():
    account, positions, context, bridge = setup_bridge()
    bridge.apply_reports([report(plan(bridge))])
    sell = plan(bridge, day='2026-11-03', target=0, side='close')
    bridge.apply_reports([report(sell, stamp='2026-11-04 09:25:00', fill_id='sell')])
    assert bridge.guard.active == {} and positions == {}
    buy = plan(bridge, day='2026-11-04')
    assert bridge.guard.active == {}
    bridge.apply_reports([report(buy, price=21., stamp='2026-11-05 09:25:00', fill_id='reentry')])
    assert bridge.guard.active[CODE]['decision_date'] == '2026-11-04'
    assert bridge.guard.active[CODE]['first_fill_unit_cost'] == pytest.approx(10.51)


def test_real_signal_store_date_keys_and_unqueued_order_rejected():
    from datetime import date
    from core.signal_store import SignalStore

    account, positions, context, bridge = setup_bridge()
    store = SignalStore()
    store.set(CODE, date(2026, 11, 2), SimpleNamespace(chan_buy_point=2, bi_buy_point=1))
    entries = bridge.prepare_entries('2026-11-02', {CODE: 2000.}, store)
    assert entries[CODE]['constraint']['ATR20'] == 1.
    with pytest.raises(ValueError, match='queue_orders'):
        bridge.bind_orders([dict(stock_code=CODE + '.SZ', action='open', shares=100)], entries)


def test_portfolio_input_units_gross_parity_and_explicit_all_in_cost():
    account, positions, context, bridge = setup_bridge()
    bridge.apply_reports([report(plan(bridge))])
    raw_prices = {CODE: 24.}
    inputs = bridge.portfolio_inputs('2026-11-03', raw_prices)
    assert inputs == dict(prices={CODE: 12.}, current_positions={CODE: 2400.},
                          cost={CODE: [200., 10.]})
    assert raw_prices == {CODE: 24.} and positions[CODE]['shares'] == 100
    all_in = bridge.portfolio_inputs('2026-11-03', raw_prices, include_fees=True)
    assert all_in['cost'][CODE] == pytest.approx([200., 10.01])
    assert bridge.guard.active[CODE]['first_fill_unit_cost'] == pytest.approx(10.01)


def test_portfolio_cost_keeps_fill_day_scale_adds_and_partial_sales():
    def scale(code, day):
        return 2. if day <= '2026-11-03' else 4.

    account, positions, context, bridge = setup_bridge(scale)
    bridge.apply_reports([report(plan(bridge))])
    add = plan(bridge, day='2026-11-03', target=200, side='adjust')
    bridge.apply_reports([report(add, stamp='2026-11-04 09:25:00', fill_id='add')])
    # Two actual buys: 200 + 400 QFQ units; neither old fill is repriced.
    inputs = bridge.portfolio_inputs('2026-11-04', {CODE: 24.})
    assert inputs['prices'][CODE] == 6.
    assert inputs['current_positions'][CODE] == 4800.
    assert inputs['cost'][CODE] == pytest.approx([600., 4000. / 600.])
    sell = plan(bridge, day='2026-11-04', target=100, side='adjust')
    bridge.apply_reports([report(sell, stamp='2026-11-05 09:25:00', fill_id='sell')])
    bridge = restart(account, positions, context, scale)
    assert bridge.portfolio_inputs('2026-11-05', {CODE: 24.})['cost'][CODE] == pytest.approx([300., 4000. / 600.])


def test_unknown_decision_ATR_stays_unknown_after_real_fill():
    account, positions, context, bridge = setup_bridge()
    context.prepare = lambda code, day: dict(known=False, reason='entry_ATR_unknown')
    bridge.apply_reports([report(plan(bridge))])
    assert bridge.guard.active[CODE]['known'] is False
    context.close = 100.
    bridge.observe('2026-11-03', {CODE: 200.})
    assert bridge.guard.exits_for('2026-11-03') == {}


def test_wrong_context_and_unknown_bootstrap_cost_fail_without_fabrication():
    account, positions, context, bridge = setup_bridge()
    before = deepcopy(account)
    with pytest.raises(ValueError, match='context identity'):
        LiveAccepted465(account, positions, context, lambda c, d: 2., context_id='different')
    assert account == before
    account.pop(STATE_KEY)
    positions[CODE] = dict(shares=100, entry_price=20.)
    bridge = LiveAccepted465(account, positions, context, lambda c, d: 2.,
                             context_id='frozen-test-context', bootstrap_existing=True)
    with pytest.raises(ValueError, match='verified actual fill cost'):
        bridge.portfolio_inputs('2026-11-03', {CODE: 20.})


def test_provider_quote_mismatch_cannot_partially_advance_guard():
    account, positions, context, bridge = setup_bridge()
    bridge.apply_reports([report(plan(bridge))])
    before = deepcopy(account)
    context.close = 13.
    with pytest.raises(ValueError, match='quotes disagree'):
        bridge.observe('2026-11-03', {CODE: 24.})
    assert account == before and bridge.guard.active[CODE]['peak_close'] is None


@pytest.fixture
def live_tree(tmp_path, monkeypatch):
    """Isolated synthetic files, never the production account or market tree."""
    import os
    import pandas as pd

    # Profile activation mutates environment keys; isolate all such mutations.
    monkeypatch.setattr(os, 'environ', {key: value for key, value in os.environ.items()
                                       if not key.startswith('QUANT_')})
    data = tmp_path / 'data/stock_data/backtrader_data'
    raw = tmp_path / 'data/stock_data/raw_data' / CODE
    peer = tmp_path / 'data/research/accepted465_peer_context'
    for path in (data, raw, peer):
        path.mkdir(parents=True)
    # Synthetic test sessions only; the factory always reads supplied sessions.
    sessions = pd.bdate_range('2026-10-01', '2026-11-05')
    pd.DataFrame({'datetime': sessions}).to_csv(data / 'sh000001_qfq.csv', index=False)
    pd.DataFrame(dict(datetime=sessions, high=10.5, low=9.5, close=10., volume=1000.)).to_csv(
        data / f'{CODE}_qfq.csv', index=False)
    pd.DataFrame({'日期': sessions, '收盘': 20.}).to_csv(raw / 'none.csv', index=False)
    manifest = dict(start='2026-10-01', end='2026-11-05', all_market_universe=True,
                    validation=dict(known_mask_mismatches=0, ebb_mismatches=0), output_sha256={})
    (peer / 'manifest.json').write_text(json.dumps(manifest))
    cfg = {'paths.data': 'data/stock_data/backtrader_data', 'backtest.fromdate': '2026-10-01'}
    return tmp_path, cfg, data, raw / 'none.csv', peer


def seed_pending_tree(live_tree):
    from core.live_accepted465 import create_live_bridge

    root, cfg, data, raw_path, peer = live_tree
    account = dict(initial_capital=100000., cash=100000., last_date='2026-11-02',
                   pending_orders={}, processed_reports={}, ledger_version=2)
    positions = {}
    bridge = create_live_bridge(account, positions, '2026-11-02', config=cfg, root=root)
    order = plan(bridge)
    (root / 'live_account.json').write_text(json.dumps(account))
    (root / 'current_positions.json').write_text(json.dumps(positions))
    return account, positions, order


def test_factory_scans_and_reuses_maps_with_stable_identity_across_refresh(live_tree):
    import pandas as pd
    from core.live_accepted465 import create_live_bridge, LIVE_CONTEXT_ID

    root, cfg, data, raw_path, peer = live_tree
    account, positions = dict(cash=100000.), {}
    first = create_live_bridge(account, positions, '2026-11-02', config=cfg, root=root)
    assert first.context_id == LIVE_CONTEXT_ID
    assert first.raw_per_qfq(CODE, '2026-11-02') == 2.
    assert first.provider.start_date == '2026-10-01'
    assert first.provider.calendar[-1] == pd.Timestamp('2026-11-02')
    before = deepcopy(account)
    manifest = json.loads((peer / 'manifest.json').read_text())
    manifest['end'] = '2026-11-06'
    (peer / 'manifest.json').write_text(json.dumps(manifest))
    second = create_live_bridge(account, positions, '2026-11-03', config=cfg, root=root,
                                raw_map={CODE: raw_path},
                                stock_file_map={CODE: data / f'{CODE}_qfq.csv'},
                                calendar=first.provider.calendar.append(
                                    pd.DatetimeIndex(['2026-11-03'])))
    assert second.context_id == LIVE_CONTEXT_ID and second.provider is not first.provider
    assert second.raw_per_qfq(CODE, '2026-11-03') == 2.
    assert account == before  # factory does not observe or advance the lifecycle


@pytest.mark.parametrize('failure', ['stale_peer', 'future_peer_start', 'duplicate_calendar', 'missing_session'])
def test_factory_rejects_bad_coverage_without_initializing_account(live_tree, failure):
    import pandas as pd
    from core.live_accepted465 import create_live_bridge

    root, cfg, data, raw_path, peer = live_tree
    calendar = None
    manifest = json.loads((peer / 'manifest.json').read_text())
    if failure == 'stale_peer':
        manifest['end'] = '2026-11-02'
    elif failure == 'future_peer_start':
        manifest['start'] = '2026-11-04'
    elif failure == 'duplicate_calendar':
        calendar = pd.DatetimeIndex(['2026-11-03', '2026-11-03'])
    else:
        calendar = pd.DatetimeIndex(['2026-11-02'])
    (peer / 'manifest.json').write_text(json.dumps(manifest))
    account = dict(cash=100000.)
    before = deepcopy(account)
    with pytest.raises(ValueError):
        create_live_bridge(account, {}, '2026-11-03', config=cfg, root=root, calendar=calendar)
    assert account == before


def test_reconcile_persisted_policy_without_env_partial_restart_duplicate_and_close(live_tree, monkeypatch):
    import os
    import pandas as pd
    import reconcile_live_account as reconciliation
    from core.live_accepted465 import create_live_bridge, LIVE_CONTEXT_ID

    root, cfg, data, raw_path, peer = live_tree
    account, positions, order = seed_pending_tree(live_tree)
    monkeypatch.setattr(reconciliation, 'load_config', lambda: cfg)
    # Any fallback to the direct ledger is a test failure.
    monkeypatch.setattr(reconciliation, 'apply_execution_reports',
                        lambda *a, **kw: pytest.fail('accepted465 bypassed bridge'))
    os.environ['QUANT_B2_PROFIT_CONTEXT'] = str(peer)
    assert 'QUANT_TRADING_PROFILE' not in os.environ
    path = root / 'reports.csv'
    pd.DataFrame([report(order, qty=40, fees=.8, status='PARTIAL')]).to_csv(path, index=False)
    result = reconciliation.reconcile(path, root)
    assert result['reports_applied'] == 1 and result['unresolved'] == 1
    assert os.environ['QUANT_TRADING_PROFILE'] == 'accepted465'
    acct = json.loads((root / 'live_account.json').read_text())
    book = json.loads((root / 'current_positions.json').read_text())
    assert book[CODE]['shares'] == 40 and acct['cash'] == pytest.approx(99199.2)
    assert acct[STATE_KEY]['context_id'] == LIVE_CONTEXT_ID
    assert acct[STATE_KEY]['active'][CODE]['first_fill_unit_cost'] == pytest.approx(10.01)
    assert acct[STATE_KEY]['observed_date'] is None  # reconciliation never decides
    assert reconciliation.reconcile(path, root)['reports_applied'] == 0
    # Terminate the residual buy without inventing a fill.
    pd.DataFrame([report(order, qty=0, fees=0, status='UNFILLED_CANCEL',
                         stamp='', fill_id='cancel')]).to_csv(path, index=False)
    reconciliation.reconcile(path, root)
    acct = json.loads((root / 'live_account.json').read_text())
    bridge = create_live_bridge(acct, book, '2026-11-03', config=cfg, root=root)
    sell = plan(bridge, day='2026-11-03', target=0, side='close')
    (root / 'live_account.json').write_text(json.dumps(acct))
    pd.DataFrame([report(sell, qty=40, stamp='2026-11-04 09:25:00',
                         fill_id='sell')]).to_csv(path, index=False)
    reconciliation.reconcile(path, root)
    acct = json.loads((root / 'live_account.json').read_text())
    assert json.loads((root / 'current_positions.json').read_text()) == {}
    assert acct[STATE_KEY]['active'] == {} and acct[STATE_KEY]['cost_book'] == {}
    assert acct[STATE_KEY]['observed_date'] is None


def test_reconcile_stale_data_never_commits_or_falls_back(live_tree, monkeypatch):
    import pandas as pd
    import reconcile_live_account as reconciliation

    root, cfg, data, raw_path, peer = live_tree
    account, positions, order = seed_pending_tree(live_tree)
    manifest = json.loads((peer / 'manifest.json').read_text())
    manifest['end'] = '2026-11-02'
    (peer / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setenv('QUANT_B2_PROFIT_CONTEXT', str(peer))
    monkeypatch.setattr(reconciliation, 'load_config', lambda: cfg)
    monkeypatch.setattr(reconciliation, 'apply_execution_reports',
                        lambda *a, **kw: pytest.fail('stale context bypassed bridge'))
    files = [root / 'live_account.json', root / 'current_positions.json']
    before = [file.read_bytes() for file in files]
    path = root / 'reports.csv'
    pd.DataFrame([report(order)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='peer context does not cover'):
        reconciliation.reconcile(path, root)
    assert [file.read_bytes() for file in files] == before


def test_reconcile_explicit_profile_zero_fill_preserves_empty_inventory(live_tree, monkeypatch):
    import pandas as pd
    import reconcile_live_account as reconciliation

    root, cfg, data, raw_path, peer = live_tree
    account, positions, order = seed_pending_tree(live_tree)
    monkeypatch.setenv('QUANT_TRADING_PROFILE', 'accepted465')
    monkeypatch.setenv('QUANT_B2_PROFIT_CONTEXT', str(peer))
    monkeypatch.setattr(reconciliation, 'load_config', lambda: cfg)
    path = root / 'reports.csv'
    pd.DataFrame([report(order, qty=0, fees=0, status='REJECTED', stamp='')]).to_csv(path, index=False)
    reconciliation.reconcile(path, root)
    acct = json.loads((root / 'live_account.json').read_text())
    assert acct['cash'] == 100000. and acct[STATE_KEY]['active'] == {}
    assert json.loads((root / 'current_positions.json').read_text()) == {}


def test_reconcile_default_account_preserves_existing_ledger_path(live_tree, monkeypatch):
    import pandas as pd
    import reconcile_live_account as reconciliation

    root, cfg, data, raw_path, peer = live_tree
    account, positions, order = seed_pending_tree(live_tree)
    account.pop(STATE_KEY)
    (root / 'live_account.json').write_text(json.dumps(account))
    monkeypatch.setattr(reconciliation, 'load_config', lambda: cfg)
    monkeypatch.setattr(reconciliation, 'create_live_bridge',
                        lambda *a, **kw: pytest.fail('default account changed policy'))
    path = root / 'reports.csv'
    pd.DataFrame([report(order)]).to_csv(path, index=False)
    result = reconciliation.reconcile(path, root)
    assert result['reports_applied'] == 1 and result['positions'] == 1
    assert STATE_KEY not in json.loads((root / 'live_account.json').read_text())


def planner_report_phase(live_tree, *, dry_run=False, acknowledge=False):
    """Run the actual main() report/ACK/gate statements without market setup.

    Extract production statements rather than reimplementing their ordering.
    Use the real bridge, ledger and atomic commit, all under pytest's temp root;
    no import of the planner (which initializes production paths) is needed.
    This covers the report phase, not the full signal/decision/order pipeline.
    """
    import ast
    import os
    import pandas as pd
    from core.execution_ledger import (acknowledge_orders, apply_execution_reports,
                                       unresolved_orders)
    from core.live_accepted465 import create_live_bridge
    from core.live_state import commit_json_files

    root, cfg, data, raw_path, peer = live_tree
    account, positions, order = seed_pending_tree(live_tree)
    bridge = create_live_bridge(account, positions, '2026-11-03', config=cfg, root=root)
    reports = root / 'planner_fills.csv'
    pd.DataFrame([report(order, qty=40, fees=.8, status='PARTIAL')]).to_csv(reports, index=False)
    source = Path(__file__).resolve().parents[1] / 'generate_trade_orders.py'
    main = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_n_applied' for t in node.targets))
    end = next(i for i, node in enumerate(main.body[start:], start)
               if isinstance(node, ast.For) and ast.unparse(node.target) == '(_code, _info)')
    code = compile(ast.Module(body=main.body[start:end], type_ignores=[]), str(source), 'exec')
    commits, events = [], []

    def commit(updates, journal):
        events.append('commit')
        commits.append(deepcopy(updates))
        commit_json_files(updates, journal)

    def unresolved(*args, **kwargs):
        events.append('unresolved')
        return unresolved_orders(*args, **kwargs)

    scope = dict(os=os, pd=pd, args=SimpleNamespace(
        fills=str(reports), deviations=str(root / 'absent_deviations.csv'),
        dry_run=dry_run, ack_stale_unresolved=order['order_id'] if acknowledge else ''),
        account=account, prev_positions=positions, _accepted_bridge=bridge,
        config=cfg, target_date='2026-11-03', _PROJECT_DIR=str(root),
        POSITIONS_FILE=str(root / 'current_positions.json'),
        ACCOUNT_FILE=str(root / 'live_account.json'), commit_json_files=commit,
        acknowledge_orders=acknowledge_orders, unresolved_orders=unresolved,
        stale_orders=lambda *a, **kw: [], apply_execution_reports=apply_execution_reports)
    return lambda: exec(code, scope), scope, order, commits, events


def test_planner_partial_fill_commits_before_unresolved_rejects_new_plan(live_tree):
    run, scope, order, commits, events = planner_report_phase(live_tree)
    with pytest.raises(RuntimeError, match='未完成订单'):
        run()
    assert events == ['commit', 'unresolved']
    assert len(commits) == 1
    assert set(commits[0]) == {scope['ACCOUNT_FILE'], scope['POSITIONS_FILE']}
    account = json.loads(Path(scope['ACCOUNT_FILE']).read_text())
    positions = json.loads(Path(scope['POSITIONS_FILE']).read_text())
    assert positions[CODE]['shares'] == 40
    assert account['cash'] == pytest.approx(99199.2)
    assert account['pending_orders'][order['order_id']]['status'] == 'PARTIAL'
    assert account[STATE_KEY]['active'][CODE]['first_fill_unit_cost'] == pytest.approx(10.01)
    assert account[STATE_KEY]['history_count'] == len(account['execution_history']) == 1
    assert account['last_date'] == '2026-11-02'  # report commit does not advance decisions


def test_planner_ack_commits_both_files_without_inventing_residual_fill(live_tree):
    run, scope, order, commits, events = planner_report_phase(live_tree, acknowledge=True)
    run()
    assert events == ['commit', 'commit', 'unresolved']
    assert all(set(update) == {scope['ACCOUNT_FILE'], scope['POSITIONS_FILE']} for update in commits)
    first, last = (update[scope['ACCOUNT_FILE']] for update in commits)
    assert first['pending_orders'][order['order_id']]['status'] == 'PARTIAL'
    assert last['pending_orders'][order['order_id']]['status'] == 'EXPIRED_UNFILLED'
    assert last['cash'] == first['cash'] == pytest.approx(99199.2)
    assert last[STATE_KEY] == first[STATE_KEY]
    assert last['pending_orders'][order['order_id']]['unfilled_shares'] == 60
    assert json.loads(Path(scope['POSITIONS_FILE']).read_text())[CODE]['shares'] == 40


def test_planner_dry_run_partial_and_ack_never_write_live_files(live_tree):
    run, scope, order, commits, events = planner_report_phase(
        live_tree, dry_run=True, acknowledge=True)
    files = [Path(scope['ACCOUNT_FILE']), Path(scope['POSITIONS_FILE'])]
    before = [file.read_bytes() for file in files]
    run()
    assert commits == [] and events == ['unresolved']
    assert [file.read_bytes() for file in files] == before
    assert not (live_tree[0] / '.live_state_journal.json').exists()
    assert scope['account']['pending_orders'][order['order_id']]['status'] == 'EXPIRED_UNFILLED'
    assert scope['prev_positions'][CODE]['shares'] == 40  # memory-only demonstration
