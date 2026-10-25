"""Causality, actual-fill accounting and crash recovery regression tests.

Run with stdlib unittest; no broker, network, or production state is required.
"""
import ast
import contextlib
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.execution_ledger import (queue_orders, apply_execution_reports, unresolved_orders,
                                  stale_orders, acknowledge_orders)
from core.live_state import atomic_json, commit_json_files, recover_commit
from core.factor_neutralizer import neutralize_factor_df, neutralize_factors
from core.ml_predictor import MLFactorPredictor
from core.multi_timeframe import MultiTimeframeAnalyzer
from core.trading_calendar import sessions_elapsed
from core.portfolio_risk import constrain_targets


class ActualLedgerTests(unittest.TestCase):
    def test_missing_account_with_holdings_cannot_invent_cash(self):
        from core.live_account import load_account
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                load_account(100000, {'600000': {'shares': 100, 'entry_price': 10}},
                             account_file=Path(d) / 'account.json')

    def test_reconciliation_uses_isolated_root_and_is_idempotent(self):
        from reconcile_live_account import reconcile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            account = {'cash': 100000., 'initial_capital': 100000.}
            positions = {}
            orders = [{'stock_code': '600000.SH', 'action': 'open', 'shares': 1000}]
            queue_orders(account, orders, positions, '2026-10-22')
            atomic_json(root / 'live_account.json', account)
            report = {'run_id': '20261022', 'seq': 1, 'stock_code': '600000.SH',
                      'order_type': 'BUY', 'price': 10, 'volume': 1000, 'fees': 5,
                      'fill_id': 'isolated', 'order_status': 'FILLED_ALL', 'ts': '2026-10-23 09:25:00'}
            fills = root / 'fills.csv'; pd.DataFrame([report]).to_csv(fills, index=False)
            first = reconcile(fills, root)
            second = reconcile(fills, root)
            self.assertEqual(first['reports_applied'], 1)
            self.assertEqual(second['reports_applied'], 0)
            self.assertEqual(second['cash'], 89995)
            self.assertEqual(json.loads((root / 'current_positions.json').read_text())['600000']['shares'], 1000)

    def setUp(self):
        self.account = {'cash': 100000.0, 'pending_orders': {}, 'processed_reports': {}}
        self.book = {}
        # price 参与计划金额上限 (券商按金额换算整手 ⇒ 金额才是真约束)
        self.orders = [{'stock_code': '600000.SH', 'action': 'open', 'shares': 1000,
                        'price': 10.0}]
        queue_orders(self.account, self.orders, self.book, '2026-10-22')

    def report(self, **kwargs):
        return {'run_id': '20261022', 'seq': 1, 'stock_code': '600000.SH',
                'order_type': 'BUY', 'price': 10, 'volume': 1000,
                'fill_id': 'fill1', 'order_status': 'FILLED_ALL',
                'ts': '2026-10-23 09:25:00', 'fees': 5, **kwargs}

    def test_plan_does_not_change_cash_or_actual_holdings(self):
        self.assertEqual(self.account['cash'], 100000)
        self.assertEqual(self.book, {})
        self.assertEqual(len(unresolved_orders(self.account, '2026-10-23')), 1)

    def test_fill_is_idempotent(self):
        r = self.report()
        self.assertEqual(apply_execution_reports(self.account, self.book, [r]), 1)
        self.assertEqual(self.account['cash'], 89995)
        self.assertEqual(self.book['600000']['shares'], 1000)
        self.assertEqual(apply_execution_reports(self.account, self.book, [r]), 0)
        self.assertFalse(unresolved_orders(self.account))

    def test_partial_then_cancel_keeps_only_actual_shares(self):
        apply_execution_reports(self.account, self.book, [self.report(
            volume=400, order_status='PARTIAL', fees=2)])
        self.assertEqual(self.book['600000']['shares'], 400)
        self.assertEqual(len(unresolved_orders(self.account)), 1)
        apply_execution_reports(self.account, self.book, [self.report(
            fill_id='cancel', volume=0, price=0, fees=0, order_status='UNFILLED_CANCEL')])
        self.assertFalse(unresolved_orders(self.account))
        self.assertEqual(self.account['cash'], 95998)

    def test_rejection_is_not_a_fill(self):
        apply_execution_reports(self.account, self.book, [self.report(
            volume=0, price=0, fees=0, order_status='REJECTED')])
        self.assertEqual(self.book, {})
        self.assertEqual(self.account['cash'], 100000)

    def test_addition_updates_weighted_cost(self):
        apply_execution_reports(self.account, self.book, [self.report()])
        more = [{'stock_code': '600000.SH', 'action': 'adjust', 'shares': 1500}]
        queue_orders(self.account, more, self.book, '2026-10-23')
        apply_execution_reports(self.account, self.book, [self.report(
            run_id='20261023', fill_id='fill2', volume=500, price=16,
            ts='2026-10-26 09:25:00')])
        self.assertAlmostEqual(self.book['600000']['entry_price'], 12)

    def test_sell_updates_cash_and_remaining_cost(self):
        apply_execution_reports(self.account, self.book, [self.report()])
        queue_orders(self.account, [{'stock_code': '600000.SH', 'action': 'adjust', 'shares': 400}],
                     self.book, '2026-10-23')
        apply_execution_reports(self.account, self.book, [self.report(
            run_id='20261023', fill_id='sell', volume=600, order_type='SELL', price=12, fees=7.2)])
        self.assertEqual(self.book['600000']['shares'], 400)
        self.assertEqual(self.book['600000']['entry_price'], 10)
        self.assertAlmostEqual(self.account['cash'], 97187.8)

    def test_invalid_batch_is_atomic(self):
        """真矛盾 (回报代码与挂单不符) ⇒ 整批拒绝, 已处理的部分不落盘。"""
        initial = json.dumps(self.account, sort_keys=True)
        with self.assertRaises(ValueError):
            apply_execution_reports(self.account, self.book,
                                    [self.report(volume=500, order_status='PARTIAL'),
                                     self.report(fill_id='bad', volume=600,
                                                 stock_code='000001.SH')])
        self.assertEqual(json.dumps(self.account, sort_keys=True), initial)
        self.assertEqual(self.book, {})

    def test_broker_share_count_may_differ_from_plan(self):
        """券商按金额换算整手: 成交股数≠计划股数是常态, 不能拒收整批。"""
        applied = apply_execution_reports(self.account, self.book,
                                          [self.report(volume=1100, price=9.5, fees=5)])
        self.assertEqual(applied, 1)
        self.assertEqual(self.book['600000']['shares'], 1100)
        self.assertAlmostEqual(self.account['cash'], 100000 - 1100 * 9.5 - 5)
        self.assertFalse(unresolved_orders(self.account))
        self.assertTrue(any(d['kind'] == 'FILLED_ALL_RESIDUAL'
                            for d in self.account['execution_deviations']))

    def test_overfill_beyond_planned_amount_is_rejected(self):
        """金额上限是真约束: 超额成交 ⇒ 整批拒绝, 账户不变。"""
        initial = json.dumps(self.account, sort_keys=True)
        with self.assertRaises(ValueError):
            apply_execution_reports(self.account, self.book,
                                    [self.report(volume=2000, price=10, fees=0)])
        self.assertEqual(json.dumps(self.account, sort_keys=True), initial)
        self.assertEqual(self.book, {})

    def test_unknown_run_id_is_skipped_not_fatal(self):
        """fills_today.csv 是 append-only: 昨天的键混进来只跳过, 不中断整批。"""
        applied = apply_execution_reports(self.account, self.book,
                                          [self.report(run_id='20260101', seq=9, fill_id='old'),
                                           self.report()])
        self.assertEqual(applied, 1)
        self.assertEqual(self.book['600000']['shares'], 1000)
        self.assertTrue(any('unknown order' in s['reason']
                            for s in self.account['skipped_reports']))

    def test_minimal_deviation_row_terminates_order(self):
        """偏差账本的最小行 (只带 run_id/seq/status) 也能终结挂单。"""
        applied = apply_execution_reports(self.account, self.book, [
            {'run_id': '20261022', 'seq': 1, 'order_status': 'SKIPPED_NOT_ARMED'}])
        self.assertEqual(applied, 1)
        self.assertFalse(unresolved_orders(self.account))
        self.assertEqual(self.book, {})

    def test_stale_order_needs_explicit_acknowledgement(self):
        """过期挂单不自动假定成交: 必须人工核对后显式确认。"""
        cal = pd.bdate_range('2026-10-01', '2026-12-31')
        stale = stale_orders(self.account, '2026-11-10', max_age_sessions=3, calendar=cal)
        self.assertEqual([o['run_id'] + ':' + str(o['seq']) for o in stale], ['20261022:1'])
        acked = acknowledge_orders(self.account, ['20261022:1'], reason='券商账户已核对: 未成交')
        self.assertEqual(acked, ['20261022:1'])
        self.assertFalse(unresolved_orders(self.account))
        self.assertEqual(stale_orders(self.account, '2026-11-10', max_age_sessions=3,
                                      calendar=cal), [])
        self.assertTrue(any(d['kind'] == 'ACKNOWLEDGED_UNFILLED'
                            for d in self.account['execution_deviations']))

    def test_acknowledged_order_still_accepts_broker_fill(self):
        """人工判定未成交后券商仍报成交 ⇒ 以券商为准入账并留痕。"""
        acknowledge_orders(self.account, ['20261022:1'], reason='核对后以为未成交')
        apply_execution_reports(self.account, self.book, [self.report()])
        self.assertEqual(self.book['600000']['shares'], 1000)

    def test_conflicting_fill_id_fails(self):
        apply_execution_reports(self.account, self.book, [self.report()])
        with self.assertRaises(ValueError):
            apply_execution_reports(self.account, self.book, [self.report(price=11)])

    def test_old_unresolved_orders_survive_new_plan(self):
        queue_orders(self.account, [], self.book, '2026-10-23')
        self.assertEqual(len(unresolved_orders(self.account)), 1)


class StateTests(unittest.TestCase):
    def test_buy_batch_matches_scalar_get_order_and_duplicate_semantics(self):
        from core.signal_store import SignalStore
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'signals.csv'
            pd.DataFrame({'code': ['000001', '000001', '600000', '300001'],
                          'date': ['2026-10-23'] * 4, 'buy': [False, True, True, True],
                          'sell': [False] * 4, 'industry': ['A', 'A', None, 'B']}).to_csv(p, index=False)
            store = SignalStore(); store.finalize(p)
            universe = ['300001', 'missing', '000001', '600000']
            expected = []
            for code in universe:
                sig = store.get(code, '2026-10-23')
                if sig and sig.buy: expected.append((code, sig.industry))
            self.assertEqual(store.buy_signals_for_date('2026-10-23', universe), expected)
            self.assertEqual(store.buy_signals_for_date('2026-10-24', universe), [])

    def test_eligible_no_trade_step_matches_full_empty_bear_state(self):
        import live_no_trade_precheck as precheck
        import core.live_account as accounts
        from core.portfolio import PortfolioConstructor
        day = pd.Timestamp('2026-10-23').date()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); state_file = root / 'state.json'; account_file = root / 'account.json'
            p = PortfolioConstructor(); p._entry_dates['old'] = pd.Timestamp('2026-10-22').date()
            p._entry_reasons['old'] = {}; p.save_tracking_state(state_file, '2026-10-22')
            atomic_json(account_file, {'initial_capital': 100000, 'cash': 100000,
                                      'last_date': '2026-10-22', 'nav_history': [
                                          {'date': '2026-10-22', 'nav': 99000}], 'pending': None})
            full = PortfolioConstructor(); full.restore_tracking_state(state_file)
            full.update_returns(100000 / 99000 - 1)
            full.build(date=day, universe=[], current_positions={}, signal_store=None,
                       cash=100000, prices={}, bear_risk=True, trend_score=-1.)
            with patch.object(precheck, 'PORTFOLIO_STATE_FILE', str(state_file)), \
                 patch.object(accounts, 'ACCOUNT_FILE', str(account_file)), \
                 contextlib.redirect_stdout(io.StringIO()):
                acct, observed, _ = precheck._append_flat_state(str(day))
            self.assertEqual(full.tracking_state(day), observed)
            self.assertEqual(acct['nav_history'][-1]['nav'], 100000)

    def test_signal_columnar_cache_preserves_csv_values_and_invalidates(self):
        from core.signal_store import SignalStore
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'signals.csv'
            pd.DataFrame({'code': ['000001', '600000'], 'date': ['2026-10-23'] * 2,
                          'buy': [True, False], 'sell': [False, True],
                          'score': [.12345678901234567, .12345678901234565],
                          'chan_buy_point': [1, None], 'industry': ['A', 'B']}).to_csv(p, index=False)
            a, b = SignalStore(), SignalStore(); a.finalize(p); b.finalize(p)
            pd.testing.assert_frame_equal(a._df, b._df, check_exact=True)
            changed = pd.read_csv(p, dtype={'code': str}); changed.loc[0, 'score'] = .9
            changed.to_csv(p, index=False)
            c = SignalStore(); c.finalize(p)
            self.assertEqual(c.get('000001', '2026-10-23').score, .9)

    def test_live_planner_commits_plan_without_fabricating_positions_and_replays(self):
        import generate_trade_orders as live
        import core.live_account as accounts
        day = pd.Timestamp('2026-10-23').date()
        class Store:
            def finalize(self, *args): pass
            def has_date(self, *args): return True
            def max_date(self): return str(day)
            def get(self, *args): return None
        class Portfolio:
            current_exposure = 1.; peak_equity = None; _stop_loss_triggered = False
            _clb_triggered = False; _daily_returns = []; vol_lookback = 20
            _last_rebalance_date = day; last_selection = []
            _sector_rotation = None
            def __init__(self, **kwargs): pass
            def restore_tracking_state(self, *args): pass
            def tracking_state(self, **kwargs): return {'as_of': str(day)}
            def build(self, **kwargs): return {'600000': 10000.}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'signals.csv').write_text('date\n2026-10-23\n')
            changes = {'_PROJECT_DIR': d, '_STRATEGY_DIR': d,
                       'POSITIONS_FILE': str(root / 'positions.json'),
                       'ORDERS_FILE': str(root / 'orders.json'),
                       'ACCOUNT_FILE': str(root / 'account.json'),
                       'PORTFOLIO_STATE_FILE': str(root / 'portfolio.json'),
                       'REBUILD_STATE_FILE': str(root / 'rebuild.json'),
                       'SIGNALS_CSV': str(root / 'signals.csv'),
                       'SignalStore': Store, 'PortfolioConstructor': Portfolio,
                       'config': SimpleNamespace(get=lambda key, default=None:
                                                False if key == 'stock_pool.enabled' else default)}
            with contextlib.ExitStack() as stack:
                for key, value in changes.items(): stack.enter_context(patch.object(live, key, value))
                stack.enter_context(patch.object(accounts, 'ACCOUNT_FILE', str(root / 'account.json')))
                stack.enter_context(patch.object(live, 'build_stock_file_map', return_value={'600000': 'unused'}))
                stack.enter_context(patch.object(live, 'build_raw_file_map', return_value={'600000': 'unused'}))
                stack.enter_context(patch.object(live, 'load_prices_for_date', return_value={'600000': 10.}))
                stack.enter_context(patch.object(live, 'build_st_filter', return_value=(set(), set())))
                stack.enter_context(patch.object(live, 'get_index_regime', return_value=(1, .1, 1., False, False)))
                stack.enter_context(patch.object(live, 'get_market_features', return_value={}))
                stack.enter_context(patch.object(live, 'load_context', return_value={'industry_stock_counts': {}}))
                stack.enter_context(patch.object(live, 'compute_support_resistance', return_value=(8., 12., {'price': 10.})))
                stack.enter_context(patch.object(live, 'build_monitor_list', return_value=[]))
                stack.enter_context(patch.object(sys, 'argv', ['orders', '--date', str(day), '--skip-freshness']))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                live.main()
                before = {p.name: p.read_bytes() for p in root.glob('*.json')}
                live.main()
                self.assertEqual(before, {p.name: p.read_bytes() for p in root.glob('*.json')})
            account = json.loads((root / 'account.json').read_text())
            self.assertEqual(account['cash'], 500000)
            self.assertEqual(json.loads((root / 'positions.json').read_text()), {})
            self.assertEqual(len(account['pending_orders']), 1)
            self.assertEqual(json.loads((root / 'orders.json').read_text())['accounting_mode'], 'actual_fills')

    def test_execution_context_roundtrip_and_signal_binding(self):
        from core.execution_context import save_context, load_context
        with tempfile.TemporaryDirectory() as d:
            signal = Path(d) / 'signals.csv'; signal.write_text('one')
            frame = pd.DataFrame({'datetime': pd.to_datetime(['2026-10-23']), 'style_score': [.75]})
            strategy = SimpleNamespace(index_data=frame, industry_stock_counts={'A': 3})
            save_context(strategy, signal)
            context = load_context(signal)
            pd.testing.assert_frame_equal(context['index_data'], frame)
            self.assertEqual(context['industry_stock_counts'], {'A': 3})
            signal.write_text('different signals')
            self.assertIsNone(load_context(signal))

    def test_sector_history_roundtrip(self):
        from core.sector_rotation import SectorRotation
        a = SectorRotation()
        a.compute_signal_density([('x', 'A')], {'A': 3, 'B': 3})
        b = SectorRotation(); b.restore_tracking_state(json.loads(json.dumps(a.tracking_state())))
        a.compute_signal_density([('y', 'B')], {'A': 3, 'B': 3})
        b.compute_signal_density([('y', 'B')], {'A': 3, 'B': 3})
        self.assertEqual(a._rotation_speed, b._rotation_speed)
        self.assertEqual(a._signal_density_rank, b._signal_density_rank)

    def test_crash_recovery_rolls_forward(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); journal = root / 'journal.json'
            atomic_json(journal, {'version': 1, 'updates': [['account.json', {'cash': 7}],
                                                        ['orders.json', {'run_id': 'x'}]]})
            self.assertTrue(recover_commit(journal))
            self.assertEqual(json.loads((root / 'account.json').read_text()), {'cash': 7})
            self.assertFalse(recover_commit(journal))

    def test_failed_multi_file_commit_recovers(self):
        import core.live_state as mod
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); original = mod.atomic_json
            def fail_orders(path, value):
                if Path(path).name == 'orders.json':
                    raise OSError('injected disk error')
                return original(path, value)
            with patch.object(mod, 'atomic_json', side_effect=fail_orders):
                with self.assertRaises(OSError):
                    commit_json_files({root / 'account.json': {'cash': 1},
                                       root / 'orders.json': {'run_id': 'same'}}, root / 'journal.json')
            recover_commit(root / 'journal.json')
            self.assertEqual(json.loads((root / 'orders.json').read_text())['run_id'], 'same')

    def test_nonfinite_state_cannot_replace_valid_state(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.json'; atomic_json(p, {'cash': 1})
            with self.assertRaises(ValueError):
                atomic_json(p, {'cash': float('nan')})
            self.assertEqual(json.loads(p.read_text())['cash'], 1)

    def test_calendar_counts_sessions_not_holidays(self):
        dates = pd.to_datetime(['2026-09-30', '2026-10-09', '2026-10-12'])
        self.assertEqual(sessions_elapsed('2026-09-30', '2026-10-09', dates), 1)
        self.assertEqual(sessions_elapsed('2026-09-30', '2026-10-12', dates), 2)

    def test_rebalance_unit_preserves_calendar_cadence(self):
        from core.portfolio import PortfolioConstructor
        p = PortfolioConstructor()
        p._trading_calendar = pd.to_datetime(['2026-09-30', '2026-10-09', '2026-10-12'])
        p._rebalance_unit = 'calendar'
        self.assertEqual(p._rebalance_elapsed('2026-09-30', '2026-10-12'), 12)
        p._rebalance_unit = 'trading'
        self.assertEqual(p._rebalance_elapsed('2026-09-30', '2026-10-12'), 2)

    def test_bear_state_roundtrip_and_idempotence(self):
        from core.portfolio import PortfolioConstructor
        p = PortfolioConstructor()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'state.json'
            for day in pd.bdate_range('2026-01-01', periods=61):
                p.advance_market_state(day, True)
                p.advance_market_state(day, True)
                p.save_tracking_state(path, day)
                p = PortfolioConstructor(); p.restore_tracking_state(path)
            self.assertEqual(p._bear_streak, 61)
            self.assertFalse(p.advance_market_state(day, True))


class CausalityTests(unittest.TestCase):
    def test_short_target_excludes_unattainable_overnight_gap(self):
        from core.forward_targets import short_net_targets
        dates = pd.bdate_range('2026-01-01', periods=10)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '600000.csv'
            pd.DataFrame({'datetime': dates, 'open': [10.] + [10.5] * 9,
                          'close': [10.] + [10.5] * 9, 'volume': [1000] * 10}).to_csv(path, index=False)
            frame = pd.DataFrame({'code': ['600000'] * 10, 'date': dates, 'feature': np.arange(10),
                                  'future_ret': [.05] * 10})
            out = short_net_targets(frame, {'600000': str(path)}, dates)
        self.assertEqual(len(out), len(frame))
        np.testing.assert_array_equal(out['feature'], frame['feature'])
        self.assertLess(out['future_ret'].iloc[0], 0)  # 买不到的隔夜+5%不能算入盈利标签
        self.assertEqual(out['label_end_date'].iloc[0], dates[5])
        self.assertTrue(out['future_ret'].iloc[-5:].isna().all())
        self.assertEqual(frame['future_ret'].iloc[0], .05)

    def test_changing_future_labels_cannot_change_current_features_or_universe(self):
        from core.factor_preparer import prepare_factor_data
        day = pd.Timestamp('2026-01-12')
        frame = pd.DataFrame({'code': [f'{i:06}' for i in range(20)], 'date': [day] * 20,
                              'industry': ['A'] * 20, 'value': np.arange(20, dtype=float),
                              'future_ret': np.linspace(-.9, .9, 20),
                              'label_end_date': [day + pd.offsets.BDay(10)] * 20})
        changed = frame.copy(); changed['future_ret'] = np.linspace(-100., 100., 20)
        dates = list(pd.bdate_range('2024-01-01', '2026-10-23'))
        stock_map = {c: 'unused' for c in frame['code']}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('core.factor_preparer._data_fingerprint', return_value='fixture'))
            stack.enter_context(patch('core.factor_preparer._raw_code_fingerprint', return_value='fixture'))
            stack.enter_context(patch('core.factor_preparer._signal_code_fingerprint', return_value='fixture'))
            cache = stack.enter_context(patch('core.cache_manager.load_factor_cache', side_effect=[frame, changed]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            a, _, _ = prepare_factor_data(stock_map, None, {}, dates, num_workers=1)
            b, _, _ = prepare_factor_data(stock_map, None, {}, dates, num_workers=1)
        self.assertEqual(len(a), 20)
        pd.testing.assert_frame_equal(a[['code', 'value', 'value_rank']], b[['code', 'value', 'value_rank']])

    def test_final_book_limits_include_old_positions_and_retain_cash(self):
        target = constrain_targets({'old': 60., 'new': 50., 'third': 20.}, 100,
                                   {'enabled': True, 'max_single_weight': .35,
                                    'max_industry_weight': .50, 'max_gross': .70},
                                   {'old': 'A', 'new': 'A', 'third': 'B'})
        self.assertLessEqual(target['old'], 35)
        self.assertLessEqual(target['old'] + target['new'], 50)
        self.assertLessEqual(sum(target.values()), 70)

    def test_mtf_all_outputs_are_prefix_invariant(self):
        rng = np.random.default_rng(27); n = 520
        dates = pd.bdate_range('2024-01-01', periods=n)
        close = 10 * np.exp(np.cumsum(rng.normal(0, .018, n)))
        vol = rng.uniform(100, 1000, n)
        analyzer = MultiTimeframeAnalyzer()
        def calc(k):
            x = close[:k]
            return analyzer.analyze(x, x * 1.02, x * .98, x, vol[:k], dates[:k])
        full = calc(n)
        for cut in [1, 5, 59, 120, 180, 350, 402, 445, 499, 519]:
            partial = calc(cut)
            for key in vars(full):
                np.testing.assert_allclose(getattr(full, key)[:cut], getattr(partial, key),
                                           atol=1e-12, equal_nan=True, err_msg=f'{key}, cut={cut}')

    def test_missing_values_are_filled_with_cross_sectional_mean(self):
        """方案A (2026-10-25 用户裁定): 无效值填**当日截面**均值。

        钉的是当前生产语义: 均值填充只用当日截面 (决策时可得), 不含未来信息,
        因此不是前视。已知代价 = 池掩码被一并抹掉 (见下一条), 属知情取舍。
        """
        values = np.arange(20, dtype=float); values[-1] = np.nan
        result = neutralize_factors(values, ['A'] * 10 + ['B'] * 10)
        self.assertFalse(np.isnan(result[-1]))
        self.assertAlmostEqual(result[-1], values[:-1].mean(), places=10)

    def test_pool_mask_is_erased_by_mean_fill(self):
        """记录已知事实: 池掩码(NaN)在中性化后被均值填回 ⇒ 掩码不再传递到
        因子排名/ML训练。改动这一行为必须同时改 ml_predictor 的成员资格过滤。
        """
        rng = np.random.default_rng(11)
        frame = pd.DataFrame({
            'date': ['2026-10-23'] * 12,
            'industry': ['A'] * 6 + ['B'] * 6,
            'f1': list(rng.normal(size=12)),
        })
        frame.loc[11, 'f1'] = np.nan
        out = neutralize_factor_df(frame, ['f1'], industry_col='industry')
        self.assertFalse(out['f1_neu'].isna().any())
        self.assertAlmostEqual(out['f1_neu'].iloc[11],
                               out['f1_neu'].iloc[:11].mean(), places=6)

    def test_fill_indicator_identifies_pool_masked_rows(self):
        """缺失指示位 (2026-10-26): 均值填充后, "被填的格子" = 当日均值。

        池外掩码行 = 池成员之外所有因子列被置NaN ⇒ 中性化后全部等于当日均值
        ⇒ 模型必须知道哪些值是假的, 且这些非成员行不得进训练集。
        """
        from core.ml_predictor import MLFactorPredictor
        pred = MLFactorPredictor({'ml': {'use_fill_indicator': True,
                                         'feature_whitelist': ['f1', 'f2']}})
        rows = []
        for day in ('2026-10-22', '2026-10-23'):
            for i in range(10):
                rows.append({'date': day, 'code': f'{i:06d}',
                             'f1': float(i), 'f2': float(i) * 2.0})
        frame = pd.DataFrame(rows)
        for day in ('2026-10-22', '2026-10-23'):
            d = frame['date'] == day
            def code(i):
                return d & (frame['code'] == f'{i:06d}')
            # 自然缺失行(000008): f1 缺(会被填), f2 是真值 99
            frame.loc[code(8), 'f2'] = 99.0
            # 中性化按列填: f1 的有效行 = 0..7 ⇒ 填充值 = mean(0..7);
            # f2 的有效行 = 0..8 (含 99) ⇒ 填充值 = mean(0..8)。
            f1_fill = frame.loc[d & (frame['code'] < '000008'), 'f1'].mean()
            f2_fill = frame.loc[d & (frame['code'] <= '000008'), 'f2'].mean()
            # 池外行(000009): 两列都被填 ⇒ 都等于当日均值
            frame.loc[code(9), 'f1'] = f1_fill
            frame.loc[code(9), 'f2'] = f2_fill
            # 自然缺失行(000008): f1 被填, f2 保持真值
            frame.loc[code(8), 'f1'] = f1_fill
        z = pred._cross_sectional_zscore(frame[['date', 'code', 'f1', 'f2']].copy(),
                                         ['f1', 'f2'])
        out = pred._mark_filled_train(z, ['f1', 'f2'])
        # 池外行被剔除, 自然缺失行保留
        self.assertNotIn('000009', set(out['code']))
        self.assertIn('000008', set(out['code']))
        self.assertEqual(len(out), 18)
        # 指示位: 自然缺失行的 f1 被标记, f2 没被标记
        self.assertIn('f1__filled', out.columns)
        self.assertIn('f2__filled', out.columns)
        nat = out[out['code'] == '000008'].iloc[0]
        self.assertAlmostEqual(float(nat['f1__filled']), 1.0, places=6)
        self.assertAlmostEqual(float(nat['f2__filled']), 0.0, places=6)

    def test_fill_indicator_columns_survive_model_roundtrip(self):
        """载入模型后必须仍知道"哪些列有指示位" ⇒ 预测期特征矩阵与模型一致。"""
        from core.ml_predictor import MLFactorPredictor
        rng = np.random.default_rng(23)
        dates = pd.bdate_range('2025-01-01', periods=60)
        frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), 60),
                              'a': rng.normal(size=720), 'b': rng.normal(size=720),
                              'c': rng.normal(size=720), 'future_ret': rng.normal(size=720)})
        frame['label_end_date'] = frame['date'] + pd.offsets.BDay(10)
        # 每个日期的最后一行 = "全填行"(池外掩码行): 三列都等于**其余有效行**的均值
        # (中性化按列填: 填充值 = 该列当日有效行的均值 ⇒ 填完后的当日均值不变 ⇒ z=0)
        for day in dates:
            m = frame['date'] == day
            for col in ('a', 'b', 'c'):
                fill_val = frame.loc[m & (frame['code'] != 11), col].mean()
                frame.loc[m & (frame['code'] == 11), col] = fill_val
        cfg = {'ml': {'feature_whitelist': ['a', 'b', 'c'],
                      'use_fill_indicator': True, 'log_context': 'REGRESSION_RANDOM_DATA'}}
        p = MLFactorPredictor(cfg)
        p.xgb_params.update(n_jobs=1, n_estimators=4, early_stopping_rounds=2)
        ic = p.train(frame, {'by_date': {}})
        self.assertTrue(np.isfinite(ic))
        self.assertTrue(p._fill_indicator_src_cols)
        self.assertIn('a__filled', p._trained_features)
        last = frame.loc[frame['date'] == dates[-1]]
        pred = p.predict(last, {'regime': 0})
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'ensemble.json')
            p.save_model(path)
            restored = MLFactorPredictor(cfg)
            restored.load_model(path)
            self.assertEqual(restored._fill_indicator_src_cols, p._fill_indicator_src_cols)
            np.testing.assert_array_equal(restored.predict(last, {'regime': 0}), pred)

    def test_fill_indicator_off_is_byte_identical(self):
        """默认关 = 逐位零改动 (生产基线不能被这个开关碰到)。"""
        from core.ml_predictor import MLFactorPredictor
        pred = MLFactorPredictor({'ml': {'feature_whitelist': ['f1', 'f2']}})
        self.assertFalse(pred.use_fill_indicator)
        frame = pd.DataFrame({'date': ['2026-10-22'] * 5,
                               'code': [f'{i:06d}' for i in range(5)],
                               'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
                               'f2': [2.0, 4.0, 6.0, 8.0, 10.0]})
        z = pred._cross_sectional_zscore(frame.copy(), ['f1', 'f2'])
        before = z.copy()
        out = pred._mark_filled_train(z, ['f1', 'f2'])
        pd.testing.assert_frame_equal(before, out)

    def test_size_neutralization_removes_exposure(self):
        cap = np.exp(np.linspace(18, 25, 40)); signal = np.log(cap) + np.sin(np.arange(40)) * .1
        frame = pd.DataFrame({'date': ['2026-10-23'] * 40, 'industry': ['A'] * 40,
                              'cap': cap, 'value': signal})
        out = neutralize_factor_df(frame, ['value'], market_cap_col='cap')
        self.assertLess(abs(np.corrcoef(out['value_neu'], np.log(cap))[0, 1]), 1e-10)

    def test_size_only_does_not_demean_industry(self):
        vals = np.r_[np.arange(20), np.arange(20) + 100.0]
        out = neutralize_factors(vals, ['A'] * 20 + ['B'] * 20, neutralize_industry=False)
        np.testing.assert_allclose(out, vals)

    def test_recent_samples_have_greater_weight(self):
        w = MLFactorPredictor.time_weights([0, 250, 500])
        self.assertTrue(w[0] > w[1] > w[2])

    def test_exact_label_end_purges_train_validation_overlap(self):
        p = MLFactorPredictor()
        dates = pd.bdate_range('2026-01-01', periods=30)
        frame = pd.DataFrame({'date': dates, 'label_end_date': dates + pd.offsets.BDay(10)})
        train, val = p.split_masks(frame, dates[20])
        self.assertTrue((frame.loc[train, 'label_end_date'] < dates[20]).all())
        self.assertEqual(int(val.sum()), 10)

    def test_train_selects_features_without_validation_labels(self):
        rng = np.random.default_rng(11); dates = pd.bdate_range('2025-01-01', periods=100)
        frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), 100),
                              'a': rng.normal(size=1200), 'b': rng.normal(size=1200),
                              'c': rng.normal(size=1200), 'future_ret': rng.normal(size=1200)})
        frame['label_end_date'] = frame['date'] + pd.offsets.BDay(10)
        p = MLFactorPredictor({'ml': {'feature_whitelist': ['a', 'b', 'c'],
                                     'log_context': 'REGRESSION_RANDOM_DATA'}})
        p.xgb_params.update(n_jobs=1, n_estimators=4, early_stopping_rounds=2)
        seen = []
        original = p._select_top_features
        def capture(df, features, k):
            seen.append(df.copy()); return original(df, features, k)
        with patch.object(p, '_select_top_features', side_effect=capture):
            ic = p.train(frame, {'by_date': {}})
        self.assertTrue(np.isfinite(ic))
        self.assertLess(seen[0]['label_end_date'].max(), dates[80])
        pred = p.predict(frame.loc[frame['date'] == dates[-1]], {'regime': 0})
        self.assertEqual(pred.shape, (12,))
        self.assertTrue(np.isfinite(pred).all())
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'ensemble.json')
            p.save_model(path)
            restored = MLFactorPredictor(p.config)
            restored.load_model(path)
            np.testing.assert_array_equal(restored.predict(frame.loc[frame['date'] == dates[-1]],
                                                          {'regime': 0}), pred)
            member = Path(path + '.1.json')
            payload = member.read_bytes()
            member.write_bytes(payload + b' ')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                restored.load_model(path)
            member.unlink()
            with self.assertRaises(FileNotFoundError):
                restored.load_model(path)
            self.assertEqual(len(restored._ensemble_models), 3)

    def test_missing_future_label_cannot_change_other_stocks_cross_features(self):
        dates = pd.bdate_range('2025-01-01', periods=100)
        rng = np.random.default_rng(17)
        frame = pd.DataFrame({'date': np.repeat(dates, 12), 'code': np.tile(np.arange(12), 100),
                              'a': rng.normal(size=1200), 'b': rng.normal(size=1200),
                              'c': rng.normal(size=1200), 'future_ret': rng.normal(size=1200)})
        frame['label_end_date'] = frame['date'] + pd.offsets.BDay(10)
        changed = frame.copy(); changed.loc[0, 'future_ret'] = np.nan
        captured = []
        for source in (frame, changed):
            p = MLFactorPredictor({'ml': {'feature_whitelist': ['a', 'b', 'c'],
                                         'log_context': 'REGRESSION_RANDOM_DATA'}})
            p.xgb_params.update(n_jobs=1, n_estimators=2, early_stopping_rounds=1)
            original = p.prepare_features
            def record(*args, **kwargs):
                result = original(*args, **kwargs)
                captured.append(result.copy())
                return result
            with patch.object(p, '_select_top_features', return_value=['a', 'b', 'c']), \
                 patch.object(p, 'prepare_features', side_effect=record):
                p.train(source)
        common = lambda df: df.loc[(df['date'] == dates[0]) & (df['code'] != 0),
                                  ['code', 'a', 'b', 'c', 'cross_a_b', 'cross_a_c', 'cross_b_c']].reset_index(drop=True)
        pd.testing.assert_frame_equal(common(captured[0]), common(captured[1]))


class ExecutionTests(unittest.TestCase):
    def replay(self, closes, volumes=None, goals=None):
        # Execute the real engine function with fixture data, isolated output paths
        # and a tiny strategy whose decisions are known in advance.
        src = Path(__file__).resolve().parents[1] / 'bt_execution.py'
        tree = ast.parse(src.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_vectorized_backtest')
        from core.config_loader import load_config
        from core.pipeline_logger import plog
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = root / 'prices'; data.mkdir()
            out = root / 'rolling_validation_results'; out.mkdir()
            dates = pd.bdate_range('2026-01-01', periods=len(closes))
            pd.DataFrame({'datetime': dates}).to_csv(data / 'sh000001_qfq.csv', index=False)
            pd.DataFrame({'datetime': dates, 'open': [10.] * len(dates),
                          'low': np.minimum(closes, 10), 'close': closes,
                          'volume': volumes if volumes is not None else [100000.] * len(dates),
                          'amount': [100000000.] * len(dates)}).to_csv(data / '600000_qfq.csv', index=False)
            raw = root / 'raw/600000'; raw.mkdir(parents=True)
            pd.DataFrame({'日期': dates, '收盘': closes}).to_csv(raw / 'none.csv', index=False)
            targets = goals or {5: {'600000': 10000}, 6: {}}
            state = SimpleNamespace(update_returns=lambda x: None, _dbg={})
            seen = []
            def generate(**kw):
                seen.append(kw)
                return targets.get(len(seen) - 1, kw['current_positions'])
            strategy = SimpleNamespace(index_data=None, portfolio=state,
                                       signal_store=SimpleNamespace(get=lambda *x: None),
                                       generate_positions=generate)
            ns = {'pd': pd, 'np': np, 'os': os, '__file__': str(root / 'bt_execution.py'),
                  'config': load_config(), 'DATA_PATH': str(data), 'RAW_DATA_PATH': str(root / 'raw'),
                  'COMMISSION': .0005, 'plog': plog, 'defaultdict': defaultdict}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ns)
            with patch('core.stock_pool._is_non_stock_series', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                result = ns['_vectorized_backtest'](strategy, None, dates[0], dates[-1], 100000)
            return result, seen

    def test_empty_target_liquidates(self):
        result, _ = self.replay([10.] * 10)
        self.assertEqual(result['fill_stats']['buy_filled'], 1)
        self.assertEqual(result['fill_stats']['sell_filled'], 1)
        self.assertLess(result['realized_trades'][0]['ret'], 0)
        self.assertLess(result['realized_trades'][0]['pnl'], 0)

    def test_open_fill_does_not_depend_on_same_day_close_or_low_volume(self):
        closes = [10.] * 10; closes[6] = 1.5
        vols = [100000.] * 10; vols[6] = 50
        result, seen = self.replay(closes, vols)
        self.assertEqual(result['fill_stats']['buy_filled'], 1)
        self.assertIn('600000', seen[6]['current_positions'])

    def test_low_price_exit_is_not_blocked_by_entry_filters(self):
        closes = [10.] * 10; closes[7] = 1.5
        result, _ = self.replay(closes)
        self.assertEqual(result['fill_stats']['sell_filled'], 1)


if __name__ == '__main__':
    unittest.main()
