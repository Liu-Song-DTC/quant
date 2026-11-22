"""Actual-fill adapter for accepted465's primary-B2 profit guard.

Caller contract (no file I/O or estimated executions in this module)::

    bridge = LiveAccepted465(account, actual_positions, qfq_context,
                             raw_per_qfq, context_id=sealed_context_identity)
    bridge.apply_reports(broker_reports, commission=..., stamp_tax=...)
    bridge.observe(decision_date, fresh_raw_closes)  # completed, traded bars only
    portfolio.b2_profit_guard = bridge.guard
    inputs = bridge.portfolio_inputs(decision_date, raw_valuation_prices)
    targets = portfolio.build(..., **inputs)  # orders STILL use raw prices
    entries = bridge.prepare_entries(decision_date, targets, signal_store)
    queue_orders(account, final_orders, actual_positions, decision_date, ...)
    bridge.bind_orders(final_orders, entries)
    # Commit account + actual_positions + portfolio state + orders with the
    # existing live_run_lock/commit_json_files transaction, orders last.

State lives inside account[STATE_KEY], including pending entry provenance. Both
generate_trade_orders and standalone reconciliation must use apply_reports.
An adapter cannot retrofit missing historical entry provenance: existing stock
without a saved bridge requires explicit bootstrap_existing=True and remains
unclassified until actually sold. Never import simulated inventory or guard state.

raw_per_qfq(code, day) must return raw(day)/frozen_qfq(day), strictly positive,
using verified as-of data. Do not pass price_scale's fallback-to-one behavior.
The provider's ATR, close and MA are QFQ; actual inventory remains broker shares.
This adapter does not implement accepted465 ranking or the execution no-add cap.
"""
from copy import deepcopy
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from .execution_ledger import apply_execution_reports
from .profit_guard import B2ContextProfitGuard


STATE_KEY = 'accepted465_live'
POLICY_ID = 'accepted465-primary-B2-MA20-v1'
LIVE_CONTEXT_ID = 'accepted465-primary-B2-MA20-peer-kernel-v1'


def create_live_bridge(account, positions, decision_date, *, config=None, root=None,
                       raw_map=None, stock_file_map=None, calendar=None,
                       data_path=None, peer_path=None):
    """Construct the shared live adapter; never refresh data or write files.

    decision_date is the required as-of coverage date. Supply the existing raw
    and QFQ maps/calendar from generate_trade_orders, or omit them to scan the
    same production directories. Relative configured paths resolve under root.
    Calendar input must be exchange sessions, never a generated weekday range.
    A new provider/basis is constructed each call so daily dataset extensions do
    not reuse cached files. The persisted context identity is the stable kernel
    identity, not the daily manifest hash. Historical data immutability remains
    the data publisher's responsibility.
    """
    import pandas as pd

    from .config_loader import load_config
    from .deployment_profile import activate_profile
    from .live_price_basis import LivePriceBasis
    from .profit_guard import DailyExitContext

    profile = activate_profile()
    cfg = load_config() if config is None else config
    project = Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else project / path

    prices_path = resolve(data_path or cfg.get('paths.data', 'data/stock_data/backtrader_data'))
    peers_path = resolve(peer_path or os.environ.get('QUANT_B2_PROFIT_CONTEXT') or
                         (profile['peer_context'] if profile else 'data/research/accepted465_peer_context'))
    day = _day(decision_date)
    start = _day(cfg.get('backtest.fromdate', '2021-01-01'))
    if calendar is None:
        calendar = pd.read_csv(prices_path / 'sh000001_qfq.csv', usecols=['datetime'])['datetime']
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    if sessions.hasnans or sessions.has_duplicates or sessions.tz is not None:
        raise ValueError('Live calendar requires unique, valid local exchange dates')
    sessions = sessions.sort_values()
    sessions = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(day))]
    if sessions.empty or sessions[-1] != pd.Timestamp(day):
        raise ValueError(f'Live exchange calendar does not cover required session: {day}')
    provider = DailyExitContext(prices_path, peers_path, sessions, start)
    if not provider.manifest['start'] <= day <= provider.manifest['end']:
        raise ValueError(f'accepted465 peer context does not cover required session: {day}')
    if stock_file_map is None:
        stock_file_map = {path.name[:-8]: path for path in prices_path.glob('*_qfq.csv')
                          if not path.name.startswith('._')}
    if raw_map is None:
        raw_root = project / 'data/stock_data/raw_data'
        raw_map = {path.parent.name: path for path in raw_root.glob('*/none.csv')}
    basis = LivePriceBasis(raw_map, stock_file_map)
    return LiveAccepted465(account, positions, provider, basis.raw_per_qfq,
                           context_id=LIVE_CONTEXT_ID)


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _day(value):
    text = str(value)
    if not re.match(r'^\d{4}-\d{2}-\d{2}(?:$|[T ])', text):
        raise ValueError('Actual fill/decision requires a full calendar date')
    parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(ZoneInfo('Asia/Shanghai'))
    return parsed.date().isoformat()


def _shares(positions):
    result = {}
    for code, info in positions.items():
        value = float(info.get('shares', 0))
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError('Actual inventory requires nonnegative integer broker shares')
        if value:
            result[str(code)] = int(value)
    return result


class LiveAccepted465:
    """Persist guard state in the same transaction as the actual account.

    Methods update memory only. apply_reports stages the ledger and guard
    together, then publishes both only after every report passes validation.
    Repeated reports rely on the ledger's real fill-ID deduplication. Repeated
    observations of the same session cannot change already observed quotes.
    """

    def __init__(self, account, positions, provider, raw_per_qfq, *, context_id,
                 bootstrap_existing=False):
        if not isinstance(context_id, str) or not context_id.strip():
            raise ValueError('A versioned QFQ/peer context identity is required')
        self.account, self.positions = account, positions
        self.provider, self.raw_per_qfq = provider, raw_per_qfq
        self.context_id = context_id
        held = _shares(positions)
        state = account.get(STATE_KEY)
        if state is None:
            if (held or account.get('execution_history')) and not bootstrap_existing:
                raise ValueError('Existing live account requires explicit bootstrap; no inferred lifecycle')
            state = dict(version=1, policy_id=POLICY_ID, context_id=context_id,
                         active={}, pending_entries={}, observed_date=None,
                         current_exits={}, observation=None,
                         cost_book={},
                         history_count=len(account.get('execution_history', [])),
                         unclassified_initial_positions=sorted(held))
        if (state.get('version') != 1 or state.get('policy_id') != POLICY_ID
                or state.get('context_id') != context_id):
            raise ValueError('Incompatible accepted465 live state or context identity')
        if state['history_count'] != len(account.get('execution_history', [])):
            raise ValueError('Execution reports bypassed the live bridge; reconcile lifecycle evidence')
        if any(not held.get(code) for code in state['active']):
            raise ValueError('Guard lifecycle without actual inventory')
        self.guard = self._restore_guard(state)
        account[STATE_KEY] = _json_copy(state)

    def _restore_guard(self, state):
        guard = B2ContextProfitGuard(self.provider, primary_points=(2,))
        guard.active = deepcopy(state['active'])
        guard.observed_date = state['observed_date']
        guard.current_exits = deepcopy(state['current_exits'])
        return guard

    def _state(self):
        state = self.account[STATE_KEY]
        if state['history_count'] != len(self.account.get('execution_history', [])):
            raise ValueError('Execution reports bypassed the live bridge')
        return deepcopy(state)

    def _save_guard(self, state, guard):
        state.update(active=deepcopy(guard.active), observed_date=guard.observed_date,
                     current_exits=deepcopy(guard.current_exits))
        return _json_copy(state)

    def _scale(self, code, day):
        scale = float(self.raw_per_qfq(code, day))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Missing/invalid verified raw-per-QFQ scale')
        return scale

    def prepare_entries(self, decision_date, targets, signal_store):
        """Freeze primary origin/decision ATR; creates no active lifecycle.

        targets are currency values. Pass final targets; only actual fresh
        holdings qualify. Missing origin is an error, never inferred from BI2.
        """
        day = _day(decision_date)
        held = _shares(self.positions)
        prepared = self.guard.prepare(date.fromisoformat(day), targets, held, signal_store)
        entries = {}
        for code, value in targets.items():
            if value <= 0 or held.get(code, 0):
                continue
            signal = signal_store.get(code, date.fromisoformat(day))
            if signal is None:
                raise ValueError(f'Missing actual decision signal origin: {code}')
            point = int(getattr(signal, 'chan_buy_point', 0))
            entries[code] = dict(decision_date=day, source_chan_buy_point=point,
                                 constraint=prepared.get(code))
            if point == 2 and code not in prepared:
                raise ValueError(f'Missing primary-B2 decision constraint: {code}')
        return _json_copy(entries)

    def bind_orders(self, orders, entries):
        """Bind frozen evidence AFTER queue_orders, only to actual BUY intents.

        Call after all cash/share caps. Pending entries are persisted even
        before any fill, keyed by run_id:seq, never just by stock code.
        """
        state = self._state()
        held = _shares(self.positions)
        for order in orders:
            key = order.get('order_id')
            pending = self.account.get('pending_orders', {}).get(key)
            if pending is None:
                code = str(order['stock_code']).split('.')[0].zfill(6)
                target = 0 if order['action'] == 'close' else int(order.get('shares', 0))
                if target != held.get(code, 0):
                    raise ValueError('Order must be registered with queue_orders first')
                continue
            code = pending['code']
            if pending['side'] != 'BUY' or held.get(code, 0):
                continue
            entry = entries.get(code)
            if entry is None or entry['decision_date'] != pending['decision_date']:
                raise ValueError(f'Missing matching pending entry evidence: {key}')
            entry = _json_copy(entry)
            previous = state['pending_entries'].get(key)
            if previous is not None and previous != entry:
                raise ValueError(f'Pending entry evidence changed: {key}')
            state['pending_entries'][key] = entry
        self.account[STATE_KEY] = _json_copy(state)

    def apply_reports(self, reports, *, commission=0.0005, stamp_tax=0.0005):
        """Book only broker reports and commit lifecycle on first positive fill.

        A PARTIAL first buy is sufficient. Later buys never reset frozen basis.
        Rejection/cancellation with zero volume creates no lifecycle. Only
        shares reaching zero closes it; a terminal order alone is insufficient.
        The caller must atomically persist account AND positions afterward.
        """
        state = self._state()
        acct, positions = deepcopy(self.account), deepcopy(self.positions)
        before = len(acct.get('execution_history', []))
        count = apply_execution_reports(acct, positions, reports,
                                        commission=commission, stamp_tax=stamp_tax)
        guard = self._restore_guard(state)
        held = _shares(self.positions)
        for fill in acct.get('execution_history', [])[before:]:
            quantity = fill['quantity']
            if not quantity:
                continue
            code, key = fill['code'], fill['order']
            day = _day(fill['ts'])  # actual timestamp, never import/plan date
            if guard.observed_date is not None and day < guard.observed_date:
                raise ValueError('Late fill predates guard observations; replay actual evidence first')
            old = held.get(code, 0)
            if fill['side'] == 'BUY':
                scale = self._scale(code, day)
                if old == 0:
                    entry = state['pending_entries'].get(key)
                    if entry is None:
                        raise ValueError(f'First fill lacks frozen pending entry evidence: {key}')
                    if day <= entry['decision_date']:
                        raise ValueError('Actual first fill must follow its decision session')
                    constraint = entry['constraint']
                    if entry['source_chan_buy_point'] == 2:
                        if not constraint:
                            raise ValueError('Primary-B2 first fill lacks decision ATR evidence')
                        all_in_raw = fill['price'] + fill['booked_fees'] / quantity
                        basis = guard.first_fill(code, day, constraint, all_in_raw / scale)
                        basis.update(first_fill_order=key, first_fill_id=fill['fill_id'],
                                     first_fill_raw_per_qfq=scale,
                                     first_fill_fees_estimated=fill['fees_estimated'])
                        guard.active[code] = basis
                # Cost tracker uses QFQ-equivalent units, not raw shares with
                # QFQ prices. Freeze the fill-day scale; a future ex-rights
                # scale must never be applied retroactively to old fills.
                costs = state['cost_book']
                if old == 0:
                    costs[code] = dict(qfq_shares=0., gross=0., all_in=0.)
                if code in costs:
                    costs[code]['qfq_shares'] += quantity * scale
                    costs[code]['gross'] += quantity * fill['price']
                    costs[code]['all_in'] += quantity * fill['price'] + fill['booked_fees']
                held[code] = old + quantity
            else:
                held[code] = old - quantity
                if code in state['cost_book']:
                    fraction = held[code] / old
                    state['cost_book'][code] = {
                        key: value * fraction for key, value in state['cost_book'][code].items()}
                if held[code] == 0:
                    guard.closed(code)
                    state['cost_book'].pop(code, None)
                    held.pop(code)
                    if code in state['unclassified_initial_positions']:
                        state['unclassified_initial_positions'].remove(code)
        if held != _shares(positions):
            raise ValueError('Lifecycle inventory disagrees with actual ledger')
        state['history_count'] = len(acct.get('execution_history', []))
        acct[STATE_KEY] = self._save_guard(state, guard)
        self.account.clear()
        self.account.update(acct)
        self.positions.clear()
        self.positions.update(positions)
        self.guard = guard
        return count

    def portfolio_inputs(self, decision_date, raw_prices, *, include_fees=False):
        """Return build kwargs: QFQ prices/cost, actual currency holding values.

        Default cost=[QFQ-equivalent shares, average actual QFQ execution price]
        matches bt_execution's cost_tracker (excludes fees). include_fees=True
        explicitly selects remaining purchase cost including booked fees; it is
        NOT the original465 portfolio cost convention. B2 profit protection
        always uses its own frozen first-fill ALL-IN cost, independently.

        No inferred entry_price fallback: bootstrapped positions without actual
        fill cost history require separately verified migration evidence.
        raw_prices remain untouched for build_orders/support-price conversion.
        Valuation quotes may be stale; never pass them as fresh observe inputs.
        """
        day = _day(decision_date)
        state = self._state()
        qfq = {}
        for code, value in raw_prices.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Valuation raw price must be finite and positive')
            qfq[code] = value / self._scale(code, day)
        current, cost = {}, {}
        for code, quantity in _shares(self.positions).items():
            if code not in raw_prices:
                raise ValueError(f'Missing actual holding valuation: {code}')
            current[code] = quantity * float(raw_prices[code])
            basis = state['cost_book'].get(code)
            if not basis or basis['qfq_shares'] <= 0:
                raise ValueError(f'Missing verified actual fill cost basis: {code}')
            amount = basis['all_in' if include_fees else 'gross']
            cost[code] = [basis['qfq_shares'], amount / basis['qfq_shares']]
        return dict(prices=qfq, current_positions=current, cost=cost)

    def observe(self, decision_date, fresh_raw_closes):
        """Observe completed traded closes before portfolio.build (never stale MTM).

        Omit suspended/missing quotes. Existing committed exits survive unknown
        quotes/peers and partial/blocked sales. Same-session repeated calls are
        idempotent; revised closes for an already observed lifecycle are rejected.
        A same-day newly reported fill can then be observed using that day's close.
        """
        day = _day(decision_date)
        state = self._state()
        guard = self._restore_guard(state)
        held = _shares(self.positions)
        qfq = {}
        for code in guard.active:
            raw = fresh_raw_closes.get(code)
            if raw is not None:
                raw = float(raw)
                if not math.isfinite(raw) or raw <= 0:
                    raise ValueError('Fresh raw close must be finite and positive')
                qfq[code] = raw / self._scale(code, day)
        previous = state['observation']
        identities = {code: row['first_fill_order'] + ':' + str(row['first_fill_id'])
                      for code, row in guard.active.items()}
        if previous and previous['date'] == day:
            for code, identity in identities.items():
                if previous['identities'].get(code) == identity:
                    old, new = previous['qfq_closes'].get(code), qfq.get(code)
                    if old != new:
                        raise ValueError('Same-session completed close changed; explicit recovery required')
        # Reuse the entire first observation, including unknown peer evidence.
        # A same-day retry must not acquire a different answer from a refreshed
        # provider; a newly filled lifecycle may still get its first observation.
        same_session = previous and previous['date'] == day
        provider = self.provider

        class SessionContext:
            def observe(self, code, when):
                if (same_session and previous['identities'].get(code) == identities[code]):
                    return deepcopy(previous['contexts'][code])
                return provider.observe(code, when)

        guard.provider = SessionContext()
        rows = guard.observe(day, held, qfq)
        guard.provider = provider
        fields = ('quote_known', 'close', 'price_two_below', 'peer_known', 'peer_ebb',
                  'peer_return5', 'peer_relative5', 'peer_breadth5')
        contexts = {}
        for row in rows:
            context = {key: row[key] for key in fields if key in row}
            for key in ('peer_return5', 'peer_relative5', 'peer_breadth5'):
                if key in context:
                    value = float(context[key]) if context[key] is not None else float('nan')
                    context[key] = value if math.isfinite(value) else None
            contexts[row['code']] = context
        state['observation'] = dict(date=day, qfq_closes=qfq, identities=identities,
                                    contexts=contexts)
        self.account[STATE_KEY] = self._save_guard(state, guard)
        self.guard = guard
        return deepcopy(rows)
