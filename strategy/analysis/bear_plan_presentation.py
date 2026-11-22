"""Present accepted465 bear observations after build, before the cash gate.

This module never builds a portfolio or creates executable orders. The caller
must supply the same raw bear flag used by the completed same-day build.
Only the effective bear gate is known here; macro-adjusted FAST/NORM is not.
"""

from math import isfinite
from numbers import Real
from operator import index

import numpy as np
import pandas as pd


_OBSERVATION_NOTE = (
    '【有效熊市 BEAR：不买入】仅为当日源信号候选观察名单；'
    '不承诺通过全部组合准入，不代表持仓或目标权重，不生成可执行买单。'
)


def _shares(value):
    """Reject unknown quantities rather than infer a trade direction."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('open/adjust requires known nonnegative integer shares') from exc
    if (isinstance(value, (bool, np.bool_)) or not isfinite(number)
            or number < 0 or not number.is_integer()):
        raise ValueError('open/adjust requires known nonnegative integer shares')
    return number


def present_bear_plan(date, portfolio, raw_bear_risk, universe, signal_store,
                      orders, actual_positions, top_n) -> dict:
    """Return orders, watchlist, market_status, buy_enabled, watchlist_note.

    ``raw_bear_risk`` is a bool, not an inferred market regime. The portfolio
    must have v_recovery_enabled=False and _market_state_date equal to date.
    ``signal_store.get(code, datetime.date)`` must perform an exact-day lookup.
    Orders use ``stock_code`` (or ``code``), ``action`` and target ``shares``;
    actual_positions uses bare six-digit codes and records with ``shares``.
    A missing position means zero shares; an unknown quantity is rejected.

    buy_enabled describes this bear gate alone, not overall entry admission.
    Non-bear orders are returned unchanged, including their original ordering.
    Inputs and source signal scores are never edited.
    """
    if getattr(portfolio, 'v_recovery_enabled', None) is not False:
        raise ValueError('accepted465 requires v_recovery_enabled=False; cannot infer state')
    day = pd.Timestamp(date).date()
    if pd.isna(day) or getattr(portfolio, '_market_state_date', None) != day:
        raise ValueError('same-day build must advance _market_state_date to date first')
    if not isinstance(raw_bear_risk, (bool, np.bool_)):
        raise ValueError('raw_bear_risk must be a known boolean from the same-day build')
    if isinstance(top_n, (bool, np.bool_)):
        raise ValueError('top_n must be a nonnegative integer')
    try:
        limit = index(top_n)
    except TypeError as exc:
        raise ValueError('top_n must be a nonnegative integer') from exc
    if limit < 0:
        raise ValueError('top_n must be a nonnegative integer')

    # The date guard makes this a same-day idempotent read of the source rule.
    # Do not duplicate its threshold or reinterpret its FAST exception.
    effective_bear = bool(portfolio.advance_market_state(day, raw_bear_risk))
    market_status = {
        'status': 'BEAR' if effective_bear else 'NOT_BEAR',
        'effective_bear': effective_bear,
        'raw_bear_risk': bool(raw_bear_risk),
        'bear_streak': portfolio._bear_streak,
        'market_state_date': day.isoformat(),
        'v_recovery_enabled': False,
        'strategy_exception': 'FAST' if raw_bear_risk and not effective_bear else None,
    }
    if not effective_bear:
        return dict(
            orders=orders, watchlist=[], market_status=market_status,
            buy_enabled=True,
            watchlist_note=(
                '非有效熊市 NOT_BEAR：不生成熊市观察名单；buy_enabled仅表示熊市闸允许，'
                '其他组合准入仍由原策略决定。'
                + ('原策略 FAST 例外，保留原订单。' if market_status['strategy_exception'] else '')
            ),
        )

    filtered_orders = []
    for order in orders:
        if order['action'] in ('open', 'adjust'):
            code = order.get('stock_code', order.get('code'))
            if code is None:
                raise ValueError('open/adjust requires stock_code or code')
            code = str(code).split('.')[0].zfill(6)
            actual = actual_positions.get(code, {'shares': 0})
            if _shares(order.get('shares')) > _shares(actual.get('shares')):
                continue
        filtered_orders.append(order)

    candidates = []
    for code in dict.fromkeys(universe):
        signal = signal_store.get(code, day)
        if signal is None:
            continue
        buy, sell = getattr(signal, 'buy', None), getattr(signal, 'sell', None)
        if (not isinstance(buy, (bool, np.bool_)) or not bool(buy)
                or not isinstance(sell, (bool, np.bool_)) or bool(sell)):
            continue
        score = getattr(signal, 'score', None)
        if (isinstance(score, (bool, np.bool_)) or not isinstance(score, Real)
                or not isfinite(score)):
            continue
        candidates.append(dict(
            code=str(code), score=float(score),
            industry=getattr(signal, 'industry', ''),
            factor_name=getattr(signal, 'factor_name', ''),
            signal_date=day.isoformat(), observation_only=True, buy_enabled=False,
        ))
    candidates.sort(key=lambda row: (-row['score'], row['code']))
    watchlist = candidates[:limit]
    note = _OBSERVATION_NOTE
    if not candidates:
        note += '观察名单为空：universe内无当日buy=True、sell=False且score为有限数值的源信号。'
    elif not limit:
        note += '观察名单为空：top_n=0。'
    return dict(orders=filtered_orders, watchlist=watchlist,
                market_status=market_status, buy_enabled=False, watchlist_note=note)
