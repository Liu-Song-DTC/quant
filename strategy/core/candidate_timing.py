"""Causal, default-off research priority within already eligible buy-point groups.

This neither extends an expired signal nor infers freshness across a missing
session. Scores still determine sizing; the explicit ordering intervention is
recorded separately. Existing holdings never exchange their list positions.
"""
from collections import defaultdict

import numpy as np
import pandas as pd


def source_state(signal):
    values = [getattr(signal, name, None) for name in ('buy', 'sell')]
    known = all(isinstance(value, (bool, np.bool_)) for value in values)
    return dict(known=known, buy=bool(values[0]) if known else None,
                sell=bool(values[1]) if known else None,
                active=bool(values[0] and not values[1]) if known else None)


def prioritize_new_source_buys(qualified, date, calendar, signal_store, held_codes, point_types=None):
    if point_types is not None and frozenset(point_types) != frozenset({2}):
        raise ValueError('Research point scope must be all types or native B2 only')
    if calendar is None:
        raise ValueError('Signal timing requires the actual market calendar')
    cal = pd.DatetimeIndex(calendar)
    if not cal.is_monotonic_increasing or cal.has_duplicates:
        raise ValueError('Signal timing requires unique ordered sessions')
    day = pd.Timestamp(date).normalize()
    loc = cal.get_indexer([day])[0]
    if loc < 0:
        raise ValueError('Decision date is not a market session')
    previous = cal[loc - 1].date() if loc > 0 else None
    groups = defaultdict(list)
    evidence = []
    for i, candidate in enumerate(qualified):
        code = candidate['code']
        held = code in held_codes or bool(candidate.get('is_held'))
        locked = bool(candidate.get('_locked'))
        current = source_state(candidate.get('sig'))
        prior = source_state(signal_store.get(code, previous)) if previous is not None and not held and not locked else source_state(None)
        fresh = bool(current['known'] and current['active'] and prior['known'] and not prior['active'])
        evidence.append(dict(code=code, buy_point=int(candidate.get('chan_buy_point', 0)),
            held=held, locked=locked, current=current, previous=prior, fresh=fresh,
            original_index=i, effective_score=float(candidate['effective_score'])))
        if not held and not locked and (point_types is None or evidence[-1]['buy_point'] in point_types):
            groups[evidence[-1]['buy_point']].append(i)
    result = list(qualified)
    new_indices = {}
    for indices in groups.values():
        # Stable priority retains the complete original score order within each class.
        order = sorted(indices, key=lambda i: not evidence[i]['fresh'])
        for destination, origin in zip(indices, order):
            result[destination] = qualified[origin]
            new_indices[origin] = destination
    for row in evidence:
        row['new_index'] = new_indices.get(row['original_index'], row['original_index'])
    return result, dict(previous_session=previous, rows=evidence,
        priority_points=None if point_types is None else sorted(point_types),
        changed=sum(row['original_index'] != row['new_index'] for row in evidence),
        scope='After original day gates and strategy weighting; same-type unheld ordering only. Original numeric scores, holding slots and risk rules retained.')
