"""Candidate numeric kernel for exact prefix-quality evaluation.

Not connected to the trading path. Geometry mirrors detect_fractals and
detect_strokes for finite float64 inputs; the quality formula is reused from
chan_theory rather than reimplemented. Equivalence must precede activation.
"""
import numpy as np
from numba import njit

from .chan_theory import process_inclusions, _perf_compiled_quality


@njit(cache=True, fastmath=False)
def geometry_arrays(high, low, min_stroke_bars=4):
    n = len(high)
    bottom = np.zeros(n, dtype=np.bool_)
    direction = np.zeros(n, dtype=np.int64)
    if n < 3:
        return bottom, direction
    ph, pl, original = process_inclusions(high, low)
    pn = len(ph)
    filtered = np.empty(pn, dtype=np.int64)
    kinds = np.empty(pn, dtype=np.int64)
    count = 0
    for i in range(2, pn):
        kind = 0
        if ph[i-1] > ph[i-2] and ph[i-1] > ph[i] and pl[i] < pl[i-1]:
            kind = 1
        if pl[i-1] < pl[i-2] and pl[i-1] < pl[i] and ph[i] > ph[i-1]:
            kind = -1
        if kind == 0:
            continue
        center = i-1
        if count and kinds[count-1] == kind:
            old = filtered[count-1]
            if (kind == 1 and ph[center] > ph[old]) or (kind == -1 and pl[center] < pl[old]):
                filtered[count-1] = center
        else:
            filtered[count] = center
            kinds[count] = kind
            count += 1
    positions = np.empty(count, dtype=np.int64)
    for i in range(count):
        positions[i] = original[filtered[i]]
        if kinds[i] == -1:
            bottom[positions[i]] = True
    starts = np.empty(count, dtype=np.int64)
    ends = np.empty(count, dtype=np.int64)
    signs = np.empty(count, dtype=np.int64)
    strokes = 0
    i = 0
    while i < count-1:
        first = positions[i]
        kind = kinds[i]
        price = high[first] if kind == 1 else low[first]
        j = i+1
        while j < count:
            other = positions[j]
            other_kind = kinds[j]
            other_price = high[other] if other_kind == 1 else low[other]
            if other_kind != kind:
                if other-first >= min_stroke_bars:
                    if (kind == 1 and price > other_price) or (kind == -1 and price < other_price):
                        starts[strokes] = first
                        ends[strokes] = other
                        signs[strokes] = -kind
                        strokes += 1
                        i = j
                        break
                j += 1
            else:
                if (kind == 1 and other_price > price) or (kind == -1 and other_price < price):
                    if strokes and ends[strokes-1] == first and signs[strokes-1] == -kind:
                        ends[strokes-1] = other
                    i = j
                    first = positions[i]
                    kind = kinds[i]
                    price = high[first] if kind == 1 else low[first]
                j += 1
        if j >= count:
            break
    for j in range(strokes):
        for k in range(starts[j], min(ends[j]+1, n)):
            direction[k] = signs[j]
    if strokes:
        for k in range(ends[strokes-1], n):
            direction[k] = -signs[strokes-1]
    return bottom, direction


@njit(cache=True, fastmath=False, error_model='numpy')
def quality_prefix_arrays(close, high, low, volume, ema20, ema60, macd_hist, vol_sma20):
    n = len(close)
    result = np.zeros((5, n))
    for stop in range(3, n+1):
        bottom, direction = geometry_arrays(high[:stop], low[:stop])
        quality = _perf_compiled_quality(close[:stop], high[:stop], low[:stop], volume[:stop],
            ema20[:stop], ema60[:stop], macd_hist[:stop], bottom, direction, vol_sma20[:stop])
        result[0, stop-1] = quality['bottom_fractal_quality'][-1]
        result[1, stop-1] = quality['bottom_fractal_strength'][-1]
        result[2, stop-1] = quality['bottom_fractal_vol_ratio'][-1]
        result[3, stop-1] = quality['bottom_fractal_vol_spike'][-1]
        result[4, stop-1] = quality['bottom_fractal_ema_dist'][-1]
    return result
