"""Point-in-time B2 evaluation using the maintained detector formulas.

Primary B1 labels must already be causal; cached EMA/MACD values are
prefix-causal. Confirmed fractal clocks
permit skipping days on which the existing detector cannot emit a new B2.
This retains each day's existing evaluation semantics, not a new persistent
one-emission-per-zone policy.
"""
import numpy as np


def _finite_macd_windows(values):
    """Cache identical trailing reductions; retain scalar handling otherwise.

    Columns are recent six bars, the earlier five bars (ending six bars
    before the decision), and recent four bars. No cumulative sums or
    future values participate in a row's reductions.
    """
    if (not isinstance(values, np.ndarray) or values.ndim != 1
            or values.dtype != np.dtype('float64') or not np.isfinite(values).all()):
        return None
    n = len(values)
    windows = {}
    for width in (4, 5, 6):
        result = np.full(n, np.nan)
        for i in range(min(width - 1, n)):
            result[i] = np.nanmean(values[:i + 1])
        if n >= width:
            view = np.lib.stride_tricks.sliding_window_view(values, width)
            result[width - 1:] = np.nanmean(view, axis=1)
        windows[width] = result
    earlier = np.full(n, np.nan)
    if n > 6:
        earlier[6:] = windows[5][:-6]
    return np.column_stack((windows[6], earlier, windows[4]))


def empty_observation():
    return dict(second_buy_point=False,second_buy_confidence=0.,second_buy_b1_ref=-1,
                second_buy_origin=0)


def observe_second_buy_at(close,high,low,volume,ema20,ema60,macd_hist,primary_buy_point,index,
                          _macd_windows=None):
    """Observe only index's output using inputs bounded by that daily close."""
    if not isinstance(index,(int,np.integer)) or isinstance(index,bool) or index<0 or index>=len(close):
        raise ValueError('Expected an available integer decision index')
    if index<59 or volume is None:
        return empty_observation()
    from . import chan_theory as ct
    stop=index+1
    close,high,low,volume,ema20,ema60,macd_hist=[x[:stop] for x in
        (close,high,low,volume,ema20,ema60,macd_hist)]
    geometry=ct.detect_strokes(high,low,close)
    clock=ct.detect_fractals(high,low)
    quality=ct.analyze_bottom_fractal(close,high,low,volume,ema20,ema60,macd_hist,
        geometry['bottom_fractals'],geometry['stroke_direction'])
    result=ct.detect_second_buy_point(close,high,low,geometry['bottom_fractals'],
        quality['bottom_fractal_quality'],macd_hist,primary_buy_point[:stop],
        geometry['stroke_direction'],geometry['strokes'],clock['fractal_vis'],
        _macd_windows=_macd_windows[:stop] if _macd_windows is not None else None)
    return {key:np.asarray(value)[-1].item() for key,value in result.items()}


def compute_second_buy_asof(close,high,low,volume,ema20,ema60,macd_hist,
                            primary_buy_point,fractal_vis,bottom_fractals):
    """Retain the current-close output on each confirmed bottom-fractal clock."""
    n=len(close)
    result=dict(second_buy_point=np.zeros(n,dtype=bool),second_buy_confidence=np.zeros(n),
                second_buy_b1_ref=np.full(n,-1,dtype=int),second_buy_origin=np.zeros(n,dtype=np.int16))
    if n<60 or volume is None:return result
    macd_windows = _finite_macd_windows(macd_hist)
    clocks=np.unique(fractal_vis[bottom_fractals])
    for index in clocks[(clocks>=59)&(clocks<n)]:
        observation=observe_second_buy_at(close,high,low,volume,ema20,ema60,macd_hist,
                                         primary_buy_point,int(index),_macd_windows=macd_windows)
        for key,value in observation.items():result[key][index]=value
    return result
