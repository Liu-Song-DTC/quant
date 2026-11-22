"""Causal entry observations and isolated research controls.

No forward returns, backfilled fractal dates, or position-exit decisions belong here.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def entry_confirmation_observations(bars):
    """Use completed bars only; a three-bar bottom is stamped on its right bar."""
    required = ['open', 'high', 'low', 'close', 'volume']
    x = bars[required].astype(float)
    valid = (np.isfinite(x).all(axis=1) & x.gt(0).all(axis=1)
             & x.high.ge(x[['open', 'close', 'low']].max(axis=1))
             & x.low.le(x[['open', 'close']].min(axis=1)))
    close, high, low = x.close, x.high, x.low
    pair_valid = valid & valid.shift(1, fill_value=False)
    recovery = (pair_valid & close.gt(close.shift(1)) & low.ge(low.shift(1))
                & close.ge((high + low) / 2))
    confirmed_bottom = (pair_valid & valid.shift(2, fill_value=False)
                        & low.shift(1).lt(low.shift(2)) & low.shift(1).lt(low)
                        & high.shift(1).lt(high.shift(2)) & high.shift(1).lt(high))
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema12 - ema26
    hist = dif - dif.ewm(span=9, adjust=False, min_periods=9).mean()
    improving = pair_valid & hist.gt(hist.shift(1))
    return pd.DataFrame({
        'quote_valid': valid, 'recovery': recovery, 'confirmed_bottom': confirmed_bottom,
        'macd_improving': improving, 'fractal_macd': confirmed_bottom & improving,
        'close': close, 'previous_close': close.shift(1), 'low': low,
        'previous_low': low.shift(1), 'midpoint': (high + low) / 2,
        'macd_hist': hist, 'previous_macd_hist': hist.shift(1),
    }, index=bars.index)


class EntryConfirmationTable:
    """Load a sealed observation table. Missing research input is an error, not a pass."""
    def __init__(self, path, expected_sha256, mode):
        if mode not in ('recovery', 'fractal_macd'):
            raise ValueError('Unknown B4 confirmation mode')
        path = Path(path)
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected_sha256:
            raise ValueError('B4 confirmation table hash mismatch')
        self.mode = mode
        data = pd.read_parquet(path)
        if data.duplicated(['date', 'code']).any():
            raise ValueError('Duplicate B4 confirmation keys')
        data['date'] = pd.to_datetime(data.date).dt.date
        data['code'] = data.code.astype(str).str.zfill(6)
        self.rows = data.set_index(['date', 'code']).to_dict('index')

    def observation(self, date, code):
        key = (pd.Timestamp(date).date(), str(code).zfill(6))
        if key not in self.rows:
            raise ValueError(f'Missing B4 confirmation observation: {key}')
        row = self.rows[key]
        if not isinstance(row[self.mode], (bool, np.bool_)):
            raise ValueError(f'Invalid confirmation flag: {key}')
        return bool(row[self.mode]), row


def rerank_fresh_b4(qualified, ranks, held_codes):
    """Reassign only fresh B4's existing rank values by ML, preserving other ranks."""
    result = np.array(ranks, copy=True)
    indices = [i for i,c in enumerate(qualified)
               if c.get('chan_buy_point') == 4 and c['code'] not in held_codes]
    if len(indices) < 2:
        return result
    values = {i: float(getattr(qualified[i]['sig'], 'ml_score', np.nan)) for i in indices}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('B4 reranking requires finite ML scores')
    # Equal ML preserves legacy order; code is only a deterministic last tie break.
    order = sorted(indices, key=lambda i: (values[i], ranks[i], qualified[i]['code']))
    result[order] = np.sort(result[indices])
    return result


def added_capital_confirmation(signal, weights, minimum, require_s4=False):
    """Reuse the existing four structural votes for incremental capital only.

    This covers source buy/sell, structural votes and the new-position gate
    floor, not the entire sector/rank/portfolio admission policy.
    """
    def number(name, default=0.):
        try:
            value=float(getattr(signal,name,default))
            return value if np.isfinite(value) else default
        except (ValueError,TypeError):return default
    bp=int(number('chan_buy_point'));bi=int(number('bi_buy_point'));level=int(number('signal_level'))
    divergence=getattr(signal,'chan_divergence_type','')
    votes=[bp>0 and bp!=5,bi>0,level>=1,divergence in ('bottom','bottom_fx','bottom_fx_3x','B2')]
    if len(weights)!=4 or not np.isfinite(np.asarray(weights,dtype=float)).all():
        raise ValueError('Expected four finite buy-point weights')
    confirmation=sum(float(w)*int(v) for w,v in zip(weights,votes))
    gate=number('_gate_quality',.5)
    failures=[]
    if not bool(getattr(signal,'buy',False)):failures.append('source_buy_absent')
    if bool(getattr(signal,'sell',False)):failures.append('source_sell_present')
    if minimum>0 and confirmation<minimum:failures.append('buy_confirmation_below_new_entry_minimum')
    if require_s4 and not votes[3]:failures.append('required_S4_absent')
    if gate<.65:failures.append('below_new_entry_gate_floor')
    return dict(allowed=not failures,reasons=failures,buy_point=bp,bi_buy_point=bi,signal_level=level,
                confirmation=confirmation,minimum=float(minimum),votes=[bool(v) for v in votes],gate_quality=gate)


def cap_blocked_addition(target_shares,current_shares,blocked):
    """An add restriction must neither force extra selling nor allow new buying."""
    if target_shares<0 or current_shares<0:raise ValueError('Negative share quantity')
    return min(target_shares,current_shares) if blocked else target_shares
