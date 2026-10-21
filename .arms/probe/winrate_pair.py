#!/usr/bin/env python3
"""Paired win-rate significance for two arm snapshots.

User directive 2026-10-21: NAV up but win rate down => fragile, needs stronger evidence.
This quantifies whether a win-rate gap is distinguishable from month/period granularity.

For each horizon, signs are compared on the SAME calendar/trading-period grid (paired):
  - monthly: calendar month return sign
  - 20d/5d: non-overlapping blocks anchored at the common first trading day
  - daily: daily return sign
Discordant cells (A wins >0 & B loses<=0, and vice versa) are counted; exact binomial
p-value tests H0: the two arms are equally likely to be the positive one, conditional on
discordant cells (McNemar exact). n_disc is the effective evidence, NOT total periods.

Usage: python winrate_pair.py A B [HORIZONS...]   (A=baseline/incumbent, B=challenger)
"""
import sys
import numpy as np
import pandas as pd
from math import comb

R = '/mnt/d/quant/strategy/rolling_validation_results'


def load(tag):
    p = f'{R}/equity_curve.csv.{tag}' if tag else f'{R}/equity_curve.csv'
    d = pd.read_csv(p)
    d['date'] = pd.to_datetime(d['date'])
    return d.set_index('date')['nav'].sort_index()


def sign_series(nav, kind):
    if kind == 'daily':
        r = nav.pct_change().dropna()
        return np.sign(r)
    if kind == 'monthly':
        m = nav.resample('ME').last()
        r = m.pct_change()
        r.iloc[0] = m.iloc[0] / nav.iloc[0] - 1
        return np.sign(r)
    k = int(kind.rstrip('d'))
    v = nav.values
    n = len(v) // k
    ends = v[k - 1::k][:n]
    prev = np.r_[v[0], ends[:-1]]
    r = ends / prev - 1
    idx = nav.index[k - 1::k][:n]
    return pd.Series(np.sign(r), index=idx)


def mcnemar_exact(b, c):
    """two-sided exact binomial for discordant counts b vs c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def compare(a, b, kind):
    sa = sign_series(a, kind)
    sb = sign_series(b, kind)
    j = sa.index.intersection(sb.index)
    sa, sb = sa.loc[j], sb.loc[j]
    a_pos = sa > 0
    b_pos = sb > 0
    a_only = int((a_pos & ~b_pos).sum())   # A win, B lose
    b_only = int((~a_pos & b_pos).sum())   # B win, A lose
    both_pos = int((a_pos & b_pos).sum())
    both_neg = int((~a_pos & ~b_pos).sum())
    p = mcnemar_exact(a_only, b_only)
    return dict(h=kind, n=len(j), a_win=float(a_pos.mean()), b_win=float(b_pos.mean()),
                both_pos=both_pos, both_neg=both_neg, a_only=a_only, b_only=b_only,
                diff=float(b_pos.mean() - a_pos.mean()), p=p)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    ta, tb = args[0], args[1]
    kinds = args[2:] or ['daily', '5d', '20d', 'monthly']
    a, b = load(ta), load(tb)
    print(f'A(baseline)={ta}  B(challenger)={tb}')
    rows = [compare(a, b, k) for k in kinds]
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n判读: p<0.05 且 diff<0 ⇒ B 的胜率显著更差 (NAV 升也按脆弱处理);')
    print('      p>=0.05 ⇒ 胜率差在期数粒度噪声内, 不能作为否决/采纳证据.')


if __name__ == '__main__':
    main()
