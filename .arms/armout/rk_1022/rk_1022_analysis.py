#!/usr/bin/env python3
"""rank_decay 面 (2026-10-22) 判读: 指标 + 选股Jaccard + 权重Δ + 配对胜率.

用法: python rk_1022_analysis.py [BASE_TAG] TAG...
读 rolling_validation_results/*.csv.$tag (回测链落盘快照).
"""
import sys
import math
import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'


def eq(tag):
    d = pd.read_csv(f'{R}/equity_curve.csv.{tag}')
    d['date'] = pd.to_datetime(d['date'])
    return d.set_index('date')['nav'].astype(float).sort_index()


def tr(tag):
    return pd.read_csv(f'{R}/trade_realized.csv.{tag}', dtype={'code': str})


def sel(tag):
    return pd.read_csv(f'{R}/portfolio_selections.csv.{tag}', dtype={'code': str},
                       parse_dates=['date'])


def period_win(nav, k):
    v = nav.values
    n = len(v) // k
    if n < 2:
        return np.nan, []
    ends = v[k - 1::k][:n]
    starts = np.r_[v[0], ends[:-1]]
    rets = ends / starts - 1
    return float((rets > 0).mean()), rets


def monthly_ret(nav):
    m = nav.resample('ME').last()
    r = m.pct_change()
    r.iloc[0] = m.iloc[0] / nav.iloc[0] - 1
    return r


def metrics(nav):
    r = nav.pct_change().dropna()
    peak = nav.cummax()
    mdd = float((nav / peak - 1).min())
    yrs = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    w5, _ = period_win(nav, 5)
    w20, _ = period_win(nav, 20)
    mr = monthly_ret(nav)
    yr = nav.resample('YE').last()
    yret = yr.pct_change()
    yret.iloc[0] = yr.iloc[0] / nav.iloc[0] - 1
    return dict(nav=nav.iloc[-1], cagr=cagr, sharpe=sharpe, mdd=mdd,
                d_win=(r > 0).mean(), w5=w5, w20=w20,
                m_win=(mr > 0).mean(), y_win=(yret > 0).mean(),
                yearly={int(i.year): v for i, v in yret.items()})


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    p0 = 0.5
    probs = [math.comb(n, i) * p0 ** n for i in range(n + 1)]
    obs = probs[k]
    return float(sum(p for p in probs if p <= obs + 1e-12))


def mcnemar(a, b):
    """a,b: paired boolean arrays. return (n01, n10, p)."""
    a = np.asarray(a); b = np.asarray(b)
    n01 = int(((~a) & b).sum())   # base lose, arm win
    n10 = int((a & (~b)).sum())   # base win, arm lose
    return n01, n10, binom_two_sided(n01, n01 + n10)


def jaccard(x, y):
    sx = set(zip(x.date.dt.strftime('%Y-%m-%d'), x.code))
    sy = set(zip(y.date.dt.strftime('%Y-%m-%d'), y.code))
    inter = len(sx & sy)
    return inter / len(sx | sy) if (sx | sy) else np.nan, len(sx & sy), len(sx), len(sy)


def main():
    base_tag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith('base') else 'base_rk1022'
    tags = [base_tag] + [t for t in sys.argv[1:] if t != base_tag]
    base_nav = eq(base_tag)
    base_tr = tr(base_tag)
    base_sel = sel(base_tag)
    base_m = metrics(base_nav)
    base_mr = monthly_ret(base_nav)
    _, base_20 = period_win(base_nav, 20)

    print(f'BASE {base_tag}: NAV={base_m["nav"]:.2f} CAGR={base_m["cagr"]:.2%} '
          f'Sharpe={base_m["sharpe"]:.4f} MDD={base_m["mdd"]:.2%}')
    print(f'{"tag":10s} {"NAV":>12s} {"ΔNAV":>10s} {"CAGR":>7s} {"Sharpe":>7s} '
          f'{"MDD":>7s} {"dW":>6s} {"w5":>6s} {"w20":>6s} {"mW":>6s} {"yW":>6s} '
          f'{"ntr":>4s} {"tW":>6s} {"tMean":>7s} {"PF":>5s} {"Jacc":>5s} {"wΔ":>7s}')
    rows = {}
    for tag in tags:
        nav = eq(tag); t = tr(tag); s = sel(tag); m = metrics(nav)
        j, inter, nx, ny = jaccard(base_sel, s)
        common = base_sel.merge(s, on=['date', 'code'], suffixes=('_b', '_a'))
        wd = (common.weight_a - common.weight_b).abs().mean() if len(common) else np.nan
        wins = t.ret[t.ret > 0].sum(); loss = -t.ret[t.ret < 0].sum()
        print(f'{tag:10s} {m["nav"]:12.2f} {m["nav"]-base_m["nav"]:10.2f} '
              f'{m["cagr"]:7.2%} {m["sharpe"]:7.4f} {m["mdd"]:7.2%} '
              f'{m["d_win"]:6.2%} {m["w5"]:6.2%} {m["w20"]:6.2%} {m["m_win"]:6.2%} '
              f'{m["y_win"]:6.2%} {len(t):4d} {(t.ret > 0).mean():6.2%} '
              f'{t.ret.mean():7.2%} {wins/loss if loss else np.inf:5.2f} {j:5.2f} {wd:7.4f}')
        rows[tag] = (nav, m, s)

    print('\n年度收益 (arm vs base Δpp):')
    for tag, (nav, m, s) in rows.items():
        if tag == base_tag:
            continue
        ds = []
        for y in base_m['yearly']:
            ds.append(f'{y}:{m["yearly"].get(y, np.nan)-base_m["yearly"][y]:+.2%}')
        print(f'  {tag:10s} ' + ' '.join(ds))

    print('\n配对胜率 (McNemar, 只算不一致期):')
    for tag, (nav, m, s) in rows.items():
        if tag == base_tag:
            continue
        mr = monthly_ret(nav)
        # align months
        idx = base_mr.index.intersection(mr.index)
        n01, n10, p = mcnemar(base_mr.loc[idx] > 0, mr.loc[idx] > 0)
        _, r20 = period_win(nav, 20)
        n = min(len(base_20), len(r20))
        n01b, n10b, pb = mcnemar(base_20[:n] > 0, r20[:n] > 0)
        print(f'  {tag:10s} month: base_lose/arm_win={n01} base_win/arm_lose={n10} p={p:.3f} | '
              f'20d: {n01b}/{n10b} p={pb:.3f}')


if __name__ == '__main__':
    main()
