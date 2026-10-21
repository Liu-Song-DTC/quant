#!/usr/bin/env python3
"""Ex-top counterfactual for an arm snapshot (lottery check).

Motivation (2026-10-21, be75): an arm can pass NAV + win-rate criteria while its
entire gain rides on one trade. We reconstruct each top trade's daily contribution
from selection weight × stock daily returns, subtract it (cash substitute) from the
arm's daily portfolio return, propagate to terminal NAV, and compare with baseline.

Method:
  - selection weight: latest portfolio_selections row for (code, date<=entry, <=10d)
  - stock path: data/stock_data/backtrader_data/{code}_qfq.csv close
  - contribution_t = pos_{t-1} * r_t / equity_{t-1}; pos evolves with price
  - cf ratio at exit = prod(1 + (1+r_port)/(1+c) - 1); final_cf = final_arm * ratio
This is an approximation (assumes rest-of-book returns unchanged, cash earns 0),
sufficient for the binary lottery test: final_cf <= baseline => lottery.

Usage: python ex_top.py ARM BASE [N]
"""
import sys
import os
import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'


def load_eq(tag):
    d = pd.read_csv(f'{R}/equity_curve.csv.{tag}')
    d['date'] = pd.to_datetime(d['date'])
    return d.set_index('date')['nav'].sort_index()


def load_sel(tag):
    d = pd.read_csv(f'{R}/portfolio_selections.csv.{tag}', dtype={'code': str})
    d['date'] = pd.to_datetime(d['date'])
    return d


def load_tr(tag):
    d = pd.read_csv(f'{R}/trade_realized.csv.{tag}', dtype={'code': str})
    d['entry_date'] = pd.to_datetime(d['entry_date'])
    d['exit_date'] = pd.to_datetime(d['exit_date'])
    return d


def weight_for(sel_code, entry):
    x = sel_code[sel_code.date <= entry]
    if not len(x):
        return np.nan
    x = x[x.date >= entry - pd.Timedelta(days=10)]
    return float(x.weight.iloc[-1]) if len(x) else np.nan


def counterfactual(arm, base_tag, trade, w):
    eq = load_eq(arm)
    code, entry, exit_ = trade.code, trade.entry_date, trade.exit_date
    p = f'{BT}/{code}_qfq.csv'
    if not os.path.exists(p):
        return None
    px = pd.read_csv(p, parse_dates=['datetime']).set_index('datetime')['close']
    px = px.loc[entry:exit_]
    if len(px) < 3 or entry not in eq.index:
        return None
    e0 = float(eq.loc[entry])
    pos = w * e0
    prev = float(px.iloc[0])
    cs, idx = [], []
    for dt, pr in px.items():
        if dt == px.index[0]:
            continue
        if dt not in eq.index:
            prev = float(pr)
            continue
        r = float(pr) / prev - 1
        cs.append(pos * r / float(eq.loc[dt]))
        pos *= 1 + r
        prev = float(pr)
        idx.append(dt)
    if not cs:
        return None
    c = pd.Series(cs, index=idx)
    rport = eq.pct_change().reindex(c.index)
    rcf = (1 + rport) / (1 + c) - 1
    eq_exit = float(eq.loc[c.index[-1]])
    cf_exit = e0 * float((1 + rcf).prod())
    final_cf = float(eq.iloc[-1]) * cf_exit / eq_exit
    return final_cf, float(c.sum())


def main():
    arm, base = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    eq_arm, eq_base = load_eq(arm), load_eq(base)
    sel, tr = load_sel(arm), load_tr(arm)
    base_tr = load_tr(base)
    kb = set(zip(base_tr.entry_date, base_tr.code))
    final_arm, final_base = float(eq_arm.iloc[-1]), float(eq_base.iloc[-1])
    print(f'arm={arm} final={final_arm:,.0f} | base={base} final={final_base:,.0f}')
    print(f'{"code":8s} {"entry":10s} {"ret":>8s} {"w":>6s} {"unique":>6s} {"cf_final":>12s} {"cf/base":>8s} {"contrib":>8s}')
    for _, t in tr.nlargest(n, 'ret').iterrows():
        sc = sel[sel.code == t.code]
        w = weight_for(sc, t.entry_date)
        if not np.isfinite(w):
            print(f'{t.code:8s} {t.entry_date.date()} NO_SEL')
            continue
        out = counterfactual(arm, base, t, w)
        if out is None:
            print(f'{t.code:8s} {t.entry_date.date()} NO_PX')
            continue
        cf, contrib = out
        uniq = 'Y' if (t.entry_date, t.code) not in kb else '.'
        print(f'{t.code:8s} {str(t.entry_date.date()):10s} {t.ret:8.1%} {w:6.3f} {uniq:>6s} {cf:12,.0f} {cf/final_base:8.3f} {contrib:8.1%}')
    print('\n判读: cf/base <= 1.000 且 unique=Y ⇒ 该单笔承载全部增益, 彩票臂;')
    print('      多个 top 单笔各自 cf/base>1 ⇒ 非单笔承载.')


if __name__ == '__main__':
    main()
