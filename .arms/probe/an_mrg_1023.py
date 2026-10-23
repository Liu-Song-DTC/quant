#!/usr/bin/env python3
"""mrg 臂判读 (2026-10-23): NAV/逐年/胜率/触发分解 vs anchor_1022.

输入: rolling_validation_results/*.{anchor_1022,mrgT010,mrgT020,mrgT030}
用法: .venv/bin/python .arms/probe/an_mrg_1023.py
"""
import sys

import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'
TAGS = ['anchor_1022', 'mrgT010', 'mrgT020', 'mrgT030']


def load(tag):
    eq = pd.read_csv(f'{R}/equity_curve.csv.{tag}', parse_dates=['date'])
    tr = pd.read_csv(f'{R}/trade_realized.csv.{tag}', dtype={'code': str})
    sel = pd.read_csv(f'{R}/portfolio_selections.csv.{tag}', dtype={'code': str})
    return eq, tr, sel


def yearly(eq):
    e = eq.set_index('date')['nav']
    out = {}
    for y, g in e.groupby(e.index.year):
        out[y] = g.iloc[-1] / g.iloc[0] - 1
    return out


def main():
    data = {}
    for t in TAGS:
        try:
            data[t] = load(t)
        except FileNotFoundError:
            print(f'⚠ 缺 {t} 产物, 跳过')
    tags = list(data)
    if 'anchor_1022' not in tags:
        print('⛔ 无锚')
        return
    a_eq, a_tr, a_sel = data['anchor_1022']
    a_nav = a_eq['nav'].iloc[-1]
    a_ret = a_eq.set_index('date')['nav'].pct_change()
    a_dd = (a_eq['nav'] / a_eq['nav'].cummax() - 1).min()
    print(f'anchor_1022: NAV {a_nav:,.0f} | 交易 {len(a_tr)} 笔 tW '
          f'{(a_tr.ret > 0).mean()*100:.2f}% | MDD {a_dd*100:.2f}%')
    yy = yearly(a_eq)
    for t in tags[1:]:
        eq, tr, sel = data[t]
        nav = eq['nav'].iloc[-1]
        ret = eq.set_index('date')['nav'].pct_change()
        dd = (eq['nav'] / eq['nav'].cummax() - 1).min()
        sr = np.sqrt(252) * ret.mean() / ret.std()
        tw = (tr.ret > 0).mean() * 100
        d = nav - a_nav
        # 逐年差
        y2 = yearly(eq)
        ydiff = ' '.join(f'{y}:{(y2.get(y,0)-yy.get(y,0))*100:+.1f}' for y in sorted(yy))
        # Jaccard: 选股集合是否完全一致 (纯加权签名)
        ka = set(zip(a_sel.date, a_sel.code.str.zfill(6)))
        kb = set(zip(sel.date, sel.code.str.zfill(6)))
        jac = len(ka & kb) / max(len(ka | kb), 1)
        print(f'\n{t}: NAV {nav:,.0f} (Δ{d:+,.0f}) | Sharpe {sr:.3f} | MDD {dd*100:.2f}% '
              f'(锚 {a_dd*100:.2f}%)')
        print(f'  交易 {len(tr)} 笔 tW {tw:.2f}% (锚 {(a_tr.ret>0).mean()*100:.2f}%) | '
              f'Jaccard {jac:.4f} | 逐年差(pp): {ydiff}')
    # 触发分解: 臂内新增权重的入场笔 vs 锚同笔 (权重对比)
    if len(tags) > 1:
        t = tags[1]  # 第一个臂
        eq, tr, sel = data[t]
        m = tr.merge(a_tr[['code', 'entry_date', 'weight', 'ret']],
                     on=['code', 'entry_date'], how='inner', suffixes=('_arm', '_anc'))
        if len(m) and 'weight_arm' in m:
            dw = m['weight_arm'] - m['weight_anc']
            hi = m[dw > 1e-9]
            print(f'\n[{t}] 同笔入场 {len(m)}, 臂加权笔 {len(hi)} (Δw 中位 '
                  f'{hi.dw.median()*100 if len(hi) else 0:.2f}pp): '
                  f'这些笔 tW {(hi.ret_arm > 0).mean()*100 if len(hi) else float("nan"):.1f}% '
                  f'均ret {hi.ret_arm.mean()*100 if len(hi) else float("nan"):+.2f}% | '
                  f'锚同笔均ret {hi.ret_anc.mean()*100 if len(hi) else float("nan"):+.2f}%')


if __name__ == '__main__':
    main()
