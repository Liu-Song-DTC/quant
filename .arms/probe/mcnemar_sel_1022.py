#!/usr/bin/env python3
"""选股层配对检验 (2026-10-22): 同一 (调仓日, 代码) 上的前瞻/已实现胜率 McNemar.

用法: python mcnemar_sel_1022.py base_tr1022 new2 ah1 ...
口径:
  - 前瞻: 次日开盘入场 → +H 日收盘 (只对两臂都选中的 (date,code) 配对)
  - 已实现: 两臂都成交的 (entry_date,code) 配对, ret>0
  - 精确 McNemar = 二项检验 (双侧), 只算不一致样本
  - ex-top: 剔除各臂 r20 最大的单笔后重算 (W5)
"""
import sys
from math import comb
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
import winrate_diag_1022 as W

R = '/mnt/d/quant/strategy/rolling_validation_results'


def load(tag):
    sel = pd.read_csv(f'{R}/portfolio_selections.csv.{tag}', dtype={'code': str},
                     parse_dates=['date'])
    sel['code'] = sel['code'].str.zfill(6)
    eq = pd.read_csv(f'{R}/equity_curve.csv.{tag}', parse_dates=['date'])
    cal = list(pd.to_datetime(eq['date']))
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    sel['exec_date'] = sel['date'].map(nxt)
    sel = W.add_forward(sel)
    trd = pd.read_csv(f'{R}/trade_realized.csv.{tag}', dtype={'code': str},
                      parse_dates=['entry_date'])
    trd['code'] = trd['code'].str.zfill(6)
    return sel, trd


def binom_two_sided(k, n):
    """精确 McNemar 双侧 p (n 个不一致样本中 k 个方向A)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar_pair(a, b):
    """a, b: 对齐的 0/1 序列 (1=win). 返回 (arm_only, base_only, p)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    arm_only = int((a & ~b).sum()); base_only = int((~a & b).sum())
    n = arm_only + base_only
    return arm_only, base_only, binom_two_sided(arm_only, n)


def wrate(x):
    x = np.asarray(x)
    return float(x.mean()) if len(x) else np.nan


def main():
    base = sys.argv[1]; tags = sys.argv[2:]
    bsel, btrd = load(base)
    print(f'# base={base}  nSel={len(bsel)} nTrd={len(btrd)}')
    print(f'# base tW={wrate(btrd.ret > 0)*100:.1f}% w5={wrate(bsel.w5 > 0)*100:.1f}% '
          f'w20={wrate(bsel.w20 > 0)*100:.1f}%')
    for tag in tags:
        sel, trd = load(tag)
        print(f'\n=== {tag}  nSel={len(sel)} nTrd={len(trd)} ===')
        print(f'  自身: tW={wrate(trd.ret > 0)*100:.1f}% w5={wrate(sel.w5 > 0)*100:.1f}% '
              f'w20={wrate(sel.w20 > 0)*100:.1f}% r20={sel.r20.mean()*100:+.2f}%')
        # 前瞻配对 (同一 date,code)
        m = bsel[['date', 'code', 'w5', 'w20', 'r20']].merge(
            sel[['date', 'code', 'w5', 'w20', 'r20']], on=['date', 'code'],
            suffixes=('_b', '_a'))
        for h in ('w5', 'w20'):
            sub = m[[f'{h}_b', f'{h}_a']].dropna()
            ao, bo, p = mcnemar_pair(sub[f'{h}_a'] > 0, sub[f'{h}_b'] > 0)
            print(f'  前瞻 {h}: 配对n={len(sub)} base={wrate(sub[f"{h}_b"]>0)*100:.1f}% '
                  f'arm={wrate(sub[f"{h}_a"]>0)*100:.1f}% '
                  f'臂独赢={ao} 基独赢={bo} McNemar p={p:.4f}')
        # 已实现配对 (entry_date,code)
        mt = btrd[['entry_date', 'code', 'ret']].merge(
            trd[['entry_date', 'code', 'ret']], on=['entry_date', 'code'],
            suffixes=('_b', '_a'))
        ao, bo, p = mcnemar_pair(mt.ret_a > 0, mt.ret_b > 0)
        print(f'  已实现: 配对n={len(mt)} base={wrate(mt.ret_b>0)*100:.1f}% '
              f'arm={wrate(mt.ret_a>0)*100:.1f}% '
              f'臂独赢={ao} 基独赢={bo} McNemar p={p:.4f}')
        # ex-top (各臂剔除自身 r20 最大单笔; 与基线同口径比较)
        for name, t in (('base', None), ('arm', sel)):
            pass
        top_b = bsel.r20.idxmax(); top_a = sel.r20.idxmax()
        bs = bsel.drop(index=top_b); as_ = sel.drop(index=top_a)
        tb = btrd.drop(index=btrd.ret.idxmax()); ta = trd.drop(index=trd.ret.idxmax())
        print(f'  ex-top: tW base={wrate(tb.ret>0)*100:.1f}% arm={wrate(ta.ret>0)*100:.1f}% '
              f'(Δ{(wrate(ta.ret>0)-wrate(tb.ret>0))*100:+.1f}pp) | '
              f'w20 base={wrate(bs.w20>0)*100:.1f}% arm={wrate(as_.w20>0)*100:.1f}% '
              f'(Δ{(wrate(as_.w20>0)-wrate(bs.w20>0))*100:+.1f}pp)')
        # 分年 (自身口径)
        sel2 = sel.copy(); sel2['year'] = sel2['date'].dt.year
        yrs = []
        for y, g in sel2.groupby('year'):
            yrs.append(f'{y}:{wrate(g.w20>0)*100:.0f}/{g.r20.mean()*100:+.1f}')
        print('  分年 w20/r20: ' + ' '.join(yrs))


if __name__ == '__main__':
    main()
