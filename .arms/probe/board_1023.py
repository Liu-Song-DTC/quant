#!/usr/bin/env python3
"""★板块(科创板/创业板/主板) × 选股胜率 — 只读微探针 (2026-10-23夜).

动机: 账簿688重仓 + §22 灾难腿688063 ⇒ 板块是否是未测的选股维度?
池级: w5/w20 by board + 分年 + 簇bootstrap; realized 166笔 tW/均ret by board.
注意: 纯权重倾斜不改变tW (§21定理); 只有"板块跳过"改变选股集合才可能动tW.
用法: .venv/bin/python .arms/probe/board_1023.py
"""
import sys

import numpy as np
import pandas as pd

AR = '/mnt/d/quant/.arms/armout'
sys.path.insert(0, '/mnt/d/quant/.arms/probe')

BOARD = [('68', '科创板'), ('30', '创业板'), ('60', '沪主板'), ('00', '深主板')]


def board_of(c):
    c = str(c).zfill(6)
    for p, nm in BOARD:
        if c.startswith(p):
            return nm
    return '其他'


def cluster_boot(d, ycol='w5'):
    d = d[['date', 'board', ycol]].dropna()
    days = sorted(d.date.unique())
    di = {v: i for i, v in enumerate(days)}
    idx = d.date.map(di).values
    y = (d[ycol].values > 0).astype(float)
    bos = {}
    for b in ['科创板', '创业板', '沪主板']:
        m = (d.board.values == b)
        if m.sum() < 50:
            continue
        n = np.bincount(idx[m], minlength=len(days))
        k = np.bincount(idx[m], weights=y[m], minlength=len(days))
        bos[b] = (n, k)
    if '科创板' not in bos or '沪主板' not in bos:
        return
    na, ka = bos['科创板']
    nb, kb = bos['沪主板']
    obs = ka.sum() / na.sum() - kb.sum() / nb.sum()
    rng = np.random.default_rng(20261023)
    B = rng.integers(0, len(days), size=(5000, len(days)))
    bd = B @ ka / (B @ na) - B @ kb / (B @ nb)
    bd = bd[np.isfinite(bd)]
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p = 2 * min((bd >= 0).mean(), (bd <= 0).mean())
    print(f'  簇bootstrap Δw5(科创−沪主板)={obs * 100:+.2f}pp CI[{lo * 100:+.2f},{hi * 100:+.2f}] p={p:.4f}')


def main():
    sys.argv = [sys.argv[0]]
    from shortfeat_1022 import load as pool_load
    p = pool_load()
    p['board'] = p['code'].map(board_of)

    print(f'池 {len(p)} 行 / {p["date"].nunique()} 日')
    print('  板块行数:', p.board.value_counts().to_dict())
    print('\n══ 池级 w5/w20 by 板块 ══')
    for b in ['科创板', '创业板', '沪主板', '深主板', '其他']:
        g = p[p.board == b]
        if len(g) < 100:
            continue
        print(f'  {b:4s} n={len(g):5d} w5={(g.w5 > 0).mean() * 100:5.1f}% w20={(g.w20 > 0).mean() * 100:5.1f}% '
              f'均score={g.score.mean():+.3f}')
    print('  [cap2候选 nr≤2]')
    nm = p[p.nr <= 2]
    for b in ['科创板', '创业板', '沪主板', '深主板']:
        g = nm[nm.board == b]
        if len(g) < 30:
            continue
        print(f'  {b:4s} n={len(g):4d} w5={(g.w5 > 0).mean() * 100:5.1f}% w20={(g.w20 > 0).mean() * 100:5.1f}%')
    cluster_boot(p)

    print('\n  分年 w5 by 板块:')
    p['year'] = p.date.dt.year
    tab = p.groupby(['year', 'board']).apply(lambda g: pd.Series(
        {'n': len(g), 'w5': (g.w5 > 0).mean() * 100}), include_groups=False).reset_index()
    for y, g in tab.groupby('year'):
        parts = [f'{r.board}{r.n:.0f}:{r.w5:.1f}' for r in g.itertuples() if r.n >= 40]
        print(f'    {y}: ' + ' | '.join(parts))

    print('\n══ realized 166笔 by 板块 ══')
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['board'] = mm['code'].map(board_of)
    for b in ['科创板', '创业板', '沪主板', '深主板', '其他']:
        g = mm[mm.board == b]
        if not len(g):
            continue
        print(f'  {b:4s} n={len(g):3d} tW={(g.ret > 0).mean() * 100:5.1f}% 均ret={g.ret.mean() * 100:+6.2f}% '
              f'中位ret={g.ret.median() * 100:+6.2f}%')
    # top1 权重/评分: 板块分布 (用户问询相关)
    if 'selected' in p.columns:
        sel = p[p.selected == 1] if p.selected.dtype != object else p[p.selected.astype(str).isin(['1', 'True', 'true'])]
        print('  [selected] 板块分布:', sel.board.value_counts().to_dict())

    # 年度板块构成 (池 vs 选中)
    print('\n  分年板块占比 (池):')
    for y, g in p.groupby('year'):
        vc = g.board.value_counts(normalize=True) * 100
        print(f'    {y}: ' + ' | '.join(f'{k}{vc.get(k, 0):.0f}%' for k in ['科创板', '创业板', '沪主板', '深主板']))


if __name__ == '__main__':
    main()
