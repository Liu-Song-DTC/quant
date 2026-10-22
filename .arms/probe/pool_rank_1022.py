#!/usr/bin/env python3
"""池级名次检验 (2026-10-22, 只读): 用 QUANT_DBG_POOL 全池扩大 rank3 样本.

背景: 组合层 cap=2 只在"当日 3 个新入场"的 45 天上绑定, 样本太少
(随机化 p=0.058 / 同日配对 p=0.096). 池 `_dump_pool_debug` 落盘了每次
选股前的**完整 qualified 池**, 且行序 = 引擎顺序 (locked 先, 其余 eff_score 降序;
实测 rho=0.992, 被选中新入场全部位于非持仓行序 1-3).

⇒ 对每个调仓日的非持仓候选按行序编名次, 得到 ~150 天的 rank3 前瞻样本
(而不是只有 45 天被选中的). 检验假设: 第3名候选的前瞻胜率是否低于第1/2名.

注意: 这是**信号质量**口径 (候选池), 不是账簿口径; 未入选的第3名没有交易成本.
用法: python pool_rank_1022.py [pool.csv]
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from winrate_diag_1022 import add_forward
from mcnemar_sel_1022 import binom_two_sided
from rank3_paired_1022 import wilson

POOL = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/quant/.arms/pool_1022.csv'
EQ = '/mnt/d/quant/strategy/rolling_validation_results/equity_curve.csv.base_tr1022'
SEL = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv.base_tr1022'


def load():
    p = pd.read_csv(POOL, dtype={'code': str})
    p['code'] = p['code'].str.zfill(6)
    p['date'] = pd.to_datetime(p['date'])
    # 行序 = 引擎顺序; 非持仓名次
    p['row'] = p.groupby('date').cumcount() + 1
    p['nr'] = np.nan
    nh = p[p['is_held'] == 0]
    p.loc[nh.index, 'nr'] = nh.groupby('date').cumcount() + 1
    eq = pd.read_csv(EQ, parse_dates=['date'])
    cal = list(eq['date'])
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    p['exec_date'] = p['date'].map(nxt)
    p['year'] = p['date'].dt.year
    p['half'] = np.where(p['year'] <= 2023, 'H1', 'H2')
    p = add_forward(p)
    return p


def pair(m, hi, lo, col, label, days=None):
    a = m[m['nr'] == lo].set_index('date')[[col]]
    b = m[m['nr'] == hi].set_index('date')[[col]]
    x = a.join(b, lsuffix='_lo', rsuffix='_hi').dropna()
    if days is not None:
        x = x[x.index.isin(days)]
    if not len(x):
        print(f'  {label}: 无可配对'); return
    lo_w = (x[col + '_lo'] > 0).values; hi_w = (x[col + '_hi'] > 0).values
    honly = int((hi_w & ~lo_w).sum()); lonly = int((~hi_w & lo_w).sum())
    p = binom_two_sided(honly, honly + lonly)
    print(f'  {label}: n日={len(x)} 第{hi}名={hi_w.mean()*100:5.1f}% '
          f'第{lo}名={lo_w.mean()*100:5.1f}% Δ={(hi_w.mean()-lo_w.mean())*100:+5.1f}pp '
          f'仅高名赢={honly} 仅低名赢={lonly} McNemar p={p:.4f}')


def main():
    p = load()
    print(f'池: {POOL}  行={len(p)} 日={p["date"].nunique()} '
          f'区间={p["date"].min().date()}..{p["date"].max().date()}')
    q = p[p['nr'].notna()]
    print('\n名次前瞻 (非持仓候选, 引擎行序):')
    print(f'{"名次":>4s} {"n":>5s} {"w5%":>6s} {"r5%":>7s} {"w10%":>6s} {"w20%":>6s} {"r20%":>7s}')
    for r in range(1, 9):
        g = q[q['nr'] == r]
        if len(g) < 10:
            continue
        print(f'{r:4d} {len(g):5d} {g.w5.mean()*100:6.1f} {g.r5.mean()*100:+7.2f} '
              f'{g.w10.mean()*100:6.1f} {g.w20.mean()*100:6.1f} {g.r20.mean()*100:+7.2f}')

    print('\n同日配对 (全池):')
    for hi, lo in ((1, 2), (2, 3), (1, 3), (2, 4), (1, 4)):
        for col in ('w5', 'w20'):
            pair(q, hi, lo, col, f'第{hi}名 vs 第{lo}名 {col}')

    print('\n分 regime:')
    for reg, g in q.groupby('regime'):
        if g['date'].nunique() < 20:
            continue
        print(f'  [{reg}] 日={g["date"].nunique()}')
        for r in (1, 2, 3, 4):
            gg = g[g['nr'] == r]
            if len(gg) >= 10:
                print(f'    名次{r}: n={len(gg):4d} w5={gg.w5.mean()*100:5.1f} '
                      f'w20={gg.w20.mean()*100:5.1f} r20={gg.r20.mean()*100:+6.2f}')
        for col in ('w5', 'w20'):
            pair(g, 2, 3, col, f'  {reg} 第2 vs 第3 {col}')

    print('\n分年 (w20% / r20%):')
    for y, g in q.groupby('year'):
        line = f'  {y}:'
        for r in (1, 2, 3):
            gg = g[g['nr'] == r]
            if len(gg):
                line += f'  R{r} n={len(gg):3d} {gg.w20.mean()*100:5.1f}%/{gg.r20.mean()*100:+5.1f}%'
        print(line)

    print('\n分半 (w5/w20/r20):')
    for h, g in q.groupby('half'):
        line = f'  {h}:'
        for r in (1, 2, 3):
            gg = g[g['nr'] == r]
            if len(gg):
                line += (f'  R{r} n={len(gg):3d} {gg.w5.mean()*100:5.1f}/'
                         f'{gg.w20.mean()*100:5.1f}/{gg.r20.mean()*100:+5.2f}')
        print(line)

    # 仅 45 个"3新入场"日 (与组合层臂口径对齐)
    sel = pd.read_csv(SEL, dtype={'code': str}, parse_dates=['date'])
    sel['code'] = sel['code'].str.zfill(6)
    days3 = set(sel.groupby('date').size().loc[lambda s: s == 3].index)
    print(f'\n仅 {len(days3)} 个"当日3个新入场"日 (与 new2 绑定日对齐):')
    for hi, lo in ((2, 3), (1, 3)):
        for col in ('w5', 'w20'):
            pair(q, hi, lo, col, f'第{hi}名 vs 第{lo}名 {col}', days=days3)

    # 与实选一致率: 被选中新入场是否都在 nr<=2
    s = p[(p['selected'] == 1) & (p['is_held'] == 0)]
    print(f'\n被选中新入场: {len(s)} 行, nr 分布 {s["nr"].value_counts().sort_index().to_dict()}')
    print(f'  nr<=2 占比 {(s["nr"]<=2).mean()*100:.1f}%  (cap=2 会丢掉的: n={int((s["nr"]>2).sum())})')
    q.to_csv('/mnt/d/quant/.arms/armout/pool_rank_panel_1022.csv', index=False)


if __name__ == '__main__':
    main()
