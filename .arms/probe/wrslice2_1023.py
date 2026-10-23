#!/usr/bin/env python3
"""Q5 倒U结构稳健性探针 (2026-10-23晚, 只读) — wrslice D 切片深挖.

发现 (wrslice_1023): realized 166 笔按入场信号 score 五分位, tW/均ret 呈倒U:
  Q1 52.9%/+1.72 | Q2 69.7%/+6.79 | Q3 72.7%/+9.18 | Q4 60.6%/+4.97 | Q5 57.6%/−0.66
超高分桶 = 唯一负桶, 且三笔大灾难 (−47/−29/−13%) 全在 score>1.0。机制假设:
极端 score = 多因子同时极端 = 共识亢奋 = 追高尾部风险。

本探针杀混杂/验稳健:
  R1 入场日簇 bootstrap: Q5 vs Q2-Q4 的 ΔtW / Δ均ret
  R2 分年×Q5 (是否 2025-26 情绪彩票)
  R3 阈值曲线 score∈{0.85,0.95,1.05,1.2} 之上桶的 n/tW/均ret/左尾(VaR5)
  R4 混杂: Q5 的 daily_ret/mom_60d/volume_ratio/exhaustion/trend_type vs 其余
     (高分是否=当日已暴涨? 若 score 与 daily_ret 强耦合, "高分排除"≈"追高排除",
      而 chase/tt1chase 罚已存在却没拦住 ⇒ 看罚后 eff_score 是否同样倒U)
  R5 eff_score (罚后) 五分位是否同构 — 决定闸放哪一层
  R6 FAST/NORM × Q5; w5/w20 前瞻口径复核

用法: .venv/bin/python .arms/probe/wrslice2_1023.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')

AR = '/mnt/d/quant/.arms/armout'
RNG = np.random.default_rng(20261023)
NBOOT = 5000


def clus_boot_delta(mm, mask, col):
    """入场日簇 bootstrap: 桶 vs 补集 的均值差 (col=ret 或 win)."""
    mm = mm.copy()
    if col == 'win':
        mm['y'] = (mm['ret'] > 0).astype(float)
    else:
        mm['y'] = mm['ret']
    obs = mm[mask]['y'].mean() - mm[~mask]['y'].mean()
    grp = {d: g for d, g in mm.groupby('entry_date')}
    days = sorted(grp)
    deltas = []
    for _ in range(NBOOT):
        pick = RNG.choice(days, size=len(days), replace=True)
        b = pd.concat([grp[d] for d in pick])
        m = mask.reindex(b.index).fillna(False)
        if m.sum() < 4 or (~m).sum() < 4:
            continue
        deltas.append(b[m]['y'].mean() - b[~m]['y'].mean())
    d = np.array(deltas)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, lo, hi, p


def main():
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['sig_date'] = pd.to_datetime(mm['sig_date'])
    # 池列 (混杂变量 + eff_score)
    q = pd.read_csv('/mnt/d/quant/.arms/pool_1022.csv', dtype={'code': str},
                    parse_dates=['date'])
    q['code'] = q['code'].str.zfill(6)
    keep = ['code', 'date', 'eff_score', 'daily_ret', 'mom_60d', 'volume_ratio',
            'turnover', 'exhaustion', 'trend_type', 'stroke_phase', 'regime',
            'chase_penalty', 'tt1chase_penalty', 'ah_penalty', 'bp', 'bp_confirm']
    mm = mm.merge(q[keep], left_on=['code', 'sig_date'], right_on=['code', 'date'],
                  how='left').drop(columns=['date'])

    thr5 = mm['score'].quantile(0.8)
    q5 = mm['score'] >= thr5
    mid = (mm['score'] >= mm['score'].quantile(0.2)) & ~q5

    # ── R1 簇 bootstrap ──
    print(f'══ R1 入场日簇 bootstrap: Q5(score>={thr5:.2f}, n={q5.sum()}) vs Q2-Q4 ══')
    for col, lab in [('win', 'ΔtW'), ('ret', 'Δ均ret')]:
        obs, lo, hi, p = clus_boot_delta(mm, q5, col)
        u = 'pp' if col == 'win' else 'pp'
        print(f'  {lab:7s} obs={obs*100:+7.2f}{u}  95%CI [{lo*100:+.2f},{hi*100:+.2f}]  p={p:.4f}')

    # ── R2 分年 ──
    print(f'\n══ R2 分年 × Q5 ══')
    mm['year'] = mm['entry_date'].dt.year
    for y, g in mm.groupby('year'):
        a, b = g[g['score'] >= thr5], g[g['score'] < thr5]
        print(f'  {y}: Q5 {len(a):2d}笔 tW {(a.ret>0).mean()*100 if len(a) else np.nan:5.1f}% '
              f'均ret {a.ret.mean()*100 if len(a) else np.nan:+6.2f}% | '
              f'其余 {len(b):3d}笔 tW {(b.ret>0).mean()*100:5.1f}% '
              f'均ret {b.ret.mean()*100:+6.2f}%')

    # ── R3 阈值曲线 ──
    print(f'\n══ R3 阈值曲线 (score>thr 桶) ══')
    print(f'  {"thr":>5} {"n":>3} {"tW":>6} {"均ret":>7} {"VaR5":>7} {"min":>7} | 其余tW 其余ret')
    for thr in [0.85, 0.95, 1.05, 1.20]:
        a, b = mm[mm['score'] > thr], mm[mm['score'] <= thr]
        print(f'  {thr:5.2f} {len(a):3d} {(a.ret>0).mean()*100:6.1f}% '
              f'{a.ret.mean()*100:+7.2f}% {a.ret.quantile(0.05)*100:+7.1f}% '
              f'{a.ret.min()*100:+7.1f}% | {(b.ret>0).mean()*100:5.1f}% {b.ret.mean()*100:+6.2f}%')

    # ── R4 混杂: 高分=追高? ──
    print(f'\n══ R4 Q5 vs 其余 的入场日特征 (池列, 选股时点已知) ══')
    for c in ['daily_ret', 'mom_60d', 'volume_ratio', 'turnover', 'chase_penalty',
              'tt1chase_penalty', 'exhaustion']:
        a, b = mm.loc[q5, c], mm.loc[~q5, c]
        if a.notna().sum() < 5:
            continue
        print(f'  {c:16s} Q5中位 {a.median():+.3f} vs 其余 {b.median():+.3f} '
              f'(corr(score,{c})={mm["score"].corr(mm[c]) if c!="exhaustion" else float("nan"):+.3f})')
    for c in ['trend_type', 'regime']:
        ct = pd.crosstab(mm[c], q5, normalize='columns')
        print(f'  {c}: Q5内分布 {ct[True].round(2).to_dict()}')
        print(f'        其余分布 {ct[False].round(2).to_dict()}')

    # ── R5 eff_score (罚后) 五分位 ──
    print(f'\n══ R5 eff_score 五分位 (罚后最终排序键) ══')
    mm['eq'] = pd.qcut(mm['eff_score'].rank(method='first'), 5,
                       labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高'])
    for lab, g in mm.groupby('eq', observed=True):
        print(f'  {str(lab):5s} n={len(g):3d} eff中位={g.eff_score.median():.3f} '
              f'tW={(g.ret>0).mean()*100:5.1f}% 均ret={g.ret.mean()*100:+6.2f}% '
              f'w5={(g.w5>0).mean()*100:5.1f}% w20={(g.w20>0).mean()*100:5.1f}%')

    # ── R6 FAST/NORM + 前瞻口径 ──
    print(f'\n══ R6 Q5 × regime; Q5 前瞻口径 ══')
    for rg, g in mm.groupby('regime'):
        a, b = g[g['score'] >= thr5], g[g['score'] < thr5]
        print(f'  {rg}: Q5 n={len(a)} tW {(a.ret>0).mean()*100 if len(a)>=3 else np.nan:5.1f}% '
              f'| 其余 n={len(b)} tW {(b.ret>0).mean()*100:5.1f}%')
    a, b = mm[q5], mm[~q5]
    print(f'  Q5: w5 {(a.w5>0).mean()*100:.1f}% w20 {(a.w20>0).mean()*100:.1f}% '
          f'均w5 {a.w5.mean()*100:+.2f}% 均w20 {a.w20.mean()*100:+.2f}%')
    print(f'  其余: w5 {(b.w5>0).mean()*100:.1f}% w20 {(b.w20>0).mean()*100:.1f}% '
          f'均w5 {b.w5.mean()*100:+.2f}% 均w20 {b.w20.mean()*100:+.2f}%')

    # 反事实粗算: 剔 Q5 后账簿 (直接效应, 无补位)
    print(f'\n══ 反事实粗算: 剔 score>{thr5:.2f} (直接, 无补位) ══')
    keep_b = mm[mm['score'] <= thr5]
    print(f'  剩 {len(keep_b)} 笔 tW {(keep_b.ret>0).mean()*100:.2f}% '
          f'(原 62.65%) 均ret {keep_b.ret.mean()*100:+.2f}% (原 +4.39%)')
    print(f'  被剔 {int(q5.sum())} 笔: 赢家 {int((mm[q5].ret > 0).sum())} 笔 '
          f'合计 {mm[q5][mm[q5].ret > 0].ret.sum() * 100:+.1f}pp, '
          f'输家 {int((mm[q5].ret <= 0).sum())} 笔 '
          f'合计 {mm[q5][mm[q5].ret <= 0].ret.sum() * 100:+.1f}pp')


if __name__ == '__main__':
    main()
