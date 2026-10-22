#!/usr/bin/env python3
"""特征加成反事实 (2026-10-22, 只读).

shortfeat_1022 在 cap2 候选上发现三个正特征 (池级边际):
  stroke_phase==0.4 → w5 56.2% vs 43.8%
  div_type==bottom_fx → 58.9% vs 44.4%
  bp==1 → 51.9% vs 43.4%
但池级边际≠选择边际 (动量轴已证). 本探针把特征作为 eff_score 加成,
逐日重选前2名, 日级配对 bootstrap 比较 w5/w20.

加成尺度: 相邻 eff_score 差中位 0.088 (p25 .038/p75 .185).
用法: python bonus_1022.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load

CAP = 2
NBOOT = 5000
RNG = np.random.default_rng(20261022)


def prep():
    p = load()
    q = p[p['nr'].notna()].copy()
    q['stroke04'] = np.isclose(q['stroke_phase'], 0.4).astype(float)
    q['bottomfx'] = q['div_type'].astype(str).eq('bottom_fx').astype(float)
    q['bp1'] = q['bp'].eq(1).astype(float)
    q['bp2'] = q['bp'].eq(2).astype(float)
    q['hidden_bottom'] = q['div_type'].astype(str).eq('hidden_bottom').astype(float)
    return q


def select(q, score_col, cap=CAP):
    rows = []
    for d, g in q.groupby('date', sort=True):
        g = g.sort_values(score_col, ascending=False, kind='mergesort').head(cap)
        rows.append(dict(date=d, n=len(g), w5=g['w5'].mean(), w20=g['w20'].mean(),
                         r5=g['r5'].mean(), r20=g['r20'].mean(),
                         codes=set(g['code'])))
    return pd.DataFrame(rows)


def paired(base, alt, col):
    m = base.set_index('date')[[col]].join(alt.set_index('date')[[col]],
                                           lsuffix='_b', rsuffix='_a').dropna()
    d = (m[col + '_a'] - m[col + '_b']).values
    if not len(d):
        return (np.nan,) * 4
    idx = RNG.integers(0, len(d), size=(NBOOT, len(d)))
    bm = d[idx].mean(axis=1)
    return (d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5),
            float((bm > 0).mean()))


def main():
    q = prep()
    base = select(q, 'eff_score')
    print(f'基线: 选取={int(base.n.sum())}只 日={len(base)} '
          f'w5={base.w5.mean()*100:.2f} w20={base.w20.mean()*100:.2f} '
          f'r5={base.r5.mean()*100:+.2f} r20={base.r20.mean()*100:+.2f}')
    rules = [
        ('stroke04 +0.03', 'eff_score + 0.03*stroke04'),
        ('stroke04 +0.06', 'eff_score + 0.06*stroke04'),
        ('stroke04 +0.12', 'eff_score + 0.12*stroke04'),
        ('bottomfx +0.03', 'eff_score + 0.03*bottomfx'),
        ('bottomfx +0.06', 'eff_score + 0.06*bottomfx'),
        ('bottomfx +0.12', 'eff_score + 0.12*bottomfx'),
        ('bp1 +0.06', 'eff_score + 0.06*bp1'),
        ('bp1 +0.12', 'eff_score + 0.12*bp1'),
        ('bp2 -0.06', 'eff_score - 0.06*bp2'),
        ('bp2 -0.12', 'eff_score - 0.12*bp2'),
        ('stroke04+bfx +0.06', 'eff_score + 0.06*(stroke04+bottomfx)'),
        ('stroke04+bfx +0.12', 'eff_score + 0.12*(stroke04+bottomfx)'),
    ]
    out = []
    for name, expr in rules:
        q['_alt'] = q.eval(expr.replace('eff_score', '`eff_score`')) if False else None
        # 手写避免 eval 精度/引用问题
        if name in ('stroke04 +0.03', 'stroke04 +0.06', 'stroke04 +0.12'):
            b = float(name.split('+')[1])
            q['_alt'] = q['eff_score'] + b * q['stroke04']
        elif name in ('bottomfx +0.03', 'bottomfx +0.06', 'bottomfx +0.12'):
            b = float(name.split('+')[1])
            q['_alt'] = q['eff_score'] + b * q['bottomfx']
        elif name.startswith('bp1'):
            b = float(name.split('+')[1])
            q['_alt'] = q['eff_score'] + b * q['bp1']
        elif name.startswith('bp2'):
            b = float(name.split('-')[1])
            q['_alt'] = q['eff_score'] - b * q['bp2']
        elif name.startswith('stroke04+bfx'):
            b = float(name.split('+')[2])
            q['_alt'] = q['eff_score'] + b * (q['stroke04'] + q['bottomfx'])
        alt = select(q, '_alt')
        d5 = paired(base, alt, 'w5')
        d20 = paired(base, alt, 'w20')
        # 换手构成
        bset = base.set_index('date')['codes']
        aset = alt.set_index('date')['codes']
        keep = sum(len(bset[d] & aset[d]) for d in bset.index)
        out.append(dict(rule=name, n=int(alt.n.sum()), same=keep,
                        w5=alt.w5.mean() * 100, d5=d5[0] * 100,
                        lo5=d5[1] * 100, hi5=d5[2] * 100, p5=d5[3],
                        w20=alt.w20.mean() * 100, d20=d20[0] * 100,
                        lo20=d20[1] * 100, hi20=d20[2] * 100, p20=d20[3]))
    r = pd.DataFrame(out)
    print('\n' + '=' * 110)
    print(r.round(2).to_string(index=False))
    r.to_csv('/mnt/d/quant/.arms/armout/bonus_1022.csv', index=False)
    print('\n落盘: .arms/armout/bonus_1022.csv')


if __name__ == '__main__':
    main()
