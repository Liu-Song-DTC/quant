#!/usr/bin/env python3
"""目标敞口 × 新入场数 反事实 (2026-10-22, 只读).

线索: 日级 target_exposure (te) 高的调仓日, cap2 候选 w5 胜率显著更低
(te=0.85: 40.6% vs te<0.7: 52.9-60.0%; 见 expo_time_1022.log).
规则候选: te>=thr 只买第1名 (cap1), 否则 cap2 — 保留低敞口日的 R2, 高敞口日避开 R2.

本探针用日级重采样 (5000) 给 w5/w20 的规则差 CI; 日等权 (与 cluster bootstrap 同口径).
注意: 忽略持仓占用槽位, 仅作规则筛选.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load

NBOOT = 5000
RNG = np.random.default_rng(20261022)


def day_panel():
    p = load()
    q = p[p['nr'] <= 2].copy()
    q['te'] = q.groupby('date')['target_exposure'].transform('first')
    rows = []
    for d, g in q.groupby('date'):
        g = g.sort_values('nr')
        r = dict(date=d, te=g['te'].iloc[0], n=len(g))
        for h in (5, 20):
            vals = g[f'w{h}'].values
            rr = g[f'r{h}'].values
            r[f'w{h}_1'] = vals[0] if len(vals) > 0 else np.nan
            r[f'w{h}_2'] = vals[1] if len(vals) > 1 else np.nan
            r[f'w{h}_all'] = np.nanmean(vals) if len(vals) else np.nan
            r[f'r{h}_1'] = rr[0] if len(rr) > 0 else np.nan
            r[f'r{h}_2'] = rr[1] if len(rr) > 1 else np.nan
            r[f'r{h}_all'] = np.nanmean(rr) if len(rr) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def rule_panel(d, thr):
    """返回每日规则下的 w5/w20 与选取只数."""
    out = d.copy()
    hi = out['te'] >= thr
    for h in (5, 20):
        out[f'rw{h}'] = np.where(hi, out[f'w{h}_1'], out[f'w{h}_all'])
        out[f'rr{h}'] = np.where(hi, out[f'r{h}_1'], out[f'r{h}_all'])
    out['k'] = np.where(hi, 1, np.minimum(out['n'], 2))
    return out


def boot(d, col, B):
    v = d[col].values
    ok = np.isfinite(v)
    v = v[ok]
    if not len(v):
        return (np.nan,) * 4
    idx = RNG.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return (v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5),
            float((means > 0).mean()))


def main():
    d = day_panel()
    print(f'调仓日={len(d)} te>=0.8 日={int((d.te>=0.8).sum())}')
    base = d['w5_all']
    print(f"基线 cap2: w5={base.mean()*100:.2f} w20={d['w20_all'].mean()*100:.2f} "
          f"选取={d['n'].clip(upper=2).sum()}只")
    for thr in (0.80, 0.70, 0.65, 0.60, 0.50):
        r = rule_panel(d, thr)
        # 日级配对差
        dd = (r['rw5'] - d['w5_all']).dropna()
        m, lo, hi, pw = boot(pd.DataFrame({'x': dd}), 'x', NBOOT)
        dd20 = (r['rw20'] - d['w20_all']).dropna()
        m20, lo20, hi20, pw20 = boot(pd.DataFrame({'x': dd20}), 'x', NBOOT)
        print(f'\nte>={thr:.2f}→cap1 else cap2: 选取={int(r.k.sum())}只 (基线 {int(d.n.clip(upper=2).sum())})')
        print(f"  w5={r.rw5.mean()*100:5.2f} Δ={(r.rw5.mean()-base.mean())*100:+.2f}pp "
              f"CI[{lo*100:+.2f},{hi*100:+.2f}] P>0={pw:.3f}")
        print(f"  w20={r.rw20.mean()*100:5.2f} Δ={(r.rw20.mean()-d.w20_all.mean())*100:+.2f}pp "
              f"CI[{lo20*100:+.2f},{hi20*100:+.2f}] P>0={pw20:.3f} "
              f"r20={r.rr20.mean()*100:+.2f}% (基线 {d.r20_all.mean()*100:+.2f}%)")
    # 分年: 规则(0.80) vs 基线
    r = rule_panel(d, 0.80)
    r['year'] = r['date'].dt.year
    print('\n分年 (te>=0.8→cap1):')
    for y, g in r.groupby('year'):
        print(f"  {y}: 日={len(g):2d} w5={g.rw5.mean()*100:5.1f} (基线 {g.w5_all.mean()*100:5.1f}, "
              f"Δ{(g.rw5.mean()-g.w5_all.mean())*100:+.1f}pp) w20={g.rw20.mean()*100:5.1f} "
              f"(基线 {g.w20_all.mean()*100:5.1f}, Δ{(g.rw20.mean()-g.w20_all.mean())*100:+.1f}pp)")


if __name__ == '__main__':
    main()
