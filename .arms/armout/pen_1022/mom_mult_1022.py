#!/usr/bin/env python3
"""mom_adj / 动量乘数 的离线重排扫描 (2026-10-22).

两个从未在修复态 sweep 过的残差成分:
  mom_adj  = NORM: +0.18·mom_60d ; FAST: −0.10·|mom_60d|   (硬编码)
  multiplier: mom>0.60→×0.85, >0.45→×0.92; dist_ma60>0.30→×0.40
rank_term 列 = 截面百分位 rank ∈ (0,1] (乘 multiplier 后进 eff_scored).
池级 top-k 代理已验证 (实际 vs 离线 Jacc=0.99, Δ10 −0.03%).
仅量化, 不改生产.
"""
import sys
import numpy as np
import pandas as pd

POOL = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/quant/.arms/pool_1022.csv'
PX = '/mnt/d/quant/data/research/qfq_kline.parquet'
RNG = np.random.default_rng(20261025)

d = pd.read_csv(POOL, dtype={'code': str})
d['date'] = pd.to_datetime(d['date'])
print(f'池 rows={len(d):,} days={d.date.nunique()}')

px = pd.read_parquet(PX, columns=['code', 'date', 'open', 'close'])
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['code', 'date'])
g = px.groupby('code', sort=False)
px['open_next'] = g['open'].shift(-1)
for k in (5, 10, 20):
    px[f'fwd{k}'] = g['close'].shift(-k) / px['open_next'] - 1
m = d.merge(px[['code', 'date', 'fwd5', 'fwd10', 'fwd20']],
            on=['code', 'date'], how='left', validate='many_to_one')
m = m[m.fwd10.notna()].reset_index(drop=True)
print(f'可用 {len(m):,} 行 {m.date.nunique()} 日')

mom = m.mom_60d.fillna(0.0).values
rank = m.rank_term.values
mult = m.multiplier.values
dist = m.dist_ma60.fillna(0.0).values
fast = (m.regime == 'FAST').values
mom_adj = m.mom_adj.fillna(0.0).values
eff = m.eff_scored.values

# 拆出 dist_mult 与 mom_mult
dist_mult = np.where(dist > 0.30, 0.40, 1.0)
mom_mult = np.round(mult / dist_mult, 6)

print('mom_mult 分布', pd.Series(mom_mult).value_counts().to_dict())
print('dist_mult 行数', int((dist_mult < 1).sum()))

base_sel = {}
for dt, gd in m.groupby('date', sort=True):
    k = int(gd.n_target.iloc[0])
    if k > 0:
        base_sel[dt] = set(gd.nlargest(k, 'eff_scored').code)


def evaluate(v):
    rows = []
    for dt, gd in m.groupby('date', sort=True):
        bs = base_sel.get(dt)
        if not bs:
            continue
        tmp = gd.copy()
        tmp['_v'] = v[tmp.index.values]
        top = tmp.nlargest(len(bs), '_v')
        b = tmp[tmp.code.isin(bs)]
        rows.append((dt, top.fwd5.mean() - b.fwd5.mean(),
                     top.fwd10.mean() - b.fwd10.mean(),
                     top.fwd20.mean() - b.fwd20.mean(),
                     len(set(top.code) & bs) / max(len(set(top.code) | bs), 1)))
    r = pd.DataFrame(rows, columns=['date', 'd5', 'd10', 'd20', 'jacc'])
    r['year'] = r.date.dt.year
    return r


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    return float(np.quantile(x[idx].mean(axis=1), .025)), float(np.quantile(x[idx].mean(axis=1), .975))


def report(r, label):
    mean10 = r.d10.mean(); lo, hi = boot_ci(r.d10)
    ys = r.groupby('year').d10.mean()
    same = int((np.sign(ys) == np.sign(mean10)).sum()) if mean10 != 0 else 0
    print(f'{label:26s} Δ5={r.d5.mean():+6.2%} Δ10={mean10:+6.2%} '
          f'CI[{lo:+.2%},{hi:+.2%}] Δ20={r.d20.mean():+6.2%} '
          f'Jacc={r.jacc.mean():.2f} 年同向={same}/{len(ys)}')
    return dict(label=label, d10=mean10, lo=lo, hi=hi,
                jacc=r.jacc.mean(), yearly=';'.join(f'{y}:{x:+.4f}' for y, x in ys.items()))


res = [report(evaluate(eff), 'baseline')]

# ---------- mom_adj 整体 scale ----------
print('\n=== mom_adj scale (两 regime 同乘) ===')
for s in (0.0, 0.5, 1.5, 2.0, 3.0):
    v = eff - mom_adj + s * mom_adj
    res.append(report(evaluate(v), f'momadj×{s}'))

# ---------- regime 分拆 ----------
print('\n=== regime 分拆 ===')
for lab, mask, val in (
        ('FAST off', fast, 0.0),
        ('FAST −0.10→0', fast, 0.0),
        ('NORM ×2', ~fast, None),
        ('NORM off', ~fast, 0.0)):
    if lab == 'NORM ×2':
        v = eff - mom_adj + np.where(~fast, 2 * mom_adj, mom_adj)
    else:
        v = eff - np.where(mask, mom_adj, 0.0)
    res.append(report(evaluate(v), lab))

# ---------- 动量乘数 ----------
print('\n=== 动量乘数面 (mom_mult 替换) ===')
mom_new = {
    'mom_mult off': np.where(mom > 0.60, 1.0, np.where(mom > 0.45, 1.0, 1.0)),
    'mom_mult mild': np.where(mom > 0.60, 0.92, np.where(mom > 0.45, 0.95, 1.0)),
    'mom_mult strong': np.where(mom > 0.60, 0.75, np.where(mom > 0.45, 0.85, 1.0)),
    'mom_mult hard60': np.where(mom > 0.60, 0.70, np.where(mom > 0.45, 0.85, 1.0)),
}
for lab, nm in mom_new.items():
    v = eff - rank * mult + rank * nm * dist_mult
    res.append(report(evaluate(v), lab))

print('\n=== dist 乘数面 (仅 47 行触发, 参考) ===')
for dm in (1.0, 0.6, 0.3):
    nm = np.where(dist > 0.30, dm, 1.0)
    v = eff - rank * mult + rank * mom_mult * nm
    res.append(report(evaluate(v), f'distmult={dm}'))

pd.DataFrame(res).to_csv('/mnt/d/quant/.arms/probe/mom_mult_1022.csv', index=False)
print('\n← .arms/probe/mom_mult_1022.csv')
