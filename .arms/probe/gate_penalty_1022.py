#!/usr/bin/env python3
"""活跃门控罚项 (chase/ml_gate) 的重标定探针 (2026-10-22).

池级落盘证明: 8 个 gate 罚中只有 2 个在修复态真正触发 —
  chase_penalty  (QUANT_ZG_FILTER=2, QUANT_ZG_PEN=0.20)  触发 36.8%
  ml_gate_penalty(QUANT_ML_GATE=0.15, TH=0.084)          触发 14.3%
其余 6 项(no_chan −0.20 仅剩 81 行/1.3%, divnone/tt1chase/ah/bp_c4/lowsh=0)
在 bp_confirm_min=3 下已结构性死掉。

两个探针:
  A. Fama-MacBeth: fwd10 ~ 1 + trig + base (base=eff_scored−该项值)
     问"触发该项的名字扣掉其余评分后, 实际多/少赚".
  B. 离线 top-k 重排: 把该项值乘 scale ∈ {0,.5,.75,1,1.25,1.5,2},
     与基线离线 top-k 的 fwd5/10/20 配对 (日级 bootstrap CI + 逐年).
仅量化, 不改生产.
"""
import sys
import numpy as np
import pandas as pd

POOL = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/quant/.arms/pool_1022.csv'
PX = '/mnt/d/quant/data/research/qfq_kline.parquet'
RNG = np.random.default_rng(20261024)
MIN_ROWS = 30

print(f'读池 {POOL} ...')
d = pd.read_csv(POOL, dtype={'code': str})
d['date'] = pd.to_datetime(d['date'])
print(f'  rows={len(d):,} days={d.date.nunique()} codes={d.code.nunique()}')

rec = (d.rank_term * d.multiplier + d.additive + d.turnover + d.repl_buffer
       + d.mom_adj + d.no_chan_penalty + d.chase_penalty + d.ml_gate_penalty
       + d.divnone_penalty + d.tt1chase_penalty + d.ah_penalty
       + d.bp_c4_penalty + d.lowsh_penalty)
err = (rec - d.eff_scored).abs()
print(f'identity: max|Δ|={err.max():.3e} 超1e-9={int((err > 1e-9).sum())}')
if err.max() > 1e-6:
    print('⛔ identity 失败'); sys.exit(2)

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
print(f'  可用行 {len(m):,} 日 {m.date.nunique()}')

TERMS = ['chase_penalty', 'ml_gate_penalty', 'no_chan_penalty']
y_all = m.fwd10.values
dates = m.date.values
day_idx = {}
for dt, idx in m.groupby('date', sort=True).groups.items():
    day_idx[dt] = np.asarray(list(idx))


def day_ols(idx, trig, base):
    if len(idx) < MIN_ROWS:
        return None
    t = trig[idx]
    if t.min() == t.max():
        return None
    X = np.column_stack([np.ones(len(idx)), t.astype(float), base[idx]])
    y = y_all[idx]
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[1], beta[2]


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    mu = x[idx].mean(axis=1)
    return float(np.quantile(mu, .025)), float(np.quantile(mu, .975))


print('\n=== A. Fama-MacBeth 独立信息检验 ===')
for name in TERMS:
    v = m[name].fillna(0.0).values
    trig = (v != 0).astype(float)
    if trig.sum() == 0:
        print(f'{name:16s} 零触发, 跳过'); continue
    base = m.eff_scored.values - v
    bt, bb, yrs = [], [], []
    for dt, idx in day_idx.items():
        r = day_ols(idx, trig, base)
        if r is None:
            continue
        bt.append(r[0]); bb.append(r[1]); yrs.append(pd.Timestamp(dt).year)
    bt = np.asarray(bt); bb = np.asarray(bb); yrs = np.asarray(yrs)
    if len(bt) < 20:
        print(f'{name:16s} 有效日={len(bt)} 太少'); continue
    mean_b = bt.mean(); se = bt.std(ddof=1) / np.sqrt(len(bt))
    lo, hi = boot_ci(bt)
    coef = float(v[trig == 1].mean())
    lam = float(np.median(bb))
    ys = pd.Series(bt).groupby(yrs).mean()
    same = int((np.sign(ys) == np.sign(mean_b)).sum())
    print(f'{name:16s} 有效日={len(bt):3d} 触发={trig.mean():6.1%} '
          f'c={coef:+.3f} b_trig={mean_b:+.4f} CI[{lo:+.4f},{hi:+.4f}] '
          f't={mean_b/se:+.2f} λ={lam:+.4f} 隐含={coef*lam:+.4f} 年同向={same}/{len(ys)}')
    print('    逐年: ' + ' '.join(f'{y}:{x:+.4f}' for y, x in ys.items()))

# ---------- B. 离线 top-k 重排 ----------
base_sel = {}
for dt, gd in m.groupby('date', sort=True):
    k = int(gd.n_target.iloc[0])
    if k <= 0:
        continue
    base_sel[dt] = set(gd.nlargest(k, 'eff_scored').code)


def evaluate(score_col):
    rows = []
    for dt, gd in m.groupby('date', sort=True):
        bs = base_sel.get(dt)
        if not bs:
            continue
        top = gd.nlargest(len(bs), score_col)
        b = gd[gd.code.isin(bs)]
        a = top
        rows.append((dt, a.fwd5.mean() - b.fwd5.mean(),
                     a.fwd10.mean() - b.fwd10.mean(),
                     a.fwd20.mean() - b.fwd20.mean(),
                     len(set(a.code) & bs) / max(len(set(a.code) | bs), 1)))
    r = pd.DataFrame(rows, columns=['date', 'd5', 'd10', 'd20', 'jacc'])
    r['year'] = r.date.dt.year
    return r


def report(r, label):
    mean10 = r.d10.mean(); lo, hi = boot_ci(r.d10)
    ys = r.groupby('year').d10.mean()
    same = int((np.sign(ys) == np.sign(mean10)).sum()) if mean10 != 0 else 0
    print(f'{label:24s} Δ5={r.d5.mean():+6.2%} Δ10={mean10:+6.2%} '
          f'CI[{lo:+.2%},{hi:+.2%}] Δ20={r.d20.mean():+6.2%} '
          f'Jacc={r.jacc.mean():.2f} 年同向={same}/{len(ys)}')
    return dict(label=label, d10=mean10, lo=lo, hi=hi, jacc=r.jacc.mean(),
                yearly=';'.join(f'{y}:{x:+.4f}' for y, x in ys.items()))


print('\n=== B. 离线 top-k 重排: 单项 scale ===')
res = [report(evaluate('eff_scored'), 'baseline')]
for name in ('chase_penalty', 'ml_gate_penalty'):
    v = m[name].fillna(0.0).values
    for s in (0.0, 0.5, 0.75, 1.25, 1.5, 2.0):
        m['_v'] = m.eff_scored.values - v + s * v
        res.append(report(evaluate('_v'), f'{name}×{s}'))
# 两项同删
m['_v'] = (m.eff_scored.values - m.chase_penalty.fillna(0).values
           - m.ml_gate_penalty.fillna(0).values)
res.append(report(evaluate('_v'), 'drop_both'))
# 两项同放大
m['_v'] = (m.eff_scored.values + m.chase_penalty.fillna(0).values
           + m.ml_gate_penalty.fillna(0).values)  # c<0 ⇒ 加回=放大
res.append(report(evaluate('_v'), 'scale2_both'))

pd.DataFrame(res).to_csv('/mnt/d/quant/.arms/probe/gate_penalty_1022.csv', index=False)
print('\n← .arms/probe/gate_penalty_1022.csv')
