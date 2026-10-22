#!/usr/bin/env python3
"""选股残差罚项离线重排反事实 (2026-10-22).

输入: QUANT_DBG_POOL 落盘的池 (`/mnt/d/quant/.arms/pool_1022.csv`):
每个调仓日的全部 qualified 候选 + 评分分量 + 特征 + eff_score + selected.
输出: 对 v2.04 继承的 12 条件罚项 / bp1·bp2·BOM 加成, 逐项做"消融"重排,
      与基线离线 top-k 的 fwd5/10/20 配对比较 (日级 bootstrap CI + 逐年).
判据: 某项 CI 排除 0 且逐年同向 → 才进引擎臂; 否则闭合.

口径: 入场=T+1 开盘, fwdN=close(T+N)/open(T+1)-1 (与回测成交一致).
"""
import sys
import numpy as np
import pandas as pd

POOL = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/quant/.arms/pool_1022.csv'
PX = '/mnt/d/quant/data/research/qfq_kline.parquet'
RNG = np.random.default_rng(20261022)

# 生产常量 (factor_config.yaml)
B3_VOL_PEN = -0.06
BP2_BOOST = 0.15

print(f'读池 {POOL} ...')
d = pd.read_csv(POOL, dtype={'code': str})
d['date'] = pd.to_datetime(d['date'])
print(f'  rows={len(d):,} days={d.date.nunique()} codes={d.code.nunique()}')

# ---------- 1. identity 烟测: 分量重算 == eff_score ----------
rec = (d.rank_term * d.multiplier + d.additive + d.turnover + d.repl_buffer
       + d.mom_adj + d.no_chan_penalty + d.chase_penalty + d.ml_gate_penalty
       + d.divnone_penalty + d.tt1chase_penalty + d.ah_penalty
       + d.bp_c4_penalty + d.lowsh_penalty)
err = (rec - d.eff_scored).abs()
print(f'identity: max|Δ|={err.max():.3e} p99={err.quantile(.99):.3e} '
      f'超1e-9={int((err > 1e-9).sum())}')
if err.max() > 1e-6:
    print('⛔ identity 失败'); sys.exit(2)

# ---------- 2. 前瞻收益 (先 merge, 保证 terms 与 m 逐行对齐) ----------
print('读价格镜像 ...')
px = pd.read_parquet(PX, columns=['code', 'date', 'open', 'close'])
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['code', 'date'])
g = px.groupby('code', sort=False)
px['open_next'] = g['open'].shift(-1)
for k in (5, 10, 20):
    px[f'fwd{k}'] = g['close'].shift(-k) / px['open_next'] - 1
px = px[['code', 'date', 'open_next', 'fwd5', 'fwd10', 'fwd20']]
m = d.merge(px, on=['code', 'date'], how='left', validate='many_to_one')
cov = m.fwd10.notna().mean()
print(f'  前瞻覆盖 {cov:.1%} (缺失=停牌/镜像尾部)')
m = m[m.fwd10.notna()].reset_index(drop=True)

# ---------- 3. 重算继承罚项 (v2.04 12条件 + bp1/bp2/BOM) ----------
div = m.div_type.fillna('').str.lower()
is_top = div.str.contains('top')
is_bottom = div.str.contains('bottom')
terms = {}
terms['top_div'] = np.where(is_top, -0.08, 0.0)
terms['profit_decl'] = np.where(m.profit_declining == 1, -0.10, 0.0)
terms['vol_down'] = np.where((m.daily_ret < -0.015) & (m.volume_ratio > 0.5), -0.06, 0.0)
terms['limit_chase'] = np.where(m.daily_ret > 0.09, -0.05, 0.0)
terms['exhaustion'] = np.where(m.exhaustion > 0.5, -0.10, 0.0)
terms['bp1_nobottom'] = np.where((m.bp == 1) & ~is_bottom, -0.06, 0.0)
terms['downtrend_accel'] = np.where((m.trend_type == -2) & (m.stroke_phase > 0.2)
                                    & ~((m.bp == 3) & (m.sl >= 2)), -0.05, 0.0)
terms['weak_drift'] = np.where((m.trend_type != 2) & (m.stroke_phase > 0.2)
                               & (m.daily_ret < 0), -0.04, 0.0)
terms['extreme_chase'] = np.where((m.mom_60d > 0.50) & (m.bp == 0), -0.06, 0.0)
terms['dist45'] = np.where((m.dist_ma60 > 0.45) & (m.bp == 0), -0.06, 0.0)
terms['b3_vol'] = np.where((m.bp == 3) & (m.volume_ratio > -0.30)
                           & (m.volume_ratio <= -0.20), B3_VOL_PEN, 0.0)
terms['bp1_bonus'] = np.where((m.bp == 1) & (m.sl >= 1), 0.08, 0.0)
terms['bp2_bonus'] = np.where((m.bp == 2) & (m.sl >= 2), 0.03, 0.0)
terms['bom_bonus'] = np.where(m.bom > 0.70, 0.06,
                              np.where(m.bom > 0.55, 0.03, 0.0))
inh = sum(terms.values())
other = m.additive.values - inh
print(f'继承罚项重算: residual(additive-inh) 唯一值样本 '
      f'{np.unique(np.round(other, 4))[:8]} (期望 0 和 {BP2_BOOST})')
print('  各项触发率: ' + ', '.join(f'{k}={np.mean(v != 0):.1%}' for k, v in terms.items()))

# ---------- 4. 基线/变体重排 ----------
days = sorted(m.date.unique())
base_sel = {}
for dt, gd in m.groupby('date', sort=True):
    k = int(gd.n_target.iloc[0])
    if k <= 0:
        continue
    top = gd.nlargest(k, 'eff_scored')
    base_sel[dt] = set(top.code)

# 实际选中 (engine)
act = m[m.selected == 1].groupby('date').code.apply(set).to_dict()


def evaluate(label, score_col, verbose=False):
    """重排并返回日级 fwd 差值 vs 基线 (仅共同日)."""
    rows = []
    for dt, gd in m.groupby('date', sort=True):
        bs = base_sel.get(dt)
        if not bs:
            continue
        k = len(bs)
        top = gd.nlargest(k, score_col)
        if score_col != 'eff_scored':
            aset = set(top.code)
            inter = len(aset & bs) / max(len(aset | bs), 1)
        else:
            aset = bs
            inter = 1.0
        # 只比较两边都有 fwd 的名字, 等权; 共同基准=基线集
        b = gd[gd.code.isin(bs)]
        a = top
        rows.append((dt, a.fwd5.mean() - b.fwd5.mean(),
                     a.fwd10.mean() - b.fwd10.mean(),
                     a.fwd20.mean() - b.fwd20.mean(),
                     a.fwd10.mean(), inter))
    r = pd.DataFrame(rows, columns=['date', 'd5', 'd10', 'd20', 'a10', 'jacc'])
    r['year'] = r.date.dt.year
    return r, label


def boot_ci(x, n=10000):
    if len(x) < 5:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    mu = x.values[idx].mean(axis=1)
    return float(np.quantile(mu, .025)), float(np.quantile(mu, .975))


def report(r, label):
    m10 = r.d10.mean()
    lo, hi = boot_ci(r.d10)
    # 逐年
    ys = r.groupby('year').d10.mean()
    same = (np.sign(ys) == np.sign(m10)).sum() if m10 != 0 else 0
    print(f'{label:22s} Δ5={r.d5.mean():+6.2%} Δ10={m10:+6.2%} '
          f'CI[{lo:+.2%},{hi:+.2%}] Δ20={r.d20.mean():+6.2%} '
          f'arm10={r.a10.mean():+6.2%} Jacc={r.jacc.mean():.2f} '
          f'年同向={same}/{len(ys)}')
    return dict(label=label, d10=m10, lo=lo, hi=hi, jacc=r.jacc.mean(),
                yearly=ys.to_dict())


print('\n=== 基线 vs 实际选中 (约束代价) ===')
rb, _ = evaluate('base', 'eff_scored')
rows = []
for dt, gd in m.groupby('date', sort=True):
    bs = base_sel.get(dt); ac = act.get(dt)
    if not bs or not ac:
        continue
    bb = gd[gd.code.isin(bs)]; aa = gd[gd.code.isin(ac)]
    rows.append((dt, aa.fwd10.mean() - bb.fwd10.mean(),
                 len(ac & bs) / max(len(ac | bs), 1)))
rr = pd.DataFrame(rows, columns=['date', 'd10', 'jacc']); rr['year'] = rr.date.dt.year
print(f'实际 vs 离线top-k: Δ10={rr.d10.mean():+.2%} Jacc={rr.jacc.mean():.2f} '
      f'(接近0/Jacc高 ⇒ 离线口径可信)')

print('\n=== 继承罚项整块 scale ===')
res = []
for s in (0.0, 0.5, 1.5, 2.0):
    m['_v'] = m.eff_scored - inh + s * inh
    r, lab = evaluate(f'inh_scale_{s}', '_v')
    res.append(report(r, f'inh_scale_{s}'))

print('\n=== 单项消融 (把该项贡献置0) ===')
for name, v in terms.items():
    m['_v'] = m.eff_scored - v
    r, lab = evaluate(f'drop_{name}', '_v')
    res.append(report(r, f'drop_{name}'))

print('\n=== bp2_boost 值 (已知最优0.15, 作为校准) ===')
for bv in (0.0, 0.10, 0.20):
    m['_v'] = m.eff_scored + (bv - BP2_BOOST) * (
        (m.bp == 2) & (m.chan_sell == 0)).astype(float)
    r, lab = evaluate(f'bp2_{bv}', '_v')
    res.append(report(r, f'bp2_boost_{bv}'))

out = pd.DataFrame(res)
out.to_csv('/mnt/d/quant/.arms/probe/penalty_counterfactual_1022.csv', index=False)
print('\n← .arms/probe/penalty_counterfactual_1022.csv')
