#!/usr/bin/env python3
"""继承罚项的 Fama-MacBeth 校准检验 (2026-10-22).

与 penalty_counterfactual_1022.py 互补:
  - 反事实 top-k 重排受引擎机制(持仓锁/bp2槽/行业上限)扰动;
  - 本脚本做**机制无关**的日级截面回归, 逐项问:
        fwd10 ~ 1 + trig + base      (base = eff_scored − 该项贡献)
    b_trig = 触发该罚项的名字在扣掉其余评分后, 实际多/少赚多少;
    λ     = b_base = 每 1 分 eff_score 对应的 fwd10;
    该项设定值 c 隐含的收益差 = c·λ。
  判据: b_trig 的日级 bootstrap CI 排除 0 且逐年同号 → 该项有独立信息;
        若 b_trig 与 c·λ 差距大 → 定价错(过罚/欠罚/方向反)。
  仅量化, 不改生产。

用法: .venv/bin/python .arms/probe/penalty_fm_1022.py [pool.csv]
"""
import sys
import numpy as np
import pandas as pd

POOL = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/quant/.arms/pool_1022.csv'
PX = '/mnt/d/quant/data/research/qfq_kline.parquet'
RNG = np.random.default_rng(20261023)
MIN_ROWS = 30

B3_VOL_PEN = -0.06

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

print('读价格镜像 ...')
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

div = m.div_type.fillna('').str.lower()
is_top = div.str.contains('top')
is_bottom = div.str.contains('bottom')
# (名字, 触发条件, 设定分值) — 顺序/系数与 portfolio.py L1793-1892 一致
SPEC = [
    ('top_div',        is_top.values,                                         -0.08),
    ('profit_decl',    (m.profit_declining == 1).values,                      -0.10),
    ('vol_down',       ((m.daily_ret < -0.015) & (m.volume_ratio > 0.5)).values, -0.06),
    ('limit_chase',    (m.daily_ret > 0.09).values,                           -0.05),
    ('exhaustion',     (m.exhaustion > 0.5).values,                           -0.10),
    ('bp1_nobottom',   ((m.bp == 1) & ~is_bottom).values,                     -0.06),
    ('downtrend_accel', ((m.trend_type == -2) & (m.stroke_phase > 0.2)
                         & ~((m.bp == 3) & (m.sl >= 2))).values,              -0.05),
    ('weak_drift',     ((m.trend_type != 2) & (m.stroke_phase > 0.2)
                        & (m.daily_ret < 0)).values,                          -0.04),
    ('extreme_chase',  ((m.mom_60d > 0.50) & (m.bp == 0)).values,             -0.06),
    ('dist45',         ((m.dist_ma60 > 0.45) & (m.bp == 0)).values,           -0.06),
    ('b3_vol',         ((m.bp == 3) & (m.volume_ratio > -0.30)
                        & (m.volume_ratio <= -0.20)).values,                  B3_VOL_PEN),
    ('bp1_bonus',      ((m.bp == 1) & (m.sl >= 1)).values,                     0.08),
    ('bp2_bonus',      ((m.bp == 2) & (m.sl >= 2)).values,                     0.03),
    ('bom_bonus',      (m.bom > 0.70) | ((m.bom > 0.55) & (m.bom <= 0.70)),    0.045),
]
# bom 实际为 0.06/0.03 两档, 用均值 0.045 仅用于 base 去项; 单独处理:
bom_val = np.where(m.bom > 0.70, 0.06, np.where(m.bom > 0.55, 0.03, 0.0))
SPEC = [s for s in SPEC if s[0] != 'bom_bonus']
SPEC.append(('bom_bonus', (bom_val != 0), np.nan))  # c 用逐行值, 见下

dates = m.date.values
y_all = m.fwd10.values


def day_coef(idx, trig, base):
    """单日 OLS: y ~ 1 + trig + base. 返回 (b_trig, b_base) 或 None."""
    if len(idx) < MIN_ROWS:
        return None
    t = trig[idx]
    if t.min() == t.max():      # 当日无变异
        return None
    X = np.column_stack([np.ones(len(idx)), t.astype(float), base[idx]])
    y = y_all[idx]
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[1], beta[2]


def boot_ci(x, n=10000):
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    mu = x[idx].mean(axis=1)
    return float(np.quantile(mu, .025)), float(np.quantile(mu, .975))


order = np.argsort(dates, kind='stable')
day_groups = {}
for i in order:
    day_groups.setdefault(dates[i], []).append(i)

rows = []
for name, trig_raw, coef in SPEC:
    trig = np.asarray(trig_raw, dtype=float)
    if name == 'bom_bonus':
        term_val = bom_val
        coef_eff = float(np.mean(bom_val[trig == 1])) if trig.sum() else np.nan
    else:
        term_val = trig * coef
        coef_eff = coef
    base = m.eff_scored.values - term_val
    bt, bb, years = [], [], []
    for dt, idx in day_groups.items():
        idx = np.asarray(idx)
        r = day_coef(idx, trig, base)
        if r is None:
            continue
        bt.append(r[0]); bb.append(r[1]); years.append(pd.Timestamp(dt).year)
    bt = np.asarray(bt); bb = np.asarray(bb); years = np.asarray(years)
    if len(bt) < 20:
        print(f'{name:16s} n_days={len(bt)} 太少, 跳过'); continue
    mean_b = bt.mean()
    se = bt.std(ddof=1) / np.sqrt(len(bt))
    lo, hi = boot_ci(bt)
    lam = float(np.median(bb))
    implied = coef_eff * lam if np.isfinite(coef_eff) else np.nan
    ys = pd.Series(bt).groupby(years).mean()
    same = int((np.sign(ys) == np.sign(mean_b)).sum()) if mean_b != 0 else 0
    rows.append(dict(term=name, n_days=len(bt), trig_rate=float((trig == 1).mean()),
                     coef=coef_eff, b_trig=mean_b, t=mean_b / se if se > 0 else np.nan,
                     ci_lo=lo, ci_hi=hi, lam_med=lam, implied=implied,
                     gap=mean_b - implied if np.isfinite(implied) else np.nan,
                     yearly_same=f'{same}/{len(ys)}',
                     yearly=';'.join(f'{y}:{v:+.4f}' for y, v in ys.items())))
    print(f'{name:16s} 天数={len(bt):3d} 触发={rows[-1]["trig_rate"]:6.1%} '
          f'c={coef_eff:+.3f} b_trig={mean_b:+.4f} CI[{lo:+.4f},{hi:+.4f}] '
          f't={rows[-1]["t"]:+.2f} λ={lam:+.4f} 隐含={implied:+.4f} '
          f'缺口={rows[-1]["gap"]:+.4f} 年同向={same}/{len(ys)}')

out = pd.DataFrame(rows)
path = '/mnt/d/quant/.arms/probe/penalty_fm_1022.csv'
out.to_csv(path, index=False)
print(f'\n← {path}')
print('\nλ 为每日截面 b_base 的中位数: eff_score 每 +1 分 ≈ fwd10 +λ.')
print('判读: |b_trig| CI 排除 0 ⇒ 该项有独立信息; gap≈0 ⇒ 定价正确; '
      'gap 与 c 反号 ⇒ 罚项在惩罚好票.')
