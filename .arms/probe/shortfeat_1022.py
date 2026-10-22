#!/usr/bin/env python3
"""短周期买入胜率特征扫描 (2026-10-22, 只读).

目标: 在 new_entries_cap=2 新锚 (1,098,168.62) 下, 找**可再提高 w5 胜率**的
"跳过"规则. 只扫描选股时点已知的池列, 不做名次重排 (R2 已是最强桶).

样本: `.arms/pool_1022.csv` (6,053 行 / 156 调仓日, 行序=引擎顺序).
  主样本 = 非持仓行序 nr<=2 (cap2 会考虑的候选, ~300 行)
  副样本 = nr<=3 (含被 cap2 砍掉的第3名)
前瞻: 次日开盘入场 → +H 日收盘 (与回测执行同构, 不扣成本).

统计: 日级簇 bootstrap (156 调仓日, 5000 次) — 同日候选高度相关, 行级检验会高估显著性.
  - 连续特征: 三分位 hi/lo 的 w5 胜率差; 另做 nr 内绝对值标准化后再分位 (剔除名次代理)
  - 二元/类别特征: 各水平 vs 其余
  - 多重检验: BH (仅对可操作特征, 不含 year/half 等描述量)
判据: |Δw5|>=8pp 且 p_bh<=0.10 且 w20 同向 (不劣化) 才进候选.

用法: python shortfeat_1022.py [nr_max]   (默认 2)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from winrate_diag_1022 import add_forward

POOL = '/mnt/d/quant/.arms/pool_1022.csv'
EQ = '/mnt/d/quant/strategy/rolling_validation_results/equity_curve.csv.base_tr1022'
NRMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
RNG = np.random.default_rng(20261022)
NBOOT = 5000

# 选股时点已知的连续/有序池列 (排除 date/code/selected/weight/is_held 与结果列)
NUM = ['nr', 'score', 'rank_pct', 'eff_score', 'rank_term', 'multiplier', 'additive',
       'turnover', 'repl_buffer', 'mom_adj', 'no_chan_penalty', 'chase_penalty',
       'ml_gate_penalty', 'divnone_penalty', 'tt1chase_penalty', 'ah_penalty',
       'bp_c4_penalty', 'lowsh_penalty', 'bp', 'sl', 'bp_confirm', 'chan_sell',
       'mom_60d', 'dist_ma60', 'volume_ratio', 'daily_ret', 'trend_type',
       'stroke_phase', 'exhaustion', 'div_strength', 'gap_breakout',
       'profit_declining', 'bom', 'bi_buy_point', 'pivot_zg', 'price',
       'n_target', 'target_exposure', 'gap', 'mom60', 'vol20']
CAT = ['regime', 'industry', 'trend_type', 'stroke_phase', 'div_type', 'bp',
       'bi_buy_point', 'gap_breakout', 'exhaustion', 'bom', 'profit_declining']
DESC_ONLY = {'year', 'half', 'nr'}  # 不进 BH 池


def load():
    p = pd.read_csv(POOL, dtype={'code': str})
    p['code'] = p['code'].str.zfill(6)
    p['date'] = pd.to_datetime(p['date'])
    p['nr'] = np.nan
    nh = p[p['is_held'] == 0]
    p.loc[nh.index, 'nr'] = nh.groupby('date').cumcount() + 1
    eq = pd.read_csv(EQ, parse_dates=['date'])
    cal = list(eq['date'])
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    p['exec_date'] = p['date'].map(nxt)
    p['year'] = p['date'].dt.year
    p['half'] = np.where(p['year'] <= 2023, 'H1', 'H2')
    # add_forward 会写同名 price (信号日收盘=池 price); 先去重避免 concat 产生重复列
    p = p.drop(columns=[c for c in ('price',) if c in p.columns])
    p = add_forward(p)
    p = p.loc[:, ~p.columns.duplicated()]
    return p


def wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


class ClusterBoot:
    """按调仓日重采样, 比较两组二值结局的胜率差."""

    def __init__(self, days):
        self.days = list(days)
        self.di = {d: i for i, d in enumerate(self.days)}
        self.D = len(self.days)
        self.B = RNG.integers(0, self.D, size=(NBOOT, self.D))

    def diff(self, df, mask_a, mask_b, col):
        """返回 (point_pp, lo, hi, p_two_sided)."""
        d = df.loc[df[col].notna(), ['date', col]].copy()
        a = mask_a.reindex(d.index).fillna(False).values
        b = mask_b.reindex(d.index).fillna(False).values
        y = (d[col].values > 0).astype(float)
        day = d['date'].map(self.di).values
        na = np.bincount(day[a], minlength=self.D)
        ka = np.bincount(day[a], weights=y[a], minlength=self.D)
        nb = np.bincount(day[b], minlength=self.D)
        kb = np.bincount(day[b], weights=y[b], minlength=self.D)
        if na.sum() < 5 or nb.sum() < 5:
            return (np.nan,) * 4
        point = ((ka.sum() / na.sum()) - (kb.sum() / nb.sum())) * 100
        sa = self.B @ ka; sna = self.B @ na
        sb = self.B @ kb; snb = self.B @ nb
        with np.errstate(divide='ignore', invalid='ignore'):
            bd = (sa / sna - sb / snb) * 100
        bd = bd[np.isfinite(bd)]
        if not len(bd):
            return (point, np.nan, np.nan, np.nan)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p = 2 * min((bd >= 0).mean(), (bd <= 0).mean())
        return (point, lo, hi, min(1.0, p))


def terciles(s, col):
    x = s[col].astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    try:
        q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    except Exception:
        return None, None
    if not np.isfinite(q1) or q1 == q2:
        return None, None
    return x <= q1, x >= q2


def zwithin(s, col, nr):
    x = s[col].astype(float).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=s.index)
    for _, idx in s.groupby(nr).groups.items():
        v = x.loc[idx]
        sd = v.std()
        out.loc[idx] = (v - v.mean()) / sd if sd and np.isfinite(sd) else 0.0
    return out


def bh(pvals):
    """Benjamini-Hochberg q values, NaN 保留."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    n = ok.sum()
    if not n:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def main():
    p = load()
    q = p[(p['nr'].notna()) & (p['nr'] <= NRMAX)].copy()
    print(f'池 {POOL}; 非持仓 nr<={NRMAX}: n={len(q)} 日={q["date"].nunique()} '
          f'区间 {q["date"].min().date()}..{q["date"].max().date()}')
    for h in (1, 3, 5, 20):
        v = q[f'w{h}'].dropna()
        lo, hi = wilson(int(v.sum()), len(v))
        print(f'  基线 w{h}: n={len(v)} win={v.mean()*100:.2f}% CI[{lo:.0f},{hi:.0f}] '
              f'mean={q[f"r{h}"].mean()*100:+.2f}%')
    cb = ClusterBoot(sorted(q['date'].unique()))
    rows = []

    print('\n' + '=' * 110)
    print('A. 连续/有序特征: 三分位 hi vs lo (w5 主判据, w20 护栏)')
    print(f'{"feature":22s} {"n":>4s} {"rho":>6s} {"lo%":>6s} {"hi%":>6s} '
          f'{"Δw5pp":>7s} {"CI":>16s} {"p":>7s} {"q":>7s} {"Δw20pp":>8s} {"p20":>7s} '
          f'{"zΔw5":>7s} {"zp":>7s}')
    for col in NUM:
        if col not in q.columns:
            continue
        lo_m, hi_m = terciles(q, col)
        if lo_m is None:
            continue
        n = int((lo_m.fillna(False) | hi_m.fillna(False)).sum())
        if n < 30:
            continue
        d5, l5, h5, p5 = cb.diff(q, hi_m, lo_m, 'w5')
        d20, _, _, p20 = cb.diff(q, hi_m, lo_m, 'w20')
        w5lo = q.loc[lo_m.fillna(False), 'w5'].mean() * 100
        w5hi = q.loc[hi_m.fillna(False), 'w5'].mean() * 100
        rho = q[[col, 'w5']].dropna().corr(method='spearman').iloc[0, 1]
        # 名次内标准化后再分位 (剔除 rank 代理)
        z = zwithin(q, col, 'nr')
        qz = q.assign(_z=z)
        lo_z, hi_z = terciles(qz, '_z')
        zd, zl, zh, zp = cb.diff(qz, hi_z, lo_z, 'w5') if lo_z is not None else (np.nan,) * 4
        rows.append(dict(kind='num', feature=col, n=n, rho=rho,
                         lo=w5lo, hi=w5hi, d5=d5, p5=p5, d20=d20, p20=p20,
                         zd=zd, zp=zp))
        print(f'{col:22s} {n:4d} {rho:+6.2f} {w5lo:6.1f} {w5hi:6.1f} '
              f'{d5:+7.1f} [{l5:+.1f},{h5:+.1f}] {p5:7.4f} {"":>7s} '
              f'{d20:+8.1f} {p20:7.4f} {zd:+7.1f} {zp:7.4f}')

    print('\n' + '=' * 110)
    print('B. 类别特征: 各水平 vs 其余 (n>=20; w5 主判据)')
    for col in CAT:
        if col not in q.columns:
            continue
        vc = q[col].astype(str).value_counts()
        cand = [lv for lv, n in vc.items() if 20 <= n < 0.8 * len(q)]
        if not cand:
            continue
        print(f'  [{col}] 水平 {dict((lv, int(vc[lv])) for lv in cand)}')
        for lv in cand[:12]:
            m_lv = q[col].astype(str) == lv
            m_rest = ~m_lv
            d5, l5, h5, p5 = cb.diff(q, m_lv, m_rest, 'w5')
            d20, _, _, p20 = cb.diff(q, m_lv, m_rest, 'w20')
            w5 = q.loc[m_lv, 'w5'].mean() * 100
            w20 = q.loc[m_lv, 'w20'].mean() * 100
            rows.append(dict(kind='cat', feature=f'{col}={lv}', n=int(m_lv.sum()),
                             rho=np.nan, lo=w5, hi=np.nan, d5=d5, p5=p5,
                             d20=d20, p20=p20, zd=np.nan, zp=np.nan))
            print(f'    {lv[:24]:24s} n={int(m_lv.sum()):3d} w5={w5:5.1f}% '
                  f'Δ={d5:+6.1f}pp [{l5:+.1f},{h5:+.1f}] p={p5:.4f} | '
                  f'w20={w20:5.1f}% Δ20={d20:+6.1f} p={p20:.4f}')

    r = pd.DataFrame(rows)
    # BH: 只对可操作特征 (连续 + 类别水平); 去掉 year/half (不在 NUM/CAT)
    r['q5'] = bh(r['p5'].values)
    r['q20'] = bh(r['p20'].values)

    print('\n' + '=' * 110)
    print('C. 候选短名单: |Δw5|>=8pp 且 q5<=0.10 且 (w20 同号或 p20>0.5)')
    short = r[(r['d5'].abs() >= 8) & (r['q5'] <= 0.10) &
              ((np.sign(r['d20']) == np.sign(r['d5'])) | (r['p20'] > 0.5))]
    if short.empty:
        print('  (空 — 本池/本样本没有过线的可操作单特征)')
    else:
        cols = ['feature', 'n', 'lo', 'hi', 'd5', 'p5', 'q5', 'd20', 'p20', 'zd', 'zp']
        print(short[cols].sort_values('d5').to_string(index=False))
    print('\nD. 全表按 q5 排序 (前 15):')
    cols = ['kind', 'feature', 'n', 'd5', 'p5', 'q5', 'd20', 'p20', 'zd', 'zp']
    print(r.dropna(subset=['q5']).sort_values('q5')[cols].head(15).to_string(index=False))
    r.to_csv('/mnt/d/quant/.arms/armout/shortfeat_1022.csv', index=False)
    print('\n落盘: .arms/armout/shortfeat_1022.csv')


if __name__ == '__main__':
    main()
