"""评分链最后悬疑点闭合探针 (2026-09-22, 只读, 零生产写入)

前序 (probe_qweight_coverage_20260922.py): 交集概念可切换集上, 全局权重IC +0.0464
vs 季度 +0.0290, 配对差 −0.0175 (t=−5.4), 95%CI [−0.0313,−0.0030] 排除零 →
"P0中性/牛市切季度权重"已关闭。

本探针补三个悬疑点, 使评分链裁决完备:
  (A) λ-blend连续体响应面: score_λ=(1−λ)·score_g+λ·score_q, λ∈{0,.25,.5,.75,1}
      — 若权重向量均归一(sum≈1), 这正是两权重向量的凸组合连续体。
      只在内点存在优于端点时才值得重开"切换"方向 (非熊日, 即生产消费全局的域)。
  (B) 熊日原位裁决: 生产在 regime==−1 or trend_score<−0.05 日用季度bear_factors —
      这是季度标定在交集概念上唯一的live消费通道。反事实=同日用全局中性权重。
      若熊日原位 ic_qb ≤ ic_g → 熊分支切换本身也是负期望 (可操作: 简化分支读全局);
      若 ic_qb > ic_g → 熊分支获正证据背书, 方向以阳性闭合。
  (C) 牛日原位裁决: 生产在 regime==1 日用全局bull_factors。反事实=全局中性权重。
      同理裁决bull分支价值。

regime/trend_score 精确重建: bt_execution.py:592-631 同款 — 从冻结指数CSV
(sh000001_qfq/sh000852_qfq/399006_qfq, 数据态9/17) 经 MarketRegimeDetector().generate
生成, 窗口过滤 fromdate 2021-01-01 / todate 2026-09-17 与生产一致。
零写入: 只读CSV/parquet, 输出仅 /tmp/probe_qweight_blend_bear_20260922.csv。
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.market_regime_detector import MarketRegimeDetector  # noqa: E402

CACHE_F = '/mnt/d/quant/strategy/cache/factor_df_5190s_811d_85f064b9.parquet'
QD = '/mnt/d/quant/strategy/config/quarterly_factors'
GLOBAL_Y = '/mnt/d/quant/strategy/config/factor_config.yaml'
MAP_P = '/mnt/d/quant/data/stock_concept_map.pkl'
INCEP_P = '/mnt/d/quant/data/concept_inception.pkl'
BTD = '/mnt/d/quant/data/stock_data/backtrader_data'
FROMDATE = pd.Timestamp('2021-01-01')
TODATE = pd.Timestamp('2026-09-17')

STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
            '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
            '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
            '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
            '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
            '转债标的', '长江三角', '深圳特区', '破净股', '创投']


def qid_of(ds):
    d = pd.Timestamp(ds)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


def load_maps():
    with open(MAP_P, 'rb') as f:
        raw = pickle.load(f)
    cmap = {}
    for code, concepts in raw.items():
        filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
        if filtered:
            cmap[code] = filtered
    if os.path.exists(INCEP_P):
        with open(INCEP_P, 'rb') as f:
            raw_i = pickle.load(f)
        incep = {k: pd.Timestamp(v) for k, v in raw_i.items()}
    else:
        incep = {}
    return cmap, incep


def rebuild_regime():
    """bt_execution.py:592-631 同款重建 regime/trend_score"""
    idx = pd.read_csv(os.path.join(BTD, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    idx = idx[(idx['datetime'] >= FROMDATE) & (idx['datetime'] <= TODATE)]
    small = pd.read_csv(os.path.join(BTD, 'sh000852_qfq.csv'), parse_dates=['datetime'])
    growth = pd.read_csv(os.path.join(BTD, '399006_qfq.csv'), parse_dates=['datetime'])
    det = MarketRegimeDetector()
    det.generate(idx, small_cap_df=small, growth_df=growth)
    out = det.index_data.copy()
    out = out.set_index('datetime')
    assert {'regime', 'trend_score'} <= set(out.columns), out.columns.tolist()
    return out


def ic_by_date(d, col, min_n=20):
    def _ic(sub):
        if len(sub) < min_n or sub[col].nunique() < 5:
            return np.nan
        rho = spearmanr(sub[col], sub['future_ret']).correlation
        return rho if np.isfinite(rho) else np.nan
    return d.groupby('date').apply(_ic)


def report(label, ic_series):
    ics = ic_series.dropna()
    t = ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else 0.0
    print(f'  [{label}] n={len(ics)}, IC={ics.mean():+.5f}, t={t:.2f}, '
          f'正比例={100 * (ics > 0).mean():.1f}%')
    by = ics.groupby(ics.index.year).agg(ic=('mean'), n=('count'))
    print(by.to_string())
    return ics


def main():
    # ---- 1. regime重建 ----
    reg = rebuild_regime()
    bear_dates = set(reg.index[(reg['regime'] == -1) | (reg['trend_score'] < -0.05)])
    bull_dates = set(reg.index[reg['regime'] == 1])
    print(f'regime序列: {len(reg)}日; 熊/弱趋势日 {len(bear_dates)}, '
          f'牛市日 {len(bull_dates)}')
    print('熊日分年:', {yr: sum(1 for d in bear_dates if d.year == yr)
                      for yr in sorted({d.year for d in bear_dates})})
    print('牛日分年:', {yr: sum(1 for d in bull_dates if d.year == yr)
                      for yr in sorted({d.year for d in bull_dates})})

    # ---- 2. 配置加载 ----
    g = yaml.safe_load(open(GLOBAL_Y))['industry_factors']
    idx_cfg = yaml.safe_load(open(os.path.join(QD, 'index.yaml')))['quarters']
    qs = {qid: yaml.safe_load(open(os.path.join(QD, info['file'])))['industry_factors']
          for qid, info in idx_cfg.items()}
    cmap, incep = load_maps()

    # ---- 3. factor_df + 可切换集解析 (同前序探针) ----
    df = pd.read_parquet(CACHE_F)
    df['date'] = pd.to_datetime(df['date'])
    df['qid'] = df['date'].apply(qid_of)

    def resolve(row):
        code = row['code']
        qcfg = qs.get(row['qid'], {})
        concepts = cmap.get(code)
        if not concepts:
            return None
        ts = row['date']
        for c in concepts:
            _inc = incep.get(c)
            if _inc is not None and ts < _inc:
                continue
            if c in qcfg:
                return c if c in g else None
        return None

    df['key'] = df.apply(resolve, axis=1)
    n_switch = int(df['key'].notna().sum())
    print(f'\n可切换集: {n_switch} 行 ({100 * n_switch / len(df):.1f}%)')
    df = df[df['key'].notna()].copy()

    def score_row(r, which):
        key = r['key']
        if which == 'g':
            cfg = g.get(key, {})
            fk, wk = 'factors', 'weights'
        elif which == 'q':
            cfg = qs.get(r['qid'], {}).get(key, {})
            fk, wk = 'factors', 'weights'
        elif which == 'qb':
            cfg = qs.get(r['qid'], {}).get(key, {})
            fk, wk = 'bear_factors', 'bear_weights'
        else:  # 'gb' 全局bull
            cfg = g.get(key, {})
            fk, wk = 'bull_factors', 'bull_weights'
        if not cfg:
            return np.nan
        fl, wl = cfg.get(fk, []), cfg.get(wk, [])
        if not fl or len(fl) != len(wl):
            return np.nan
        s = 0.0
        for f, w in zip(fl, wl):
            if f in df.columns:
                v = r.get(f, np.nan)
                if pd.notna(v):
                    s += float(v) * w
        return s

    for col, which in [('score_g', 'g'), ('score_q', 'q'),
                       ('score_qb', 'qb'), ('score_gb', 'gb')]:
        df[col] = df.apply(lambda r: score_row(r, which), axis=1)
    df = df.dropna(subset=['score_g', 'score_q', 'future_ret'])
    print(f'双score可算: {len(df)} 行')

    # 权重归一断言 (blend=权重向量凸组合的前提)
    wsums = {}
    for key, cfg in g.items():
        if 'weights' in cfg and cfg.get('weights'):
            wsums[key] = sum(cfg['weights'])
    bad = [k for k, v in wsums.items() if abs(v - 1) > 0.01]
    print(f'全局权重sum≠1的概念数: {len(bad)} (n={len(wsums)})')

    # ---- 4. (A) λ-blend响应面 (非熊日) ----
    df['year'] = df['date'].dt.year
    nb = df[~df['date'].isin(bear_dates)].copy()
    print(f'\n=== A. λ-blend响应面 (非熊日, n={len(nb)}行, '
          f'{nb["date"].nunique()}日) ===')
    lam_ics = {}
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        nb['score_blend'] = (1 - lam) * nb['score_g'] + lam * nb['score_q']
        ic = ic_by_date(nb, 'score_blend')
        lam_ics[lam] = ic
        ics = ic.dropna()
        t = ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else 0.0
        print(f'  λ={lam:.2f}: n={len(ics)}, IC={ics.mean():+.5f}, t={t:.2f}, '
              f'正比例={100 * (ics > 0).mean():.1f}%')
    print('  → 单调性: 每步 λ↑ 的 IC 变化 =',
          [f'{lam_ics[l2].mean() - lam_ics[l1].mean():+.4f}'
           for l1, l2 in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]])

    # λ=0.5 vs λ=0 内点bump分类: 分年 + 块bootstrap CI
    d_blend = (lam_ics[0.5] - lam_ics[0.0]).dropna()
    print(f'\n  [λ=0.5内点bump vs λ=0] 配对差 n={len(d_blend)}, '
          f'mean={d_blend.mean():+.5f}, '
          f't={d_blend.mean() / d_blend.std() * np.sqrt(len(d_blend)):.2f}')
    print('  分年:', {yr: f'{sub.mean():+.4f}'
                    for yr, sub in d_blend.groupby(d_blend.index.year)})
    rng = np.random.default_rng(42)
    v = d_blend.values
    n, blk, B = len(v), 20, 2000
    nblk = int(np.ceil(n / blk))
    out = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, n - blk + 1, nblk)
        out[b] = np.concatenate([v[i:i + blk] for i in ii])[:n].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    print(f'  block BS(block=20,B=2000,seed=42): 95%CI [{lo:+.5f}, {hi:+.5f}], '
          f'P(mean>0)={(out > 0).mean():.4f}')
    pd.DataFrame({f'lam_{lam}': lam_ics[lam] for lam in lam_ics}).to_csv(
        '/tmp/probe_qweight_blend_20260922_lams.csv')

    # bump定位: 非熊日 = 严格中性日 ∪ 牛日 (生产在牛日消费bull权重, 非全局中性)
    def bump_bs(days_mask, label):
        sub = df[df['date'].isin(days_mask)].copy()
        ics = {}
        for lam in [0.0, 0.5]:
            sub['sb'] = (1 - lam) * sub['score_g'] + lam * sub['score_q']
            ics[lam] = ic_by_date(sub, 'sb')
        d = (ics[0.5] - ics[0.0]).dropna()
        if not len(d):
            print(f'  [{label}] 无有效日')
            return
        rng = np.random.default_rng(42)
        v = d.values
        n, blk, B = len(v), 20, 2000
        nblk = int(np.ceil(n / blk))
        out = np.empty(B)
        for b in range(B):
            ii = rng.integers(0, n - blk + 1, nblk)
            out[b] = np.concatenate([v[i:i + blk] for i in ii])[:n].mean()
        lo, hi = np.percentile(out, [2.5, 97.5])
        t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        print(f'  [{label}] λ0.5−λ0: n={len(d)}, mean={d.mean():+.5f}, t={t:.2f}, '
              f'95%CI [{lo:+.5f},{hi:+.5f}], P(>0)={(out > 0).mean():.3f}')
        print('   分年:', {yr: f'{sub.mean():+.4f}'
                        for yr, sub in d.groupby(d.index.year)})

    neutral_days = set(reg.index) - bear_dates - bull_dates
    bump_bs(neutral_days, '严格中性日')
    bump_bs(bull_dates, '牛日(生产消费bull)')

    # ---- 5. (B) 熊日原位 ----
    bd = df[df['date'].isin(bear_dates)].copy()
    print(f'\n=== B. 熊日原位裁决 (n={len(bd)}行, {bd["date"].nunique()}日) ===')
    if len(bd):
        ic_qb = report('生产=季度bear', ic_by_date(bd, 'score_qb'))
        ic_gb_ = report('反事实=全局中性', ic_by_date(bd, 'score_g'))
        ic_qb_ = report('参考=季度中性', ic_by_date(bd, 'score_q'))
        d1 = (ic_qb - ic_gb_).dropna()
        t1 = d1.mean() / d1.std() * np.sqrt(len(d1)) if len(d1) and d1.std() > 0 else 0.0
        print(f'  配对差 季度bear−全局中性: n={len(d1)}, mean={d1.mean():+.5f}, '
              f't={t1:.2f}, 正比例={100 * (d1 > 0).mean():.1f}%')
        print('  分年配对差:',
              {yr: f'{sub.mean():+.4f}'
               for yr, sub in d1.groupby(d1.index.year)})
        pd.DataFrame({'bear_ic_qb': ic_qb, 'bear_ic_g': ic_gb_,
                      'bear_ic_q': ic_qb_}).to_csv(
            '/tmp/probe_qweight_blend_bear_20260922_bear.csv')

    # ---- 6. (C) 牛日原位 ----
    bu = df[df['date'].isin(bull_dates)].copy()
    print(f'\n=== C. 牛日原位裁决 (n={len(bu)}行, {bu["date"].nunique()}日) ===')
    if len(bu):
        ic_bull = report('生产=全局bull', ic_by_date(bu, 'score_gb'))
        ic_neu = report('反事实=全局中性', ic_by_date(bu, 'score_g'))
        d2 = (ic_bull - ic_neu).dropna()
        t2 = d2.mean() / d2.std() * np.sqrt(len(d2)) if len(d2) and d2.std() > 0 else 0.0
        print(f'  配对差 全局bull−全局中性: n={len(d2)}, mean={d2.mean():+.5f}, '
              f't={t2:.2f}, 正比例={100 * (d2 > 0).mean():.1f}%')
        print('  分年配对差:',
              {yr: f'{sub.mean():+.4f}'
               for yr, sub in d2.groupby(d2.index.year)})
        pd.DataFrame({'bull_ic_gb': ic_bull, 'bull_ic_g': ic_neu}).to_csv(
            '/tmp/probe_qweight_blend_bear_20260922_bull.csv')

    print('\n探针完成 — 只读, 零生产写入。')


if __name__ == '__main__':
    main()
