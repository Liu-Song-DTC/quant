#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M5探针 (2026-09-20): multi_strategy权重臂前置检验。
机制: portfolio.build选股后处理, compute_weights给每个候选 multiplier=
trend_w×trend + rev_w×rev + def_w×def, 乘到effective_score再重排; mult≤0.05
强制排尾。**注意: bear regime使用硬编码0.3/0.2/0.5, 权重bracket只影响
bull/neutral日**。
本探针: 对每个选股日, 取当日buy信号集(生产候选上界), 比较生产权重(0.4/0.3/0.3)
与变体(V1=0.5/0.3/0.2趋势倾斜, V2=0.33/0.33/0.33等权)的multiplier:
  a) |Δmult|分布; b) mult排序重排幅度(Kendall τ / 相邻交换距离);
  c) 穿越0.05抹杀线或top-6边界的候选日数。
若排名几乎不动(|Δmult|中位<0.02且τ>0.95) → 权重臂=噪声, M5关闭不花贵成本。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS = os.path.join(BASE, 'arms_20260919', '_baseline_sig', 'backtest_signals.csv')
REG = os.path.join(BASE, 'arms_20260919', 'C1_faithful_0_55', 'pre_regime_state.csv')
SEL = os.path.join(BASE, 'arms_20260919', 'C1_faithful_0_55', 'pre_portfolio_selections.csv')

USE = ['date', 'code', 'buy', 'chan_buy_point', 'chan_sell_point', 'signal_level',
       'trend_type', 'exhaustion_risk']
W_PROD = (0.4, 0.3, 0.3)
W_V1 = (0.5, 0.3, 0.2)
W_V2 = (1/3, 1/3, 1/3)
MIN_CHAN = 2
MAX_EXH = 0.30


def mult(bp, cs, sl, tt, ex, w, regime_tilt):
    if regime_tilt == 'defensive':  # 硬编码, 权重变体无效
        tw, rw, dw = 0.3, 0.2, 0.5
    else:
        tw, rw, dw = w
    trend = 0.5
    if sl >= MIN_CHAN or bp >= 2:
        trend = min(1.0, 0.5 + 0.25 * sl + 0.15 * (tt == 2))
    elif bp == 1 and sl >= 1:
        trend = 0.7
    if ex <= MAX_EXH:
        rev = 0.6 + 0.4 * (1.0 - ex / max(MAX_EXH, 0.01))
    elif bp == 1:
        rev = 0.6
    elif ex <= 0.15:
        rev = 0.7
    else:
        rev = 0.3
    dfn = 1.0
    if cs >= 3:
        dfn = 0.0
    elif cs >= 2:
        dfn = 0.3
    elif ex > 0.4:
        dfn = 0.3
    elif cs == 1:
        dfn = 0.5
    elif tt == -2:
        dfn = 0.4
    return float(np.clip(tw * trend + rw * rev + dw * dfn, 0.0, 1.5))


def main():
    reg = pd.read_csv(REG, parse_dates=['date'])[['date', 'regime']]
    reg = reg.dropna(subset=['regime'])
    reg['regime'] = reg['regime'].map({'NORM': 0, 'FAST': 1, 'BEAR': -1}).fillna(0).astype(int)
    sel = pd.read_csv(SEL, parse_dates=['date'])
    sel['code'] = sel['code'].astype(str).str.zfill(6)
    sel_dates = set(sel['date'])
    print(f'选股日: {len(sel_dates)}')

    # 单遍扫signals, 只留选股日的buy行
    rows = []
    for chunk in pd.read_csv(SIGNALS, usecols=USE, dtype={'code': str}, chunksize=4_000_000):
        cd = pd.to_datetime(chunk['date']).dt.normalize()
        m = cd.isin(sel_dates) & chunk['buy'].astype(str).str.lower().eq('true')
        if m.any():
            sub = chunk.loc[m].copy()
            sub['date'] = cd[m]
            for c in ['chan_buy_point', 'chan_sell_point', 'signal_level', 'trend_type',
                      'exhaustion_risk']:
                sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0)
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    print(f'选股日buy行: {len(df)}, 日期数: {df["date"].nunique()}')

    # 每日regime → tilt (1/0=bull/neutral→config权重, -1=bear→硬编码)
    reg_map = dict(zip(reg['date'], reg['regime']))
    df['regime'] = df['date'].map(reg_map).fillna(0).astype(int)
    df['tilt'] = np.where(df['regime'] == -1, 'defensive', 'config')
    bear_rows = (df['tilt'] == 'defensive').sum()
    print(f'bear日候选行: {bear_rows} ({bear_rows/len(df)*100:.1f}%) — 权重bracket对它们无效')

    d = df[df['tilt'] == 'config'].copy()
    mp = d.apply(lambda r: mult(r['chan_buy_point'], r['chan_sell_point'],
                                r['signal_level'], r['trend_type'],
                                r['exhaustion_risk'], W_PROD, r['tilt']), axis=1)
    mv1 = d.apply(lambda r: mult(r['chan_buy_point'], r['chan_sell_point'],
                                 r['signal_level'], r['trend_type'],
                                 r['exhaustion_risk'], W_V1, r['tilt']), axis=1)
    mv2 = d.apply(lambda r: mult(r['chan_buy_point'], r['chan_sell_point'],
                                 r['signal_level'], r['trend_type'],
                                 r['exhaustion_risk'], W_V2, r['tilt']), axis=1)
    mp.name, mv1.name, mv2.name = 'mp', 'mv1', 'mv2'
    d['mp'], d['mv1'], d['mv2'] = mp, mv1, mv2

    print('\n=== multiplier分布(生产 vs 变体, config日) ===')
    print(f"生产: p50 {mp.median():.3f} p10 {mp.quantile(.1):.3f} p90 {mp.quantile(.9):.3f} "
          f"非1.0比例 {(mp!=1.0).mean()*100:.1f}%")
    for nm, mv in [('V1 0.5/0.3/0.2', mv1), ('V2 等权', mv2)]:
        dm = (mv - mp).abs()
        print(f"{nm}: |Δmult| p50 {dm.median():.3f} p90 {dm.quantile(.9):.3f} "
              f"max {dm.max():.3f}  超0.05: {(dm>0.05).mean()*100:.1f}%")
        # 抹杀线穿越
        kill = ((mp <= 0.05) != (mv <= 0.05)).sum()
        print(f"  0.05抹杀线穿越: {kill} 候选日")

    # 排名稳定性: 每日按effective_score近似(用mult排序, 每日top-k=min(6, n)边界)
    print('\n=== 每日mult排序: 生产top6 vs 变体top6 重叠 ===')
    for nm, mv in [('V1', mv1), ('V2', mv2)]:
        ovs = []
        for day, g in d.groupby('date'):
            k = min(6, len(g))
            tp = set(g.nlargest(k, 'mp')['code'])
            tv = set(g.nlargest(k, mv.name)['code'])
            ovs.append(len(tp & tv) / max(k, 1))
        print(f"  {nm}: top6重叠均值 {np.mean(ovs):.2f} "
              f"(完全不变={np.mean([1 if o == 1 else 0 for o in ovs]) * 100:.0f}%的日)")

    # 生产选择集对这些候选的覆盖度: 生产选中的code有多少落在buy候选集
    sel_codes = set(sel['code'].str.zfill(6))
    d['code'] = d['code'].str.zfill(6)
    cov = d.groupby('date').apply(lambda g: len(set(g['code']) & sel_codes), include_groups=False)
    print(f'\n生产选中集与当日buy候选集交集: 均值 {cov.mean():.1f} 只/日 (上下文: 选择~5-6只)')


if __name__ == '__main__':
    main()
