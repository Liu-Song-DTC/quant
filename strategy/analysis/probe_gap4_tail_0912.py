#!/usr/bin/env python3
"""2026-09-13 差距4尾部探针: ml_score在池内顶部的单调性/反转检验 (hump假说)

背景: 池内重排名器探针(NO否决)证池内秩标签阶段2无效 — parent模型池IC_f20=0.0668
已与全市场IC同阶, 标签工程到顶。但gap4敏感性校准留下一个未解释异常:
  ρ=0.065的良态排名器应拿+4.43% (top-N f20), 实测top_ml只有+0.79%;
  而top_score(ρ=0.036)实测+3.24%≈模拟+3.28% — score尾部良态, ml_score尾部畸形。
选股只取池内top-6/500 ≈ top 1.2% — 顶部1-2%的单调性直接决定选股结果。
本探针按ml_score/score的池内百分位分桶, 测每桶f20均值+桶内spearman:
  若顶桶均值显著低于次顶桶且桶内spearman为负 → 顶部反转确认 → 修法=尾部重标定
  (portfolio层effective_score尾部处理, 非再训练)。
只读。轻量。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap4_tail_0912.py > logs/probe_gap4_tail_0912.log 2>&1
"""
import os
import time
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
IDX = os.path.join(BT, 'sh000001_qfq.csv')
FWD = 20
END_OK = '2026-08-13'
BUCKETS = [(0, 50), (50, 80), (80, 90), (90, 95), (95, 98), (98, 98.8), (98.8, 100)]


def build_close():
    idx = pd.read_csv(IDX, usecols=['datetime'], parse_dates=['datetime'])
    idx = idx[(idx.datetime >= '2020-12-01') & (idx.datetime <= '2026-10-15')]
    D = idx['datetime'].values.astype('datetime64[ns]')
    T = len(D)
    codes = []
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        c = fn[:-len('_qfq.csv')]
        if c.startswith(('sh', 'sz')) or c.startswith(('4', '8', '92')):
            continue
        codes.append(c)
    colmap = {c: i for i, c in enumerate(codes)}
    print(f'[0] 日期 {T} 天 x 股票 {len(codes)} 只', flush=True)
    close = np.full((T, len(codes)), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, c in enumerate(codes):
        try:
            df = pd.read_csv(os.path.join(BT, f'{c}_qfq.csv'),
                             usecols=['datetime', 'close'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(D, dt)
        pos = pos[pos < T]
        if len(pos) == 0:
            continue
        close[pos, i] = df['close'].values[:len(pos)].astype(np.float32)
    print(f'[0] 加载 {time.time()-t0:.0f}s', flush=True)
    return D, codes, colmap, pd.DataFrame(close).ffill(axis=0).values


def main():
    t0 = time.time()
    D, codes, colmap, close = build_close()
    f20 = (pd.DataFrame(close).shift(-FWD) / pd.DataFrame(close) - 1).values

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'score', 'ml_score'],
                      dtype={'code': str})
    sig = sig[sig.buy == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[(sig.date >= '2021-04-01') & (sig.date <= END_OK)]
    tmap = {d: i for i, d in enumerate(pd.to_datetime(D))}
    sig = sig[sig.date.isin(tmap)]
    sig['ti'] = sig['date'].map(tmap)
    sig = sig[sig.code.isin(colmap)]
    sig['ci'] = sig['code'].map(colmap)
    sig['f20'] = f20[sig['ti'].values.astype(int), sig['ci'].values.astype(int)]
    sig = sig[np.isfinite(sig.f20)]
    print(f'[1] 池行 {len(sig)}, 日期 {sig.date.nunique()}', flush=True)

    # === 分桶均值 (逐日算桶均值再跨日平均, 与gap1口径一致) ===
    print(f'\n[2] 池内分桶 mean f20 (n日期={sig.date.nunique()})')
    print(f'  {"桶(百分位)":>14s} {"ml桶均值":>10s} {"score桶均值":>12s} {"桶内IC_ml":>10s}')
    for lo, hi in BUCKETS:
        ml_vals, sc_vals, ic_vals = [], [], []
        for d, g in sig.groupby('date'):
            rk_m = g.ml_score.rank(pct=True)
            rk_s = g.score.rank(pct=True)
            m = (rk_m >= lo / 100) & (rk_m < hi / 100)
            if m.sum() >= 3:
                ml_vals.append(g.f20[m].mean())
            m2 = (rk_s >= lo / 100) & (rk_s < hi / 100)
            if m2.sum() >= 3:
                sc_vals.append(g.f20[m2].mean())
            if m.sum() >= 10:
                v = g.loc[m, 'ml_score'].corr(g.loc[m, 'f20'], method='spearman')
                if np.isfinite(v):
                    ic_vals.append(v)
        lab = f'{lo}-{hi}%'
        print(f'  {lab:>14s} {np.mean(ml_vals)*100:+9.2f}% {np.mean(sc_vals)*100:+11.2f}% '
              f'{np.mean(ic_vals):+9.3f} (n={len(ic_vals)})')

    # === 顶部反转检验: top5%内逐日spearman(ml_score, f20) ===
    top_ic, sub_ic = [], []
    for d, g in sig.groupby('date'):
        rk_m = g.ml_score.rank(pct=True)
        t5 = g[rk_m >= 0.95]
        if len(t5) >= 10:
            v = t5.ml_score.corr(t5.f20, method='spearman')
            if np.isfinite(v):
                top_ic.append(v)
        t95 = g[(rk_m >= 0.90) & (rk_m < 0.95)]
        if len(t95) >= 10:
            v = t95.ml_score.corr(t95.f20, method='spearman')
            if np.isfinite(v):
                sub_ic.append(v)
    print(f'\n[3] 顶部反转检验 (top5%内 spearman(ml_score, f20))')
    print(f'  top5%桶内IC: mean={np.mean(top_ic):+.4f} 正比例='
          f'{(np.array(top_ic)>0).mean()*100:.0f}% n={len(top_ic)}')
    print(f'  90-95%桶内IC: mean={np.mean(sub_ic):+.4f} 正比例='
          f'{(np.array(sub_ic)>0).mean()*100:.0f}% n={len(sub_ic)}')

    # === 顶桶 vs 次顶桶逐日配对差 (ml vs score) ===
    print(f'\n[4] 顶桶(98.8-100) vs 次顶桶(98-98.8) 配对差')
    for col in ['ml_score', 'score']:
        diffs = []
        for d, g in sig.groupby('date'):
            rk = g[col].rank(pct=True)
            top = g.f20[(rk >= 98.8 / 100)].mean()
            sub = g.f20[(rk >= 98 / 100) & (rk < 98.8 / 100)].mean()
            if np.isfinite(top) and np.isfinite(sub):
                diffs.append(top - sub)
        diffs = np.array(diffs)
        print(f'  {col:>9s}: 顶-次顶 mean={diffs.mean()*100:+.2f}pp 中位='
              f'{np.median(diffs)*100:+.2f}pp 正比例={(diffs>0).mean()*100:.0f}% n={len(diffs)}')

    # === 逐年顶桶验证 (反转是否稳定) ===
    print(f'\n[5] 逐年: top2%桶 mean f20 (ml_score) vs 次顶桶')
    for y in sorted(sig.date.dt.year.unique()):
        top_vals, sub_vals = [], []
        for d, g in sig[sig.date.dt.year == y].groupby('date'):
            rk = g.ml_score.rank(pct=True)
            t = g.f20[rk >= 0.98].mean()
            s = g.f20[(rk >= 0.95) & (rk < 0.98)].mean()
            if np.isfinite(t):
                top_vals.append(t)
            if np.isfinite(s):
                sub_vals.append(s)
        print(f'  {y}: top2%桶={np.mean(top_vals)*100:+6.2f}% '
              f'95-98%桶={np.mean(sub_vals)*100:+6.2f}% n={len(top_vals)}')
    print(f'\n[总耗时 {time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
