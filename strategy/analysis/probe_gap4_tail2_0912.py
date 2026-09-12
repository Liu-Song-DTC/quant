#!/usr/bin/env python3
"""2026-09-13 差距4尾部探针2: 复合score尾部健康度 + ML blend在顶部的净贡献

背景: probe_gap4_tail证明 — 复合score(CSV 'score'列=signal_engine 697-698行
  score=gate_score=result['adjusted_score'], 已含BOM乘数+0.4 ML blend)顶部良态:
  顶1.2%桶f20=+3.51%且配对+0.96pp(55%正); 而独立ml_score顶部平坦(配对-0.10pp),
  其IC集中在底部桶(0-50%: +0.070)→ ml是避雷器不是选美裁判。
  残留问题: 0.4 blend加进复合score后, 对顶部排序是净帮助还是净伤害?
  若blend伤害顶部 → 修法=降blend权重(冷跑裁决); 若中性/帮助 → 差距4关闭。

方法: 用CSV列重构无blend分数 U = (score/bom − 0.4·ml·I)/0.6 (active), = score/bom
  (inactive), bom = 0.7+0.6·bom_quality_score。blend前后在截面内的唯一排序差异=blend项
  (alt_market逐日常数秩不变, dt_sig股级罕见)。对比 top-N by U vs top-N by score。
判定(跑前定): U-top6 − score-top6 ≥ +0.10pp 且 正比例≥55% 且 ≥4/6年稳定 → blend
  伤害顶部(下一步降权重冷跑); 否则blend中性/有益 → 差距4关闭。
只读。轻量。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap4_tail2_0912.py > logs/probe_gap4_tail2_0912.log 2>&1
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
BLEND_W = 0.4
NS = [3, 6, 10]


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

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'score', 'ml_score',
                                    'bom_quality_score'],
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

    # 无blend重构 U (engine 1551-1552: adjusted = 0.6u + 0.4ml, active=|ml|>0.01;
    #  1693: adjusted *= (0.7+0.6*bom_score) 在blend之后 → 先除bom再解blend)
    bom = (0.7 + 0.6 * sig.bom_quality_score.fillna(0.3)).values
    sc_adj = sig.score.values / bom
    ml = sig.ml_score.fillna(0.0).values
    active = np.abs(ml) > 0.01
    U = np.where(active, (sc_adj - BLEND_W * ml) / (1 - BLEND_W), sc_adj)
    sig['U'] = U
    print(f'[1] ml active行占比 {(active).mean()*100:.0f}%', flush=True)

    # === [2] 分桶IC: composite(score) + ml + U ===
    print(f'\n[2] 分桶IC spearman(x, f20) (n日期={sig.date.nunique()})')
    print(f'  {"桶":>12s} {"IC_score":>10s} {"IC_ml":>10s} {"IC_U":>10s}')
    for lo, hi in BUCKETS:
        ic_s, ic_m, ic_u = [], [], []
        for d, g in sig.groupby('date'):
            for col, lst in [('score', ic_s), ('ml_score', ic_m), ('U', ic_u)]:
                rk = g[col].rank(pct=True)
                m = (rk >= lo / 100) & (rk < hi / 100)
                if m.sum() >= 10:
                    v = g.loc[m, col].corr(g.loc[m, 'f20'], method='spearman')
                    if np.isfinite(v):
                        lst.append(v)
        print(f'  {f"{lo}-{hi}%":>12s} {np.mean(ic_s):+10.3f} '
              f'{np.mean(ic_m):+10.3f} {np.mean(ic_u):+10.3f}')

    # === [3] top-N 对比: score vs U vs ml (逐日配对) ===
    print(f'\n[3] top-N mean f20 配对差 (vs 复合score)')
    print(f'  {"N":>3s} {"score":>9s} {"U":>9s} {"ml":>9s} '
          f'{"U-score":>9s} {"正比%":>6s} {"ml-score":>9s} {"正比%":>6s}')
    for N in NS:
        d_score, d_u, d_ml = [], [], []
        for d, g in sig.groupby('date'):
            if len(g) < max(10, N):
                continue
            top_s = g.nlargest(N, 'score').f20.mean()
            top_u = g.nlargest(N, 'U').f20.mean()
            top_m = g.nlargest(N, 'ml_score').f20.mean()
            if np.isfinite(top_s):
                d_score.append(top_s)
                d_u.append(top_u)
                d_ml.append(top_m)
        ds = np.mean(d_score)
        du = np.mean(d_u)
        dm = np.mean(d_ml)
        du_diff = np.array(d_u) - np.array(d_score)
        dm_diff = np.array(d_ml) - np.array(d_score)
        print(f'  {N:>3d} {ds*100:+8.2f}% {du*100:+8.2f}% {dm*100:+8.2f}% '
              f'{du_diff.mean()*100:+8.2f}pp {(du_diff>0).mean()*100:>5.0f}% '
              f'{dm_diff.mean()*100:+8.2f}pp {(dm_diff>0).mean()*100:>5.0f}%')

    # === [4] 顶6重叠: U vs score ===
    print(f'\n[4] top-6选择重叠 (U vs score, 逐日)')
    ovs = []
    for d, g in sig.groupby('date'):
        if len(g) < 20:
            continue
        a = set(g.nlargest(6, 'score').code)
        b = set(g.nlargest(6, 'U').code)
        ovs.append(len(a & b) / 6)
    print(f'  平均重叠 {np.mean(ovs)*100:.0f}% (n={len(ovs)}日)')

    # === [5] 逐年 U-score top-6 配对差 (稳定性) ===
    print(f'\n[5] 逐年 top-6 f20 配对差 U-score')
    for y in sorted(sig.date.dt.year.unique()):
        diffs = []
        for d, g in sig[sig.date.dt.year == y].groupby('date'):
            if len(g) < 20:
                continue
            a = g.nlargest(6, 'score').f20.mean()
            b = g.nlargest(6, 'U').f20.mean()
            if np.isfinite(a) and np.isfinite(b):
                diffs.append(b - a)
        if diffs:
            diffs = np.array(diffs)
            print(f'  {y}: mean={diffs.mean()*100:+.2f}pp 正比例='
                  f'{(diffs>0).mean()*100:.0f}% n={len(diffs)}')
    print(f'\n[总耗时 {time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
