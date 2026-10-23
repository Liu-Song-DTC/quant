#!/usr/bin/env python3
"""★市场宽度 (breadth) 日型闸 × 选股胜率 — 只读探针 (2026-10-23夜).

动机: 已采纳 ahday(行业日型) 是日级闸正先例; 宽度是**未测过的日级市场状态**。
PIT: 宽度用**信号日 date 收盘**计算 (池的 exec_date=次日开盘, date=信号日) ⇒ 完全已知。
  ⚠ 绝不能用 exec_date 当日收盘 (那是未来).

特征 (universe = pool_1022 出现过的全部股票, 即策略实际投资域):
  b20   = 收盘>MA20 的股票占比
  r5pos = 5日收益>0 的股票占比
  up1   = 当日上涨股票占比
  disp  = 当日收益截面标准差
市场态: 上证收盘 vs MA20 (usnight.pkl['sh']) — 与体制检测器同源, 用于检验宽度是否只是它的影子.

层:
  P1 池级五/三分位 × w5/w20 + 簇bootstrap(signal date) + 分年
  P2 市场态交叉: 指数在MA20上/下 × 宽度三分位 (宽度是否有增量)
  P3 realized 166笔 (entry_date→信号日映射) × 宽度

用法: .venv/bin/python .arms/probe/breadth_1023.py
"""
import sys

import numpy as np
import pandas as pd

AR = '/mnt/d/quant/.arms/armout'
PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
AD = '/mnt/d/quant/data/alternative_data'

sys.path.insert(0, '/mnt/d/quant/.arms/probe')


def cluster_boot(d, col, ycol='w5', nb=5000, seed=20261023):
    d = d[['date', col, ycol]].dropna()
    x = d[col].values
    lo_m = x <= np.quantile(x, 1 / 3)
    hi_m = x >= np.quantile(x, 2 / 3)
    days = sorted(d.date.unique())
    di = {v: i for i, v in enumerate(days)}
    day_idx = d.date.map(di).values
    y = (d[ycol].values > 0).astype(float)
    na = np.bincount(day_idx[hi_m], minlength=len(days))
    ka = np.bincount(day_idx[hi_m], weights=y[hi_m], minlength=len(days))
    nb_ = np.bincount(day_idx[lo_m], minlength=len(days))
    kb = np.bincount(day_idx[lo_m], weights=y[lo_m], minlength=len(days))
    if na.sum() < 50 or nb_.sum() < 50:
        return None
    obs = ka.sum() / na.sum() - kb.sum() / nb_.sum()
    rng = np.random.default_rng(seed)
    B = rng.integers(0, len(days), size=(nb, len(days)))
    bd = B @ ka / (B @ na) - B @ kb / (B @ nb_)
    bd = bd[np.isfinite(bd)]
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p = 2 * min((bd >= 0).mean(), (bd <= 0).mean())
    return obs, lo, hi, p


def main():
    sys.argv = [sys.argv[0]]  # shortfeat 解析 argv[1]
    from shortfeat_1022 import load as pool_load
    pool = pool_load()
    pool['date'] = pd.to_datetime(pool['date'])
    uni = set(pool['code'].unique())
    print(f'池 {len(pool)} 行 / {pool["date"].nunique()} 信号日 / {len(uni)} 只')

    px = pd.read_parquet(PQ, columns=['code', 'date', 'close'])
    px['date'] = pd.to_datetime(px['date'])
    px = px[px.code.isin(uni)].sort_values(['code', 'date'])
    g = px.groupby('code', sort=False)['close']
    px['r1'] = g.pct_change()
    px['ma20'] = g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    px['r5'] = g.transform(lambda s: s.pct_change(5))
    px['above20'] = (px['close'] > px['ma20']).astype(float)
    px.loc[px['ma20'].isna(), 'above20'] = np.nan
    px['r5pos'] = (px['r5'] > 0).astype(float)
    px.loc[px['r5'].isna(), 'r5pos'] = np.nan
    px['up1'] = (px['r1'] > 0).astype(float)
    px.loc[px['r1'].isna(), 'up1'] = np.nan
    br = px.groupby('date').agg(b20=('above20', 'mean'), r5pos=('r5pos', 'mean'),
                                up1=('up1', 'mean'), disp=('r1', 'std'),
                                n=('r1', 'count')).reset_index()
    print(f'宽度序列 {len(br)} 日, 覆盖 {br.date.min().date()}→{br.date.max().date()}')

    d = pool.merge(br, on='date', how='left')
    d = d.dropna(subset=['b20'])
    print(f'合并后 {len(d)} 行')

    def quint(df, col, lab):
        x = df[col]
        q = pd.qcut(x, 5, labels=False, duplicates='drop')
        print(f'  [{lab}]')
        for i in range(5):
            gg = df[q == i]
            if len(gg) < 100:
                continue
            print(f'    Q{i + 1} n={len(gg):5d} {col}∈[{gg[col].min():.3f},{gg[col].max():.3f}] '
                  f'w5={(gg.w5 > 0).mean() * 100:5.1f}% w20={(gg.w20 > 0).mean() * 100:5.1f}%')

    for col, lab in [('b20', '收盘>MA20占比'), ('r5pos', '5日上涨占比'),
                     ('up1', '当日上涨占比'), ('disp', '截面离散度')]:
        quint(d, col, lab)
        r = cluster_boot(d, col)
        if r:
            obs, lo, hi, p = r
            print(f'    Δw5(hi−lo)={obs * 100:+.2f}pp CI[{lo * 100:+.2f},{hi * 100:+.2f}] p={p:.4f}')
            r20 = cluster_boot(d, col, 'w20')
            if r20:
                print(f'    Δw20(hi−lo)={r20[0] * 100:+.2f}pp CI[{r20[1] * 100:+.2f},{r20[2] * 100:+.2f}] p={r20[3]:.4f}')

    # ── P2 指数MA20 交叉 ──
    sh = pd.read_pickle(f'{AD}/usnight.pkl')['sh'][['date', 'close']].copy()
    sh['date'] = pd.to_datetime(sh['date'])
    sh['ma20'] = sh['close'].rolling(20, min_periods=20).mean()
    sh['ix_above'] = sh['close'] > sh['ma20']
    d = d.merge(sh[['date', 'ix_above']], on='date', how='left')
    print('\n══ P2 指数MA20 × 宽度三分位 (w5) ══')
    for above, lab in [(True, '指数>MA20'), (False, '指数<MA20')]:
        sub = d[d.ix_above == above].copy()
        if len(sub) < 200:
            continue
        x = sub['b20']
        sub['bt3'] = np.where(x <= x.quantile(1 / 3), 'lo',
                              np.where(x >= x.quantile(2 / 3), 'hi', 'mid'))
        print(f'  [{lab}] n={len(sub)}')
        for k in ['lo', 'mid', 'hi']:
            gg = sub[sub.bt3 == k]
            print(f'    {k:3s} n={len(gg):5d} w5={(gg.w5 > 0).mean() * 100:5.1f}% '
                  f'w20={(gg.w20 > 0).mean() * 100:5.1f}%')

    # ── P3 realized ──
    print('\n══ P3 realized 166笔 × 信号日宽度 ══')
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['code'] = mm['code'].str.zfill(6)
    cal = sorted(pool['date'].unique())
    ex = pool[['date', 'exec_date']].dropna().drop_duplicates()
    ex['exec_date'] = pd.to_datetime(ex['exec_date'])
    mp = dict(zip(ex.exec_date, ex.date))
    mm['signal_date'] = mm['entry_date'].map(mp)
    miss = mm.signal_date.isna().sum()
    if miss:
        # entry_date 不在池exec日历 (如建仓日非调仓日) → 用日历上一个交易日
        cal_arr = np.array(cal)
        mm.loc[mm.signal_date.isna(), 'signal_date'] = [
            cal_arr[np.searchsorted(cal_arr, t, side='left') - 1] if (np.searchsorted(cal_arr, t, side='left') > 0) else pd.NaT
            for t in mm.loc[mm.signal_date.isna(), 'entry_date']]
    mm = mm.merge(br, left_on='signal_date', right_on='date', how='left', suffixes=('', '_br'))
    r = mm[['b20', 'r5pos', 'ret']].dropna()
    print(f'  n={len(r)} (映射缺失 {mm.b20.isna().sum()})')
    for col in ['b20', 'r5pos']:
        x = r[col]
        med = x.median()
        a, b = r[x >= med], r[x < med]
        print(f'  {col}: ≥中位 n={len(a):3d} tW={(a.ret > 0).mean() * 100:5.1f}% 均ret={a.ret.mean() * 100:+6.2f}% | '
              f'<中位 n={len(b):3d} tW={(b.ret > 0).mean() * 100:5.1f}% 均ret={b.ret.mean() * 100:+6.2f}%')
        for thr, tlab in [(x.quantile(0.75), 'top25%'), (x.quantile(0.25), 'bot25%')]:
            aa, bb = r[x >= thr], r[x <= thr]
            print(f'    {tlab}: n={len(aa):3d} tW={(aa.ret > 0).mean() * 100:5.1f}% 均ret={aa.ret.mean() * 100:+6.2f}% | '
                  f'其余 n={len(r) - len(aa):3d} tW={(r[~(x >= thr)].ret > 0).mean() * 100:5.1f}%')

    # ── 稳健: b20 分年 ──
    print('\n  b20 分年 Δw5(hi−lo):')
    x = d['b20']
    d['hi'] = x >= x.quantile(2 / 3)
    d['lo'] = x <= x.quantile(1 / 3)
    d['year'] = d.date.dt.year
    for y, gg in d.groupby('year'):
        a, b = gg[gg.hi], gg[gg.lo]
        if len(a) < 50 or len(b) < 50:
            print(f'    {y}: 不足 (hi {len(a)} lo {len(b)})')
            continue
        print(f'    {y}: hi {(a.w5 > 0).mean() * 100:5.1f}% lo {(b.w5 > 0).mean() * 100:5.1f}% '
              f'Δ={(a.w5 > 0).mean() * 100 - (b.w5 > 0).mean() * 100:+6.1f}pp')

    d.to_csv(f'{AR}/breadth_pool_1023.csv', index=False)
    br.to_csv(f'{AR}/breadth_daily_1023.csv', index=False)
    print(f'\n落盘 {AR}/breadth_pool_1023.csv / breadth_daily_1023.csv')


if __name__ == '__main__':
    main()
