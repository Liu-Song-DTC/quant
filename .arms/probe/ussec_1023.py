#!/usr/bin/env python3
"""★美股板块轮动 × A股对应行业 — 四层探针 (2026-10-23夜, 用户问询第二层, 只读).

问 (用户): 美股每天有板块涨跌 → 在A股找对应标的, 有没有溢价?
与§25的区别: §25=美股大盘聚合隔夜(日级); 本探针=**截面/轮动**信号 (板块→行业映射)。
先验: A股"映射行情"(英伟达→算力/CPO) 在**跳空**层是公认现象; 待测=跳空之后 open→+5/+20
有没有残余可吃, 以及能否进选股层。

数据: us_sector_ret.pkl = 11×SPDR板块ETF + SOXX费城半导体 + NVDA + SPX (2001→2026-10-22)。
A股侧: 股票→所处行业(基本面)→20类(industry_mapping) → 美股板块映射表。
PIT: 美股收盘 北京凌晨4-5点 < A股开盘9:30。

映射 (20类→美股):
  SOXX: 半导体/光伏, 电子 (人工智能/算力含其中)
  XLK: 互联网/软件, 通信/计算机
  XLC: 传媒
  XLF: 金融
  XLV: 医药
  XLY: 消费, 新能源车/风电
  XLP: 农业
  XLI: 自动化/制造, 军工, 交运
  XLB: 有色/钢铁/煤炭, 化工, 建材
  XLE: 基建/地产/石油石化
  XLU: 电力设备, 环保/公用
附加: NVDA 单票映射 (半导体/电子 替代SOXX) — 已知最强映射链。

层:
  T1 传导: 美股板块隔夜 ret ↔ 次日A股对应行业等权收益 (corr, 2021起)。
  T2 池级: exec日 ussec_ret 三分位 × w5/w20 (主判读) + 相对版 (ussec−SPX) + NVDA版。
  T3 realized: 166笔 × 同特征 × ret。
  T4 稳健: 分年 + 簇bootstrap + lag(D−1)。

用法: .venv/bin/python .arms/probe/ussec_1023.py
"""
import sys

import numpy as np
import pandas as pd

AR = '/mnt/d/quant/.arms/armout'
PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
AD = '/mnt/d/quant/data/alternative_data'

CAT2US = {
    '半导体/光伏': 'SOXX', '电子': 'SOXX', '人工智能/算力': 'SOXX',
    '互联网/软件': 'XLK', '通信/计算机': 'XLK', '传媒': 'XLC',
    '金融': 'XLF', '医药': 'XLV', '消费': 'XLY', '新能源车/风电': 'XLY',
    '农业': 'XLP', '自动化/制造': 'XLI', '军工': 'XLI', '交运': 'XLI',
    '有色/钢铁/煤炭': 'XLB', '化工': 'XLB', '建材': 'XLB',
    '基建/地产/石油石化': 'XLE', '电力设备': 'XLU', '环保/公用': 'XLU',
}


def us_ret_series(sec, col):
    """col的隔夜收益, 索引=美股日."""
    return sec[col].dropna()


def align_us(dates, u):
    """A股日/执行日序列 ← 最近美股日(lag≤4天)的美股收益."""
    dates = pd.to_datetime(pd.Series(dates))
    pos = np.searchsorted(u.index.values, dates.values, side='left') - 1
    ok = pos >= 0
    rets = np.where(ok, u.values[np.clip(pos, 0, len(u) - 1)], np.nan)
    lag = dates.values - u.index.values[np.clip(pos, 0, len(u) - 1)]
    return pd.Series(np.where(ok & (lag <= np.timedelta64(4, 'D')), rets, np.nan), index=dates.index)


def align_us_lag1(dates, u):
    """滞后一晚的美股收益 (D−1), 测A股反应延迟."""
    dates = pd.to_datetime(pd.Series(dates))
    pos = np.searchsorted(u.index.values, dates.values, side='left') - 1
    pos = np.clip(pos - 1, 0, len(u) - 1)  # 再往前一个美股日
    lag = dates.values - u.index.values[pos]
    return pd.Series(np.where(lag <= np.timedelta64(6, 'D'), u.values[pos], np.nan), index=dates.index)


def main():
    sec = pd.read_pickle(f'{AD}/us_sector_ret.pkl')
    cat = pd.read_pickle(f'{AR}/stock_cat20_1023.pkl')  # code→cat20

    # ── T1 传导: 行业等权日收益 × 美股板块隔夜 ──
    print('══ T1 传导: 美股板块隔夜 → 次日A股对应行业等权收益 (2021起) ══')
    px = pd.read_parquet(PQ, columns=['code', 'date', 'close'])
    px['date'] = pd.to_datetime(px['date'])
    px = px.merge(cat, on='code', how='inner')
    px['ret'] = px.groupby('code')['close'].pct_change()
    ind = px.groupby(['date', 'cat20'])['ret'].mean().reset_index()
    ind = ind[ind.cat20.isin(CAT2US)]
    rows = []
    for c20, g in ind.groupby('cat20'):
        u = us_ret_series(sec, CAT2US[c20])
        r = align_us(g['date'], u).values
        m = pd.DataFrame({'us': r, 'a': g['ret'].values}).dropna()
        m = m[m.index >= 0]
        if len(m) < 200:
            continue
        rows.append((c20, CAT2US[c20], len(m), m.us.corr(m.a)))
    t1 = pd.DataFrame(rows, columns=['cat20', 'us', 'n', 'corr']).sort_values('corr', ascending=False)
    print(t1.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    print(f'  中位corr={t1["corr"].median():+.3f} | corr>0.05的行业 {(t1["corr"] > 0.05).sum()}/{len(t1)}')

    # ── T2 池级 (主判读) ──
    print('\n══ T2 池级: exec日 ussec_ret 三分位 × w5/w20 ══')
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, '/mnt/d/quant/.arms/probe')
    from shortfeat_1022 import load as pool_load
    q = pool_load()
    q['code'] = q['code'].str.zfill(6)
    q = q.merge(cat, on='code', how='left')
    q = q[q.cat20.isin(CAT2US)].copy()
    q['ussec'] = np.nan
    q['usrel'] = np.nan
    for c20, g in q.groupby('cat20'):
        u = us_ret_series(sec, CAT2US[c20])
        us = align_us(g['exec_date'], u)
        sp = align_us(g['exec_date'], us_ret_series(sec, 'SPX'))
        q.loc[g.index, 'ussec'] = us.values
        q.loc[g.index, 'usrel'] = (us - sp).values
    m = q[['exec_date', 'cat20', 'ussec', 'usrel', 'w5', 'w20']].dropna(subset=['ussec'])
    print(f'  n={len(m)} ({m.cat20.nunique()}类) 基线 w5 {(m.w5 > 0).mean() * 100:.1f}% '
          f'w20 {(m.w20 > 0).mean() * 100:.1f}%')

    def tercile_table(df, col, lab):
        x = df[col]
        try:
            lo_m = x <= x.quantile(1 / 3)
            hi_m = x >= x.quantile(2 / 3)
        except Exception:
            return
        print(f'  [{lab}]')
        for nm, msk in [('lo', lo_m), ('mid', ~(lo_m | hi_m)), ('hi', hi_m)]:
            g = df[msk & df[col].notna()]
            if len(g) < 100:
                continue
            print(f'    {nm:3s} n={len(g):5d} w5={(g.w5 > 0).mean() * 100:5.1f}% '
                  f'w20={(g.w20 > 0).mean() * 100:5.1f}%')

    tercile_table(m, 'ussec', '美股板块隔夜绝对')
    tercile_table(m, 'usrel', '相对版 (板块−SPX)')

    # 簇bootstrap: hi vs lo (exec日为簇)
    def clus_boot(df, col):
        d = df[['exec_date', col, 'w5']].dropna()
        x = d[col]
        lo_m = (x <= x.quantile(1 / 3)).values
        hi_m = (x >= x.quantile(2 / 3)).values
        days = sorted(d.exec_date.unique())
        di = {v: i for i, v in enumerate(days)}
        day_idx = d.exec_date.map(di).values
        y = (d.w5.values > 0).astype(float)
        na = np.bincount(day_idx[hi_m], minlength=len(days))
        ka = np.bincount(day_idx[hi_m], weights=y[hi_m], minlength=len(days))
        nb = np.bincount(day_idx[lo_m], minlength=len(days))
        kb = np.bincount(day_idx[lo_m], weights=y[lo_m], minlength=len(days))
        if na.sum() < 50 or nb.sum() < 50:
            return None
        obs = ka.sum() / na.sum() - kb.sum() / nb.sum()
        rng = np.random.default_rng(20261023)
        B = rng.integers(0, len(days), size=(5000, len(days)))
        bd = (B @ ka / (B @ na) - B @ kb / (B @ nb))
        bd = bd[np.isfinite(bd)]
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p = 2 * min((bd >= 0).mean(), (bd <= 0).mean())
        return obs, lo, hi, p

    for col, lab in [('ussec', '绝对'), ('usrel', '相对')]:
        r = clus_boot(m, col)
        if r:
            obs, lo, hi, p = r
            print(f'    簇bootstrap Δw5(hi−lo) [{lab}] = {obs * 100:+.2f}pp '
                  f'CI[{lo * 100:+.2f},{hi * 100:+.2f}] p={p:.4f}')

    # 分年 (usrel hi−lo)
    print('  分年 Δw5(hi−lo, 相对版):')
    x = m['usrel']
    m2 = m.assign(hi=(x >= x.quantile(2 / 3)), lo=(x <= x.quantile(1 / 3)))
    m2['year'] = pd.to_datetime(m2.exec_date).dt.year
    for y, g in m2.groupby('year'):
        a, b = g[g.hi], g[g.lo]
        if len(a) < 50 or len(b) < 50:
            print(f'    {y}: 不足 (hi {len(a)} lo {len(b)})')
            continue
        print(f'    {y}: hi w5 {(a.w5 > 0).mean() * 100:5.1f}% lo w5 {(b.w5 > 0).mean() * 100:5.1f}% '
              f'Δ={(a.w5 > 0).mean() * 100 - (b.w5 > 0).mean() * 100:+6.1f}pp')

    # SOXX特殊: 半导体/电子子样本 + NVDA替代
    print('  [SOXX子样本: 半导体/光伏+电子]')
    sm = m[m.cat20.isin(['半导体/光伏', '电子'])]
    if len(sm) > 300:
        x = sm['ussec']
        hi = sm[x >= x.quantile(2 / 3)]
        lo = sm[x <= x.quantile(1 / 3)]
        print(f'    n={len(sm)} hi w5 {(hi.w5 > 0).mean() * 100:.1f}% w20 {(hi.w20 > 0).mean() * 100:.1f}% '
              f'| lo w5 {(lo.w5 > 0).mean() * 100:.1f}% w20 {(lo.w20 > 0).mean() * 100:.1f}%')

    # ── T3 realized ──
    print('\n══ T3 realized: 166笔 × ussec/usrel ══')
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['code'] = mm['code'].str.zfill(6)
    mm = mm.merge(cat, on='code', how='left')
    mm = mm[mm.cat20.isin(CAT2US)].copy()
    for c20, g in mm.groupby('cat20'):
        u = us_ret_series(sec, CAT2US[c20])
        us = align_us(g['entry_date'], u)
        sp = align_us(g['entry_date'], us_ret_series(sec, 'SPX'))
        mm.loc[g.index, 'ussec'] = us.values
        mm.loc[g.index, 'usrel'] = (us - sp).values
    r = mm[['ussec', 'usrel', 'ret']].dropna()
    print(f'  n={len(r)}')
    for col, lab in [('ussec', '绝对'), ('usrel', '相对')]:
        x = r[col]
        for thr, tlab in [(0, '≥0'), (x.quantile(0.75), 'top25%')]:
            a, b = r[x >= thr], r[x < thr]
            if len(a) < 8:
                continue
            print(f'  {lab} {tlab:6s}: n={len(a):3d} tW={(a.ret > 0).mean() * 100:5.1f}% '
                  f'均ret={a.ret.mean() * 100:+6.2f}% | 其余 n={len(b):3d} '
                  f'tW={(b.ret > 0).mean() * 100:5.1f}% 均ret={b.ret.mean() * 100:+6.2f}%')

    # ── T4 lag稳健 ──
    print('\n══ T4 lag(D−1) 稳健: 池级 usrel 滞后一晚 ══')
    q2 = q.dropna(subset=['exec_date']).copy()
    q2['usrel_l1'] = np.nan
    for c20, g in q2.groupby('cat20'):
        u = us_ret_series(sec, CAT2US[c20])
        us = align_us_lag1(g['exec_date'], u)
        sp = align_us_lag1(g['exec_date'], us_ret_series(sec, 'SPX'))
        q2.loc[g.index, 'usrel_l1'] = (us - sp).values
    ml = q2[['exec_date', 'usrel_l1', 'w5']].dropna()
    x = ml['usrel_l1']
    hi = ml[x >= x.quantile(2 / 3)]
    lo = ml[x <= x.quantile(1 / 3)]
    print(f'  n={len(ml)} hi w5 {(hi.w5 > 0).mean() * 100:.1f}% | lo w5 {(lo.w5 > 0).mean() * 100:.1f}%')

    m.to_csv(f'{AR}/ussec_pool_1023.csv', index=False)
    print(f'\n落盘 {AR}/ussec_pool_1023.csv')


if __name__ == '__main__':
    main()
