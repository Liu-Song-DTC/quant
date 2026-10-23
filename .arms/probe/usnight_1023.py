#!/usr/bin/env python3
"""★美股隔夜信息 × A股次日/后续溢价 — 四层探针 (2026-10-23夜, 用户问询, 只读).

问题 (用户): 接入美股信息, 对次日A股开盘或后续几天有没有溢价(预测力)?
结构定位: **日级/时序信号** (全球风险因子) ≠ 8例个股截面死家族; 与已采纳 ahday 日型闸
同族 (正先例)。PIT: 美股收盘 北京04:00(夏)/05:00(冬) < A股开盘09:30, 且 < 每日链05:00。
   账本是688重仓 ⇒ 纳指相关性可能 > 标普 (tech beta)。

对齐规则: A股交易日 T ← 最近的美股日 D (D < T 日历日; 美股D的盘在北京T凌晨收盘)。
  周末/美股假日自动回退到上一美股日。us_ret = 美股 close(D)/close(前一美股日) − 1。

四层:
  T1 传导: corr(us_ret, 上证 gap=open(T)/close(T−1)−1) + 分位表。
  T2 指数级: us_ret 五分位 → 上证 open(T)→close(T+5/T+20) (我们的执行同构)。
  T3 池级: pool_1022 (exec日对齐) w5/w20 × us_ret 桶 (主判读, n大)。
  T4 realized: 166笔 entry日的 us_ret × ret (次判读)。
  T5 防混杂: us_ret 与 A股自身动量/T−1涨幅的相关 (是不是只是"昨日A股涨→美股涨"回声)。

用法: .venv/bin/python .arms/probe/usnight_1023.py [--dl]
"""
import sys

import numpy as np
import pandas as pd

AR = '/mnt/d/quant/.arms/armout'
AD = '/mnt/d/quant/data/alternative_data'
R = '/mnt/d/quant/strategy/rolling_validation_results'


def download():
    import akshare as ak
    out = {}
    for sym, name in [('.INX', 'spx'), ('.IXIC', 'ixic')]:
        df = ak.index_us_stock_sina(symbol=sym)[['date', 'close']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates('date')
        df['ret'] = df['close'].pct_change()
        out[name] = df
        print(f'{name}: {len(df)} 行 {df.date.min().date()}→{df.date.max().date()}')
    idx = ak.stock_zh_index_daily(symbol='sh000001')[['date', 'open', 'close']].copy()
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx.sort_values('date').drop_duplicates('date')
    out['sh'] = idx
    print(f'sh000001: {len(idx)} 行 {idx.date.min().date()}→{idx.date.max().date()}')
    pd.to_pickle(out, f'{AD}/usnight.pkl')
    return out


def main():
    import os
    if '--dl' in sys.argv or not os.path.exists(f'{AD}/usnight.pkl'):
        D = download()
    else:
        D = pd.read_pickle(f'{AD}/usnight.pkl')

    sh = D['sh'].reset_index(drop=True)
    sh['gap'] = sh['open'] / sh['close'].shift(1) - 1
    sh['o5'] = sh['close'].shift(-5) / sh['open'] - 1
    sh['o20'] = sh['close'].shift(-20) / sh['open'] - 1
    sh['pc1'] = sh['close'].pct_change()  # T−1 A股涨幅 (混杂变量)

    def align(key):
        """A股日T ← 最近美股日D<T 的隔夜收益."""
        u = D[key].dropna(subset=['ret']).reset_index(drop=True)
        dates = u['date'].values
        pos = np.searchsorted(dates, sh['date'].values, side='left') - 1
        ok = pos >= 0
        rets = np.where(ok, u['ret'].values[np.clip(pos, 0, len(u) - 1)], np.nan)
        lag = (sh['date'].values - dates[np.clip(pos, 0, len(u) - 1)])
        # 只保留滞后≤4天 (否则是假日错位)
        rets = np.where(ok & (lag <= np.timedelta64(4, 'D')), rets, np.nan)
        return pd.Series(rets, index=sh.index)

    for key, lab in [('spx', '标普'), ('ixic', '纳指')]:
        sh['us_' + key] = align(key)

    # ── T1 传导 ──
    print('══ T1 传导: us_ret(D) → 上证 gap(T) ══')
    for key, lab in [('spx', '标普'), ('ixic', '纳指')]:
        m = sh[['us_' + key, 'gap']].dropna()
        m = m[m.index >= sh.index[sh.date >= '2021-01-01'][0]]
        r = m.corr().iloc[0, 1]
        print(f'  {lab}: n={len(m)} corr={r:+.3f} | gap分位均值 '
              f'{m.groupby(pd.qcut(m["us_" + key], 5, labels=False, duplicates="drop"))["gap"].mean().round(4).tolist()}')

    # ── T2 指数级 open→+5/+20 ──
    print('\n══ T2 指数级: us_ret 五分位 → 上证 open(T)→+5/+20 (2021起) ══')
    sub = sh[sh.date >= '2021-01-01'].copy()
    for key, lab in [('spx', '标普'), ('ixic', '纳指')]:
        m = sub[['us_' + key, 'o5', 'o20']].dropna()
        m['q'] = pd.qcut(m['us_' + key], 5, labels=['Q1崩', 'Q2', 'Q3', 'Q4', 'Q5涨'], duplicates='drop')
        print(f'  [{lab}]')
        for q, g in m.groupby('q', observed=True):
            print(f'    {str(q):4s} n={len(g):4d} us_ret中位={g["us_" + key].median() * 100:+5.2f}% '
                  f'o5={g.o5.mean() * 100:+5.2f}% (win {(g.o5 > 0).mean() * 100:4.0f}%) '
                  f'o20={g.o20.mean() * 100:+5.2f}% (win {(g.o20 > 0).mean() * 100:4.0f}%)')

    # ── T3 池级 (主判读) ──
    print('\n══ T3 池级: exec日 us_ret 桶 × w5/w20 (pool_1022 全史) ══')
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, '/mnt/d/quant/.arms/probe')
    from shortfeat_1022 import load as pool_load
    q = pool_load()
    q['code'] = q['code'].str.zfill(6)

    def us_at(exec_dates, key):
        u = D[key].dropna(subset=['ret']).reset_index(drop=True)
        dates = u['date'].values
        pos = np.searchsorted(dates, pd.to_datetime(exec_dates).values, side='left') - 1
        ok = pos >= 0
        rets = np.where(ok, u['ret'].values[np.clip(pos, 0, len(u) - 1)], np.nan)
        lag = pd.to_datetime(exec_dates).values - dates[np.clip(pos, 0, len(u) - 1)]
        return pd.Series(np.where(ok & (lag <= np.timedelta64(4, 'D')), rets, np.nan),
                         index=exec_dates.index if hasattr(exec_dates, 'index') else None)

    ed = q['exec_date'].copy()
    for key, lab in [('spx', 'spx'), ('ixic', 'ixic')]:
        q['us_' + lab] = us_at(ed, key).values
    print(f'  池行 {len(q)} (exec日 {q.exec_date.min().date()}→{q.exec_date.max().date()})')
    for key, lab in [('spx', '标普'), ('ixic', '纳指')]:
        m = q[['us_' + ('spx' if key == 'spx' else 'ixic'), 'w5', 'w20']].dropna()
        col = 'us_' + ('spx' if key == 'spx' else 'ixic')
        print(f'  [{lab}] n={len(m)} 基线 w5 {(m.w5 > 0).mean() * 100:.1f}% '
              f'w20 {(m.w20 > 0).mean() * 100:.1f}%')
        m2 = m.copy()
        m2['q'] = pd.qcut(m2[col], 5, labels=['Q1崩', 'Q2', 'Q3', 'Q4', 'Q5涨'], duplicates='drop')
        for qq_, g in m2.groupby('q', observed=True):
            print(f'    {str(qq_):4s} n={len(g):5d} w5={(g.w5 > 0).mean() * 100:5.1f}% '
                  f'w20={(g.w20 > 0).mean() * 100:5.1f}%')
        for thr in [-0.015, -0.01, -0.02]:
            a, b = m[m[col] <= thr], m[m[col] > thr]
            print(f'    崩盘闸 us≤{thr * 100:.1f}%: n={len(a):5d} w5={(a.w5 > 0).mean() * 100:5.1f}% '
                  f'w20={(a.w20 > 0).mean() * 100:5.1f}% | 其余 w5={(b.w5 > 0).mean() * 100:5.1f}% '
                  f'w20={(b.w20 > 0).mean() * 100:5.1f}%')

    # ── T4 realized ──
    print('\n══ T4 realized: 166笔 entry日 us_ret × ret ══')
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['code'] = mm['code'].str.zfill(6)
    for key in ['spx', 'ixic']:
        mm['us_' + key] = us_at(mm['entry_date'], key).values
    for key, lab in [('spx', '标普'), ('ixic', '纳指')]:
        m = mm[['us_' + key, 'ret']].dropna()
        med = m['us_' + key].median()
        a, b = m[m['us_' + key] < med], m[m['us_' + key] >= med]
        print(f'  {lab}: n={len(m)} | us<中位 tW={(a.ret > 0).mean() * 100:5.1f}% '
              f'均ret={a.ret.mean() * 100:+6.2f}% | us≥中位 tW={(b.ret > 0).mean() * 100:5.1f}% '
              f'均ret={b.ret.mean() * 100:+6.2f}%')
        a2 = m[m['us_' + key] <= -0.01]
        if len(a2) >= 10:
            print(f'    崩盘夜入场 (us≤−1%): n={len(a2)} tW={(a2.ret > 0).mean() * 100:.1f}% '
                  f'均ret={a2.ret.mean() * 100:+.2f}%')

    # ── T5 防混杂: us_ret vs A股信号日自身涨幅 (回声) ──
    print('\n══ T5 回声检验: corr(us_ret, 上证pc1) + 池级正交化后残差 ══')
    m = sub[['us_spx', 'us_ixic', 'pc1']].dropna()
    print(f'  corr(标普, 上证T−1)={m.us_spx.corr(m.pc1):+.3f} corr(纳指, 上证T−1)={m.us_ixic.corr(m.pc1):+.3f}')
    # 池级: exec日的 pc1 (=信号日A股涨幅) 正交化 us_ixic 后取残差分位
    shp = sh[['date', 'pc1']].rename(columns={'date': 'exec_date'})
    q = q.merge(shp, on='exec_date', how='left')
    mm3 = q[['us_ixic', 'pc1', 'w5', 'w20']].dropna()
    x = np.column_stack([np.ones(len(mm3)), mm3.pc1])
    beta = np.linalg.lstsq(x, mm3.us_ixic, rcond=None)[0]
    mm3 = mm3.assign(resid=mm3.us_ixic - x @ beta)
    rr = mm3.copy()
    rr['q'] = pd.qcut(rr['resid'], 5, labels=['R1', 'R2', 'R3', 'R4', 'R5'], duplicates='drop')
    for qq_, g in rr.groupby('q', observed=True):
        print(f'    正交残差 {str(qq_):3s} n={len(g):5d} w5={(g.w5 > 0).mean() * 100:5.1f}% '
              f'w20={(g.w20 > 0).mean() * 100:5.1f}%')

    q.to_csv(f'{AR}/usnight_pool_1023.csv', index=False)
    print(f'\n落盘 {AR}/usnight_pool_1023.csv')


if __name__ == '__main__':
    main()
