#!/usr/bin/env python
"""业绩预告 × 买入信号叠加探针 (2026-09-09): 靴子落地效应能否叠加在缠论买点上

yjyg_event_study_v3结论: A股预告反向 — NEG预告后续跑赢POS (raw fwd20 NEG+4.02%
vs POS+1.95%; 深亏<-100%梯度最强+1.14% exc20; 5/6年NEG胜)。
本探针问: 该系统交易的缠论买点信号上, 预告靴子落地是否提供正交增量?
即 bp×预告bucket × 信号日fwd5 — 若NEG深亏桶在同类买点上fwd5更高, 则预告信息
未被缠论结构吸收, 是因子层/组合层的候选增量。
PIT: 预告notice_date必须严格早于信号日(保守, 不用当日)。
输入: rolling_validation_results/backtest_signals.csv (E-E2消融run)
      data/alternative_data/yjyg_records.pkl
      data/stock_data/backtrader_data/*_qfq.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.alternative_data import get_provider

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_sell_point']
IND_PRIO = ['归属于上市公司股东的净利润', '扣除非经常性损益后的净利润', '净利润',
            '每股收益', '营业收入']
TYPE_COHORT = {
    '预增': 'POS', '扭亏': 'POS', '略增': 'POS',
    '预减': 'NEG', '首亏': 'NEG', '略减': 'NEG', '增亏': 'NEG', '续亏': 'NEG',
    '减亏': 'MIX', '续盈': 'NEU', '不确定': 'NEU',
}


def load_events():
    df = get_provider().load_yjyg()
    df = df[df['indicator'].isin(IND_PRIO)].copy()
    df['prio'] = df['indicator'].map({k: i for i, k in enumerate(IND_PRIO)})
    df = df.sort_values(['code', 'report_period', 'prio', 'notice_date'])
    ev = df.groupby(['code', 'report_period'], as_index=False).apply(
        lambda g: g.sort_values(['prio', 'notice_date']).iloc[-1],
        include_groups=False)
    ev = ev[['code', 'notice_date', 'forecast_type', 'change_pct']]
    ev['cohort'] = ev['forecast_type'].map(TYPE_COHORT)
    ev['code'] = ev['code'].astype(str).str.zfill(6)
    ev['cp'] = pd.to_numeric(ev['change_pct'], errors='coerce')
    ev = ev.drop_duplicates(['code', 'notice_date'])
    return ev.reset_index(drop=True)


def main():
    ev = load_events()
    ev['nd'] = pd.to_datetime(ev['notice_date'])
    ev = ev.sort_values('nd')
    print(f"预告事件: {len(ev):,}")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    b = sig[sig['buy']].copy().reset_index(drop=True)
    b['bp'] = b['chan_buy_point'].astype(int)
    print(f"buy信号: {len(b):,}")

    # 每信号取最近一条严格早于信号日的预告 (窗口90天), 向量化merge+分组
    e2 = ev.rename(columns={'nd': 'd'})[['code', 'd', 'cohort', 'cp']]
    e2['d'] = e2['d'].astype('datetime64[us]')
    m = pd.merge_asof(b.sort_values('d'), e2.sort_values('d'), on='d', by='code',
                      direction='backward', allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=90))
    def bucket(r):
        if pd.isna(r['cohort']):
            return '无预告'
        if r['cohort'] == 'NEG':
            return 'NEG深亏' if r['cp'] < -50 else 'NEG浅亏'
        if r['cohort'] == 'POS':
            return 'POS高增' if r['cp'] > 100 else 'POS低增'
        return '其他'
    m['yb'] = m.apply(bucket, axis=1)
    hit = m['cohort'].notna().mean()
    print(f"90天内预告覆盖率: {hit*100:.1f}%")
    print(m['yb'].value_counts().to_string())

    # fwd5
    closes = {}
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))
    f5s = np.full(len(m), np.nan)
    for code, grp in m.groupby('code', sort=False):
        if code not in closes:
            continue
        dts, cl = closes[code]
        d64 = grp['d'].values
        i = np.searchsorted(dts, d64)
        ok = (i >= 0) & (i < len(cl) - 5)
        f5s[grp.index.values[ok]] = cl[i[ok] + 5] / cl[i[ok]] - 1
    m['fwd5'] = f5s
    mv = m[m['fwd5'].notna()]

    print("\n=== ① bp × 预告bucket 信号日fwd5 ===")
    g = mv.groupby(['bp', 'yb'], observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    g['mean5'] = (g['mean5'] * 100).round(2)
    print(g.to_string())

    print("\n=== ② 主类聚焦: NEG深亏 vs 无预告 (同bp配对) ===")
    for bpv in (0, 1, 2, 7):
        sub = mv[mv['bp'] == bpv]
        nd_ = sub[sub['yb'] == 'NEG深亏']
        none_ = sub[sub['yb'] == '无预告']
        if len(nd_) >= 30 and len(none_) >= 100:
            print(f"bp{bpv}: NEG深亏 n={len(nd_)} mean5 {nd_['fwd5'].mean()*100:+.2f}% "
                  f"vs 无预告 n={len(none_)} {none_['fwd5'].mean()*100:+.2f}% "
                  f"(Δ {nd_['fwd5'].mean()*100 - none_['fwd5'].mean()*100:+.2f}pp)")

    print("\n=== ③ NEG深亏 × 逐年 (bp0/bp1/bp2主类) ===")
    ndb = mv[mv['yb'] == 'NEG深亏']
    for bpv in (0, 1, 2):
        sub = ndb[ndb['bp'] == bpv].copy()
        sub['year'] = sub['d'].dt.year
        if len(sub) < 50:
            continue
        gy = sub.groupby('year').agg(n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
        gy['mean5'] = (gy['mean5'] * 100).round(2)
        print(f"-- bp{bpv} NEG深亏 --")
        print(gy.to_string())

    print("\n=== ④ 预告距信号日天数 × fwd5 (NEG深亏, 效应衰减曲线) ===")
    e3 = ev[['code', 'nd', 'cp']].sort_values('nd').rename(columns={'nd': 'dd'})
    e3['dd'] = e3['dd'].astype('datetime64[us]')
    m3 = pd.merge_asof(b.sort_values('d'), e3, left_on='d', right_on='dd', by='code',
                       direction='backward', allow_exact_matches=False,
                       tolerance=pd.Timedelta(days=90))
    m3['lag'] = (m3['d'] - m3['dd']).dt.days
    deep = m3[(m3['cp'] < -50) & m3['cp'].notna()].copy()
    deep['fwd5'] = f5s[deep.index.values]
    deep = deep[deep['fwd5'].notna()]
    deep['lag_b'] = pd.cut(deep['lag'], [0, 5, 15, 30, 60, 90],
                           labels=['1-5天', '6-15天', '16-30天', '31-60天', '61-90天'])
    gl = deep.groupby('lag_b', observed=True).agg(
        n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
    gl['mean5'] = (gl['mean5'] * 100).round(2)
    print(gl.to_string())


if __name__ == '__main__':
    main()
