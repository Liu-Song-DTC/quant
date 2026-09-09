#!/usr/bin/env python
"""yjyg闸门设计探针 (2026-09-09): 反向闸候选的逐年稳定性+窗口敏感性

realized配对检查结论: 已入场520笔中30天内NEG深亏仅1笔(bp2, +28.8%) — 组合层
入场太稀疏, realized关对此机制无统计功效, 定夺须走信号日证据+双态回测。
信号日叠加(①表)给出双向候选:
  正向(加成): bp1/bp7/bp9 × NEG深亏跑赢类基线 — 但入场稀疏, 加成功效存疑
  反向(闸):  bp3/4/5(续涨类) × NEG深亏跑输类基线 — 排除类机制与bearhard同型,
            且排除的是差人群, 功效方向更干净
本探针为反向闸候选补齐: 逐桶逐年稳定性(排除单年驱动) + 窗口敏感性(30/60/90天)。
输入: rolling_validation_results/backtest_signals.csv (E-E2消融run, bp8行有偏)
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
IND_PRIO = ['归属于上市公司股东的净利润', '扣除非经常性损益后的净利润', '净利润',
            '每股收益', '营业收入']
TYPE_COHORT = {
    '预增': 'POS', '扭亏': 'POS', '略增': 'POS',
    '预减': 'NEG', '首亏': 'NEG', '略减': 'NEG', '增亏': 'NEG', '续亏': 'NEG',
    '减亏': 'MIX', '续盈': 'NEU', '不确定': 'NEU',
}
GATE_BPS = (3, 4, 5)


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
    # NEG深亏必须cohort==NEG且cp<-50 (减亏/扭亏等cp<-50非NEG队列不算, 与原叠加探针同口径)
    e2 = ev[ev['cohort'] == 'NEG'][['code', 'nd', 'cp']].sort_values('nd')
    e2['nd'] = e2['nd'].astype('datetime64[us]')

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'],
                      low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    b = sig[sig['buy']].copy().reset_index(drop=True)
    b['bp'] = b['chan_buy_point'].astype(int)
    b = b[b['bp'].isin(GATE_BPS)].reset_index(drop=True)
    print(f"buy信号(bp3/4/5): {len(b):,}")

    # fwd5 (只算bp3/4/5, 快)
    closes = {}
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))
    f5s = np.full(len(b), np.nan)
    for code, grp in b.groupby('code', sort=False):
        if code not in closes:
            continue
        dts, cl = closes[code]
        d64 = grp['d'].values
        i = np.searchsorted(dts, d64)
        ok = (i >= 0) & (i < len(cl) - 5)
        f5s[grp.index.values[ok]] = cl[i[ok] + 5] / cl[i[ok]] - 1
    b['fwd5'] = f5s
    bv = b[b['fwd5'].notna()].copy()

    for win in (30, 60, 90):
        m = pd.merge_asof(bv.sort_values('d'), e2, left_on='d', right_on='nd',
                          by='code', direction='backward',
                          allow_exact_matches=False,
                          tolerance=pd.Timedelta(days=win))
        m['yb'] = np.where(pd.isna(m['cp']), '无预告',
                           np.where(m['cp'] < -50, 'NEG深亏', 'NEG浅亏'))
        m['year'] = m['d'].dt.year
        print(f"\n{'='*60}\n窗口 {win}天")
        print("\n=== ① bp × 桶 信号日fwd5 ===")
        g = m.groupby(['bp', 'yb'], observed=True).agg(
            n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
        g['mean5'] = (g['mean5'] * 100).round(2)
        print(g.to_string())

        print(f"\n=== ② NEG深亏 vs 无预告 逐年 (bp3/4/5合并) ===")
        for yb in ('NEG深亏', '无预告'):
            sub = m[m['yb'] == yb]
            gy = sub.groupby('year').agg(n=('fwd5', 'size'), mean5=('fwd5', 'mean'))
            gy['mean5'] = (gy['mean5'] * 100).round(2)
            print(f"-- {yb} --")
            print(gy.to_string())


if __name__ == '__main__':
    main()
