#!/usr/bin/env python
"""yjyg靴子落地: 已入场人群配对检查 (2026-09-09) — E-K1 realized关

背景: yjyg_signal_overlay在信号日fwd5上发现 bp1×NEG深亏 Δ+1.71pp / bp7 Δ+1.60pp /
bp9 Δ+2.45pp, 但bp3/4/5反向。E-N7/E-N10/E-N15教训: 探针桶≠机制人群, 必须查
已入场(realized)人群里预告溢价是否存活 — 若入场选择把溢价吃光, 机制候选立即降级。
本检查: 每笔trade入场日前30天内最近一条预告(严格早于入场日, PIT) × realized ret,
整体+分bp类。bp1 realized n≈81(E-E2 run), 子桶预期个位数 — 方向参考,
最终定夺必须走双态回测(E-N16设计)。
输入: rolling_validation_results/trade_realized.csv (E-E2消融run)
      rolling_validation_results/backtest_signals.csv (同run, bp8行有偏)
      data/alternative_data/yjyg_records.pkl
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.alternative_data import get_provider

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
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
    e2 = ev[['code', 'nd', 'cohort', 'cp']].sort_values('nd')
    e2['nd'] = e2['nd'].astype('datetime64[us]')

    trades = pd.read_csv(f'{BASE}/trade_realized.csv')
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    print(f"trades: {len(trades)} | {trades['entry_date'].min().date()} → "
          f"{trades['entry_date'].max().date()}")

    t = trades.sort_values('entry_date').reset_index(drop=True)
    m = pd.merge_asof(t, e2, left_on='entry_date', right_on='nd', by='code',
                      direction='backward', allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=30))

    def bucket(r):
        if pd.isna(r['cohort']):
            return '无预告'
        if r['cohort'] == 'NEG':
            return 'NEG深亏' if r['cp'] < -50 else 'NEG浅亏'
        if r['cohort'] == 'POS':
            return 'POS'
        return '其他'
    m['yb'] = m.apply(bucket, axis=1)
    m['lag'] = (m['entry_date'] - m['nd']).dt.days
    print(m['yb'].value_counts().to_string())

    print("\n=== ① 已入场 × 30天内预告桶 realized ===")
    g = m.groupby('yb', observed=True).agg(
        n=('ret', 'size'), winrate=('ret', lambda x: (x > 0).mean()),
        mean_ret=('ret', 'mean'), med_ret=('ret', 'median'))
    g['winrate'] = (g['winrate'] * 100).round(1)
    g[['mean_ret', 'med_ret']] = (g[['mean_ret', 'med_ret']] * 100).round(2)
    print(g.to_string())

    # bp类标签 (同bp7_realized_probe协议: 入场前20天最近buy信号)
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'],
                      low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sigb = sig[sig['buy']].copy()
    mm = m.merge(sigb, on='code', how='left', suffixes=('', '_sig'))
    mm = mm[(mm['d'] <= mm['entry_date']) &
            (mm['d'] >= mm['entry_date'] - pd.Timedelta(days=20))]
    mm = mm.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    mm['bp'] = mm['chan_buy_point'].fillna(0).astype(int)

    print("\n=== ② 已入场 × bp × NEG深亏/无预告 realized ===")
    for bpv in (0, 1, 2, 7):
        sub = mm[mm['bp'] == bpv]
        nd_ = sub[sub['yb'] == 'NEG深亏']
        none_ = sub[sub['yb'] == '无预告']
        if len(nd_) >= 1:
            print(f"bp{bpv}: NEG深亏 n={len(nd_)} winrate {(nd_['ret']>0).mean()*100:.0f}% "
                  f"mean {nd_['ret'].mean()*100:+.2f}% | 无预告 n={len(none_)} "
                  f"winrate {(none_['ret']>0).mean()*100:.0f}% mean "
                  f"{none_['ret'].mean()*100:+.2f}%")

    print("\n=== ③ 预告距入场天数 (NEG深亏, realized) ===")
    deep = m[m['yb'] == 'NEG深亏'].copy()
    if len(deep) >= 3:
        deep['lb'] = pd.cut(deep['lag'], [0, 10, 20, 30],
                            labels=['1-10天', '11-20天', '21-30天'])
        gl = deep.groupby('lb', observed=True).agg(
            n=('ret', 'size'), mean_ret=('ret', 'mean'))
        gl['mean_ret'] = (gl['mean_ret'] * 100).round(2)
        print(gl.to_string())
    else:
        print(f"NEG深亏已入场样本仅 {len(deep)}, 分组无意义")


if __name__ == '__main__':
    main()
