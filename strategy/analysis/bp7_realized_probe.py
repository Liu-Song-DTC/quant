#!/usr/bin/env python
"""bp5/bp7 realized探针 (2026-09-09): 普查筛出的无加成类能否通过E-K1第二关

普查(buy_point_census)结论: 9类中唯二无任何类级加成的是bp5/bp7。
  bp7: n=22,377 hit1 50.9% mean5 +1.14% 6/6年正 score四分档spread+0.65pp (类级低估特征)
  bp5: n=14,544 hit1 49.2% mean5 +1.09% 5/6年正 (2022 -0.09%)
E-K1第二关(realized确认): 信号日fwd5证据在E-N7(bp4/8/9)/E-N10(bp1)上都输在
realized层 — 信号日edge被入场等待吃掉或boost引入边际劣候选。本探针回答:
  ① 已入场trade按买入类分组的realized ret (bp7 vs bp0/bp2对照)
  ② bp7逐年realized (6/6稳定性在交易层是否还成立)
  ③ 入场年龄×ret: 等待天数是否吃掉bp7的edge
  ④ 穿透率: bp7信号22k中有多少真正入场 (机制人群规模)
注意: 当前trade_realized.csv属于E-E2消融run(770,576), bp7信号行与基线一致,
      但组合选股略偏 → 类间对照仍有效, 最终裁决须跑双态回测。
输入: rolling_validation_results/trade_realized.csv
      rolling_validation_results/backtest_signals.csv (同run)
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
TRADES = f'{BASE}/trade_realized.csv'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_sell_point', 'signal_level']


def main():
    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    print(f"trades: {len(trades)} | {trades['entry_date'].min().date()} → "
          f"{trades['entry_date'].max().date()} (E-E2消融run的逐笔)")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    print(f"signals(buy=True): {len(sig):,}")

    # 每笔trade取入场前20天内最近一条buy信号 (同bp2探针口径)
    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=20))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    matched = m['chan_buy_point'].notna().sum()
    print(f"信号匹配(20天窗): {matched}/{len(trades)} ({matched/len(trades)*100:.1f}%)")
    m = m[m['chan_buy_point'].notna()].copy()
    m['age'] = (m['entry_date'] - m['d']).dt.days
    m['bp'] = m['chan_buy_point'].astype(int)
    m['year'] = m['entry_date'].dt.year

    print("\n=== ① 已入场trade × 买入类 realized ret ===")
    g = m.groupby('bp').agg(n=('ret', 'size'),
                            winrate=('ret', lambda x: (x > 0).mean()),
                            mean_ret=('ret', 'mean'), med_ret=('ret', 'median'),
                            sum_ret=('ret', 'sum'),
                            age_med=('age', 'median'))
    print(g.round(4).to_string())

    print("\n=== ② bp7逐年 realized (6/6交易层稳定性) ===")
    b7 = m[m['bp'] == 7]
    y7 = b7.groupby('year').agg(n=('ret', 'size'),
                                winrate=('ret', lambda x: (x > 0).mean()),
                                mean_ret=('ret', 'mean'))
    print(y7.round(4).to_string())
    print(f"bp7合计: n={len(b7)} winrate={(b7['ret']>0).mean()*100:.1f}% "
          f"mean {(b7['ret'].mean())*100:+.2f}% sum {(b7['ret'].sum())*100:+.1f}%")

    print("\n=== ③ 入场年龄 × realized (edge被等待吃掉吗) ===")
    m['age_b'] = pd.cut(m['age'], [-1, 2, 5, 9, 20],
                        labels=['0-2天', '3-5天', '6-9天', '10-20天'])
    for name, sub in [('bp7', m[m['bp'] == 7]), ('bp0', m[m['bp'] == 0]),
                      ('bp2', m[m['bp'] == 2]), ('全部', m)]:
        if len(sub) < 5:
            print(f"\n--- {name}: n={len(sub)} 跳过")
            continue
        ga = sub.groupby('age_b', observed=True).agg(
            n=('ret', 'size'), winrate=('ret', lambda x: (x > 0).mean()),
            mean_ret=('ret', 'mean'))
        print(f"\n--- {name} (n={len(sub)}) ---")
        print(ga.round(4).to_string())

    print("\n=== ④ 穿透率: 信号池 vs 已入场 (机制人群规模) ===")
    sig_all = pd.read_csv(f'{BASE}/backtest_signals.csv',
                          usecols=['date', 'code', 'buy', 'chan_buy_point'], low_memory=False)
    sig_all = sig_all[sig_all['buy']].copy()
    sig_all['bp'] = sig_all['chan_buy_point'].astype(int)
    sig_n = sig_all.groupby('bp').size()
    trade_n = m.groupby('bp').size()
    pene = pd.DataFrame({'信号数': sig_n, '入场数': trade_n})
    pene['穿透率%'] = (pene['入场数'] / pene['信号数'] * 100).round(1)
    print(pene.to_string())


if __name__ == '__main__':
    main()


def paired_diag():
    """配对诊断: 已入场bp7的信号日fwd5 vs 全池bp7 (入场选择是否逆向)"""
    import glob
    DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
    closes = {}
    cal = pd.read_csv(os.path.join(DATA_DIR, 'sh000001_qfq.csv'),
                      usecols=['datetime'])['datetime']
    cal = pd.to_datetime(cal).sort_values().reset_index(drop=True)
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))

    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=20))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    m = m[m['chan_buy_point'].notna()].copy()
    m['bp'] = m['chan_buy_point'].astype(int)

    def fwd5(code, d):
        if code not in closes:
            return np.nan
        dts, cl = closes[code]
        i = np.searchsorted(dts, np.datetime64(d))
        if i + 5 >= len(cl) or i < 0:
            return np.nan
        return float(cl[i + 5] / cl[i] - 1) if cl[i] > 0 else np.nan

    ent = m[m['bp'] == 7]
    e5 = [fwd5(c, d) for c, d in zip(ent['code'], ent['d'])]
    ent = ent.copy()
    ent['fwd5'] = e5
    print("\n=== ⑤ 配对: 已入场bp7的信号日fwd5 vs 全池bp7(+1.14%) ===")
    print(f"已入场bp7: n={ent['fwd5'].notna().sum()} fwd5均值 "
          f"{np.nanmean(ent['fwd5'])*100:+.2f}% 中位 {np.nanmedian(ent['fwd5'])*100:+.2f}%")
    ent['year'] = ent['entry_date'].dt.year
    print(ent.groupby('year').agg(n=('fwd5', 'size'),
                                  fwd5=('fwd5', 'mean')).round(4).to_string())
    print(f"\n已入场bp0同口径: ", end='')
    e0 = m[m['bp'] == 0]
    e0f = np.nanmean([fwd5(c, d) for c, d in zip(e0['code'], e0['d'])])
    print(f"n={len(e0)} fwd5均值 {e0f*100:+.2f}% (全池bp0 +0.57%)")
    print(f"\n已入场bp7持有期 vs bp0: ")
    h7 = trades[trades['code'].isin(ent['code'])]
    h7 = h7.merge(ent[['entry_date', 'code']], on=['entry_date', 'code'], how='inner')
    h0 = m[m['bp'] == 0]
    for name, sub in [('bp7', h7), ('bp0', trades[trades['code'].isin(h0['code'])])]:
        print(f"  {name}: hold_days 中位 {sub['hold_days'].median():.0f} 均值 "
              f"{sub['hold_days'].mean():.0f} | ret中位 {sub['ret'].median()*100:+.2f}%")


if __name__ == '__main__':
    main()
    paired_diag()
