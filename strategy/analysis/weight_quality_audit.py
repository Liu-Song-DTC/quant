#!/usr/bin/env python
"""权重质量审计 (2026-09-09): 分仓排序权重 vs realized质量 — multi_strategy层价值检验

问题: 权重分配=rank-weighted(按effective_score排序线性衰减), 排序上叠了
multi_strategy乘数(trend/rev/def混合, bp0≈0.65× vs 结构类≈1.0×)。
如果排序权重与realized质量不相关甚至负相关, 该层=纯噪音或拖累。
审计: 每笔trade的入场权重(selections按date/code就近匹配) × realized ret:
  ① 权重五档 → realized ret (rank-weighting有没有信息)
  ② bp类 × 权重中位 (multi_strategy把谁压低了)
  ③ 低权重高收益类 = 层拖累的直接证据 (bp0是否被系统性低配)
输入: rolling_validation_results/portfolio_selections.csv (E-E2消融run, 9/9 10:55)
      rolling_validation_results/trade_realized.csv (同run)
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
SEL = f'{BASE}/portfolio_selections.csv'
TRADES = f'{BASE}/trade_realized.csv'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point']


def main():
    sel = pd.read_csv(SEL)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['code'] = sel['code'].astype(str).str.zfill(6)
    print(f"selections: {len(sel)} 行 | {sel['date'].min().date()} → {sel['date'].max().date()}")

    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)

    # 匹配: trade入场日 = selections决策日+1交易日 → 就近3天内最近一次selection
    rows = []
    for _, t in trades.iterrows():
        cand = sel[(sel['code'] == t['code']) &
                   (sel['date'] <= t['entry_date']) &
                   (sel['date'] >= t['entry_date'] - pd.Timedelta(days=6))]
        if cand.empty:
            rows.append((np.nan, np.nan))
        else:
            cand = cand.iloc[cand['date'].searchsorted(t['entry_date']) - 1] \
                if cand['date'].searchsorted(t['entry_date']) > 0 else cand.iloc[0]
            rows.append((cand['weight'], cand['score']))
    w = pd.DataFrame(rows, columns=['entry_w', 'entry_score'])
    m = pd.concat([trades.reset_index(drop=True), w], axis=1)
    print(f"权重匹配: {m['entry_w'].notna().sum()}/{len(m)}")

    # 买入类
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    mm = m.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    mm = mm[(mm['d'] <= mm['entry_date']) & (mm['d'] >= mm['entry_date'] - pd.Timedelta(days=20))]
    mm = mm.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    mm['bp'] = mm['chan_buy_point'].fillna(0).astype(int)

    print("\n=== ① 入场权重五档 × realized ===")
    mm['wq'] = pd.qcut(mm['entry_w'].rank(method='first'), 5,
                       labels=['W1低', 'W2', 'W3', 'W4', 'W5高'])
    g = mm.groupby('wq', observed=True).agg(
        n=('ret', 'size'), winrate=('ret', lambda x: (x > 0).mean()),
        mean_ret=('ret', 'mean'), med_ret=('ret', 'median'),
        w_min=('entry_w', 'min'), w_max=('entry_w', 'max'))
    print(g.round(4).to_string())

    print("\n=== ② bp类 × 权重 (multi_strategy把谁压低) ===")
    g2 = mm.groupby('bp').agg(n=('ret', 'size'),
                              w_med=('entry_w', 'median'),
                              mean_ret=('ret', 'mean'))
    print(g2.round(4).to_string())

    print("\n=== ③ 权重 vs realized 相关性 ===")
    ok = mm.dropna(subset=['entry_w', 'ret'])
    print(f"Pearson(w, ret): {ok['entry_w'].corr(ok['ret']):+.3f} (n={len(ok)})")
    print(f"Pearson(score, ret): {ok['entry_score'].corr(ok['ret']):+.3f}")
    # bp0子集 (multi_strategy压得最狠的类)
    b0 = ok[ok['bp'] == 0]
    print(f"bp0: Pearson(w, ret) {b0['entry_w'].corr(b0['ret']):+.3f} (n={len(b0)}) | "
          f"w中位 {b0['entry_w'].median()*100:.2f}%")

    print("\n=== ④ 同批(same rebalance)内: 高权重位 vs 低权重位 ===")
    ok['wrank'] = ok.groupby('entry_date')['entry_w'].rank(ascending=False, method='first')
    top = ok[ok['wrank'] <= 2]
    bot = ok[ok['wrank'] > 2]
    print(f"前2权重位: n={len(top)} winrate {(top['ret']>0).mean()*100:.1f}% "
          f"mean {top['ret'].mean()*100:+.2f}%")
    print(f"后位: n={len(bot)} winrate {(bot['ret']>0).mean()*100:.1f}% "
          f"mean {bot['ret'].mean()*100:+.2f}%")

    print("\n=== ⑤ 逐批: 排序质量一致性 (前2位是否稳定跑赢同批) ===")
    by_batch = []
    for d, batch in ok.groupby('entry_date'):
        if len(batch) < 3:
            continue
        t = batch[batch['wrank'] <= 2]['ret'].mean()
        b = batch[batch['wrank'] > 2]['ret'].mean()
        by_batch.append((t, b))
    t_arr = np.array([x[0] for x in by_batch])
    b_arr = np.array([x[1] for x in by_batch])
    wins = (t_arr > b_arr).mean()
    print(f"批次: {len(by_batch)} | 前2位均值 {t_arr.mean()*100:+.2f}% vs "
          f"后位 {b_arr.mean()*100:+.2f}% | 前2位胜出批次占比 {wins*100:.1f}%")


if __name__ == '__main__':
    main()
