#!/usr/bin/env python
"""H3诊断: 洗仓单(入场后10天内出场)的入场画像 — 系统级入场质量问题

输入: rolling_validation_results/trade_realized.csv (逐笔平仓审计)
      rolling_validation_results/backtest_signals.csv (信号层, 含入场结构特征)
输出: 洗仓单(hold<=10d) vs 正常单(hold>10d)的入场特征对照:
  chan_buy_point分布 / score分布 / gate_quality / trend_type / factor_name
  → 决定H3入口门槛设计: 哪类入场信号产生的交易几乎必然10天内死掉
用法: python analysis/wash_trade_diagnosis.py [trade_csv]
"""
import sys

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
TRADE_PATH = f'{BASE}/trade_realized.csv'
SIG_PATH = f'{BASE}/backtest_signals.csv'
WASH_DAYS = 10


def main():
    trades = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else TRADE_PATH)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    trades['date_key'] = trades['entry_date'].dt.strftime('%Y-%m-%d')
    trades['wash'] = trades['hold_days'] <= WASH_DAYS
    print(f"总笔数 {len(trades)} | 洗仓单(hold<={WASH_DAYS}d) "
          f"{trades['wash'].sum()} ({trades['wash'].mean()*100:.1f}%)")

    # 修复(2026-09-06): 信号日=入场日-1~-3(调仓日收盘出信号, 次日执行),
    # 不能按入场日精确匹配 — 按code+日期范围流式过滤, 再窗口匹配取最近
    codes = set(trades['code'])
    d_min = trades['entry_date'].min() - pd.Timedelta(days=3)
    d_max = trades['entry_date'].max()
    print(f"加载信号CSV(流式过滤 code∈{len(codes)}只, "
          f"日期{d_min.date()}~{d_max.date()})...")
    usecols = ['date', 'code', 'buy', 'chan_buy_point', 'score', 'gate_quality',
               'trend_type', 'factor_name', 'industry', 'chan_divergence_type']
    chunks = []
    for chunk in pd.read_csv(SIG_PATH, usecols=usecols, low_memory=False, chunksize=2_000_000):
        chunk['code'] = chunk['code'].astype(str).str.zfill(6)
        chunk['d'] = pd.to_datetime(chunk['date'], errors='coerce')
        mask = (chunk['buy'] & chunk['code'].isin(codes)
                & (chunk['d'] >= d_min) & (chunk['d'] <= d_max))
        if mask.any():
            chunks.append(chunk[mask])
    sig = pd.concat(chunks, ignore_index=True)
    sig = sig.drop_duplicates(subset=['date', 'code'])
    print(f"候选信号行: {len(sig)}")

    # 窗口: 信号日 ∈ [入场日-3, 入场日], 每笔取最近
    # 注意: in_win是行级布尔(每笔交易×多行信号), 匹配率须按交易笔数统计
    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m['in_win'] = (m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=3))
    m = m[m['in_win']]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    print(f"入场信号匹配率: {len(m)}/{len(trades)}笔 ({len(m)/len(trades)*100:.1f}%)")

    print(f"\n=== 入场买入点分布 (洗仓单 vs 正常单) ===")
    g = m.groupby('wash')['chan_buy_point'].value_counts(dropna=False).unstack(fill_value=0)
    g['total'] = g.sum(axis=1)
    for col in g.columns:
        if col != 'total':
            g[f'{col}_pct'] = (g[col] / g['total'] * 100).round(1)
    print(g.to_string())

    print(f"\n=== 入场score分布 ===")
    print(m.groupby('wash')['score'].describe().round(3).to_string())

    print(f"\n=== 入场gate_quality分布 ===")
    print(m.groupby('wash')['gate_quality'].describe().round(3).to_string())

    print(f"\n=== 入场trend_type分布 ===")
    print(m.groupby('wash')['trend_type'].value_counts(dropna=False).unstack(fill_value=0).to_string())

    print(f"\n=== 入场factor_name top分布 ===")
    fn = m.groupby('wash')['factor_name'].value_counts().unstack(fill_value=0)
    print(fn.T.sort_values(True, ascending=False).head(12).to_string())

    print(f"\n=== 洗仓单收益画像 ===")
    print(m.groupby('wash')['ret'].agg(['size', 'mean', 'median']).round(4).to_string())

    # 洗仓单按买入点×年
    w = m[m['wash']]
    w['y'] = pd.to_datetime(w['date_key']).dt.year
    print(f"\n=== 洗仓单逐年 × 买入点 ===")
    print(w.groupby(['y', 'chan_buy_point']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()
