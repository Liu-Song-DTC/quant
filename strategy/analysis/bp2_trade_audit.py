#!/usr/bin/env python
"""E-G1审计v2: bp2专用通道选中的交易逐年表现 vs 常规选中的交易

匹配逻辑: 每笔交易入场日往前3日内最近的buy=True信号 = 触发该笔入场的信号
(调仓日收盘后出信号, 次日执行 → 信号日=入场日或入场日-1)
输入: rolling_validation_results/trade_realized.csv (E-G1b)
      rolling_validation_results/backtest_signals.csv
用法: python analysis/bp2_trade_audit.py
"""
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'

trades = pd.read_csv(f'{BASE}/trade_realized.csv')
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['code'] = trades['code'].astype(str).str.zfill(6)
trades['y'] = trades['entry_date'].dt.year

sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                  usecols=['date', 'code', 'buy', 'chan_buy_point', 'score',
                           'industry', 'chan_divergence_type', 'trend_type'],
                  low_memory=False)
sig['code'] = sig['code'].astype(str).str.zfill(6)
sig['d'] = pd.to_datetime(sig['date'])
sig = sig[sig['buy']].sort_values('d')

m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
# 窗口: 信号日 ∈ [入场日-3, 入场日], 取最近
m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=3))]
m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)

m['is_bp2'] = m['chan_buy_point'] == 2
print(f"总笔数 {len(trades)} | 窗口内匹配 {m['chan_buy_point'].notna().sum()} | "
      f"bp2入场 {m['is_bp2'].sum()}笔")

print("\n=== 逐年: bp2入场 vs 非bp2入场 ===")
g = m.groupby(['y', 'is_bp2']).agg(n=('ret', 'size'), mean_ret=('ret', 'mean'),
                                   sum_ret=('ret', 'sum'), med_hold=('hold_days', 'median'))
print(g.round(3).to_string())

print("\n=== bp2入场明细 ===")
b2 = m[m['is_bp2']][['entry_date', 'code', 'hold_days', 'ret', 'score',
                     'industry', 'chan_divergence_type', 'trend_type', 'd']]
print(b2.to_string(index=False))
