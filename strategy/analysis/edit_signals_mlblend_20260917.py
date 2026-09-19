#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1: ML通道blend权重重标定注入 (2026-09-17)。
对|ml_score|>0.01的buy/sell行: 逆向剥离生产w0=0.4的ML分量, 再用新权重w1重blend:
  z = tanh(ml_score*3); s_clean = (adjusted - w0*z)/(1-w0); adjusted_new = (1-w1)*s_clean + w1*z
其余行(无ML分量)不动。输出 backtest_signals.mlw{XX}.csv (与生产同schema同dtype格式)。
用法: python analysis/edit_signals_mlblend_20260917.py <w1>
"""
import sys
import numpy as np
import pandas as pd

RVD = '/mnt/d/quant/strategy/rolling_validation_results'
W0 = 0.4
w1 = float(sys.argv[1])

print(f"加载生产signals...", flush=True)
sig = pd.read_csv(f'{RVD}/backtest_signals.csv', dtype={'code': str})
n = len(sig)
print(f"{n} 行加载完成", flush=True)

mask = np.abs(sig['ml_score'].to_numpy()) > 0.01
z = np.tanh(sig['ml_score'].to_numpy() * 3)
adj = sig['adjusted_score'].to_numpy().astype(np.float64)
s_clean = (adj - W0 * z) / (1 - W0)
adj_new = adj.copy()
adj_new[mask] = (1 - w1) * s_clean[mask] + w1 * z[mask]
sig['adjusted_score'] = adj_new
print(f"注入行: {mask.sum()} ({mask.mean()*100:.1f}%), w0={W0}→w1={w1}", flush=True)

out = f'{RVD}/backtest_signals.mlw{int(round(w1*100))}.csv'
sig.to_csv(out, index=False)
print(f"写入 {out}", flush=True)
