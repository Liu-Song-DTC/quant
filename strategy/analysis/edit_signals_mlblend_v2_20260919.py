#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1: ML通道blend权重重标定注入 v2 (2026-09-19) — 消除v1的A项重缩放confound。

v1的缺陷: adjusted = (1-w0)*score + w0*z + A (A=另类市场项, blend后逐行加),
  逆向剥离时 A 被计入 s_clean, 重blend后mask行A变为(1-w1)/(1-w0)*A,
  non-mask行保持A → 截面相对扰动+日期级阈值移位, 非纯ML权重变更。
v2修复: A是date的纯函数(alt_market = (nb*0.6+mg*0.4)*0.15, lag1, 仅i>=60),
  用alternative_data provider逐日精确重算并完整剥离:
    s_clean = (adj - w0*z - A)/(1-w0);  adj_new = (1-w1)*s_clean + A + w1*z
  mask行与non-mask行的A均原样保留 → 纯ML权重变更。
identity烟测: w1=0.4 (==w0) 时必须逐位复原 (max|diff| < 1e-12), 否则中止。
dragon_tiger已关闭(0g), A不含个股级项。
用法: python analysis/edit_signals_mlblend_v2_20260919.py <w1>
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.alternative_data import get_provider

RVD = '/mnt/d/quant/strategy/rolling_validation_results'
W0 = 0.4
w1 = float(sys.argv[1])

print("加载生产signals...", flush=True)
sig = pd.read_csv(f'{RVD}/backtest_signals.csv', dtype={'code': str})
n = len(sig)
print(f"{n} 行加载完成", flush=True)

# === A项逐日重算 (与signal_engine 3.6节同逻辑: lag1, i>=60) ===
provider = get_provider()
dates = pd.to_datetime(sig['date'])
uniq_dates = sorted(set(dates))
A_map = {}
for d in uniq_dates:
    prev = (d - pd.Timedelta(days=1)).date()
    nb = provider.get_northbound_signal(prev)
    mg = provider.get_margin_signal(prev)
    A_map[d] = (nb * 0.6 + mg * 0.4) * 0.15
A = np.array([A_map[d] for d in dates])
print(f"A项: {len(uniq_dates)} 个日期, |A| max={np.abs(A).max():.4f}, "
      f"mean={A.mean():.5f}, 非零日期比例={(np.abs(np.array([A_map[d] for d in uniq_dates])) > 1e-12).mean()*100:.1f}%",
      flush=True)

mask = np.abs(sig['ml_score'].to_numpy()) > 0.01
z = np.tanh(sig['ml_score'].to_numpy() * 3)
adj = sig['adjusted_score'].to_numpy().astype(np.float64)
s_clean = (adj - W0 * z - A) / (1 - W0)
adj_new = adj.copy()
adj_new[mask] = (1 - w1) * s_clean[mask] + A[mask] + w1 * z[mask]
print(f"注入行: {mask.sum()} ({mask.mean()*100:.1f}%), w0={W0}→w1={w1}", flush=True)

if abs(w1 - W0) < 1e-9:
    dmax = np.abs(adj_new - adj).max()
    print(f"[identity烟测] w1==w0, max|adj_new-adj| = {dmax:.3e}", flush=True)
    assert dmax < 1e-12, "identity烟测失败! 剥离/重blend机制与生产不一致, 中止"

sig['adjusted_score'] = adj_new
out = f'{RVD}/backtest_signals.mlw{int(round(w1*100))}.csv'
sig.to_csv(out, index=False)
print(f"写入 {out}", flush=True)
