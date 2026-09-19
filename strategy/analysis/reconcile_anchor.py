#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锚点对账(通用版, 2026-09-17): 每次刷新后冷跑四指标 + 曲线级对账。
用法: python analysis/reconcile_anchor.py <new_curve.csv> <ref_curve.csv> <prev_date> <anchor_date> [band]
检查: 1) ≤prev_date逐位一致(数据态/代码态无漂移)  2) anchor_date单日增量在合理带内
 3) 末行=anchor_date, 报告日收益
"""
import sys
import pandas as pd

new_path, ref_path = sys.argv[1], sys.argv[2]
prev_date, anchor_date = sys.argv[3], sys.argv[4]
band = float(sys.argv[5]) if len(sys.argv) > 5 else 30000.0

new = pd.read_csv(new_path)
ref = pd.read_csv(ref_path)
new = new.rename(columns={new.columns[0]: 'date', new.columns[1]: 'nav'})
ref = ref.rename(columns={ref.columns[0]: 'date', ref.columns[1]: 'nav'})
new['date'] = pd.to_datetime(new['date'])
ref['date'] = pd.to_datetime(ref['date'])

ref_upto = ref[ref['date'] <= prev_date].set_index('date')['nav']
new_upto = new[new['date'] <= prev_date].set_index('date')['nav']
common = ref_upto.index.intersection(new_upto.index)
if len(common) != len(ref_upto) or len(common) != len(new_upto):
    print(f"[FAIL] 行集不一致: 新缺{len(ref_upto)-len(common)} 参考缺{len(new_upto)-len(common)}")
    print(f"  新缺: {list(ref_upto.index.difference(new_upto.index))[:8]}")
    print(f"  参考缺: {list(new_upto.index.difference(ref_upto.index))[:8]}")
    sys.exit(1)
maxdiff = float((new_upto.loc[common] - ref_upto.loc[common]).abs().max())
print(f"连续性(≤{prev_date}): {len(common)}行, 最大偏差={maxdiff:.10f} {'✓' if maxdiff == 0.0 else '✗ FAIL'}")
if maxdiff != 0.0:
    print("[FAIL] 历史段漂移 — 停止, 取证后再出单")
    sys.exit(1)

nav_prev = float(new_upto.loc[common].iloc[-1])
d = new[new['date'] == anchor_date]
if len(d) == 0:
    print(f"[FAIL] 新曲线无{anchor_date}行! 末行={new['date'].iloc[-1].date()}")
    sys.exit(1)
nav_anchor = float(d['nav'].iloc[0])
delta = nav_anchor - nav_prev
ret = delta / nav_prev * 100
ok = abs(delta) <= band
print(f"NAV({prev_date})={nav_prev:,.2f} → NAV({anchor_date})={nav_anchor:,.2f}  Δ={delta:+,.2f} ({ret:+.2f}%)")
print(f"合理带±{band:,.0f}: {'✓ 通过' if ok else '✗ 超出 — 单日组合波动超常或存在漂移, 需深查'}")
print(f"曲线末行: {new['date'].iloc[-1].date()} NAV={new['nav'].iloc[-1]:,.2f}")
