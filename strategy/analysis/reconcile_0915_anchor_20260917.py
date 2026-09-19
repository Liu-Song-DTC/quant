#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""9/15锚点对账 (2026-09-17晨):
1. 连续性: 新跑equity_curve的≤9/14行与run-1逐位比对 (run-1参考=arms快照pre_equity_curve.csv)
2. 9/15增量: NAV(9/15) vs NAV(9/14)=868,610.96, 报告日收益与合理性带(±30k)
用法: python analysis/reconcile_0915_anchor_20260917.py <new_equity_curve.csv> <ref_equity_curve.csv>
"""
import sys
import numpy as np
import pandas as pd

new_path, ref_path = sys.argv[1], sys.argv[2]
new = pd.read_csv(new_path)
ref = pd.read_csv(ref_path)
dcol, ncol = new.columns[0], new.columns[1]
new = new.rename(columns={dcol: 'date', ncol: 'nav'})
ref = ref.rename(columns={ref.columns[0]: 'date', ref.columns[1]: 'nav'})

new['date'] = pd.to_datetime(new['date'])
ref['date'] = pd.to_datetime(ref['date'])

# 1. 连续性: 9/14及以前逐位一致
ref_upto = ref[ref['date'] <= '2026-09-14'].set_index('date')['nav']
new_upto = new[new['date'] <= '2026-09-14'].set_index('date')['nav']
common = ref_upto.index.intersection(new_upto.index)
if len(common) != len(ref_upto) or len(common) != len(new_upto):
    missing_new = ref_upto.index.difference(new_upto.index)
    missing_ref = new_upto.index.difference(ref_upto.index)
    print(f"[FAIL] 行集不一致: 新缺{len(missing_new)} 参考缺{len(missing_ref)}")
    print(f"  新缺: {list(missing_new)[:10]}")
    print(f"  参考缺: {list(missing_ref)[:10]}")
    sys.exit(1)
maxdiff = float((new_upto.loc[common] - ref_upto.loc[common]).abs().max())
print(f"连续性(≤9/14): {len(common)}行比对, 最大偏差={maxdiff:.10f} {'✓' if maxdiff == 0.0 else '✗ FAIL'}")
if maxdiff != 0.0:
    print("[FAIL] 历史段与run-1不一致 — 数据态/代码态漂移!")
    sys.exit(1)

# 2. 9/15增量
nav_914 = float(new_upto.loc[common].iloc[-1])
d15 = new[new['date'] == '2026-09-15']
if len(d15) == 0:
    print(f"[FAIL] 新曲线无9/15行! 末行={new['date'].iloc[-1].date()}")
    sys.exit(1)
nav_915 = float(d15['nav'].iloc[0])
delta = nav_915 - nav_914
ret = delta / nav_914 * 100
ok = abs(delta) <= 30000
print(f"NAV(9/14)={nav_914:.2f}  NAV(9/15)={nav_915:.2f}  Δ={delta:+.2f} ({ret:+.2f}%)")
print(f"合理性带±30k: {'✓ 通过' if ok else '✗ 超出 — 需深查(单日组合波动超4%或存在漂移)'}")
final = new.iloc[-1]
print(f"曲线末行: {final['date'].date()} NAV={final['nav']:.2f}")
