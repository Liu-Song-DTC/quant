# smoke_0f_monthly_20260918.py — 0f-v3 arm A (monthly) 烟测
# 1. 边界函数单元测试  2. 真实membership构建(写入运行缓存键, 冷跑秒级命中)
import sys, os
sys.path.insert(0, '/mnt/d/quant/strategy')
import pandas as pd

from core.stock_pool import _month_boundary_prev, _monthly_boundaries

# ── 1. 单元烟测 ──
assert _month_boundary_prev(pd.Timestamp('2026-09-30')) == pd.Timestamp('2026-09-30')
assert _month_boundary_prev(pd.Timestamp('2026-09-28')) == pd.Timestamp('2026-08-31')
assert _month_boundary_prev(pd.Timestamp('2026-03-31')) == pd.Timestamp('2026-03-31')
assert _month_boundary_prev(pd.Timestamp('2026-01-05')) == pd.Timestamp('2025-12-31')
assert _month_boundary_prev(pd.Timestamp('2024-02-29')) == pd.Timestamp('2024-02-29')  # 闰年月末
assert _month_boundary_prev(pd.Timestamp('2024-02-10')) == pd.Timestamp('2024-01-31')

import bt_execution as bt
_earliest = pd.Timestamp(bt.FROMDATE) - pd.Timedelta(days=730)
_todate = pd.Timestamp(bt.TODATE)
print(f"bt_execution: FROMDATE={bt.FROMDATE} TODATE={bt.TODATE} earliest={_earliest.date()}")

mb = _monthly_boundaries(_earliest, _todate)
assert mb[0] == _month_boundary_prev(_earliest)
assert mb[-1] == _month_boundary_prev(_todate), (mb[-1], _month_boundary_prev(_todate))
assert all(mb[i] < mb[i + 1] for i in range(len(mb) - 1))
assert all((b + pd.offsets.MonthEnd(0)) == b for b in mb)  # 全部是月末
print(f"monthly boundaries: {len(mb)} 个 ({mb[0].date()}..{mb[-1].date()}) ✓")

# ── 2. 真实membership (运行缓存键) ──
m = bt._load_pool_membership()
assert m is not None and len(m) == len(mb), f"map边界数 {len(m)} vs 预期 {len(mb)}"
assert all({'sh000001', 'sh000852', '000001', '399006'} <= v for v in m.values())
cnts = sorted((str(k.date()), len(v)) for k, v in m.items())
print(f"成员数样例: {cnts[0]}, {cnts[len(cnts)//2]}, {cnts[-1]}")
print("SMOKE PASS")
