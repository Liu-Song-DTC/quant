#!/usr/bin/env python3
"""9/30月度日历池迁移预估计探针 (2026-09-22, 只读, 零生产写入)

动机: 9/30重锚漂移 = 数据刷新delta + 9/30池边界迁移delta + A.5印花税(−4.4%±1%)。
印花税成分已预量化; 本探针预量化池迁移成分的下界代理。

生产口径 (bt_execution.py:110-146): pool_calendar=monthly,
relax_floor=0.05 / relax_momentum=None (0f-G臂, 732,689锚点), bse_exclude=true。
9/17数据态下生产membership最后边界=2026-08-31; 9/30冷跑将追加边界2026-09-30。

代理语义: 用现有数据(至9/17)评估as-of 2026-09-30边界的成员资格 —
等价于"市场9/18起冻结到9/30"的池迁移下界代理。9/30真实池还会看到9/18-9/29
新数据(13日), 实际迁移≥本代理。若代理迁移已近零 → 池成分对9/30漂移贡献小;
若代理迁移大 → 9/30归因时池迁移是主成分, 数据刷新delta需与之分离。

零写入: 无cache_key(不写生产缓存), 仅stdout + /tmp/probe_pool_transition_0930_20260922.csv
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.stock_pool import get_pool_membership_map  # noqa: E402

DATA_PATH = '/mnt/d/quant/data/stock_data/backtrader_data'
# 生产参数 (bt_execution.py:133-134 + yaml 7624-7641)
RELAX_FLOOR = 0.05
RELAX_MOMENTUM = None
MIN_PRICE = 2.0
BSE_EXCLUDE = True

BOUNDARIES = [pd.Timestamp('2026-08-31'), pd.Timestamp('2026-09-30')]


def main():
    m = get_pool_membership_map(BOUNDARIES, data_dir=DATA_PATH,
                                min_price=MIN_PRICE, bse_exclude=BSE_EXCLUDE,
                                cache_key=None,
                                relax_floor=RELAX_FLOOR,
                                relax_momentum=RELAX_MOMENTUM)
    if len(m) != 2:
        print(f'FAIL: 期望2边界, 得{len(m)}: {list(m.keys())}')
        return 1
    prev = m[BOUNDARIES[0]]
    new = m[BOUNDARIES[1]]
    enter = sorted(new - prev)
    leave = sorted(prev - new)
    common = prev & new
    print(f'生产当前边界 2026-08-31 成员: {len(prev)}')
    print(f'9/30边界代理(数据至9/17) 成员: {len(new)}')
    print(f'  留存: {len(common)} ({100*len(common)/len(prev):.1f}%)')
    print(f'  新入: {len(enter)} ({100*len(enter)/len(new):.1f}% of 9/30池)')
    print(f'  退出: {len(leave)} ({100*len(leave)/len(prev):.1f}% of 当前池)')
    print(f'  净变化: {len(new)-len(prev):+d}')
    if len(enter) <= 30:
        print(f'  新入样例: {enter}')
    if len(leave) <= 30:
        print(f'  退出样例: {leave}')
    rows = ([(c, 'stay') for c in common] + [(c, 'enter') for c in enter]
            + [(c, 'leave') for c in leave])
    pd.DataFrame(rows, columns=['code', 'status']).to_csv(
        '/tmp/probe_pool_transition_0930_20260922.csv', index=False)
    print('细节 → /tmp/probe_pool_transition_0930_20260922.csv (零生产写入)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
