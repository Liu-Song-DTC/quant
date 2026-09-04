#!/usr/bin/env python3
"""2026-09-02 补全另类数据到最新 (龙虎榜/融资融券/北向/减持/解禁/业绩预告)

- 复用 strategy/core/alternative_data.py 的下载逻辑 (schema 由项目自身代码保证)
- 每个 pkl 先备份 .preRefresh_0902, mtime 老化 72h 强制触发 load 方法重下
- 龙虎榜历史明细 dict 重建: 2024-01 ~ 2026-09 逐月 detail_daily (补齐 6/19 后缺口,
  且从"每只仅最近上榜日"升级为"全部上榜日集合")
- 北向 hist: API 2024-08 起净买额停发但行持续更新到 9/2, 照常刷新
"""
import os
import sys
import time
import pickle
import shutil

sys.path.insert(0, '/mnt/d/quant/strategy')
import pandas as pd

from core.alternative_data import AlternativeDataProvider

DATA = '/mnt/d/quant/data/alternative_data'
prov = AlternativeDataProvider(data_dir=DATA)

REFRESH_PKLS = [
    ('dragon_tiger.pkl', 'load_dragon_tiger'),
    ('margin_daily.pkl', 'load_margin'),
    ('northbound_daily.pkl', 'load_northbound'),
    ('reduction_records.pkl', 'load_reduction'),
    ('reduction_plans.pkl', 'load_reduction_plans'),
    ('unlock_schedule.pkl', 'load_unlock'),
    ('yjyg_records.pkl', 'load_yjyg'),
]


def rng(name):
    """pkl 日期范围简述"""
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return '不存在'
    df = pd.read_pickle(p)
    if isinstance(df, dict):
        return f'dict {len(df)}键'
    cols = [c for c in df.columns if 'date' in c.lower() or '日期' in str(c) or '上榜' in str(c) or '日' in str(c)]
    if not cols:
        return f'{df.shape} 无日期列'
    s = pd.to_datetime(df[cols[0]], errors='coerce')
    return f'{len(df)}行 {s.min().date()} -> {s.max().date()}'


print('=== 刷新前 ===')
for name, _ in REFRESH_PKLS:
    print(f'  {name}: {rng(name)}')
print(f'  dragon_tiger_history.pkl: {rng("dragon_tiger_history.pkl")}')

for name, meth in REFRESH_PKLS:
    p = os.path.join(DATA, name)
    try:
        if os.path.exists(p) and not os.path.exists(p + '.preRefresh_0902'):
            shutil.copy2(p, p + '.preRefresh_0902')
            print(f'备份: {p}')
        if os.path.exists(p):
            old = time.time() - 72 * 3600
            os.utime(p, (old, old))
        print(f'--- 刷新 {name} ---')
        getattr(prov, meth)()
    except Exception as e:
        print(f'[FAIL] {name}: {str(e)[:200]}')

print()
print('=== 刷新后 ===')
for name, _ in REFRESH_PKLS:
    print(f'  {name}: {rng(name)}')

# === 龙虎榜历史明细 dict 重建 (2024-01 ~ 2026-09) ===
print()
print('=== 重建 dragon_tiger_history.pkl (datacenter-web 直连) ===')
# 2026-09-04: akshare 路径(ak.stock_lhb_detail_daily_em)当天0/33月失败 → 改直连 datacenter-web.
# 该接口每页硬封顶500行, 必须按500分页(rebuild_dragon_tiger_history.py 已处理).
# 失败月份跳过, 全失败不动旧文件(fail-open).
try:
    sys.path.insert(0, '/mnt/d/quant/data')
    from rebuild_dragon_tiger_history import rebuild, write_hist
    hist, ok = rebuild(verbose=True)
    write_hist(hist, ok, min_ok=3)
except Exception as e:
    print(f'[FAIL] history重建: {str(e)[:200]}')

print('=== 补全完成 ===')
