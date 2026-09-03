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
print('=== 重建 dragon_tiger_history.pkl ===')
try:
    import akshare as ak
    all_records = []
    months = []
    for year in [2024, 2025, 2026]:
        last_m = 9 if year == 2026 else 12
        for month in range(1, last_m + 1):
            months.append(f'{year}-{month:02d}')
    ok = 0
    for m in months:
        try:
            df = ak.stock_lhb_detail_daily_em(date=m)
            if df is not None and len(df) > 0:
                all_records.append(df)
                ok += 1
            time.sleep(0.3)
        except Exception:
            pass
    print(f'明细月份: {ok}/{len(months)} 成功, 共 {sum(len(d) for d in all_records)} 行')
    if all_records:
        merged = pd.concat(all_records, ignore_index=True)
        code_col = next((c for c in merged.columns if '代码' in str(c)), None)
        date_col = next((c for c in merged.columns if '日期' in str(c)), None)
        if code_col and date_col:
            merged['code_6'] = merged[code_col].astype(str).str.extract(r'(\d{6})', expand=False)
            merged['dt'] = pd.to_datetime(merged[date_col]).dt.date
            hist = {}
            for code, grp in merged.groupby('code_6'):
                if code and len(str(code)) == 6:
                    hist[code] = set(grp['dt'].unique())
            hp = os.path.join(DATA, 'dragon_tiger_history.pkl')
            if os.path.exists(hp) and not os.path.exists(hp + '.preRefresh_0902'):
                shutil.copy2(hp, hp + '.preRefresh_0902')
            with open(hp, 'wb') as f:
                pickle.dump(hist, f)
            all_dates = sorted({d for v in hist.values() for d in v})
            print(f'历史明细: {len(hist)} 只, 日期 {all_dates[0]} -> {all_dates[-1]} 共{len(all_dates)}个上榜日')
        else:
            print('[WARN] 明细列缺失, 未重建')
except Exception as e:
    print(f'[FAIL] history重建: {str(e)[:200]}')

print('=== 补全完成 ===')
