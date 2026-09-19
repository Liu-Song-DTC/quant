#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解锁表取证 (2026-09-20): Z=git-9/5版 vs Y=当前9/20版 sort-aligned逐行diff
+ 消费者级漂移量化 (get_unlock_codes(date,30,0.05) 集合对全回测日≤9/17)。
问: Y较Z的行级/集合级差异多大 → 推断X(9/17态, 不可恢复)较Y差异上界。
若集合级差异≈0 → X≈Y在更短窗口内更是≈0, C1忠实臂与batch2臂解锁输入视为干净。
"""
import pandas as pd

Z = '/tmp/unlock_z.pkl'
Y = 'data/alternative_data/unlock_schedule.pkl'
CUT = pd.Timestamp('2026-09-05')

z = pd.read_pickle(Z)
y = pd.read_pickle(Y)
print(f'Z行数 {len(z)}  Y行数 {len(y)}')
print(f'Z列 {list(z.columns)}')
print(f'Y列 {list(y.columns)}')
for c in ['code', 'unlock_date', 'type']:
    if c in z.columns and c in y.columns:
        print(f'  {c}: Z唯一 {z[c].nunique()} / Y唯一 {y[c].nunique()}')

# 对齐键
keys = ['code', 'unlock_date']
val_cols = [c for c in z.columns if c not in keys]
val_cols = [c for c in val_cols if c in y.columns]
print(f'比对值列: {val_cols}')

zz = z.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
yy = y.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
print(f'Z去重后 {len(zz)}  Y去重后 {len(yy)}')

zk = set(map(tuple, zz[keys].values))
yk = set(map(tuple, yy[keys].values))
only_z = zk - yk
only_y = yk - zk
print(f'仅Z键: {len(only_z)}  仅Y键: {len(only_y)}')
if only_z:
    zd = pd.DataFrame(list(only_z), columns=keys)
    print(f'  仅Z按unlock_date分布: {zd["unlock_date"].min()} ~ {zd["unlock_date"].max()}')
if only_y:
    yd = pd.DataFrame(list(only_y), columns=keys)
    print(f'  仅Y按unlock_date分布: {yd["unlock_date"].min()} ~ {yd["unlock_date"].max()}')

# 共同键的值比对
mz = zz.merge(yy, on=keys, suffixes=('_z', '_y'))
print(f'共同键: {len(mz)}')
changed = []
for c in val_cols:
    a, b = mz[f'{c}_z'], mz[f'{c}_y']
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        tol = max(1e-9, (a.abs().max() * 1e-6) if len(a) else 1e-9)
        neq = ((a - b).abs() > tol) & ~(a.isna() & b.isna())
        changed.append((c, neq))
    else:
        neq = (a.fillna('') != b.fillna(''))
        changed.append((c, neq))
any_ch = pd.DataFrame({c: neq for c, neq in changed}).any(axis=1)
print(f'值列有差异的共同键行: {int(any_ch.sum())} / {len(mz)}')
if any_ch.sum():
    mzc = mz[any_ch].copy()
    mzc['_hist'] = mzc['unlock_date'] <= CUT
    print('  差异行按历史/未来(>9/5)拆分:')
    print(mzc.groupby('_hist').size())
    print(mzc[['code', 'unlock_date'] + [f'{c}_z' for c in val_cols] + [f'{c}_y' for c in val_cols]].head(10).to_string())

# 消费者级: 全回测日 ≤9/17 的 get_unlock_codes 集合
sigs = pd.read_csv('strategy/arms_20260919/_baseline_sig/backtest_signals.csv',
                   usecols=['date'], parse_dates=['date'])
dates = sorted(sigs['date'].dt.normalize().unique())
dates = [d for d in dates if d <= pd.Timestamp('2026-09-17')]
print(f'\n回测日数: {len(dates)} ({dates[0]} ~ {dates[-1]})')


def unlock_set(df, d):
    t = pd.Timestamp(d)
    m = (df['unlock_date'] >= t) & (df['unlock_date'] <= t + pd.Timedelta(days=30)) \
        & (df['ratio'] >= 0.05)
    return set(df.loc[m, 'code'])


drift_dates = []
tot_z = tot_y = 0
for d in dates:
    sz = unlock_set(zz, d)
    sy = unlock_set(yy, d)
    tot_z += len(sz)
    tot_y += len(sy)
    if sz != sy:
        drift_dates.append((d, len(sz), len(sy), len(sz - sy), len(sy - sz)))
print(f'集合漂移日: {len(drift_dates)} / {len(dates)}')
print(f'总集合规模: Z {tot_z} (日均{tot_z/len(dates):.1f})  Y {tot_y} (日均{tot_y/len(dates):.1f})')
if drift_dates:
    for d, nz, ny, zonly, yonly in drift_dates[:30]:
        print(f'  {d.date()}: Z{nz} Y{ny} 仅Z{zonly} 仅Y{yonly}')
    if len(drift_dates) > 30:
        print(f'  ... 共{len(drift_dates)}漂移日')
    import statistics
    ds = [dd[0] for dd in drift_dates]
    print(f'  漂移日区间: {min(ds).date()} ~ {max(ds).date()}')
    print(f'  2026年漂移日: {sum(1 for dd in ds if dd.year == 2026)}')
    print(f'  2025年漂移日: {sum(1 for dd in ds if dd.year == 2025)}')
