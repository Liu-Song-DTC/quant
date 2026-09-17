#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""因子细胞diff探针 (2026-09-17, 2×危机取证): 对比两个因子缓存parquet的公共
(code,date)细胞, 定位todate推进/池滑动在因子层的传导路径。
用法: python analysis/factor_cell_diff_20260917.py <parquetA> <parquetB> [--top N]
输出: 公共细胞数/分歧细胞数、逐列分歧统计、按年/按码分歧分布。
裁定口径:
  同池不同todate(d2bea51c vs 895d5e91): 公共细胞有差=因子值依赖未来数据(前视);
  同todate不同池(c8de5390 vs 895d5e91): 公共细胞有差=因子含截面成分(rank/中性化)。
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

A, B = sys.argv[1], sys.argv[2]
TOP = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 10

print(f"加载 {A} ...")
dfa = pd.read_parquet(A)
print(f"加载 {B} ...")
dfb = pd.read_parquet(B)

print(f"A: {dfa.shape}  B: {dfb.shape}")

# 对齐公共(code,date)细胞
ka = set(zip(dfa['code'], dfa['date']))
kb = set(zip(dfb['code'], dfb['date']))
common = ka & kb
print(f"公共细胞: {len(common)} / A独有 {len(ka)-len(common)} / B独有 {len(kb)-len(common)}")

a = dfa.set_index(['code', 'date']).loc[list(common)]
b = dfb.set_index(['code', 'date']).loc[list(common)]
del dfa, dfb

num_cols = [c for c in a.columns if c not in ('industry',)]
print(f"数值列: {len(num_cols)}")

# NaN一致性 + 数值分歧 (NaN==NaN视为一致)
a_nan = a[num_cols].isna().to_numpy()
b_nan = b[num_cols].isna().to_numpy()
nan_mismatch = (a_nan != b_nan)

av = a[num_cols].to_numpy(dtype=np.float64, na_value=np.nan)
bv = b[num_cols].to_numpy(dtype=np.float64, na_value=np.nan)
# 只比较两者均非NaN的细胞
both_valid = ~a_nan & ~b_nan
diff = np.zeros(av.shape, dtype=bool)
diff[both_valid] = ~np.isclose(av[both_valid], bv[both_valid], rtol=1e-9, atol=1e-12)
diff[nan_mismatch] = True

n_diff_cells = int(diff.sum())
n_diff_rows = int(diff.any(axis=1).sum())
print(f"分歧细胞: {n_diff_cells} ({n_diff_cells/max(len(common)*len(num_cols),1)*100:.4f}%)  "
      f"分歧行((code,date)): {n_diff_rows} ({n_diff_rows/max(len(common),1)*100:.4f}%)")

if n_diff_cells == 0:
    print("\n✓ 公共细胞逐位全等 — 结论见裁定口径")
    sys.exit(0)

# 逐列分歧统计
col_stats = []
for j, c in enumerate(num_cols):
    n = int(diff[:, j].sum())
    if n > 0:
        d = av[both_valid[:, j], j] - bv[both_valid[:, j], j]
        m = np.nanmax(np.abs(d)) if len(d) else 0
        col_stats.append((c, n, m))
col_stats.sort(key=lambda x: -x[1])
print(f"\n逐列分歧 top{TOP}:")
for c, n, m in col_stats[:TOP]:
    print(f"  {c}: {n} 细胞, max|Δ|={m:.6g}")

# 按年分布
row_diff = diff.any(axis=1)
idx = a.index
dates = pd.to_datetime([d for _, d in idx])
years = dates.year.to_numpy()
print("\n按年分歧行数:")
for y in sorted(set(years)):
    n = int(row_diff[years == y].sum())
    tot = int((years == y).sum())
    print(f"  {y}: {n}/{tot}")
first_diff_date = dates[row_diff].min()
print(f"\n首个分歧日期: {first_diff_date.date()}")

# 按码分布
codes = np.array([c for c, _ in idx])
code_diff = pd.Series(row_diff).groupby(codes).sum()
top_codes = code_diff.sort_values(ascending=False).head(TOP)
print(f"\n分歧最多{TOP}码:")
for c, n in top_codes.items():
    print(f"  {c}: {n} 个日期分歧")
