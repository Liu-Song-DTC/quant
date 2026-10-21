#!/usr/bin/env python3
"""#352 行业 PIT 臂逐格对比.

对 (code,date) 格:
  - industry 是否变
  - **industry 不变的格, 其余 41 列是否逐位不变** (identity smoke test)
  - 行业转移 top 表
流式: 先各自 sink 成 key+industry+hash 的小 parquet, 再 full join.
"""
import os
import sys
import polars as pl

RVD = '/mnt/d/quant/strategy/rolling_validation_results'
A = '/mnt/d/quant/.arms/armout/indpit352_1022'
PRE = f'{RVD}/backtest_signals.csv.pre352'
POST = f'{RVD}/backtest_signals.csv'
PRE_PQ = f'{A}/cells_pre.parquet'
POST_PQ = f'{A}/cells_post.parquet'

cols = pl.scan_csv(PRE, infer_schema_length=0).collect_schema().names()
nonind = [c for c in cols if c != 'industry']
print(f'[cellcmp] cols={len(cols)} non-industry={len(nonind)}', flush=True)


def build(path, out):
    lf = pl.scan_csv(path, infer_schema_length=0)
    lf.select([
        pl.col('code'),
        pl.col('date'),
        pl.col('industry').cast(pl.Utf8).fill_null('').alias('ind'),
        pl.struct(nonind).hash().alias('h'),
    ]).sink_parquet(out, row_group_size=500_000)
    print(f'[cellcmp] wrote {out} ({os.path.getsize(out)/1e6:.0f} MB)', flush=True)


for p, q in ((PRE, PRE_PQ), (POST, POST_PQ)):
    if not os.path.exists(q):
        build(p, q)

pre = pl.scan_parquet(PRE_PQ)
post = pl.scan_parquet(POST_PQ)
j = pre.join(post, on=['code', 'date'], how='full', suffix='_b', coalesce=True).select([
    pl.col('ind'), pl.col('ind_b'), pl.col('h'), pl.col('h_b'),
])
j = j.with_columns([
    pl.col('ind').fill_null('').alias('ind'),
    pl.col('ind_b').fill_null('').alias('ind_b'),
])
res = j.select([
    (pl.col('h').is_not_null() & pl.col('h_b').is_not_null()).sum().alias('common'),
    (pl.col('h_b').is_null()).sum().alias('only_pre'),
    (pl.col('h').is_null()).sum().alias('only_post'),
    ((pl.col('h').is_not_null()) & (pl.col('h_b').is_not_null()) &
     (pl.col('ind') != pl.col('ind_b'))).sum().alias('ind_changed'),
    ((pl.col('h').is_not_null()) & (pl.col('h_b').is_not_null()) &
     (pl.col('ind') != pl.col('ind_b')) & (pl.col('h') != pl.col('h_b'))).sum().alias('ind_changed_nonind_changed'),
    ((pl.col('h').is_not_null()) & (pl.col('h_b').is_not_null()) &
     (pl.col('ind') != pl.col('ind_b')) & (pl.col('h') == pl.col('h_b'))).sum().alias('ind_changed_nonind_same'),
    ((pl.col('h').is_not_null()) & (pl.col('h_b').is_not_null()) &
     (pl.col('ind') == pl.col('ind_b')) & (pl.col('h') != pl.col('h_b'))).sum().alias('ind_same_nonind_changed'),
    ((pl.col('h').is_not_null()) & (pl.col('h_b').is_not_null()) &
     (pl.col('ind') == pl.col('ind_b')) & (pl.col('h') == pl.col('h_b'))).sum().alias('identical'),
]).collect()
print(res, flush=True)

tr = (j.filter(pl.col('h').is_not_null() & pl.col('h_b').is_not_null() &
               (pl.col('ind') != pl.col('ind_b')))
       .group_by(['ind', 'ind_b']).len().sort('len', descending=True).head(25).collect())
print('[cellcmp] top industry transitions (base -> arm):', flush=True)
print(tr, flush=True)

# 同一行业但非行业列有差异的样本 (若有, 说明不是纯行业效应)
bad = (j.filter(pl.col('h').is_not_null() & pl.col('h_b').is_not_null() &
                (pl.col('ind') == pl.col('ind_b')) & (pl.col('h') != pl.col('h_b')))
       .head(20).collect())
print(f'[cellcmp] same-industry-but-different sample n={len(bad)}', flush=True)
print(bad, flush=True)
