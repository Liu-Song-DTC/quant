#!/usr/bin/env python3
"""2026-09-12 缓存逐列diff探针: 8c19c0a8/41343d0a vs d814a206 (值级, 定位指纹差异落在哪些列)

背景: IC_A在9/10晚+9/11全天稳定出现(8次运行/5个缓存), 唯一矛盾点=d814a206
(9/11晚IC_A, 9/12后IC_B)。数据→IC映射在其他缓存上处处自洽。
本探针: 逐列值比对三个缓存, 看指纹差异的列分布和量级 — 若差异列全在ML特征之外,
则三份缓存的ML训练数据实质相同 → 转8c19c0a8重训实验; 若差异列含ML特征,
则IC_A/IC_B数据驱动假说成立度更高。

只读。执行: cd strategy && python analysis/probe_cache_col_diff_0912.py
"""
import os
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE, 'cache')
REF = os.path.join(CACHE_DIR, 'factor_df_2718s_809d_d814a206.parquet')
TARGETS = ['8c19c0a8', '41343d0a']


def col_diff(ref_s: pd.Series, tgt_s: pd.Series):
    x, y = ref_s.values, tgt_s.values
    same = (x == y) | (pd.isna(x) & pd.isna(y))
    bad = int((~same).sum())
    out = {'n': bad}
    if bad and np.issubdtype(ref_s.dtype, np.number):
        idx = np.where(~same)[0]
        diffs = np.abs(x[idx].astype(float) - y[idx].astype(float))
        out['sum_abs'] = float(np.nansum(diffs))
        out['max_abs'] = float(np.nanmax(diffs))
    return out


def main():
    t0 = time.time()
    ref = pd.read_parquet(REF)
    print(f"[ref] d814a206: {len(ref)} 行 × {ref.shape[1]} 列 "
          f"({time.time()-t0:.0f}s), date={ref['date'].min().date()}~{ref['date'].max().date()}")
    for t in TARGETS:
        p = os.path.join(CACHE_DIR, f'factor_df_2718s_809d_{t}.parquet')
        t1 = time.time()
        df = pd.read_parquet(p)
        print(f"\n=== {t} vs d814a206 ({time.time()-t1:.0f}s load) ===")
        if len(df) != len(ref):
            print(f"  [行数不同] {len(df)} vs {len(ref)}")
            del df
            continue
        n_diff_cols = 0
        for c in ref.columns:
            if c not in df.columns:
                print(f"  [缺列] {c}")
                continue
            r = col_diff(ref[c], df[c])
            if r['n']:
                n_diff_cols += 1
                extra = (f" sum|Δ|={r.get('sum_abs', 0):.4g} "
                         f"max|Δ|={r.get('max_abs', 0):.4g}" if 'sum_abs' in r else "")
                print(f"  {c}: {r['n']}/{len(ref)} 处不同{extra}")
        if n_diff_cols == 0:
            print("  [全部131列值级一致]")
        del df
    print(f"\n总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    main()
