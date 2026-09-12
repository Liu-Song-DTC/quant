#!/usr/bin/env python3
"""2026-09-12 parquet往返保真度探针: d814a206 → to_parquet → read_parquet 逐列比对

背景: A(pit 9/11夜)现算因子df→保存缓存→同进程ML训练得IC_A; B(9/12晨)读同一缓存
得IC_B≠IC_A。prepare之后代码两run逐字节相同 → 分歧必在factor_df本身。
唯一可能: to_parquet/read_parquet往返改变值(理论上float32无损, 从未实测)。

本探针: 加载d814a206 → 按save_factor_cache同款to_parquet(index=False) →
read_parquet → 逐列比对dtype与值(含NaN)。任一列不一致 → 找到分歧源。
往返bit级一致 → 转跑probe_ml_determinism_v2第2遍(跨进程ML确定性实锤测试)。

只读+临时文件: 不碰rolling_validation_results, 临时文件写tmp_factor/后删除。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')
TMP = os.path.join(BASE, 'tmp_factor', 'roundtrip_test_0912.parquet')


def main():
    t0 = time.time()
    a = pd.read_parquet(CACHE)
    print(f"[load] {len(a)} 行 × {a.shape[1]} 列 ({time.time()-t0:.0f}s)")

    # dtype清单 (供背景参考)
    dt = a.dtypes.value_counts()
    print(f"[dtype] {dict(dt)}")
    for special in ('date', 'code', 'industry', 'future_ret'):
        if special in a.columns:
            print(f"[dtype] {special}: {a[special].dtype}")

    # 保存/重读 (镜像cache_manager.save_factor_cache/load_factor_cache)
    t1 = time.time()
    os.makedirs(os.path.dirname(TMP), exist_ok=True)
    a.to_parquet(TMP, index=False)
    print(f"[save] {time.time()-t1:.0f}s, {os.path.getsize(TMP)/1e6:.0f}MB "
          f"(原始 {os.path.getsize(CACHE)/1e6:.0f}MB)")
    t2 = time.time()
    b = pd.read_parquet(TMP)
    print(f"[load] {time.time()-t2:.0f}s")

    # 逐列比对
    n_dtype = n_val = 0
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype != y.dtype:
            n_dtype += 1
            print(f"  [DTYPE!] {c}: {x.dtype} -> {y.dtype}")
        if x.dtype == object or isinstance(x.dtype, pd.StringDtype):
            same = (x.astype(object) == y.astype(object)) | (x.isna() & y.isna())
        else:
            same = (x == y) | (x.isna() & y.isna())
        bad = int((~same).sum())
        if bad:
            n_val += 1
            idx = np.where(~same)[0][:3]
            print(f"  [VALUE!] {c}: {bad}/{len(x)} 不一致; "
                  f"NaN {x.isna().sum()} vs {y.isna().sum()}")
            print(f"      例: {x.iloc[idx].tolist()} -> {y.iloc[idx].tolist()}")

    if n_dtype == 0 and n_val == 0:
        print("\n[结论] parquet往返逐列bit级一致 → 分歧不在此, 转ML跨进程确定性第2遍")
    else:
        print(f"\n[结论] 往返失保真: {n_dtype} 列dtype变, {n_val} 列值变 → 分歧源找到")

    try:
        os.remove(TMP)
    except Exception:
        pass


if __name__ == '__main__':
    main()
