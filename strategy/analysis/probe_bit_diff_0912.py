#!/usr/bin/env python3
"""2026-09-12 缓存bit级diff探针: 8c19c0a8 vs d814a206 (值级已证全同, 查bit级残留差异)

背景: 值级(==)比对131列全同, 但今天8c19c0a8的ML训练=IC_B±5e-5, d814a206=IC_B精确。
值级相同仍有bit级差异可能: -0.0vs+0.0(符号位)、NaN payload(指数全1尾数不同)。
若bit级100%全同 → ±5e-5抖动=训练真非确定(同bit不同结果), 机制转线程/归约;
若存在bit差 → 文件本体差异可解释抖动, 机制转parquet内部编码。

本探针: 逐列 uint64视图比对, 分类差异为符号位-only/NaN-payload/量级, 并对比schema dtype。
只读。执行: cd strategy && python analysis/probe_bit_diff_0912.py > logs/probe_bit_diff_0912.log 2>&1
"""
import os
import sys
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE, 'cache')
REF = os.path.join(CACHE_DIR, 'factor_df_2718s_809d_d814a206.parquet')
TGT = os.path.join(CACHE_DIR, 'factor_df_2718s_809d_8c19c0a8.parquet')

SIGN_ONLY = np.uint64(0x8000000000000000)
EXP_ALL1 = np.uint64(0x7FF0000000000000)  # float64指数位全1
EXP_ALL1_32 = np.uint32(0x7F800000)
SIGN_32 = np.uint32(0x80000000)


def classify(a, b):
    """返回 (n_sign0, n_nanpayload, n_magnitude)"""
    mask = a != b
    n = int(mask.sum())
    if n == 0:
        return 0, 0, 0
    d = mask.nonzero()[0]
    da, db = a[d], b[d]
    if a.dtype == np.uint64:
        sign_only = (da ^ db) == SIGN_ONLY
        a_nan = (da & EXP_ALL1) == EXP_ALL1
        b_nan = (db & EXP_ALL1) == EXP_ALL1
        nan_payload = sign_only ^ True  # placeholder
        nan_payload = (~sign_only) & a_nan & b_nan
    else:  # uint32
        sign_only = (da ^ db) == SIGN_32
        a_nan = (da & EXP_ALL1_32) == EXP_ALL1_32
        b_nan = (db & EXP_ALL1_32) == EXP_ALL1_32
        nan_payload = (~sign_only) & a_nan & b_nan
    mag = n - int(sign_only.sum()) - int(nan_payload.sum())
    return int(sign_only.sum()), int(nan_payload.sum()), int(mag)


def main():
    t0 = time.time()
    for p in (REF, TGT):
        if not os.path.exists(p):
            print(f'[fatal] 不存在: {p}')
            sys.exit(2)
    import pyarrow.parquet as pq
    sch_ref = pq.read_schema(REF)
    sch_tgt = pq.read_schema(TGT)
    print(f'[schema] 列数: ref={len(sch_ref.names)} tgt={len(sch_tgt.names)}')
    print(f'[schema] 列序一致: {sch_ref.names == sch_tgt.names}')
    n_dt = 0
    for c in sch_ref.names:
        dt_r, dt_t = sch_ref.field(c).type, sch_tgt.field(c).type
        if str(dt_r) != str(dt_t):
            n_dt += 1
            print(f'  [DTYPE!] {c}: {dt_r} vs {dt_t}')
    print(f'[schema] dtype不同列数: {n_dt}')

    t1 = time.time()
    ref = pd.read_parquet(REF)
    print(f'[load] ref {len(ref)} 行 ({time.time()-t1:.0f}s)')
    t1 = time.time()
    tgt = pd.read_parquet(TGT)
    print(f'[load] tgt {len(tgt)} 行 ({time.time()-t1:.0f}s)')

    tot = [0, 0, 0]
    for c in ref.columns:
        if ref[c].dtype not in (np.float64, np.float32):
            continue
        if ref[c].dtype == np.float64 and tgt[c].dtype == np.float64:
            a = ref[c].values.view(np.uint64)
            b = tgt[c].values.view(np.uint64)
        elif ref[c].dtype == np.float32 and tgt[c].dtype == np.float32:
            a = ref[c].values.view(np.uint32)
            b = tgt[c].values.view(np.uint32)
        else:
            continue
        s0, npay, mag = classify(a, b)
        if s0 or npay or mag:
            tot[0] += s0
            tot[1] += npay
            tot[2] += mag
            print(f'  {c}({ref[c].dtype}): 符号位{s0} NaNpayload{npay} 量级{mag}')
    print(f'[bit] 全列合计: 符号位{tot[0]} NaNpayload{tot[1]} 量级{tot[2]}')
    if tot == [0, 0, 0]:
        print('[bit] 两文件131列bit级100%全同 → 今天±5e-5抖动=训练真非确定(同bit不同结果)')
    else:
        print('[bit] 存在bit级差异 → 文件本体可解释抖动')
    print(f'总耗时 {(time.time()-t0)/60:.1f}min')


if __name__ == '__main__':
    main()
