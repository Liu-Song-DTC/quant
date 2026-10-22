#!/usr/bin/env python3
"""conceptbf_1022 机制判据(a): 信号格改写是否**局限在 2021-01..2023-01** (2026-10-22).

输入 = sigcells.py 产出的两份逐格摘要 pkl (A=生产, B=臂).
输出:
  1. 三个日期桶的改写格数/占比: <2023-01-03 / 2023-01-03..2023-04-03 / >2023-04-03
  2. 逐年改写格数
  3. 每个桶内, 哪些列(bycol 按日摘要)发生变化 —— 归因"直接热值"vs"ML/闸门间接"
预注册: 桶1 应有大量改写; 桶2 小 (EMA 尾随); 桶3 直接改写应为 0;
   若桶3 bycol 有变, 只允许是 ML 相关列(score/ml_*)且必须与 ML 重训解释一致.
用法: python bfsig_bucket_1022.py A.pkl B.pkl
"""
import pickle
import sys

import numpy as np
import pandas as pd

PA, PB = sys.argv[1], sys.argv[2]
A = pickle.load(open(PA, 'rb'))
B = pickle.load(open(PB, 'rb'))

allc = sorted(set(A['codes']) | set(B['codes']))
alld = sorted(set(A['dates']) | set(B['dates']))
ci = pd.Index(allc).get_indexer
di = pd.Index(alld).get_indexer
nd = len(alld)


def keys(P):
    c = ci(np.asarray(P['codes'], dtype=object)[P['cid']])
    d = di(np.asarray(P['dates'], dtype=object)[P['did']])
    k = c.astype(np.int64) * nd + d
    o = np.argsort(k, kind='stable')
    return k[o], P['h'][o], d[o]


kA, hA, dA = keys(A)
kB, hB, dB = keys(B)
onlyA = np.setdiff1d(kA, kB, assume_unique=True)
onlyB = np.setdiff1d(kB, kA, assume_unique=True)
common, ia, ib = np.intersect1d(kA, kB, assume_unique=True, return_indices=True)
chg = hA[ia] != hB[ib]
cd = np.asarray(alld, dtype=object)[dA[ia][chg]]
print(f'A={PA}  B={PB}')
print(f'格: A {len(kA):,} B {len(kB):,} | 新增 {len(onlyB):,} 消失 {len(onlyA):,} '
      f'改写 {int(chg.sum()):,} ({100 * chg.sum() / max(1, len(common)):.2f}% of common)')

BUCKETS = [('< 2023-01-03', '0000-00-00', '2023-01-02'),
           ('2023-01-03..2023-04-03', '2023-01-03', '2023-04-03'),
           ('> 2023-04-03', '2023-04-04', '9999-99-99')]
print('\n=== 1. 日期桶 ===')
for name, lo, hi in BUCKETS:
    m = (cd >= lo) & (cd <= hi)
    totA = np.sum((np.asarray(alld, dtype=object)[dA] >= lo) &
                  (np.asarray(alld, dtype=object)[dA] <= hi))
    print(f'  {name:24s} 改写 {int(m.sum()):7,} 格 / A 格 {int(totA):7,} '
          f'= {100 * m.sum() / max(1, totA):6.2f}%')

print('\n=== 2. 逐年 ===')
s = pd.Series(cd).str[:4]
for y, n in s.value_counts().sort_index().items():
    print(f'  {y}: {n:,}')

print('\n=== 3. 逐列归因 (桶内出现 bycol 日摘要变化的列) ===')
cb = sorted(set(A['block']) & set(B['block']))
for name, lo, hi in BUCKETS:
    days = [d for d in cb if lo <= d <= hi]
    rows = []
    for c in A['cols']:
        ca, cbb = A['bycol'][c], B['bycol'][c]
        n = sum(1 for d in days if ca.get(d) != cbb.get(d))
        if n:
            rows.append((c, n))
    rows.sort(key=lambda r: -r[1])
    print(f'  {name}: 桶内日期 {len(days)}; 有列变化的日期块 {len(rows)} 列:')
    for c, n in rows[:15]:
        print(f'      {c:28s} {n:5d} 日 ({100 * n / max(1, len(days)):.1f}%)')

print('\n=== 4. 增删格按桶 (集合变化, 非内容) ===')
allc_arr = np.asarray(allc, dtype=object)
for lab, kk in (('新增', onlyB), ('消失', onlyA)):
    if not len(kk):
        continue
    dd = np.asarray(alld, dtype=object)[(kk % nd).astype(np.int64)]
    print(f'  {lab}: ' + ' '.join(
        f'{nm}={int(((dd >= lo) & (dd <= hi)).sum())}'
        for nm, lo, hi in BUCKETS))
