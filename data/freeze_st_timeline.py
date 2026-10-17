#!/usr/bin/env python
"""冻结 ST 时间线 (2026-10-17) — 修铁律违反: `股票简称` 是构建态快照

## 问题

`FundamentalData.get_st_timeline` 从 `fundamental_data/*.csv` 的 **`股票简称`** 列推 ST 状态。
那一列**不是 PIT 值**: 它是**文件构建时**的名称, 被写进该股的多行。后果(实测):

  · `000004` 63 行横跨 2010–2026, **每一行都写 `*ST国华`** —— 它 2025 年才戴帽
  · 5585 只中 **88 只"所有行都含ST"** ⇒ 整段历史被追溯排除
  · 基本面刷新会**改写该列** ⇒ 一只股票今天戴帽, 全部历史被改写 ⇒ **f(t;T) ≠ f(t;t)**

## 为什么"冻结"而不是"重建 PIT"

试过三条重建路, 全部不可行:

  1. **涨跌停恒等式状态机**(`core/st_timeline.py`, K=2 连续命中 ±5% 板):
     实测**只判出 18 只当前 ST / 快照说 153 只 ⇒ 漏检 138 = 90%**。
     ⚠ 记忆里"K=2 对照组 0 假阳性"的结论**只验了精确率没验召回率** —— 真 ST 股
     很少连续两天打板 ⇒ 该探测器实际近乎从不触发。**不可用**。
  2. **名称变更历史**: `ak.stock_info_change_name`(新浪) 与东财 F10 都**只给名字序列、不给日期**
     ⇒ 无法定时间线。
  3. **创业板/科创板**: ST 与常态同为 ±20% 板, 裸价**原理上无判别力**(2029 只 / 其中 ST 79 只)。

且记忆 `st-snapshot-lookahead-20261017` 已实测该前视的**影响在噪声地板下**(Δ CI 含 0)。

## 本方案

把**当前 `get_st_timeline` 的输出**固化成数据文件 `st_timeline_frozen.pkl`:

  · **逐位保持现有行为** ⇒ 不需要新锚(产物逐位相同)
  · **铁律成立**: 冻结文件不再被基本面刷新改写; 要改它 = 显式、需登记的**数据重锚**
  · 残留前视被**显式化并局部化**到一个文件, 而非散落在 5585 个基本面文件里

用法:
  python freeze_st_timeline.py            # 生成/覆盖
  python freeze_st_timeline.py --verify   # 只校验现有文件与当前逻辑是否一致
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

PROJ = '/mnt/d/quant'
sys.path.insert(0, f'{PROJ}/strategy')

# 放在 alternative_data/ — 与其它**冻结契约**(corporate_actions.pkl /
# concept_map_history.pkl) 同处, 该目录已版本化 ⇒ 可 diff = 可审计
OUT = f'{PROJ}/data/alternative_data/st_timeline_frozen.pkl'


def build():
    from core.fundamental import FundamentalData
    fd = FundamentalData(f'{PROJ}/data/stock_data/fundamental_data')
    codes = sorted(fd._file_map.keys())
    print(f'基本面 {len(codes)} 只')
    out, n_st_codes, n_st_days, n_days = {}, 0, 0, 0
    t0 = time.time()
    for i, c in enumerate(codes):
        try:
            tl = fd.get_st_timeline(c)
        except Exception:
            tl = None
        if tl is None:
            continue
        av, stf = tl
        out[c] = (np.asarray(av), np.asarray(stf, dtype=bool))
        n_days += len(av)
        n_st_days += int(stf.sum())
        if stf.any():
            n_st_codes += 1
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(codes)}  {time.time()-t0:.0f}s', flush=True)
    print(f'  完成 {len(out)} 只; 含ST的 {n_st_codes} 只; ST股票-日 {n_st_days:,}/{n_days:,} '
          f'= {100*n_st_days/max(n_days,1):.2f}%')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()

    if a.verify:
        if not os.path.exists(OUT):
            print(f'[verify] 冻结文件不存在: {OUT}')
            return 1
        with open(OUT, 'rb') as fh:
            d = pickle.load(fh)
        meta = d.get('__meta__', {})
        tl = {k: v for k, v in d.items() if k != '__meta__'}
        print(f'[verify] 冻结于 {meta.get("as_of")}  {len(tl)} 只  {meta.get("n_st_days"):,} ST股票-日')
        fresh = build()
        same = diff = 0
        for c in set(tl) | set(fresh):
            if c not in tl or c not in fresh:
                diff += 1
                continue
            a1, b1 = tl[c]
            a2, b2 = fresh[c]
            if len(a1) == len(a2) and np.array_equal(a1, a2) and np.array_equal(b1, b2):
                same += 1
            else:
                diff += 1
        print(f'[verify] 与当前逻辑逐位相同 {same}  不同 {diff}')
        return 0 if diff == 0 else 2

    tl = build()
    n_st_days = int(sum(int(v[1].sum()) for v in tl.values()))
    payload = dict(tl)
    payload['__meta__'] = {
        'as_of': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'FundamentalData.get_st_timeline (股票简称 构建态快照, 非PIT)',
        'note': '冻结即合约: 更新本文件 = 数据重锚, 必须登记台账并重跑回测',
        'n_codes': len(tl), 'n_st_days': n_st_days,
    }
    tmp = OUT + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)
    os.replace(tmp, OUT)
    print(f'落盘 {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
