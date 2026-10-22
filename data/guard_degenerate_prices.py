#!/usr/bin/env python3
"""#356 退化价格守卫 (2026-10-23): 数据层换名前 fail-closed 内容闸.

接在 refresh_all.sh 步骤3 (rebuild → 守卫 → 原子换名) — 在 BT_NEW 换名进生产
**之前**全量扫描, 任何退化 ⇒ 非零退出 ⇒ set -e 链中止, 生产目录不被污染。

零容忍依据 (2026-10-23 普查 .arms/probe/degen_census_1023.py):
  5380 个生产文件 负价/零价/NaN/倒挂/负量额/重复日期 全为 0 —— 表驱动乘性
  管线数学上不可能从正 raw 产出非正价 ⇒ 任何命中 = 上游源数据损坏/管线 bug,
  正是该大声失败的事件 (对照 10-17 换名静默丢 57 文件教训: 静默本身是缺陷)。

检查项 (逐文件):
  1. OHLC 任一 NaN / <=0
  2. high < low (倒挂)
  3. volume < 0 / amount < 0
  4. datetime 重复
  5. 数据行 == 0
  6. 行数单调性 (--prod 给定时): rows(new) < rows(prod) ⇒ 历史丢行 = 改写历史
     (铁律家族; raw 只 append ⇒ rebuild 产物行数只增不减, 沿用分支逐字节相等)。

用法: python guard_degenerate_prices.py <new_dir> [--prod <prod_dir>]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']


def scan_file(fp):
    """返回 (n_rows, issues dict)。issues 空 = 干净。"""
    try:
        d = pd.read_csv(fp, usecols=COLS)
    except Exception as e:
        return 0, {'read_err': str(e)[:120]}
    iss = {}
    ohlc = [pd.to_numeric(d[c], errors='coerce').values for c in ('open', 'high', 'low', 'close')]
    m_nan = np.zeros(len(d), dtype=bool)
    m_nonpos = np.zeros(len(d), dtype=bool)
    for a in ohlc:
        m_nan |= np.isnan(a)
        m_nonpos |= (a <= 0) & ~np.isnan(a)
    h, l = ohlc[1], ohlc[2]
    m_hilo = (h < l) & ~np.isnan(h) & ~np.isnan(l)
    v = pd.to_numeric(d['volume'], errors='coerce').values
    a_ = pd.to_numeric(d['amount'], errors='coerce').values
    if m_nan.any():
        iss['ohlc_nan'] = int(m_nan.sum())
    if m_nonpos.any():
        iss['ohlc_nonpos'] = int(m_nonpos.sum())
    if m_hilo.any():
        iss['high_lt_low'] = int(m_hilo.sum())
    if (v < 0).any():
        iss['vol_neg'] = int((v < 0).sum())
    if (a_ < 0).any():
        iss['amt_neg'] = int((a_ < 0).sum())
    dup = int(d['datetime'].duplicated().sum())
    if dup:
        iss['dup_datetime'] = dup
    if len(d) == 0:
        iss['zero_rows'] = 1
    return len(d), iss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('new_dir', help='待换名的新目录 (rebuild 产物)')
    ap.add_argument('--prod', default=None, help='生产目录 (行数单调性基准)')
    a = ap.parse_args()

    t0 = time.time()
    files = sorted(f for f in os.listdir(a.new_dir)
                   if f.endswith('_qfq.csv') and not f.startswith('._'))
    print(f"[退化守卫] 扫描 {len(files)} 个文件: {a.new_dir}"
          + (f" (行数基准: {a.prod})" if a.prod else ''), flush=True)

    prod_rows = {}
    if a.prod and os.path.isdir(a.prod):
        for f in os.listdir(a.prod):
            if f.endswith('_qfq.csv') and not f.startswith('._'):
                fp = os.path.join(a.prod, f)
                try:
                    with open(fp, 'rb') as fh:
                        prod_rows[f] = sum(1 for _ in fh) - 1  # 减表头
                except OSError:
                    pass

    bad = {}
    n_rows_issue = 0
    for i, fn in enumerate(files):
        n, iss = scan_file(os.path.join(a.new_dir, fn))
        if a.prod and fn in prod_rows and n < prod_rows[fn]:
            iss['rows_regression'] = f'{n} < prod {prod_rows[fn]}'
            n_rows_issue += 1
        if iss:
            bad[fn] = iss
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(files)}", flush=True)

    if not bad:
        print(f"[退化守卫] ✅ {len(files)} 文件全干净 "
              f"(零 NaN/非正/倒挂/负量额/重复日期/丢行), {time.time() - t0:.0f}s")
        return 0

    print(f"\n[退化守卫] ⛔ {len(bad)}/{len(files)} 文件有退化格 — 拒绝换名, 生产未动。明细:")
    for fn, iss in sorted(bad.items(), key=lambda x: x[0]):
        print(f"  {fn}: {iss}")
    # 最重的前 3 个文件抽原始行取证
    print('\n样例取证 (前 3 个问题文件的首个坏行):')
    shown = 0
    for fn in sorted(bad):
        if shown >= 3:
            break
        try:
            d = pd.read_csv(os.path.join(a.new_dir, fn), usecols=COLS)
            o = d[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
            m = (o.isna().any(axis=1) | (o <= 0).any(axis=1) | (o['high'] < o['low']))
            if m.any():
                print(f'-- {fn}:')
                print(d[m].head(2).to_string(index=False))
                shown += 1
        except Exception:
            pass
    print('\n排查指引: 负价/零价 ⇒ 查 raw 源文件当日行; 丢行 ⇒ raw 被重载收缩'
          ' (EM侧复权变化自动全量重载?); 全部 ⇒ 先跑 .arms/probe/degen_census_1023.py 比对。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
