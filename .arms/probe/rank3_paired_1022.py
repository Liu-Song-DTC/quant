#!/usr/bin/env python3
"""同日名次配对检验 (2026-10-22, 只读): 第3名 vs 第2名/第1名.

组合层过滤器(new2)的 McNemar 共用对=0 不一致 (同一 (date,code) 结果恒等),
对它没有检验力. 本探针改用**同日名次配对**:
  每个出现第3名的调仓日, 取同日第2名(或第1名)作配对, 比较前瞻胜率/已实现胜率.
  - 精确 McNemar (双侧) = 只算名次间结果不一致的日子
  - 另报 Wilson CI / 比例差

用法: python rank3_paired_1022.py base_tr1022 [new2 ...]
"""
import sys
from math import comb, sqrt
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from mcnemar_sel_1022 import load, binom_two_sided, wrate


def wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def rank_panel(sel):
    """按引擎行序(文件行序)在每个 date 内编名次."""
    s = sel.copy().reset_index(drop=True)
    s['rank'] = s.groupby('date').cumcount() + 1
    return s


def pair_test(sel, hi, lo, col, label):
    """同日 hi 名 vs lo 名配对; col 为 'w5','w20' 或 'realized'."""
    s = rank_panel(sel)
    a = s[s['rank'] == lo][['date', 'code', col]].rename(columns={col: 'lo'})
    b = s[s['rank'] == hi][['date', 'code', col]].rename(columns={col: 'hi'})
    # 名次在日内唯一(cumcount) ⇒ 按 date 配对即"同日 hi名 vs lo名", 两者是不同股票
    m = a.merge(b, on=['date'], how='inner')
    m = m.dropna(subset=['lo', 'hi'])
    if not len(m):
        print(f'  {label}: 无可配对样本')
        return
    hw = (m['hi'] > 0).values; lw = (m['lo'] > 0).values
    hi_only = int((hw & ~lw).sum()); lo_only = int((~hw & lw).sum())
    p = binom_two_sided(hi_only, hi_only + lo_only)
    klo, n = int(lw.sum()), len(m)
    lo_ci = wilson(klo, n)
    print(f'  {label}: n(日)={n}  hi名={wrate(hw)*100:5.1f}% [{wilson(int(hw.sum()), n)[0]:.0f}-'
          f'{wilson(int(hw.sum()), n)[1]:.0f}]  lo名={wrate(lw)*100:5.1f}% '
          f'[{lo_ci[0]:.0f}-{lo_ci[1]:.0f}]  Δ={ (wrate(hw)-wrate(lw))*100:+.1f}pp  '
          f'仅hi赢={hi_only} 仅lo赢={lo_only}  McNemar p={p:.4f}')


def day_sign_test(sel, col):
    """绑定日层: 每日 lo 名的胜率 vs hi 名; 符号检验 (0.5 比例)."""
    s = rank_panel(sel)
    rows = []
    for d, g in s.groupby('date'):
        if len(g) < 3:
            continue
        hi = g[g['rank'] == 2][col].dropna()
        lo = g[g['rank'] == 3][col].dropna()
        if len(hi) and len(lo):
            rows.append((d, float((hi > 0).mean()), float((lo > 0).mean())))
    if not rows:
        return
    r = pd.DataFrame(rows, columns=['date', 'hi', 'lo'])
    d = r['lo'] - r['hi']
    pos = int((d > 0).sum()); neg = int((d < 0).sum()); n = pos + neg
    p = binom_two_sided(pos, n)
    print(f'  日层符号(第3名−第2名) {col}: n={len(r)} 均值Δ={(r["lo"].mean()-r["hi"].mean())*100:+.1f}pp '
          f'第3名更好={pos} 更差={neg} 持平={len(r)-pos-neg} 符号检验p={p:.4f}')


def main():
    tags = sys.argv[1:]
    for tag in tags:
        sel, trd = load(tag)
        print(f'\n=== {tag}  nSel={len(sel)} nTrd={len(trd)} ===')
        # 把已实现结果并到入选行 (entry_date = 次一交易日; trade_realized 有 entry_date)
        if 'exec_date' not in sel.columns:
            raise SystemExit('sel 缺 exec_date')
        m = sel.merge(trd[['entry_date', 'code', 'ret', 'hold_days']],
                      left_on=['exec_date', 'code'], right_on=['entry_date', 'code'],
                      how='left')
        m['realized'] = m['ret']
        print(f'  已实现可配对 {m["realized"].notna().sum()}/{len(m)}')
        for col, name in (('w5', 'w5'), ('w20', 'w20'), ('realized', '已实现')):
            for hi in (2, 1):
                pair_test(m, hi, 3, col, f'名次{hi} vs 名次3 {name}')
        for col in ('w5', 'w20'):
            day_sign_test(m, col)


if __name__ == '__main__':
    main()
