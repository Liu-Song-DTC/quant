#!/usr/bin/env python3
"""probe_pit_audit_0913.py — P0-1 PIT断言审计 · 第一刀: 基本面可用日口径

事实链(已铁证):
  - fundamental_data/*.csv 的 数据可用日期 ≡ 报告期 (200文件/13,186行 100%)
  - 写入方 data_manager.py:1320 / refresh_fundamental_recent.py:86 明示"系统既有约定"
  - 消费方: fundamental.py _get_available_data (signal_engine基本面积分/BOM),
    factor_preparer._preload_fundamental_cache (factor_df 6个fund_列→DYN IC验证/
    季度标定/ML训练特征) — 全部按"可用日=报告期"取数
  - A股披露铁律: 年报次年1~4月(法定截止4/30), Q1截止4/30, Q2截止8/31,
    Q3截止10/31 → 回测中基本面系统性提前1~4个月可见

本探针: 量化暴露 — 回测窗口(2021-2026, factor_df日期)内, 每个 (股票, 日):
  system_idx = 系统当前可见的最新报告 (数据可用日期≤D, 即报告期≤D)
  honest_idx = 按法定截止日(保守代理: 多数公司早于截止披露)可见的最新报告
  leaked = system_idx > honest_idx → 该日该股的基本面里有未来信息
输出: 逐年/全体 leaked 股票-日占比 + 面板股覆盖率。只读, 不碰基线。

执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_pit_audit_0913.py > logs/probe_pit_audit_0913.log 2>&1
"""
import os
import glob
import time
import resource

import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
FUND_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data')
PARQUET = os.path.join(ROOT, 'strategy', 'cache',
                       'factor_df_2718s_809d_d814a206.parquet')

# 法定披露截止(保守: 多数公司更早) — 报告期(mmddyy) → 可用日(含)
DEADLINE = {
    '0331': '0430',   # Q1
    '0630': '0831',   # Q2/半年报
    '0930': '1031',   # Q3
    '1231': '0430',   # 年报: 次年4/30 → 报告期+1年
}
# 少数公司报告期不在季末(如 0331/0630 之外的个别) → 统一规则: 季末+1月末
def proxy_avail(report_dt: pd.Series) -> pd.Series:
    """法定截止日保守代理: 季末报告期 → 下个披露截止日(自然日解析)"""
    out = []
    for r in report_dt:
        mmdd = r.strftime('%m%d')
        if mmdd == '1231':
            out.append(r + pd.offsets.MonthEnd(4))          # 次年4/30
        elif mmdd == '0331':
            out.append(r + pd.offsets.MonthEnd(1))          # 当年4/30
        elif mmdd == '0630':
            out.append(r + pd.offsets.MonthEnd(2))          # 当年8/31
        elif mmdd == '0930':
            out.append(r + pd.offsets.MonthEnd(1))          # 当年10/31
        else:
            # 非标准报告期: 保守给 +2月末
            out.append(r + pd.offsets.MonthEnd(2))
    return pd.Series(out)


def main():
    t0 = time.time()
    print('=' * 70, flush=True)
    print('P0-1 PIT审计第一刀: 基本面可用日口径暴露量化 (只读)', flush=True)
    print(f'系统口径: 数据可用日期=报告期 (写入方明示约定)', flush=True)
    print(f'诚实口径: 法定披露截止日 (保守: 多数公司早于截止)', flush=True)

    # 回测日期 (面板日期)
    fdf = pd.read_parquet(PARQUET, columns=['code', 'date'])
    bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
    fdf = fdf[~bad]
    codes = sorted(fdf['code'].astype(str).str.zfill(6).unique())
    dates = pd.DatetimeIndex(sorted(fdf['date'].unique()))
    dvals = dates.values
    n_codes, T = len(codes), len(dates)
    print(f'\n[1] 面板 {n_codes}只 × {T}日 ({dates.min():%Y-%m-%d}~'
          f'{dates.max():%Y-%m-%d})', flush=True)

    files = {os.path.basename(f).replace('.csv', '') : f
             for f in glob.glob(os.path.join(FUND_DIR, '*.csv'))}
    n_panel_files = sum(1 for c in codes if c in files)
    print(f'[1] 面板股中有基本面CSV: {n_panel_files}/{n_codes}', flush=True)

    # 逐年累计
    years = dates.year
    leak_cnt = np.zeros(T, dtype=np.int64)
    vis_cnt = np.zeros(T, dtype=np.int64)   # 该日有可见报告的股票数(系统口径)

    for i, code in enumerate(codes):
        if code not in files:
            continue
        df = pd.read_csv(files[code], usecols=['报告期'])
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce').dropna()
        if len(r) == 0:
            continue
        r = r.sort_values().values
        pv = proxy_avail(pd.Series(r)).values.astype('datetime64[D]')
        r_d = r.astype('datetime64[D]')
        # system可见: 数据可用日期=报告期 → idx = searchsorted(r, d, 'right')-1
        sys_idx = np.searchsorted(r_d, dvals, side='right') - 1
        # honest可见: idx = searchsorted(proxy, d, 'right')-1
        hon_idx = np.searchsorted(pv, dvals, side='right') - 1
        vis = sys_idx >= 0
        leak = vis & (sys_idx > hon_idx)
        leak_cnt += leak
        vis_cnt += vis
        if (i + 1) % 800 == 0:
            print(f'  ... {i+1}/{n_codes} 只 (elapsed {time.time()-t0:.0f}s)',
                  flush=True)

    leak_frac = leak_cnt / np.maximum(vis_cnt, 1)
    print(f'\n[2] 全体: 泄漏股票-日 {leak_cnt.sum():,} / 可见 {vis_cnt.sum():,} '
          f'= {100*leak_cnt.sum()/vis_cnt.sum():.1f}%', flush=True)
    print(f'[2] 逐年 (泄漏股票-日占比, 系统口径可见):', flush=True)
    for y in sorted(set(years)):
        m = years == y
        if vis_cnt[m].sum() == 0:
            continue
        print(f'    {y}: 泄漏 {leak_cnt[m].sum():,} / {vis_cnt[m].sum():,} '
              f'= {100*leak_cnt[m].sum()/vis_cnt[m].sum():.1f}%', flush=True)

    # 泄漏天数分布: 平均每股票-日提前看到多少天
    print(f'\n[3] 泄漏深度 (被提前可见的报告, 其法定截止日 - 当日):', flush=True)
    # 抽样估计: 前200只
    depths = []
    for code in codes[:200]:
        if code not in files:
            continue
        df = pd.read_csv(files[code], usecols=['报告期'])
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce').dropna()
        r = r.sort_values().values
        pv = proxy_avail(pd.Series(r)).values.astype('datetime64[D]')
        r_d = r.astype('datetime64[D]')
        for d in dvals:
            si = np.searchsorted(r_d, d, side='right') - 1
            hi = np.searchsorted(pv, d, side='right') - 1
            if si > hi:
                depths.append((pv[si] - d).astype(int))
    ds = np.array(depths)
    if len(ds):
        print(f'    样本 {len(ds):,} 个泄漏股票-日: 提前天数 中位 {np.median(ds):.0f}, '
              f'P25 {np.percentile(ds, 25):.0f}, P75 {np.percentile(ds, 75):.0f}, '
              f'P90 {np.percentile(ds, 90):.0f}, max {ds.max():.0f}', flush=True)

    print(f'\n总耗时 {time.time()-t0:.0f}s, rss='
          f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB', flush=True)


if __name__ == '__main__':
    main()
