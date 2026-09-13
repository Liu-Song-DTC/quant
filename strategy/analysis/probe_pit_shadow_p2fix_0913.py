#!/usr/bin/env python3
"""probe_pit_shadow_p2fix_0913.py — P0-1 PIT审计 · P2修正版(影子目录口径)

修正 probe_pit_shadow_materiality_0913.py 中 P2 段两处bug:
  ① 滞后分布读了源目录(files)而非影子目录 → 全0
  ② 深度段把修正后可用日数组同时当"报告期"用 → sys口径错位
  另: 影子可用日非单调(年报可用日晚于Q1), searchsorted(hon)失效 → 改用
  mask+argmax 精确复刻 FundamentalData 快路径语义(报告期降序+avail≤D掩码取首)。

只读。轻量(2列读 + 300只×690日矩阵), 可与后台P3共存。
"""
import os
import glob
import time
import resource

import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
SHADOW_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data_pit')
PARQUET = os.path.join(ROOT, 'strategy', 'cache',
                       'factor_df_2718s_809d_d814a206.parquet')


def main():
    t0 = time.time()
    print('=' * 70, flush=True)
    print('P2修正版: 影子目录滞后分布 + 泄漏深度(精确快路径语义)', flush=True)

    files = sorted(glob.glob(os.path.join(SHADOW_DIR, '*.csv')))
    print(f'[P2a] 影子滞后(数据可用日期-报告期), 报告期≥2021:', flush=True)
    lags = []
    for f in files:
        df = pd.read_csv(f, usecols=['报告期', '数据可用日期'])
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce')
        a = pd.to_datetime(df['数据可用日期'].astype(str), format='%Y%m%d',
                           errors='coerce')
        m = r >= pd.Timestamp('2021-01-01')
        lags.extend(((a - r) / np.timedelta64(1, 'D'))[m].dropna().tolist())
    s = pd.Series(lags)
    print(f'    中位 {s.median():.0f}天, P25 {s.quantile(.25):.0f}, '
          f'P75 {s.quantile(.75):.0f}, P90 {s.quantile(.9):.0f}, '
          f'max {s.max():.0f}, 0天占比 {(s==0).mean()*100:.1f}%, n={len(s):,}',
          flush=True)

    print(f'[P2b] 泄漏深度(影子口径), 面板2021+窗口:', flush=True)
    fdf = pd.read_parquet(PARQUET, columns=['code', 'date'])
    bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
    fdf = fdf[~bad]
    codes = sorted(fdf['code'].astype(str).str.zfill(6).unique())
    dates = pd.DatetimeIndex(sorted(fdf['date'].unique()))
    dvals = dates.values.astype('datetime64[D]')
    years = dates.year
    filemap = {os.path.basename(f).replace('.csv', ''): f for f in files}

    depths = []
    leak_years = {y: 0 for y in set(years)}
    vis_years = {y: 0 for y in set(years)}
    for i, code in enumerate(codes):
        f = filemap.get(code)
        if not f:
            continue
        df = pd.read_csv(f, usecols=['报告期', '数据可用日期'])
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce')
        a = pd.to_datetime(df['数据可用日期'].astype(str), format='%Y%m%d',
                           errors='coerce')
        ok = r.notna() & a.notna()
        if ok.sum() == 0:
            continue
        r = r[ok].values.astype('datetime64[D]')
        a = a[ok].values.astype('datetime64[D]')
        # 报告期升序(去重)
        order = np.argsort(r, kind='stable')
        r, a = r[order], a[order]
        sys_idx = np.searchsorted(r, dvals, side='right') - 1
        # hon: {avail≤D} 中报告期最大 → mask+argmax(报告期升序)
        M = (a[None, :] <= dvals[:, None])
        idx = np.where(M, np.arange(len(r))[None, :], -1)
        hon_idx = np.where(M.any(axis=1), idx.argmax(axis=1), -1)
        vis = sys_idx >= 0
        leak = vis & (sys_idx > hon_idx)
        for y, lv, vs in zip(years, leak, vis):
            if vs:
                vis_years[y] += 1
                leak_years[y] += 1 if lv else 0
        if leak.any():
            delta = (a[sys_idx[leak]] - dvals[leak]).astype(int)
            depths.extend(delta.tolist())
    ds = np.array(depths)
    if len(ds):
        print(f'    中位 {np.median(ds):.0f}天, P25 {np.percentile(ds,25):.0f}, '
              f'P75 {np.percentile(ds,75):.0f}, P90 {np.percentile(ds,90):.0f}, '
              f'max {ds.max():.0f}, n={len(ds):,}', flush=True)
    tot = sum(vis_years.values())
    print(f'    逐年泄漏/可见:', flush=True)
    for y in sorted(leak_years):
        if vis_years[y]:
            print(f'      {y}: {leak_years[y]:,}/{vis_years[y]:,} = '
                  f'{100*leak_years[y]/vis_years[y]:.1f}%', flush=True)
    print(f'    全体: {sum(leak_years.values()):,}/{tot:,} = '
          f'{100*sum(leak_years.values())/max(tot,1):.1f}%', flush=True)
    print(f'\n总耗时 {time.time()-t0:.0f}s, rss='
          f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB', flush=True)


if __name__ == '__main__':
    main()
