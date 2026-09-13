#!/usr/bin/env python3
"""probe_pit_shadow_materiality_0913.py — P0-1 PIT审计 · 第二刀: 基本面泄漏实质影响

第一刀结论(probe_pit_audit_0913.py): 数据可用日期≡报告期(100%), 回测窗口内
58.6% 的基本面可见股票-日用未公开财报(法定截止保守代理, 真实公告更早→真实泄漏
占比低于此, 但方向确定)。

本刀三阶段(只读, 不改生产):
  P1 影子目录 data/stock_data/fundamental_data_pit/ — 逐行修正 数据可用日期 =
     max(最新公告日期, zcfz_/lrb_/xjll_公告日期), clamp≥报告期, 全缺失→法定截止
     规则(0331→0430, 0630→0831, 0930→1031, 1231→次年0430)。max取齐=整行(三表
     合并)在全部语句公开后才可用, 重述行自动获得重述日(其值确实是重述值)。
  P2 合理性: 近期行(报告期≥2021)修正后滞后分布 + 泄漏深度(修复第一刀单位bug)。
  P3 实质影响: 生产 FundamentalData 类双实例(原目录 vs 影子目录), 同函数对比
     fund_score/profit_decline 在 2021+ 策略窗口的变化分布。

执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_pit_shadow_materiality_0913.py > logs/probe_pit_shadow_materiality_0913.log 2>&1
"""
import os
import sys
import glob
import time
import resource

import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
SRC_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data')
SHADOW_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data_pit')
PARQUET = os.path.join(ROOT, 'strategy', 'cache',
                       'factor_df_2718s_809d_d814a206.parquet')

DATE_COLS = ['最新公告日期', 'zcfz_公告日期', 'lrb_公告日期', 'xjll_公告日期']


def parse_date(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().split('.')[0]
    for fmt in ('%Y-%m-%d', '%Y%m%d', '%Y/%m/%d'):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except Exception:
            continue
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def deadline_rule(report_ts: pd.Timestamp) -> pd.Timestamp:
    """法定披露截止(保守): 多数公司早于截止"""
    mmdd = report_ts.strftime('%m%d')
    if mmdd == '1231':
        return report_ts + pd.offsets.MonthEnd(4)
    if mmdd == '0331':
        return report_ts + pd.offsets.MonthEnd(1)
    if mmdd == '0630':
        return report_ts + pd.offsets.MonthEnd(2)
    if mmdd == '0930':
        return report_ts + pd.offsets.MonthEnd(1)
    return report_ts + pd.offsets.MonthEnd(2)


def correct_avail(row, report_ts: pd.Timestamp) -> pd.Timestamp:
    """修正可用日 = max(四公告日) ∪ {报告期}, 全缺失→法定截止"""
    best = report_ts
    for c in DATE_COLS:
        v = row.get(c)
        t = parse_date(v)
        if t is not None and t > best:
            best = t
    if best == report_ts and all(parse_date(row.get(c)) is None for c in DATE_COLS):
        best = deadline_rule(report_ts)
    return best


def main():
    t0 = time.time()
    print('=' * 70, flush=True)
    print('P0-1 PIT审计第二刀: 基本面泄漏实质影响 (只读)', flush=True)

    # ---------- P1 影子目录 ----------
    files = sorted(glob.glob(os.path.join(SRC_DIR, '*.csv')))
    n_avail_from_cols = 0
    n_deadline_fallback = 0
    if os.path.isdir(SHADOW_DIR) and len(os.listdir(SHADOW_DIR)) == len(files):
        print(f'[P1] 影子目录已存在({len(files)}文件), 跳过构建', flush=True)
    else:
        os.makedirs(SHADOW_DIR, exist_ok=True)
        for i, f in enumerate(files):
            df = pd.read_csv(f, dtype={'报告期': str})
            if '报告期' not in df.columns:
                df.to_csv(os.path.join(SHADOW_DIR, os.path.basename(f)),
                          index=False)
                continue
            rp = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                                format='%Y%m%d', errors='coerce')
            new_avail = []
            for _, row in df.iterrows():
                rt = rp.iloc[_]
                if pd.isna(rt):
                    new_avail.append(row.get('数据可用日期', ''))
                    continue
                ca = correct_avail(row, rt)
                if ca == deadline_rule(rt):
                    n_deadline_fallback += 1
                else:
                    n_avail_from_cols += 1
                new_avail.append(ca.strftime('%Y%m%d'))
            df['数据可用日期'] = new_avail
            df.to_csv(os.path.join(SHADOW_DIR, os.path.basename(f)), index=False)
            if (i + 1) % 1000 == 0:
                print(f'  [P1] {i+1}/{len(files)} (elapsed {time.time()-t0:.0f}s)',
                      flush=True)
        print(f'[P1] 影子目录构建完成: 公告日修正 {n_avail_from_cols:,} 行, '
              f'法定截止兜底 {n_deadline_fallback:,} 行', flush=True)

    # ---------- P2 合理性 ----------
    print('\n[P2] 修正后滞后分布 (报告期≥2021-01-01):', flush=True)
    lags, depths = [], []
    for f in files[:800]:
        df = pd.read_csv(f, usecols=['报告期', '数据可用日期'])
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce')
        a = pd.to_datetime(df['数据可用日期'].astype(str), format='%Y%m%d',
                           errors='coerce')
        m = r >= pd.Timestamp('2021-01-01')
        lags.extend(((a - r) / np.timedelta64(1, 'D'))[m].dropna().tolist())
    s = pd.Series(lags)
    if len(s):
        print(f'    修正后 可用日-报告期: 中位 {s.median():.0f}天, '
              f'P25 {s.quantile(.25):.0f}, P75 {s.quantile(.75):.0f}, '
              f'P90 {s.quantile(.9):.0f}, max {s.max():.0f}', flush=True)
    # 泄漏深度 (第一刀单位bug修复): 影子可用日 - 当日, 在泄漏日上
    fdf = pd.read_parquet(PARQUET, columns=['code', 'date'])
    bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
    fdf = fdf[~bad]
    codes = sorted(fdf['code'].astype(str).str.zfill(6).unique())
    dates = pd.DatetimeIndex(sorted(fdf['date'].unique()))
    dvals = dates.values.astype('datetime64[D]')
    dstr = dates.strftime('%Y%m%d')
    filemap = {os.path.basename(f).replace('.csv', ''): f for f in files}
    depths = []
    for code in codes[:300]:
        f = filemap.get(code)
        if not f:
            continue
        sdf = pd.read_csv(f, usecols=['数据可用日期'])
        avail = pd.to_datetime(sdf['数据可用日期'].astype(str), format='%Y%m%d',
                               errors='coerce').dropna().sort_values()
        if len(avail) == 0:
            continue
        av = avail.values.astype('datetime64[D]')
        rp = avail.values.astype('datetime64[D]')  # 报告期≈原可用日(原=报告期)
        # 泄漏日: 影子可见idx < 原口径可见idx
        hon = np.searchsorted(av, dvals, side='right') - 1
        sys_idx = np.searchsorted(rp, dvals, side='right') - 1
        leak = (sys_idx > hon) & (sys_idx >= 0)
        if leak.any():
            delta = (av[sys_idx[leak]] - dvals[leak]).astype(int)
            depths.extend(delta.tolist())
    ds = np.array(depths)
    if len(ds):
        print(f'    泄漏深度(影子可用日-当日, 泄漏日上): 中位 {np.median(ds):.0f}天, '
              f'P25 {np.percentile(ds,25):.0f}, P75 {np.percentile(ds,75):.0f}, '
              f'P90 {np.percentile(ds,90):.0f}, n={len(ds):,}', flush=True)

    # ---------- P3 实质影响 ----------
    print('\n[P3] 实质影响: 生产FundamentalData双实例同函数对比 (2021+):', flush=True)
    sys.path.insert(0, os.path.join(ROOT, 'strategy'))
    from core.fundamental import FundamentalData
    from core.factor_calculator import compute_fundamental_score

    fd_sys = FundamentalData(SRC_DIR, stock_codes=codes)
    fd_hon = FundamentalData(SHADOW_DIR, stock_codes=codes)
    dates_w = [d for d in dates if d >= pd.Timestamp('2021-01-04')]
    print(f'    窗口 {len(dates_w)}日 × {len(codes)}只', flush=True)

    score_chg = 0
    score_onoff = 0          # fund_score 0→>0 或 >0→0
    decline_flip = 0
    total = 0
    n_done = 0
    for code in codes:
        for d in dates_w:
            total += 1
            try:
                r_s = fd_sys.get_roe(code, d)
                pg_s = fd_sys.get_profit_growth(code, d)
                rg_s = fd_sys.get_revenue_growth(code, d)
                e_s = fd_sys.get_eps(code, d)
                r_h = fd_hon.get_roe(code, d)
                pg_h = fd_hon.get_profit_growth(code, d)
                rg_h = fd_hon.get_revenue_growth(code, d)
                e_h = fd_hon.get_eps(code, d)
            except Exception:
                continue
            if r_s is None and pg_s is None and rg_s is None and e_s is None:
                continue
            fs_s = compute_fundamental_score(roe=r_s, profit_growth=pg_s,
                                             revenue_growth=rg_s, eps=e_s)
            fs_h = compute_fundamental_score(roe=r_h, profit_growth=pg_h,
                                             revenue_growth=rg_h, eps=e_h)
            if abs(fs_s - fs_h) > 1e-9:
                score_chg += 1
            if (fs_s > 0) != (fs_h > 0):
                score_onoff += 1
        n_done += 1
        if n_done % 400 == 0:
            print(f'    ... {n_done}/{len(codes)} 只 (elapsed {time.time()-t0:.0f}s)',
                  flush=True)

    print(f'    fund_score 变化: {score_chg:,}/{total:,} = {100*score_chg/max(total,1):.1f}%',
          flush=True)
    print(f'    fund_score 开关翻转(0↔>0): {score_onoff:,} = '
          f'{100*score_onoff/max(total,1):.1f}%', flush=True)

    print(f'\n总耗时 {time.time()-t0:.0f}s, rss='
          f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB', flush=True)


if __name__ == '__main__':
    main()
