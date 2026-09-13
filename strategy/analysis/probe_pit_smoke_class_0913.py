#!/usr/bin/env python3
"""A段语义烟测: 真实FundamentalData类 getter vs 向量化复刻 — 逐位精确对账

纪律(用户反馈): 复刻语义必须与生产类同函数对账后才可信。
抽样10只×每37日, 比较 (sys/hon两口径) getter→compute_fundamental_score 结果
与 probe_pit_materiality_vector_0913 的向量化复刻, 必须 100% 一致。
"""
import sys
import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
sys.path.insert(0, ROOT + '/strategy')
sys.path.insert(0, ROOT + '/strategy/analysis')
from core.fundamental import FundamentalData
from core.factor_calculator import compute_fundamental_score
import probe_pit_materiality_vector_0913 as vec

SRC = ROOT + '/data/stock_data/fundamental_data'
HON = ROOT + '/data/stock_data/fundamental_data_deadline'
PARQUET = ROOT + '/strategy/cache/factor_df_2718s_809d_d814a206.parquet'

fdf = pd.read_parquet(PARQUET, columns=['code', 'date'])
bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
fdf = fdf[~bad]
codes = sorted(fdf['code'].astype(str).str.zfill(6).unique())
dates = pd.DatetimeIndex(sorted(fdf['date'].unique()))
dates_w = dates[dates >= pd.Timestamp('2021-01-04')]

rng = np.random.default_rng(7)
sample = sorted(rng.choice(codes, size=10, replace=False))
d_sample = dates_w[::37]
dstr = vec.to_obj(pd.DatetimeIndex(d_sample).strftime('%Y%m%d'))
dvals = d_sample.values.astype('datetime64[D]')

fd_sys = FundamentalData(SRC, stock_codes=sample)
fd_hon = FundamentalData(HON, stock_codes=sample)

n_chk = 0
n_mismatch = 0
for code in sample:
    df = pd.read_csv(HON + '/' + code + '.csv', dtype={'报告期': str})
    df['数据可用日期'] = df['数据可用日期'].astype(str)
    r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                       format='%Y%m%d', errors='coerce')
    ok = r.notna()
    df = df[ok].reset_index(drop=True)
    r = r[ok].values.astype('datetime64[D]')
    avail_hon = vec.to_obj(df['数据可用日期'])
    sys_idx = np.searchsorted(r, dvals, side='right') - 1
    M = avail_hon[None, :] <= dstr[:, None]
    idx = np.where(M, np.arange(len(r))[None, :], -1)
    hon_idx = np.where(M.any(axis=1), idx.max(axis=1), -1)
    roe = vec.parse_pct_col(df['净资产收益率'])
    pg = vec.parse_pct_col(df['净利润-同比增长'])
    rg = vec.parse_pct_col(df['营业总收入-同比增长'])
    eps = vec.parse_eps_col(df['每股收益'])
    sc = vec.row_signal_score(roe, pg, rg, eps)

    for j, d in enumerate(d_sample):
        r_s = fd_sys.get_roe(code, d)
        pg_s = fd_sys.get_profit_growth(code, d)
        rg_s = fd_sys.get_revenue_growth(code, d)
        e_s = fd_sys.get_eps(code, d)
        fs_s = compute_fundamental_score(roe=r_s, profit_growth=pg_s,
                                         revenue_growth=rg_s, eps=e_s)
        r_h = fd_hon.get_roe(code, d)
        pg_h = fd_hon.get_profit_growth(code, d)
        rg_h = fd_hon.get_revenue_growth(code, d)
        e_h = fd_hon.get_eps(code, d)
        fs_h = compute_fundamental_score(roe=r_h, profit_growth=pg_h,
                                         revenue_growth=rg_h, eps=e_h)
        v_s = sc[np.clip(sys_idx[j], 0, None)] if sys_idx[j] >= 0 else 0.0
        v_h = sc[np.clip(hon_idx[j], 0, None)] if hon_idx[j] >= 0 else 0.0
        n_chk += 2
        if abs(fs_s - v_s) > 1e-12:
            n_mismatch += 1
            print(f'  SYS MISMATCH {code} {d.date()}: class={fs_s:.4f} '
                  f'vec={v_s:.4f} idx={sys_idx[j]}')
        if abs(fs_h - v_h) > 1e-12:
            n_mismatch += 1
            print(f'  HON MISMATCH {code} {d.date()}: class={fs_h:.4f} '
                  f'vec={v_h:.4f} idx={hon_idx[j]}')
print(f'烟测: {n_chk} 次对比, 不一致 {n_mismatch} — '
      f'{"PASS(逐位一致)" if n_mismatch == 0 else "FAIL"}')
