"""阶段8c: 股权质押比例 IC探针 (2026-09-15)

数据源: 东财 RPT_CSDC_LIST (stock_gpzy_pledge_ratio_em), 月度快照回填
  filter (TRADE_DATE='YYYY-MM-DD'), pageSize=500, 每日期~10页
  快照日 = 每月最后交易日 (sh000001日历), 2021-01 ~ 2026-08 ≈ 68日期
测度: 月度截面 Spearman IC(质押比例, parquet future_ret=fwd5 ML同款标签)
  + 高质押桶(≥30%)统计 + 质押比例月变化(d_ratio) IC
判据: |IC|>0.02 且 IR>0.3 → 接线候选(预期负向风险信号); 否则否决
产出: rolling_validation_results/pledge_probe.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import pyarrow.parquet as pq

PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
OUT = '/mnt/d/quant/strategy/rolling_validation_results'
RAW = os.path.join(OUT, 'pledge_raw.pkl')
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'


def month_end_dates():
    idx = pd.read_csv('/mnt/d/quant/data/stock_data/backtrader_data/sh000001_qfq.csv',
                      parse_dates=['datetime'])
    idx = idx.set_index('datetime')
    mends = idx['close'].resample('ME').last().dropna().index
    mends = mends[(mends >= '2021-01-01') & (mends <= '2026-08-31')]
    return [pd.Timestamp(d) for d in mends]


def fetch_snapshot(trade_date):
    rows = []
    page = 1
    while True:
        params = {
            'sortColumns': 'PLEDGE_RATIO', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': str(page),
            'reportName': 'RPT_CSDC_LIST', 'columns': 'ALL',
            'quoteColumns': '', 'source': 'WEB', 'client': 'WEB',
            'filter': f"(TRADE_DATE='{trade_date}')",
        }
        for attempt in range(4):
            try:
                r = requests.get(URL, params=params, timeout=30,
                                 headers={'User-Agent': 'Mozilla/5.0'})
                j = r.json()
                res = j.get('result') or {}
                data = res.get('data')
                if data is None:
                    print(f'    [{trade_date} p{page}] code={j.get("code")} {j.get("message")}',
                          flush=True)
                    if j.get('code') == 9701:
                        time.sleep(15 * (attempt + 1))
                        continue
                    data = []
                break
            except Exception as e:
                if attempt == 3:
                    print(f'    [{trade_date} p{page}] FAIL {e}', flush=True)
                    return None
                time.sleep(5 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        if page >= res.get('pages', 1):
            break
        page += 1
        time.sleep(0.3)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['code'] = df['SECURITY_CODE'].astype(str).str.zfill(6)
    df['pledge_ratio'] = pd.to_numeric(df['PLEDGE_RATIO'], errors='coerce')
    df['pledge_mcap'] = pd.to_numeric(df['PLEDGE_MARKET_CAP'], errors='coerce')
    df['date'] = pd.to_datetime(trade_date)
    return df[['code', 'date', 'pledge_ratio', 'pledge_mcap']]


def main():
    dates = month_end_dates()
    print(f'月度快照日: {len(dates)} 个', flush=True)

    if os.path.exists(RAW):
        snap = pd.read_pickle(RAW)
        print(f'复用缓存 {RAW}: {len(snap)} 行', flush=True)
    else:
        frames = []
        t0 = time.time()
        for i, d in enumerate(dates):
            ds = d.strftime('%Y-%m-%d')
            df = fetch_snapshot(ds)
            if df is not None and len(df):
                frames.append(df)
            print(f'  {i+1}/{len(dates)} {ds}: {0 if df is None else len(df)} 行 '
                  f'({time.time()-t0:.0f}s)', flush=True)
            time.sleep(0.4)
        snap = pd.concat(frames, ignore_index=True)
        snap.to_pickle(RAW)
    snap = snap.dropna(subset=['pledge_ratio'])
    print(f'质押快照 {len(snap)} 行, {snap.code.nunique()} 只', flush=True)

    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'future_ret'])
    panel = t.to_pandas()
    panel['code'] = panel['code'].astype(str).str.zfill(6)
    panel['date'] = pd.to_datetime(panel['date'])
    merged = snap.merge(panel, on=['code', 'date'], how='inner')
    merged = merged.dropna(subset=['future_ret'])
    print(f'合并 {len(merged)} 行', flush=True)

    from scipy import stats as _st
    print('\n[IC] 质押比例 vs future_ret(fwd5), 按快照日截面:', flush=True)
    ics = []
    for d, sub in merged.groupby('date'):
        if len(sub) < 40:
            continue
        ics.append((d, len(sub), _st.spearmanr(sub['pledge_ratio'], sub['future_ret'])[0]))
    ics = pd.DataFrame(ics, columns=['date', 'n', 'ic'])
    print(f'  IC mean={ics.ic.mean():+.4f} IR={ics.ic.mean()/ics.ic.std():+.2f} '
          f'正率={100*(ics.ic > 0).mean():.0f}% (n截面={len(ics)})', flush=True)
    print('  逐年:')
    for y in sorted(ics.date.dt.year.unique()):
        v = ics[ics.date.dt.year == y].ic
        print(f'    {y}: n={len(v)} mean={v.mean():+.4f} IR={v.mean()/v.std():+.2f}', flush=True)

    print('\n[高质押桶] 质押比例≥30%:', flush=True)
    hi = merged[merged['pledge_ratio'] >= 30]
    lo = merged[merged['pledge_ratio'] < 30]
    print(f'  高桶 n={len(hi)} fwd5均值={100*hi.future_ret.mean():+.2f}% 胜率={100*(hi.future_ret > 0).mean():.0f}%',
          flush=True)
    print(f'  低桶 n={len(lo)} fwd5均值={100*lo.future_ret.mean():+.2f}% 胜率={100*(lo.future_ret > 0).mean():.0f}%',
          flush=True)
    # 高桶超额 (减同截面均值)
    m = merged.groupby('date')['future_ret'].transform('mean')
    merged['exc'] = merged['future_ret'] - m
    hi2 = merged[merged['pledge_ratio'] >= 30]
    print(f'  高桶截面超额 fwd5均值={100*hi2.exc.mean():+.2f}%', flush=True)

    print('\n[变化量IC] 质押比例月变化 vs future_ret:', flush=True)
    snap2 = snap.sort_values('date')
    snap2['d_ratio'] = snap2.groupby('code')['pledge_ratio'].diff()
    mg = snap2.merge(panel, on=['code', 'date'], how='inner').dropna(subset=['future_ret', 'd_ratio'])
    ics2 = []
    for d, sub in mg.groupby('date'):
        if len(sub) < 40:
            continue
        ics2.append((d, _st.spearmanr(sub['d_ratio'], sub['future_ret'])[0]))
    ics2 = pd.DataFrame(ics2, columns=['date', 'ic'])
    print(f'  d_ratio IC mean={ics2.ic.mean():+.4f} IR={ics2.ic.mean()/ics2.ic.std():+.2f} '
          f'(n={len(ics2)})', flush=True)

    pd.to_pickle({'merged': merged, 'ics': ics, 'ics_d': ics2},
                 os.path.join(OUT, 'pledge_probe.pkl'))
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
