#!/usr/bin/env python3
"""个股融资融券明细回填: SSE+SZSE 每日全标的, 2021-01 -> 最新 (~2.7M行, ~50min API限速跑).

用途: 每股融资余额变化率等截面特征进ML(feature先验: 杠杆资金拥挤度/投机需求).
交易日历: 2021年取自个股行情日期, 2022+取自margin_daily(市场级pkl已是交易日).
断点续跑: 已入库日期跳过, 每SAVE_EVERY天落盘一次. 失败日重试3次后记录跳过.
输出: data/alternative_data/margin_detail.pkl (date, code, name, rzye, rzmre, rzche, rqyl, rqmcl, rqchl, rqye)
单位: 金额=元, 余量=股. 与东财披露口径一致, 特征层用时T-1滞后(收盘后披露).
"""
import akshare as ak
import pandas as pd
import time, os, sys

PKL = '/mnt/d/quant/data/alternative_data/margin_detail.pkl'
CAL_MARGIN = '/mnt/d/quant/data/alternative_data/margin_daily.pkl'
CAL_STOCK = '/mnt/d/quant/data/stock_data/backtrader_data/000001_qfq.csv'
START, END = '2021-01-04', None
SLEEP, RETRIES, SAVE_EVERY = 0.6, 3, 20

SSE_MAP = {'信用交易日期': 'date', '标的证券代码': 'code', '标的证券简称': 'name',
           '融资余额': 'rzye', '融资买入额': 'rzmre', '融资偿还额': 'rzche',
           '融券余量': 'rqyl', '融券卖出量': 'rqmcl', '融券偿还量': 'rqchl'}
SZSE_MAP = {'证券代码': 'code', '证券简称': 'name', '融资买入额': 'rzmre', '融资余额': 'rzye',
            '融券卖出量': 'rqmcl', '融券余量': 'rqyl', '融券余额': 'rqye', '融资融券余额': 'rzye_rqye'}


def calendar():
    days = []
    df = pd.read_csv(CAL_STOCK, usecols=[0])
    col = df.columns[0]
    d = pd.to_datetime(df[col].astype(str))
    days += list(d[(d >= START) & (d < '2022-01-01')].dt.strftime('%Y-%m-%d'))
    md = pd.read_pickle(CAL_MARGIN)
    d2 = pd.to_datetime(md['date'].astype(str))
    days += list(d2[d2 >= '2022-01-01'].dt.strftime('%Y-%m-%d'))
    out, seen = [], set()
    for x in sorted(days):
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def fetch_day(ds_compact):
    frames = []
    sse = ak.stock_margin_detail_sse(date=ds_compact).rename(columns=SSE_MAP)
    sse = sse[[c for c in SSE_MAP.values() if c in sse.columns]]
    frames.append(sse)
    szse = ak.stock_margin_detail_szse(date=ds_compact).rename(columns=SZSE_MAP)
    szse['date'] = None  # SZSE无日期列, 由外层填
    frames.append(szse[[c for c in SZSE_MAP.values() if c in szse.columns]])
    df = pd.concat(frames, ignore_index=True)
    return df


def main():
    days = calendar()
    if END:
        days = [d for d in days if d <= END]
    done, buf = set(), []
    if os.path.exists(PKL):
        old = pd.read_pickle(PKL)
        done = set(old['date'].unique())
        buf = [old]
        print(f'已有 {len(old)} 行, {len(done)} 天', flush=True)
    todo = [d for d in days if d not in done]
    print(f'交易日 {len(days)} 个, 待补 {len(todo)} 天', flush=True)
    cols = ['date', 'code', 'name', 'rzye', 'rzmre', 'rzche', 'rqyl', 'rqmcl', 'rqchl', 'rqye']
    fails = []
    t0 = time.time()
    for i, d in enumerate(todo):
        got = None
        for attempt in range(RETRIES):
            try:
                got = fetch_day(d.replace('-', ''))
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    fails.append(d)
                    print(f'{d} 失败: {str(e)[:80]}', flush=True)
                else:
                    time.sleep(2 * (attempt + 1))
        if got is not None:
            got['date'] = d
            got['code'] = got['code'].astype(str).str.strip().str.zfill(6)
            for c in cols:
                if c not in got.columns:
                    got[c] = None
            buf.append(got[cols])
        if (i + 1) % SAVE_EVERY == 0 or i == len(todo) - 1:
            merged = pd.concat([b for b in buf if not b.empty], ignore_index=True) if buf else pd.DataFrame()
            merged = merged.drop_duplicates(subset=['date', 'code'], keep='last')
            merged.to_pickle(PKL)
            buf = [merged]
            el = (time.time() - t0) / 60
            print(f'[{i+1}/{len(todo)}] {d} 累计{len(merged)}行 {el:.0f}min', flush=True)
        time.sleep(SLEEP)
    print(f'完成. 失败日: {fails}', flush=True)


if __name__ == '__main__':
    main()
