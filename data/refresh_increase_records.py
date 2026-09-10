#!/usr/bin/env python3
"""2026-09-10 阶段3: 股东增持记录下载 (东财 RPT_SHARE_HOLDER_INCREASE, DIRECTION=增持)
与 load_reduction 同接口另一半。全历史 ~33.7k 条, 68页, 一次性下载到
data/alternative_data/increase_records.pkl (新文件, 不动现有pkl)。
列保留: code/name/holder/change_num(万股)/change_rate/after_change_rate/hold_ratio/
        trade_average_price/start_date/end_date/eitime/market/direction
"""
import os
import sys
import time
import pickle
import requests
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0'}
OUT = '/mnt/d/quant/data/alternative_data/increase_records.pkl'


def fetch_all():
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    frames = []
    for page in range(1, 200):
        params = {
            'sortColumns': 'EITIME', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': str(page),
            'reportName': 'RPT_SHARE_HOLDER_INCREASE',
            'columns': 'ALL', 'source': 'WEB', 'client': 'WEB',
            'filter': '(DIRECTION="增持")',
        }
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=30, headers=UA)
                j = r.json()
                res = j.get('result') or {}
                data = res.get('data') or []
                if not data:
                    print(f'第{page}页空, 共{len(frames)}页数据')
                    return frames
                frames.append(pd.DataFrame(data))
                break
            except Exception as e:
                print(f'第{page}页 retry{attempt}: {e}')
                time.sleep(3)
        else:
            print(f'第{page}页3次失败, 终止')
            break
        pages = (j.get('result') or {}).get('pages') or 0
        if page >= pages:
            print(f'拉完 {pages} 页')
            break
        time.sleep(0.4)
    return frames


def main():
    frames = fetch_all()
    if not frames:
        print('无数据')
        sys.exit(1)
    raw = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({
        'code': raw['SECURITY_CODE'].astype(str).str.zfill(6),
        'name': raw.get('SECURITY_NAME_ABBR', ''),
        'holder': raw.get('HOLDER_NAME', ''),
        'change_num': pd.to_numeric(raw.get('CHANGE_NUM'), errors='coerce'),
        'change_rate': pd.to_numeric(raw.get('CHANGE_RATE'), errors='coerce'),
        'after_change_rate': pd.to_numeric(raw.get('AFTER_CHANGE_RATE'), errors='coerce'),
        'hold_ratio': pd.to_numeric(raw.get('HOLD_RATIO'), errors='coerce'),
        'trade_average_price': pd.to_numeric(raw.get('TRADE_AVERAGE_PRICE'), errors='coerce'),
        'start_date': pd.to_datetime(raw.get('START_DATE'), errors='coerce'),
        'end_date': pd.to_datetime(raw.get('END_DATE'), errors='coerce'),
        'eitime': pd.to_datetime(raw.get('EITIME'), errors='coerce'),
        'market': raw.get('MARKET', ''),
        'direction': raw.get('DIRECTION', ''),
    }).dropna(subset=['code', 'eitime'])
    df.to_pickle(OUT)
    print(f'保存 {len(df)} 条 -> {OUT}')
    print(f'eitime范围: {df.eitime.min()} ~ {df.eitime.max()}')
    print(f'change_rate>0 占比: {100*(df.change_rate>0).mean():.1f}%  (增持公告内实际变动为负的占比)')
    print(f'market分布: {df.market.value_counts().to_dict()}')
    print(f'年份分布: {df.eitime.dt.year.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
