#!/usr/bin/env python3
"""2026-09-10 阶段4: 股东户数下载 (东财 RPT_HOLDERNUM_DET, ~439k条, ~879页)
与 increase_records 同接口。一次下载到 data/alternative_data/holdernum_det.pkl。
PIT口径: END_DATE=报告期末, HOLD_NOTICE_DATE=公告日(知识日期), 携带使用公告日。
列保留: code/name/end_date/hold_notice_date/holder_num/pre_holder_num/
        holder_num_change/holder_num_ratio/avg_hold_num/eitime
"""
import os
import sys
import time
import requests
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0'}
OUT = '/mnt/d/quant/data/alternative_data/holdernum_det.pkl'


def fetch_all():
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    frames = []
    for page in range(1, 1000):
        params = {
            'sortColumns': 'END_DATE', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': str(page),
            'reportName': 'RPT_HOLDERNUM_DET',
            'columns': 'ALL', 'source': 'WEB', 'client': 'WEB',
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
        if page % 100 == 0:
            print(f'进度: {page}页/{pages}', flush=True)
    return frames


def main():
    frames = fetch_all()
    if not frames:
        print('无数据')
        sys.exit(1)
    raw = pd.concat(frames, ignore_index=True)
    print(f'原始列: {list(raw.columns)}')
    df = pd.DataFrame({
        'code': raw['SECURITY_CODE'].astype(str).str.zfill(6),
        'name': raw.get('SECURITY_NAME_ABBR', ''),
        'end_date': pd.to_datetime(raw.get('END_DATE'), errors='coerce'),
        'hold_notice_date': pd.to_datetime(raw.get('HOLD_NOTICE_DATE'), errors='coerce'),
        'holder_num': pd.to_numeric(raw.get('HOLDER_NUM'), errors='coerce'),
        'pre_holder_num': pd.to_numeric(raw.get('PRE_HOLDER_NUM'), errors='coerce'),
        'holder_num_change': pd.to_numeric(raw.get('HOLDER_NUM_CHANGE'), errors='coerce'),
        'holder_num_ratio': pd.to_numeric(raw.get('HOLDER_NUM_RATIO'), errors='coerce'),
        'avg_hold_num': pd.to_numeric(raw.get('AVG_HOLD_NUM'), errors='coerce'),
        'eitime': pd.to_datetime(raw.get('EITIME'), errors='coerce'),
    }).dropna(subset=['code', 'hold_notice_date'])
    df = df.drop_duplicates(subset=['code', 'end_date'], keep='first')
    df.to_pickle(OUT)
    print(f'保存 {len(df)} 条 -> {OUT}')
    print(f'公告日范围: {df.hold_notice_date.min()} ~ {df.hold_notice_date.max()}')
    print(f'end_date范围: {df.end_date.min()} ~ {df.end_date.max()}')
    print(f'年份分布: {df.hold_notice_date.dt.year.value_counts().sort_index().to_dict()}')
    print(f'股票数: {df.code.nunique()}')
    print(df.head())


if __name__ == '__main__':
    main()
