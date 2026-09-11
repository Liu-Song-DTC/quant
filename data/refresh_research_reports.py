#!/usr/bin/env python3
"""2026-09-11 阶段5c: 研报评级数据下载 (reportapi.eastmoney.com/report/list, 2021-01-01起)
~9.9万条研报(2021-2026), 含publishDate/orgSName/sRatingName/ratingChange(0首次
1调低 2维持 3调高 4未知, 语义待探针校验)/emRatingName + 盈利预测字段。
保存 data/alternative_data/research_reports.pkl
"""
import sys
import time
import requests
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0'}
OUT = '/mnt/d/quant/data/alternative_data/research_reports.pkl'


def fetch_all():
    url = 'https://reportapi.eastmoney.com/report/list'
    frames = []
    page = 1
    while True:
        params = {
            'industryCode': '*', 'pageSize': '500', 'industry': '*', 'rating': '',
            'ratingChange': '', 'beginTime': '2021-01-01', 'endTime': '2026-09-11',
            'pageNo': str(page), 'fields': '', 'qType': '0', 'orgCode': '',
            'code': '*', 'rcode': '',
        }
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=30, headers=UA)
                j = r.json()
                data = j.get('data') or []
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
        hits = j.get('hits') or 0
        if page * 500 >= hits:
            print(f'拉完 {page} 页 / hits={hits}')
            break
        page += 1
        time.sleep(0.25)
    return frames


def main():
    frames = fetch_all()
    if not frames:
        print('无数据')
        sys.exit(1)
    raw = pd.concat(frames, ignore_index=True)
    keep = ['title', 'stockCode', 'stockName', 'orgSName', 'publishDate',
            'sRatingName', 'ratingChange', 'emRatingName', 'infoCode']
    for c in raw.columns:
        if c.lower().startswith('predict') and keep.count(c) == 0:
            keep.append(c)
    df = raw[[c for c in keep if c in raw.columns]].copy()
    df['code'] = df['stockCode'].astype(str).str.zfill(6)
    df = df.drop(columns=['stockCode'])
    df['publishDate'] = pd.to_datetime(df['publishDate'], errors='coerce')
    df = df.drop_duplicates(subset=['code', 'orgSName', 'publishDate', 'title'])
    df = df.sort_values('publishDate').reset_index(drop=True)
    df.to_pickle(OUT)
    print(f'保存 {len(df)} 条 -> {OUT}')
    print(f'publishDate范围: {df.publishDate.min()} ~ {df.publishDate.max()}')
    print(f'ratingChange分布: {df.ratingChange.value_counts(dropna=False).to_dict()}')
    print(f'评级分布: {df.sRatingName.value_counts().head(8).to_dict()}')
    print(f'覆盖股票数: {df.code.nunique()}')
    print(f'机构数: {df.orgSName.nunique()}')


if __name__ == '__main__':
    main()
