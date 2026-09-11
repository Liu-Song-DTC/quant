#!/usr/bin/env python3
"""2026-09-11 阶段5a: 股份回购计划全历史下载 (东财 RPTA_WEB_GETHGLIST_NEW)
5511条回购计划(含预案公告日/计划期间/金额区间/实施进度)。新文件, 不动现有pkl。
PIT事件日候选: NOTICEDATE(公告日) / REPURADVANCEDATE(预案日), 探针时对比取舍。
保存 data/alternative_data/repurchase_plans.pkl
"""
import sys
import time
import requests
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0'}
OUT = '/mnt/d/quant/data/alternative_data/repurchase_plans.pkl'
KEEP = ['DIM_SCODE', 'SECURITYSHORTNAME', 'NOTICEDATE', 'REPURADVANCEDATE',
        'REPURSTARTDATE', 'REPURENDDATE', 'FINISHDATE', 'REPURPROGRESS',
        'JEXX', 'JESX', 'REPURAMOUNTLOWER', 'REPURAMOUNTLIMIT',
        'REPURNUMLOWER', 'REPURNUMCAP', 'ZSZXX', 'ZSZSX',
        'REPURPRICECAP', 'REPUROBJECTIVE', 'SHARETYPE', 'UPDATEDATE']


def fetch_all():
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    frames = []
    page = 1
    while True:
        params = {
            'sortColumns': 'NOTICEDATE', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': str(page),
            'reportName': 'RPTA_WEB_GETHGLIST_NEW',
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
        pages = res.get('pages') or 0
        if page >= pages:
            print(f'拉完 {pages} 页')
            break
        page += 1
        time.sleep(0.3)
    return frames


def main():
    frames = fetch_all()
    if not frames:
        print('无数据')
        sys.exit(1)
    raw = pd.concat(frames, ignore_index=True)
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df['code'] = df['DIM_SCODE'].astype(str).str.zfill(6)
    df['name'] = df['SECURITYSHORTNAME']
    df = df.drop(columns=['DIM_SCODE', 'SECURITYSHORTNAME'])
    for c in ['NOTICEDATE', 'REPURADVANCEDATE', 'REPURSTARTDATE', 'REPURENDDATE',
              'FINISHDATE', 'UPDATEDATE']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    df = df.drop_duplicates(subset=['code', 'NOTICEDATE'], keep='last')
    df = df.sort_values('NOTICEDATE').reset_index(drop=True)
    df.to_pickle(OUT)
    print(f'保存 {len(df)} 条 -> {OUT}')
    print(f'NOTICEDATE范围: {df.NOTICEDATE.min()} ~ {df.NOTICEDATE.max()}')
    print(f'REPURADVANCEDATE范围: {df.REPURADVANCEDATE.min()} ~ {df.REPURADVANCEDATE.max()}')
    print(f'2021起: {(df.NOTICEDATE >= "2021-01-01").sum()} 条')
    print(f'进度分布: {df.REPURPROGRESS.value_counts().head(6).to_dict()}')
    print(f'金额上限>1亿占比: {100*(df.JESX > 1e8).mean():.1f}%')
    print(f'占比总股本上限中位数: {df.ZSZSX.median():.3f}')


if __name__ == '__main__':
    main()
