#!/usr/bin/env python3
"""2026-09-11 阶段5b: 大宗交易明细下载 (东财 RPT_DATA_BLOCKTRADE, 2021-01-01起)
全历史68万条, 只需回测窗口2021+ (~30万条)。500/页, 按TRADE_DATE倒序拉到2021-01-01止。
折价率DISCOUNT_RATIO/溢价率PREMIUM_RATIO为信号核心; 东财自带CHANGE_RATE_20DAYS
留作校验, 探针自算市场调整收益。
保存 data/alternative_data/blocktrades.pkl
"""
import sys
import time
import requests
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0'}
OUT = '/mnt/d/quant/data/alternative_data/blocktrades.pkl'
KEEP = ['TRADE_DATE', 'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'DEAL_PRICE',
        'CLOSE_PRICE', 'PRE_CLOSE_PRICE', 'DISCOUNT_RATIO', 'PREMIUM_RATIO',
        'DEAL_AMT', 'DEAL_VOLUME', 'BUYER_NAME', 'SELLER_NAME',
        'CHANGE_RATE_20DAYS', 'MARKET']


def fetch_all():
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    frames = []
    page = 1
    while True:
        params = {
            'sortColumns': 'TRADE_DATE', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': str(page),
            'reportName': 'RPT_DATA_BLOCKTRADE',
            'columns': 'ALL', 'source': 'WEB', 'client': 'WEB',
            'filter': '(TRADE_DATE>=\'2021-01-01\')',
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
        if page % 100 == 0:
            print(f'已拉 {page} 页, 最新TRADE_DATE={frames[-1].TRADE_DATE.min()}')
        pages = res.get('pages') or 0
        if page >= pages:
            print(f'拉完 {pages} 页')
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
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df['code'] = df['SECURITY_CODE'].astype(str).str.zfill(6)
    df = df.drop(columns=['SECURITY_CODE', 'SECURITY_NAME_ABBR'])
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'], errors='coerce')
    for c in ['DEAL_PRICE', 'CLOSE_PRICE', 'PRE_CLOSE_PRICE', 'DISCOUNT_RATIO',
              'PREMIUM_RATIO', 'DEAL_AMT', 'DEAL_VOLUME', 'CHANGE_RATE_20DAYS']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('TRADE_DATE').reset_index(drop=True)
    df.to_pickle(OUT)
    print(f'保存 {len(df)} 条 -> {OUT}')
    print(f'TRADE_DATE范围: {df.TRADE_DATE.min()} ~ {df.TRADE_DATE.max()}')
    print(f'折价率分布: 折价(DISCOUNT_RATIO>0) {100*(df.DISCOUNT_RATIO>0).mean():.1f}% '
          f'溢价(PREMIUM_RATIO>0) {100*(df.PREMIUM_RATIO>0).mean():.1f}%')
    print(f'深折价(>8%)占比: {100*(df.DISCOUNT_RATIO>8).mean():.1f}%')
    print(f'日均笔数: {len(df)/(df.TRADE_DATE.nunique()):.1f}')


if __name__ == '__main__':
    main()
