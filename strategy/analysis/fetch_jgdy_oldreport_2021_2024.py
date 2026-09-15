"""阶段8a补: 机构调研 2021-2024 历史回填 (旧报告 RPT_ORG_SURVEY)

为什么需要: RPT_ORG_SURVEYNEW 只保留近~20个月 (实测2025+). 旧报告有全历史
  但 pageSize 硬上限 50 (100/500 实测 9701).
过滤器(实测可用): (IS_SOURCE="1")(RECEIVE_START_DATE>'2020-12-31')(RECEIVE_START_DATE<'2025-01-01')
  → 31,852页 / 1,592,580行, 单线程 ~0.75s/页 ≈ 6.5-7h (过夜任务)
断点续跑: 状态文件 + CSV追加. 9701限流 → 指数退避, 失败即保存状态退出(可再跑).
用法: python fetch_jgdy_oldreport_2021_2024.py
"""
import os
import sys
import time
import requests
import pandas as pd

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
CSV = os.path.join(OUT, 'jgdy_raw_2021_2024.csv')
STATE = os.path.join(OUT, 'jgdy_old_fetch_state.txt')
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
BASE_PARAMS = {
    'sortColumns': 'NOTICE_DATE,RECEIVE_START_DATE,SECURITY_CODE,NUMBERNEW',
    'sortTypes': '-1,-1,1,-1',
    'pageSize': '50',
    'reportName': 'RPT_ORG_SURVEY',
    'columns': ('SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,RECEIVE_START_DATE,'
                'RECEIVE_OBJECT,RECEIVE_PLACE,RECEIVE_WAY_EXPLAIN,INVESTIGATORS,RECEPTIONIST,'
                'ORG_TYPE,NUMBERNEW'),
    'quoteColumns': 'f2~01~SECURITY_CODE~CLOSE_PRICE,f3~01~SECURITY_CODE~CHANGE_RATE',
    'quoteType': '0',
    'source': 'WEB', 'client': 'WEB',
    'filter': "(IS_SOURCE='1')(RECEIVE_START_DATE>'2020-12-31')(RECEIVE_START_DATE<'2025-01-01')",
}


def fetch_page(page):
    params = dict(BASE_PARAMS)
    params['pageNumber'] = str(page)
    for attempt in range(8):
        try:
            r = requests.get(URL, params=params, timeout=30,
                             headers={'User-Agent': 'Mozilla/5.0',
                                      'Referer': 'https://data.eastmoney.com/'})
            j = r.json()
            if j.get('result') and j['result'].get('data') is not None:
                return j['result']
            if j.get('code') == 9701:
                time.sleep(20 * (attempt + 1))
                continue
            print(f'  [页{page}] code={j.get("code")} {j.get("message")}', flush=True)
            return None
        except Exception as e:
            print(f'  [页{page} retry{attempt}] {e}', flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main():
    done = 0
    if os.path.exists(STATE):
        done = int(open(STATE).read().strip() or 0)
        print(f'断点: 已存 {done} 页', flush=True)

    r0 = fetch_page(1)
    if r0 is None:
        print('首探失败(限流?), 退出稍后重试', flush=True)
        sys.exit(1)
    total_pages = r0['pages']
    print(f'总页数 {total_pages} / 总行 {r0["count"]}', flush=True)

    t0 = time.time()
    n_pages_done = 0
    for page in range(done + 1, total_pages + 1):
        res = fetch_page(page)
        if res is None:
            print(f'页{page}失败, 保存状态退出 (重跑续传)', flush=True)
            open(STATE, 'w').write(str(page - 1))
            sys.exit(1)
        df = pd.DataFrame(res['data'])
        df.to_csv(CSV, mode='a', header=(page == 1 and not os.path.exists(CSV)), index=False)
        n_pages_done += 1
        if page % 100 == 0 or page == total_pages:
            open(STATE, 'w').write(str(page))
            el = (time.time() - t0) / 60
            eta = el / n_pages_done * (total_pages - done - n_pages_done)
            print(f'  {page}/{total_pages} 页 {el:.1f}min ETA{eta/60:.1f}h', flush=True)
        time.sleep(0.45)
    n = sum(1 for _ in open(CSV)) - 1
    print(f'DONE CSV行数={n} 耗时{(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
