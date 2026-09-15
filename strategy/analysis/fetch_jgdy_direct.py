"""阶段8a: 机构调研明细直接抓取 (EM RPT_ORG_SURVEYNEW, 已验证可用)

要点(踩坑记录):
  - 旧报告 RPT_ORG_SURVEY: NOTICE_DATE过滤不支持(9701), pageSize>50拒绝 → 弃用
  - 新报告 RPT_ORG_SURVEYNEW: pageSize=500可用, 过滤 `(NUMBERNEW='1')(IS_SOURCE='1')(NOTICE_DATE>'YYYY-MM-DD')`
    (全部单引号) 实测 code 0, 全表 574页/286,549行 (2021-01-04 ~ 今)
  - 多NOTICE_DATE区间子句实测 9201空 → 不下界过滤, 全量拉取后本地裁剪
用法: python fetch_jgdy_direct.py   (断点续跑 via state文件)
"""
import os
import sys
import time
import requests
import pandas as pd

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
CSV = os.path.join(OUT, 'jgdy_raw_2021_2026.csv')
STATE = os.path.join(OUT, 'jgdy_fetch_state.txt')
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
BASE_PARAMS = {
    'sortColumns': 'NOTICE_DATE,SUM,RECEIVE_START_DATE,SECURITY_CODE',
    'sortTypes': '-1,-1,-1,1',
    'pageSize': '500',
    'reportName': 'RPT_ORG_SURVEYNEW',
    'columns': 'ALL',
    'quoteColumns': 'f2~01~SECURITY_CODE~CLOSE_PRICE,f3~01~SECURITY_CODE~CHANGE_RATE',
    'source': 'WEB', 'client': 'WEB',
    'filter': "(NUMBERNEW='1')(IS_SOURCE='1')(NOTICE_DATE>'2021-01-04')",
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
            if j.get('code') == 9701:  # 限流, 长退避
                time.sleep(15 * (attempt + 1))
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
            print(f'页{page}失败, 保存状态退出', flush=True)
            open(STATE, 'w').write(str(page - 1))
            sys.exit(1)
        df = pd.DataFrame(res['data'])
        df.to_csv(CSV, mode='a', header=(page == 1 and not os.path.exists(CSV)), index=False)
        n_pages_done += 1
        if page % 25 == 0 or page == total_pages:
            open(STATE, 'w').write(str(page))
            el = (time.time() - t0) / 60
            eta = el / max(n_pages_done, 1) * (total_pages - done - n_pages_done)
            print(f'  {page}/{total_pages} 页 {el:.1f}min ETA{eta:.1f}min', flush=True)
        time.sleep(0.35)
    n = sum(1 for _ in open(CSV)) - 1
    print(f'DONE CSV行数={n} 耗时{(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
