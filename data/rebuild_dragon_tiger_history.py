#!/usr/bin/env python3
"""龙虎榜历史明细 dict 直接重建 (绕过 akshare)

2026-09-04: akshare 路径 (ak.stock_lhb_detail_daily_em) 当天 0/33 月失败,
但 datacenter-web.eastmoney.com 接口实测可用 → 改为直连 datacenter-web 逐月拉取.

输出格式与 akshare 版一致: {6位代码: set(上榜日期)} (2024-01 ~ 2026-09).
覆盖前备份 .preRebuild_0904 (仅一次). 失败月份跳过不写入 (fail-open, 不损坏旧数据).

用法:
    python data/rebuild_dragon_tiger_history.py        # 立即重建
被 refresh_altdata_0902.py 调用: 每日刷新链路的一部分.
"""
import calendar
import json
import os
import pickle
import shutil
import time
import urllib.parse
import urllib.request

DATA = '/mnt/d/quant/data/alternative_data'
OUT = os.path.join(DATA, 'dragon_tiger_history.pkl')
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'


def fetch_month(ym, tries=4):
    """拉单月全部上榜明细 → [(6位代码, 日期字符串)]"""
    y, m = int(ym[:4]), int(ym[5:7])
    last = calendar.monthrange(y, m)[1]
    rows = []
    page = 1
    while True:
        flt = f"(TRADE_DATE>='{ym}-01')(TRADE_DATE<='{ym}-{last:02d}')"
        params = {
            'reportName': 'RPT_DAILYBILLBOARD_DETAILSNEW',
            'columns': 'SECURITY_CODE,TRADE_DATE',
            'pageNumber': page,
            'pageSize': 500,  # 接口实际每页封顶500行, 传更大也按500返
            'sortColumns': 'TRADE_DATE',
            'sortTypes': '1',
            'filter': flt,
        }
        qs = urllib.parse.urlencode(params)
        d = None
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(URL + '?' + qs, timeout=20) as r:
                    d = json.load(r)
                break
            except Exception:
                time.sleep(2.0 * (attempt + 1))
        if d is None or not d.get('result'):
            break
        res = d['result']
        for it in res.get('data') or []:
            code = str(it.get('SECURITY_CODE', '')).strip()
            date = str(it.get('TRADE_DATE', ''))[:10]
            if len(code) == 6 and date:
                rows.append((code, date))
        total = res.get('count', 0)
        if page * 500 >= total:
            break
        page += 1
        time.sleep(0.15)
    return rows


def rebuild(months_start='2024-01', months_end=None, verbose=True):
    """重建 {code: set(dates)}. months_end=None → 当前月. 失败月份跳过. 返回 (hist, 成功月数)."""
    import datetime as _dt
    if months_end is None:
        months_end = _dt.date.today().strftime('%Y-%m')
    months = []
    y0, m0 = int(months_start[:4]), int(months_start[5:7])
    y1, m1 = int(months_end[:4]), int(months_end[5:7])
    yy, mm = y0, m0
    while (yy, mm) <= (y1, m1):
        months.append(f'{yy}-{mm:02d}')
        mm += 1
        if mm > 12:
            mm, yy = 1, yy + 1
    all_rows = []
    ok = 0
    for ym in months:
        try:
            rows = fetch_month(ym)
            all_rows.extend(rows)
            if rows:
                ok += 1
            if verbose:
                print(f'  {ym}: {len(rows)} 行')
            time.sleep(0.15)
        except Exception as e:
            if verbose:
                print(f'  [FAIL] {ym}: {str(e)[:80]}')
    hist = {}
    for code, date in all_rows:
        hist.setdefault(code, set()).add(_dt.date.fromisoformat(date))
    if verbose:
        all_dates = sorted({d for v in hist.values() for d in v})
        print(f'历史明细: {len(hist)} 只, {all_dates[0]} -> {all_dates[-1]} 共{len(all_dates)}个上榜日'
              f' (成功 {ok}/{len(months)} 月)')
    return hist, ok


def write_hist(hist, ok_months, min_ok=1):
    """ok_months >= min_ok 才写入 (fail-open: 全失败不动旧文件)"""
    if ok_months < min_ok:
        print(f'[SKIP] 成功月份 {ok_months} < {min_ok}, 保留旧文件')
        return False
    if os.path.exists(OUT):
        bak = OUT + '.preRebuild_0904'
        if not os.path.exists(bak):
            shutil.copy2(OUT, bak)
            print(f'备份: {bak}')
    with open(OUT, 'wb') as f:
        pickle.dump(hist, f)
    print(f'写入: {OUT} ({len(hist)} 只)')
    return True


if __name__ == '__main__':
    hist, ok = rebuild(verbose=True)
    write_hist(hist, ok, min_ok=3)
