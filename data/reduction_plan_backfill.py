#!/usr/bin/env python3
"""减持计划公告管道: 东财公告大全-持股变动类, 全历史月度扫描.

背景(2026-08-26用户发现301257问题): RPT_SHARE_HOLDER_INCREASE只记录已执行减持,
预披露的减持计划(拟减持未实施)不进库 -> 8/24刚预披露减持的股票8/26仍被选入.
本管道补"计划层": 标题解析计划起点(预披露/拟减持)与终点(实施完成/届满/结果/终止),
供 get_reduction_plan_codes 状态机判定"减持计划期内".

输出: data/alternative_data/reduction_plans.pkl
列: ann_date, code, name, title, kind(start/end/other)
断点续跑: 已有数据覆盖的整月跳过(无减持公告的月会重复请求, 1次/月无害).
~70次月度请求, 全量首次约5-10min; 日增量=当前月1次请求.
"""
import time, os, calendar
import pandas as pd
from akshare.stock_fundamental.stock_notice import _stock_notice_report

PKL = '/mnt/d/quant/data/alternative_data/reduction_plans.pkl'
SLEEP = 0.8
START_YEAR, START_MONTH = 2021, 1

END_KEYS = ('实施完成', '期限届满', '结果公告', '实施完毕', '终止')


def classify(title):
    t = str(title)
    if '预披露' in t or '拟减持' in t or '拟计划减持' in t:
        return 'start'
    if any(k in t for k in END_KEYS):
        return 'end'
    return 'other'


def month_range(y, m):
    last = calendar.monthrange(y, m)[1]
    return f'{y:04d}-{m:02d}-01', f'{y:04d}-{m:02d}-{last:02d}'


def covered_months(df):
    if df is None or len(df) == 0:
        return set()
    d = pd.to_datetime(df['ann_date'])
    return {(y, m) for y, m in zip(d.dt.year, d.dt.month)}


def main():
    now = pd.Timestamp.now()
    all_months = []
    y, m = START_YEAR, START_MONTH
    while (y, m) <= (now.year, now.month):
        all_months.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1

    df = None
    if os.path.exists(PKL):
        df = pd.read_pickle(PKL)
        print(f'已有 {len(df)} 条公告记录', flush=True)
    done = covered_months(df)
    # 当前月永远重拉: 月内已有记录也算"覆盖"会漏掉后续新公告(增量坑, 2026-08-26修)
    now_ym = (now.year, now.month)
    prev_ym = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
    todo = [x for x in all_months if x not in done or x >= prev_ym]
    print(f'待扫月份: {len(todo)} (共{len(all_months)})', flush=True)

    for i, (y, m) in enumerate(todo):
        b, e = month_range(y, m)
        for attempt in range(3):
            try:
                raw = _stock_notice_report(symbol='持股变动', begin_date=b, end_date=e)
                break
            except Exception as ex:
                if attempt == 2:
                    print(f'{b[:7]} 失败: {str(ex)[:70]}', flush=True)
                    raw = pd.DataFrame()
                else:
                    time.sleep(3 * (attempt + 1))
        if raw is None or len(raw) == 0:
            time.sleep(SLEEP)
            continue
        jc = raw[raw['公告标题'].astype(str).str.contains('减持')].copy()
        if len(jc):
            jc = pd.DataFrame({
                'ann_date': jc['公告日期'].astype(str).str[:10],
                'code': jc['代码'].astype(str).str.zfill(6),
                'name': jc['名称'].astype(str),
                'title': jc['公告标题'].astype(str),
            })
            jc['kind'] = jc['title'].map(classify)
            df = jc if df is None else pd.concat([df, jc], ignore_index=True)
            df = df.drop_duplicates(subset=['code', 'ann_date', 'title']).reset_index(drop=True)
        df.to_pickle(PKL)
        n_s = int((df['kind'] == 'start').sum()) if len(df) else 0
        print(f'[{i+1}/{len(todo)}] {b[:7]} 本月减持公告{len(jc)}条, 累计{len(df)}(start {n_s})', flush=True)
        time.sleep(SLEEP)
    print('完成', flush=True)


if __name__ == '__main__':
    main()
