#!/usr/bin/env python3
"""龙虎榜机构席位明细 全史回补 (2026-10-23, 新文件不触信号指纹).

用户 10/22 点名"主力资金流入" — mf5 价量代理已死(#390), dragon_tiger.pkl 只有
滚动1年快照(累计口径, 无逐日史). 本脚本回补**逐日×逐股机构席位**真实数据:
  端点: datacenter-web RPT_ORGANIZATION_TRADE_DETAILS (akshare stock_lhb_jgmmtj_em)
  覆盖: 2021-01 → 今, 月度分窗 (实测 2021-03 有 730 行, 全史可回补)
  ⚠ 多上榜原因行**金额重复**(实测 000007 两行同额) ⇒ groupby max 防双计.

输出: data/alternative_data/dragon_tiger_inst.pkl
  [code, date, name, nbuy_inst, nsell_inst, buy_amt, sell_amt, net_amt,
   mkt_amt, net_pct, turn, float_mv, reason]
  net_pct = 机构买入净额/市场总成交额; turn/float_mv 为当次快照.
**不在信号指纹 4 alt 文件清单内** (northbound/margin_daily/dragon_tiger_history/
dragon_tiger) ⇒ 新文件零指纹影响. 铁律: append-only — 重跑跳过已存在月份.

用法: .venv/bin/python data/backfill_dragon_tiger_inst.py
"""
import pickle
import time

import akshare as ak
import pandas as pd

DATA = '/mnt/d/quant/data/alternative_data'
OUT = f'{DATA}/dragon_tiger_inst.pkl'
START = (2021, 1)


def month_iter(y0, m0, y1, m1):
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_month(y, m, tries=3):
    import calendar
    last = calendar.monthrange(y, m)[1]
    # ⚠ 月末必须真实存在: 端点对 '2021-02-31' 这类非法日期返 None
    #   (实测 02/04/06/09/11 全失败的根因, 2026-10-23)
    s, e = f'{y}{m:02d}01', f'{y}{m:02d}{last:02d}'
    for a in range(tries):
        try:
            return ak.stock_lhb_jgmmtj_em(start_date=s, end_date=e)
        except Exception as ex:
            print(f'  [{y}-{m:02d}] 重试{a+1}: {ex}', flush=True)
            time.sleep(3 * (a + 1))
    return None


def main():
    import datetime as dt
    today = dt.date.today()
    done = set()
    try:
        old = pd.read_pickle(OUT)
        done = set(zip(old['date'].dt.year, old['date'].dt.month))
        print(f'已有 {len(old)} 行, 覆盖月份 {len(done)} — append-only 跳过')
    except Exception:
        old = None
    frames = [] if old is None else [old]
    failed = []
    for y, m in month_iter(*START, today.year, today.month):
        if (y, m) in done:
            continue
        df = fetch_month(y, m)
        if df is None or not len(df):
            print(f'[{y}-{m:02d}] 空/失败', flush=True)
            if df is None:
                failed.append(f'{y}-{m:02d}')
            time.sleep(0.4)
            continue
        df = df.rename(columns={'代码': 'code', '上榜日期': 'date', '名称': 'name',
                                '买方机构数': 'nbuy_inst', '卖方机构数': 'nsell_inst',
                                '机构买入总额': 'buy_amt', '机构卖出总额': 'sell_amt',
                                '机构买入净额': 'net_amt', '市场总成交额': 'mkt_amt',
                                '机构净买额占总成交额比': 'net_pct', '换手率': 'turn',
                                '流通市值': 'float_mv', '上榜原因': 'reason'})
        df = df[['code', 'date', 'name', 'nbuy_inst', 'nsell_inst', 'buy_amt',
                 'sell_amt', 'net_amt', 'mkt_amt', 'net_pct', 'turn', 'float_mv', 'reason']]
        # (code,date) 多原因行金额重复 ⇒ max 聚合防双计
        df['date'] = pd.to_datetime(df['date'])
        df = (df.groupby(['code', 'date'], as_index=False)
                .agg({'name': 'first', 'nbuy_inst': 'max', 'nsell_inst': 'max',
                      'buy_amt': 'max', 'sell_amt': 'max', 'net_amt': 'max',
                      'mkt_amt': 'max', 'net_pct': 'first', 'turn': 'first',
                      'float_mv': 'first', 'reason': 'first'}))
        frames.append(df)
        print(f'[{y}-{m:02d}] {len(df)} 股日', flush=True)
        time.sleep(0.4)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(['code', 'date'], keep='last').sort_values(['code', 'date'])
    out['code'] = out['code'].astype(str).str.zfill(6)
    with open(OUT, 'wb') as f:
        pickle.dump(out, f)
    print(f'\n落盘 {OUT}: {len(out)} 行 / {out["code"].nunique()} 只 / '
          f'{out["date"].min().date()} ~ {out["date"].max().date()}')
    if failed:
        print('失败月份(需补):', failed)


if __name__ == '__main__':
    main()
