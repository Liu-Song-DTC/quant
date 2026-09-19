#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0e成交价审计 (2026-09-20): 涨跌停日开盘成交拦截率与覆盖盲区。
生产已内置拦截: bt_execution预计算_LIMIT_DATA(收盘涨跌幅≥9.5%且振幅<20%→涨停,
ST±5%), _is_limit_up/_is_limit_down 在买单/卖单成交前拦截(逐日计数).
本审计对生产558笔已实现交易逐笔验证:
  a) 覆盖完备性: 已成交交易中, 是否存在"入场日一字板开盘(买不到)却按开盘价成交"
     或"出场日一字跌停开盘(卖不掉)却按开盘价成交" — 拦截盲区。
      判据: 开盘跳空≥板阈值(主板9.3%/创业科创19.3%) 且 当日high==low(全天封死)。
  b) 拦截器自身触发率: 交易之外, 有多少buy候选日被拦截(从signals×_LIMIT_DATA规则
     直接数, 用change_percent+amplitude列) — 拦截是"活跃阀门"还是"装饰品"。
  c) 跳空折让覆盖率(0b修复后): 成交价与当日OHLC的合理性(avg_cost/exit_px落在
     [low,high]之外的比例) — 0b修复的残余。
输入: 生产trade_realized(只读) + qfq日K(只读) + 涨跌停规则复刻。
"""
import os
import numpy as np
import pandas as pd

TR = '/mnt/d/quant/strategy/arms_20260919/C2_055/pre_trade_realized.csv'
KLINE = '/mnt/d/quant/data/stock_data/backtrader_data/{code}_qfq.csv'


def lim_thr(code):
    c = str(code).zfill(6)
    return 0.193 if c.startswith(('300', '301', '688', '689')) else 0.093


def main():
    tr = pd.read_csv(TR)
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    print(f'交易数: {len(tr)}')

    bad_entry, bad_exit, oob = [], [], []
    entry_at_limit_open = 0
    for _, t in tr.iterrows():
        p = KLINE.format(code=str(t['code']).zfill(6))
        if not os.path.exists(p):
            continue
        kl = pd.read_csv(p, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
        kl['date'] = kl['datetime'].dt.normalize()
        kl = kl.set_index('date')
        thr = lim_thr(t['code'])
        # 入场日
        if t['entry_date'] in kl.index:
            i = kl.index.get_loc(t['entry_date'])
            if i > 0:
                r = kl.iloc[i]
                gap = r['open'] / kl.iloc[i - 1]['close'] - 1
                sealed = (r['high'] <= r['low'] * 1.0005)  # 近似一字
                if gap >= thr and sealed:
                    bad_entry.append((t['code'], t['entry_date'], gap))
                if gap >= thr:
                    entry_at_limit_open += 1
        # 出场日
        if t['exit_date'] in kl.index:
            i = kl.index.get_loc(t['exit_date'])
            if i > 0:
                r = kl.iloc[i]
                gap = r['open'] / kl.iloc[i - 1]['close'] - 1
                sealed = (r['high'] <= r['low'] * 1.0005)
                if gap <= -thr and sealed:
                    bad_exit.append((t['code'], t['exit_date'], gap))
        # 成交价合理性: avg_cost=持有期VWAP(多笔加仓), 与首日range不可比 —
        # 正确判据=持有期[min(low), max(high)]; exit_px=单日成交, 与exit日range比
        en = kl.index.get_loc(t['entry_date']) if t['entry_date'] in kl.index else None
        ex = kl.index.get_loc(t['exit_date']) if t['exit_date'] in kl.index else None
        if en is not None and ex is not None and ex >= en:
            seg = kl.iloc[en:ex + 1]
            if not (seg['low'].min() * 0.995 <= t['avg_cost'] <= seg['high'].max() * 1.005):
                oob.append(('entry-vwap', t['code'], t['entry_date'], t['avg_cost'],
                            seg['low'].min(), seg['high'].max()))
        if ex is not None:
            r = kl.iloc[ex]
            if not (r['low'] * 0.995 <= t['exit_px'] <= r['high'] * 1.005):
                oob.append(('exit', t['code'], t['exit_date'], t['exit_px'],
                            r['low'], r['high']))

    print(f'\n=== a) 拦截盲区 ===')
    print(f'入场日涨停开盘: {entry_at_limit_open}笔, 其中一字封死(不可买): {len(bad_entry)}笔')
    for c, d, g in bad_entry[:10]:
        print(f'   {c} {d.date()} gap={g*100:+.1f}%')
    print(f'出场日跌停开盘且一字封死(不可卖): {len(bad_exit)}笔')
    for c, d, g in bad_exit[:10]:
        print(f'   {c} {d.date()} gap={g*100:+.1f}%')

    print(f'\n=== c) 成交价落在[low,high]外(±0.5%容差) ===')
    print(f'异常: {len(oob)}笔')
    for x in oob[:10]:
        print(f'   {x}')

    # b) 拦截器暴露面: 从bt日志的逐日计数行统计涨停跳过频率
    print('\n=== b) 拦截器触发面 (生产日志逐日计数) ===')
    import re
    import glob
    logs = glob.glob('/mnt/d/quant/strategy/logs/bt_execution_20260920_013033.log')
    if logs:
        txt = open(logs[0], encoding='utf-8', errors='ignore').read()
        ups = re.findall(r'涨停跳过(\d+)', txt)
        total_up = sum(int(x) for x in ups)
        print(f'  涨停跳过买入合计: {total_up}次 (逐日行数{len(ups)})')


if __name__ == '__main__':
    main()
