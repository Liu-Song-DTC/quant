#!/usr/bin/env python3
"""组合层披露窗口探针 — 生产实际建仓的披露距离分布 (2026-09-21)

前置: probe_disclosure_window_20260921.py 发现信号层0-5d披露桶fwd10d +2.85%/
60.1%胜率, 120+d桶仅+0.86% — 披露距离是真实条件变量。组合层问题:
生产的实际建仓事件(559笔realized)是否已集中在高alpha桶? 若生产分布≈信号池
分布(披露盲选), 且生产建仓偏斜向低alpha桶 → 披露距离tilt有headroom;
若生产已集中好桶 → 无headroom, 信号层发现不可转译(与E-seq1同签名风险)。

零成本探针, 不动生产文件。数据态9/17冻结。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_disclosure_window_20260921 import (  # noqa: E402
    build_disc_index, next_disc_idx)

DATA = '/mnt/d/quant/data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
TRD = '/mnt/d/quant/strategy/arms_20260920/C26_deviat0/post_trade_realized.csv'

BINS = [(0, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 120), (121, 10**6)]


def bin_of(dates, day):
    j = next_disc_idx(dates, np.datetime64(day))
    if j < 0:
        return len(BINS) - 1, 999
    dist = (dates[j] - np.datetime64(day)).astype('timedelta64[D]').astype(int)
    b = next(b for b, (lo, hi) in enumerate(BINS) if lo <= dist <= hi)
    return b, dist


def bucket_series(codes, dates, disc):
    bs, ds = [], []
    for c, d in zip(codes, dates):
        dts = disc.get(c)
        if dts is None:
            continue
        b, dist = bin_of(dts, d)
        bs.append(b)
        ds.append(dist)
    return np.array(bs), np.array(ds)


def main():
    trd = pd.read_csv(TRD, dtype={'code': str})
    trd['entry_date'] = pd.to_datetime(trd['entry_date'])
    codes = sorted(trd['code'].unique())
    disc, miss = build_disc_index(codes)
    print(f'生产交易: {len(trd)}笔 / {len(codes)}只  | 披露索引 {len(disc)}只 ({miss}缺失)')
    pb, pd_ = bucket_series(trd['code'], trd['entry_date'], disc)
    print(f'可测建仓: {len(pb)}笔')

    sig = pd.read_csv(SIG, dtype={'code': str}, low_memory=False)
    buys = sig[sig['buy'] == True].copy()  # noqa: E712
    buys['date'] = pd.to_datetime(buys['date'])
    pool_codes = sorted(buys['code'].unique())
    pdisc, pmiss = build_disc_index(pool_codes)
    print(f'信号池buy行: {len(buys)} / 池披露索引 {len(pdisc)}只')
    sb, sd_ = bucket_series(buys['code'], buys['date'], pdisc)
    print(f'可测池行: {len(sb)}')

    print('\n=== 生产建仓披露距离分布 vs 信号池buy行分布 ===')
    print(f'{"桶":>9s}  生产 {len(pb):5d}笔   池 {len(sb):7d}行   差pp')
    for b, (lo, hi) in enumerate(BINS):
        lbl = f'{lo}-{hi}d' if hi < 10**6 else '120+d'
        pp = (pb == b).mean() * 100
        po = (sb == b).mean() * 100
        print(f'  {lbl:>9s}:  {pp:5.1f}%      {po:5.1f}%     {pp-po:+5.1f}')
    print(f'  生产中位距披露: {np.median(pd_):.0f}d   池: {np.median(sd_):.0f}d')

    print('\n=== 生产桶内实现收益 (realized ret by entry bucket) ===')
    for b, (lo, hi) in enumerate(BINS):
        lbl = f'{lo}-{hi}d' if hi < 10**6 else '120+d'
        arr = trd['ret'].values[pb == b]
        if len(arr) < 5:
            continue
        print(f'  {lbl:>9s}: N={len(arr):4d}  mean={arr.mean()*100:+6.2f}%  '
              f'med={np.median(arr)*100:+6.2f}%  win={(arr>0).mean()*100:.1f}%')


if __name__ == '__main__':
    main()
