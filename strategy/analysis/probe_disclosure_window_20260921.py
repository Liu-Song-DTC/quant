#!/usr/bin/env python3
"""披露窗口探针 — 买入信号fwd收益按距下次披露日天数分桶 (2026-09-21)

问题: 8/9月alpha系统性为负(p=0.018)是否=中报重定价季的披露前不确定性?
方法: 基本面CSV的xjll_公告日期建披露索引 → 每笔buy(持仓绑定级)算
days-to-next-disclosure → 分桶(0-5/6-10/11-20/21-40/41-80/81-120/120+d)
→ 桶内fwd10d收益对比 + 8/9月买入的披露距离分布 vs 其他月。
零成本探针, 不动任何生产文件。数据态9/17冻结。
"""
import os
import sys

import numpy as np
import pandas as pd

DATA = '/mnt/d/quant/data'
FUND = f'{DATA}/stock_data/fundamental_data'
BT = f'{DATA}/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'

# 披露索引: 有公告日且是买方的csv文件名不全部对应有数据的股票 — 只建存在的
def build_disc_index(codes):
    disc = {}
    miss = 0
    for c in codes:
        fp = f'{FUND}/{c}.csv'
        if not os.path.exists(fp):
            miss += 1
            continue
        try:
            df = pd.read_csv(fp, usecols=['xjll_公告日期'])
        except Exception:
            miss += 1
            continue
        dts = pd.to_datetime(df['xjll_公告日期'], errors='coerce').dropna()
        if len(dts):
            disc[c] = np.sort(np.unique(dts.to_numpy(dtype='datetime64[D]')))
        else:
            miss += 1
    return disc, miss


def next_disc_idx(dates, day):
    """dates=datetime64[s] 升序; 返回 >=day 的首个索引, 无则 -1"""
    pos = np.searchsorted(dates, day, side='left')
    return pos if pos < len(dates) else -1


def preload_closes(codes):
    closes = {}
    for c in codes:
        fp = f'{BT}/{c}_qfq.csv'
        if not os.path.exists(fp):
            continue
        try:
            p = pd.read_csv(fp, usecols=['datetime', 'close'])
        except Exception:
            continue
        dts = pd.to_datetime(p['datetime']).to_numpy(dtype='datetime64[D]')
        closes[c] = (dts, p['close'].values)
    return closes


def fwd_ret(closes, code, date):
    if code not in closes:
        return np.nan
    dts, p = closes[code]
    idx = np.searchsorted(dts, np.datetime64(date), side='left')
    if idx + 10 >= len(p):
        return np.nan
    return p[idx + 10] / p[idx] - 1.0


def main():
    sig = pd.read_csv(SIG, dtype={'code': str}, low_memory=False)
    buys = sig[sig['buy'] == True].copy()  # noqa: E712
    buys['date'] = pd.to_datetime(buys['date'])
    print(f'buy行: {len(buys)}')

    codes = sorted(buys['code'].unique())
    disc, miss = build_disc_index(codes)
    print(f'披露索引: {len(disc)}/{len(codes)} 只 ({miss} 缺失)')
    closes = preload_closes(codes)
    print(f'K线预载: {len(closes)}/{len(codes)} 只')

    # 全样本披露日距
    BINS = [(0, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 120), (121, 10**6)]
    rows = []
    no_disc = 0
    for c, d in zip(buys['code'], buys['date']):
        dates = disc.get(c)
        if dates is None:
            no_disc += 1
            continue
        j = next_disc_idx(dates, np.datetime64(d))
        if j < 0:
            # 无后续披露 → 120+桶(披露真空)
            j = len(BINS) - 1
            dist = 999
        else:
            dist = (dates[j] - np.datetime64(d)).astype('timedelta64[D]').astype(int)
            j = next(b for b, (lo, hi) in enumerate(BINS) if lo <= dist <= hi)
        rows.append((c, d, dist, j))
    df = pd.DataFrame(rows, columns=['code', 'date', 'dist', 'bin'])
    print(f'可测buy: {len(df)} (无披露索引 {no_disc})')

    # fwd收益 (全量, K线已预载)
    df['m'] = df['date'].dt.month
    df['fwd10'] = [fwd_ret(closes, c, d) for c, d in zip(df['code'], df['date'])]
    valid = df.dropna(subset=['fwd10'])
    print(f'\n=== 桶内fwd10d均值 (N={len(valid)}) ===')
    for b, (lo, hi) in enumerate(BINS):
        sub = valid[valid['bin'] == b]
        if not len(sub):
            continue
        lbl = f'{lo}-{hi}d' if hi < 10**6 else '120+d'
        print(f'  {lbl:>9s}: N={len(sub):5d}  mean={sub["fwd10"].mean()*100:+6.2f}%  '
              f'med={sub["fwd10"].median()*100:+6.2f}%  win={ (sub["fwd10"]>0).mean()*100:.1f}%')

    # 8/9月 vs 其他月的披露距离分布
    cal = df[df['m'].isin([8, 9])]
    oth = df[~df['m'].isin([8, 9])]
    print(f'\n=== 披露距离分布: 8/9月(N={len(cal)}) vs 其他月(N={len(oth)}) ===')
    for b, (lo, hi) in enumerate(BINS):
        lbl = f'{lo}-{hi}d' if hi < 10**6 else '120+d'
        pc = (cal['bin'] == b).mean() * 100
        po = (oth['bin'] == b).mean() * 100
        print(f'  {lbl:>9s}: 8/9月 {pc:5.1f}%   其他 {po:5.1f}%   差 {pc-po:+5.1f}pp')
    print(f'  8/9月中位距披露: {cal["dist"].median():.0f}d   其他月: {oth["dist"].median():.0f}d')

    # 8/9月桶内收益 vs 其他月同桶
    print(f'\n=== 8/9月桶内fwd10 vs 其他月同桶 (桶间对照) ===')
    for b, (lo, hi) in enumerate(BINS):
        lbl = f'{lo}-{hi}d' if hi < 10**6 else '120+d'
        sc = valid[(valid['bin'] == b) & (valid['m'].isin([8, 9]))]['fwd10']
        so = valid[(valid['bin'] == b) & (~valid['m'].isin([8, 9]))]['fwd10']
        if len(sc) < 20 or len(so) < 20:
            continue
        print(f'  {lbl:>9s}: 8/9月 {sc.mean()*100:+6.2f}% (N={len(sc)})   '
              f'其他 {so.mean()*100:+6.2f}% (N={len(so)})   差 {(sc.mean()-so.mean())*100:+6.2f}pp')


if __name__ == '__main__':
    main()
