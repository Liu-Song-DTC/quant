#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M4探针 (2026-09-20): 调仓节奏敏感性前置检验。
审计发现: 真实调仓闸=portfolio.py硬编码 rebalance_interval=10(日历日,≈7交易日);
yaml backtest.rebalance_days只门控选股记录。若想bracket节奏需参数化代码(非豁免,
每臂~78min重生成+20min回测, 2臂~4h)。本探针先估效应量:
在真实调仓日r, 若在r+5(日历日)提前调仓, score排名前6与r日生产持有集的重叠度如何?
score序在5日历日内越稳 → 节奏变化效应越小 → M4关闭(与E-N9粘性/E-O1单日churn
零效应的先验一致)。反之重排剧烈 → 参数化bracket值得排。
输入: 基线signals(只读归档, 单遍分块扫date/code/score) + 生产选股快照。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS = os.path.join(BASE, 'arms_20260919', '_baseline_sig', 'backtest_signals.csv')
SEL = os.path.join(BASE, 'arms_20260919', 'C1_faithful_0_55', 'pre_portfolio_selections.csv')


def main():
    sel = pd.read_csv(SEL, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])
    sel = sel[sel['weight'] > 0].copy()
    sel['code'] = sel['code'].str.zfill(6)
    held_by_day = {d: set(g['code']) for d, g in sel.groupby('date')}
    days = sorted(held_by_day)
    print(f'选股快照日数: {len(days)}, 行数: {len(sel)}')

    # 相邻快照重叠 (10交易日节奏下的自然换手)
    ovs = [len(held_by_day[d0] & held_by_day[d1]) / max(len(held_by_day[d0] | held_by_day[d1]), 1)
           for d0, d1 in zip(days[:-1], days[1:])]
    print(f'相邻快照(10交易日)Jaccard: 均值 {np.mean(ovs):.2f}  中位 {np.median(ovs):.2f}  '
          f'p10 {np.quantile(ovs,0.1):.2f}  p90 {np.quantile(ovs,0.9):.2f}')

    # 所需s日期: 每个快照日r的r+5日历日之后第一个交易日
    print('单遍扫描signals (date/code/score)...')
    needed_s = set()
    pairs = {}
    all_scan = {}  # s日期 → [(r, held)]
    for r in days:
        target = r + pd.Timedelta(days=5)
        # 交易日历未知, 先收集r+5之后5个自然日内所有日期作为s候选
        for k in range(5):
            c = target + pd.Timedelta(days=k)
            if c > days[-1]:
                break
            needed_s.add(c)
            all_scan.setdefault(c, []).append(r)

    needed_s = {d for d in needed_s if d >= days[0]}
    print(f'  需扫描截面日期: {len(needed_s)}')
    cross = {d: [] for d in needed_s}
    nchunk = 0
    for chunk in pd.read_csv(SIGNALS, usecols=['date', 'code', 'score'],
                             dtype={'code': str}, chunksize=4_000_000):
        nchunk += 1
        cd = pd.to_datetime(chunk['date']).dt.normalize()
        mask = cd.isin(needed_s)
        if mask.any():
            sub = chunk.loc[mask, ['code', 'score']].copy()
            sub['date'] = cd[mask]
            for d, g in sub.groupby('date'):
                cross[d].append(g)
        if nchunk % 20 == 0:
            print(f'    {nchunk} chunks...', flush=True)

    # 每对(r, s): s日top6命中r日持有
    hits, n_pairs = [], 0
    for s, rlist in sorted(all_scan.items()):
        if s not in cross or not cross[s]:
            continue
        cs = pd.concat(cross[s], ignore_index=True)
        cs['score'] = pd.to_numeric(cs['score'], errors='coerce')
        top6 = set(cs.nlargest(6, 'score')['code'].str.zfill(6))
        for r in rlist:
            if s <= r:
                continue
            held = held_by_day.get(r)
            if not held:
                continue
            n_pairs += 1
            hits.append(len(top6 & held) / max(len(top6), 1))
    print(f'\n=== r+5日历日 top6(裸score) 命中r日生产持有的比例 ===')
    print(f'配对样本: {n_pairs}, 命中率均值 {np.mean(hits):.2f}')
    if hits:
        print(f'→ 提前5日历日调仓将换掉约 {100*(1-np.mean(hits)):.0f}% 的持仓 (方向性上界, '
              f'裸score≠生产评分全链)')
        print(f'  命中率分布: p10 {np.quantile(hits,0.1):.2f} p50 {np.median(hits):.2f} '
              f'p90 {np.quantile(hits,0.9):.2f}')


if __name__ == '__main__':
    main()
