#!/usr/bin/env python3
"""BEAR部分敞口层级探针 (2026-09-22, 只读, 零生产写入)

背景: 9/21 regime量化 — BEAR=二进制清仓 (bear_risk日exposure严格=0.0,
portfolio.py:1007+1010-1011, 260日: 2022/93, 2023/85, 2024/58, 2026/24),
25簇全部谷侧触发, 68%簇现金期市场继续涨, 平均仅−0.11%。早触发方向已否决
(H5/H6/E-N12签名), 再入场提速已采纳(E-A4) — 但**层级本身(0.0→部分)**从未
bracket过。本探针给"BEAR日保持0.3层级"臂的第一阶期望值:
  指数级: 每个bear spell期间 sh000001 的累计收益 × 0.3 = 该臂在指数近似下
  的NAV贡献。分年合计 → 铁律预判 (NAV/MDD方向)。

注意: 第一阶近似不含个股alpha与买入侧结构 (真实臂=组合层90min冷跑, 若本探针
通过)。bear_risk重建自检: 应复现≈260零敞口日 (2022/93, 2023/85, 2024/58, 2026/24)。
零写入: 输出仅 /tmp/probe_bear_tier_20260922.csv。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.market_regime_detector import MarketRegimeDetector  # noqa: E402

BTD = '/mnt/d/quant/data/stock_data/backtrader_data'
FROMDATE = pd.Timestamp('2021-01-01')
TODATE = pd.Timestamp('2026-09-17')


def rebuild():
    idx = pd.read_csv(os.path.join(BTD, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    idx = idx[(idx['datetime'] >= FROMDATE) & (idx['datetime'] <= TODATE)]
    small = pd.read_csv(os.path.join(BTD, 'sh000852_qfq.csv'), parse_dates=['datetime'])
    growth = pd.read_csv(os.path.join(BTD, '399006_qfq.csv'), parse_dates=['datetime'])
    det = MarketRegimeDetector()
    det.generate(idx, small_cap_df=small, growth_df=growth)
    out = det.index_data.copy()
    out = out.set_index('datetime')
    assert {'regime', 'trend_score', 'bear_risk'} <= set(out.columns), out.columns.tolist()
    return out


def main():
    out = rebuild()
    out = out.sort_index()
    br = out['bear_risk'].fillna(False).astype(bool)

    # 自检: 零敞口日数 ≈260
    n_zero = int(br.sum())
    yr = br.groupby(br.index.year).sum()
    print(f'bear_risk日总数: {n_zero} (期望≈260; 2022/93, 2023/85, 2024/58, 2026/24)')
    print('分年:', yr.to_dict())

    # 逐spell: 连续bear_risk日段
    s = br.astype(int)
    # spell id: 从0到1跳变处累加
    spell_id = (s.diff() > 0).cumsum()
    spells = s[s == 1].groupby(spell_id)
    rows = []
    idx_close = out['close'] if 'close' in out.columns else None
    if idx_close is None:
        # 用sh000001原CSV的close对齐 (index_data可能无close)
        raw = pd.read_csv(os.path.join(BTD, 'sh000001_qfq.csv'), parse_dates=['datetime'])
        raw = raw[(raw['datetime'] >= FROMDATE) & (raw['datetime'] <= TODATE)]
        close_map = raw.set_index('datetime')['close']
    else:
        close_map = idx_close
    close_map = close_map.sort_index()
    close_map = close_map[~close_map.index.duplicated(keep='last')]

    for sid, sub in spells:
        days = sub.index
        start, end = days[0], days[-1]
        # spell期间指数收益: end日close vs start前一日close (持有整段)
        prev_close = close_map.asof(start - pd.Timedelta(days=7))
        spell_ret = close_map[end] / prev_close - 1.0 if pd.notna(prev_close) else np.nan
        # 后20日反弹 (大簇结束后的V型右腿)
        tail_start = end + pd.Timedelta(days=1)
        tail_end = end + pd.Timedelta(days=30)
        tail = close_map[(close_map.index >= tail_start) & (close_map.index <= tail_end)]
        fwd20 = tail.iloc[19] / close_map[end] - 1.0 if len(tail) >= 20 else np.nan
        rows.append({
            'year': start.year, 'start': start, 'end': end,
            'n_days': len(days), 'spell_ret': spell_ret, 'fwd20_after': fwd20,
        })
    df = pd.DataFrame(rows)
    df['tier03'] = 0.3 * df['spell_ret']
    df['year'] = df['year'].astype(int)

    print(f'\n=== BEAR spells (n={len(df)}) 指数级0.3层级贡献 ===')
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))

    g = df.groupby('year').agg(
        n_spells=('n_days', 'size'),
        bear_days=('n_days', 'sum'),
        spell_ret_sum=('spell_ret', 'sum'),
        tier03_pp=('tier03', 'sum'),
        mean_fwd20=('fwd20_after', 'mean'),
    )
    print('\n分年合计 (tier03_pp = 0.3层级指数近似的年NAV贡献, 百分点):')
    print(g.round(4).to_string())

    tot = df['tier03'].sum()
    print(f'\n全期0.3层级指数近似合计: {tot*100:.2f}pp NAV; '
          f'最坏单spell spell_ret={df["spell_ret"].min()*100:.1f}% '
          f'(0.3层级→{df["spell_ret"].min()*30:.1f}pp); '
          f'均值fwd20(现金期后反弹)={df["fwd20_after"].mean()*100:.1f}%')
    # MDD风险面: 负spell中0.3层级的最大单笔代价
    neg = df[df['spell_ret'] < 0]
    if len(neg):
        print(f'负spell {len(neg)}个: 0.3层级代价合计 {neg["tier03"].sum()*100:.2f}pp, '
              f'最坏 {neg["tier03"].min()*100:.2f}pp @ {neg.loc[neg["tier03"].idxmin(), "start"].date()}')
    df.to_csv('/tmp/probe_bear_tier_20260922.csv', index=False)
    print('\n细节 → /tmp/probe_bear_tier_20260922.csv')


if __name__ == '__main__':
    main()
