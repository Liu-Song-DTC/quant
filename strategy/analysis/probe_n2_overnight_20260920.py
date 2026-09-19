#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N2探针 (2026-09-20): 隔夜收益分解 + 尾盘成交反事实。
机制: 回测买入在信号次日开盘成交(现制 open t+1), 因此每笔交易在入场时
支付了"信号日收盘→次日开盘"的隔夜跳空。A股文献一致: 正收益集中在隔夜,
日内平均为负 — 若入场跳空系统性地吞噬收益, "尾盘成交"机制臂有真实EV。
本探针对生产558笔已实现交易逐笔:
  a) 入场跳空 gap_entry = open[entry]/close[entry-1] - 1 的分布(逐年);
  b) 出场定价模式: exit_px 与当日open/close的匹配度(确认执行模型);
  c) 反事实: 若以entry前一日收盘买入(免入场跳空), ret_ctf = (1+ret)/(1+gap) - 1,
     总净值与逐年的变化;
  d) 持有期日内/隔夜分解: 隔夜占比(若隔夜是正收益主源 → 现制已捕获, 无需改)。
输入: 生产trade_realized(只读) + backtrader_data/{code}_qfq.csv(只读)。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = '/mnt/d/quant/strategy/arms_20260919/C2_055/pre_trade_realized.csv'
KLINE = '/mnt/d/quant/data/stock_data/backtrader_data/{code}_qfq.csv'


def load_kl(code):
    p = KLINE.format(code=str(code).zfill(6))
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['date'] = df['datetime'].dt.normalize()
    return df[['date', 'open', 'close']].set_index('date')


def main():
    tr = pd.read_csv(TR)
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    print(f'交易数: {len(tr)}')

    rows = []
    skipped = 0
    for _, t in tr.iterrows():
        kl = load_kl(t['code'])
        if kl is None:
            skipped += 1
            continue
        pos = kl.index.get_loc(t['entry_date']) if t['entry_date'] in kl.index else None
        if pos is None or pos == 0:
            skipped += 1
            continue
        px = kl.iloc[pos]
        prev_close = kl.iloc[pos - 1]['close']
        if prev_close <= 0:
            skipped += 1
            continue
        gap_entry = px['open'] / prev_close - 1.0
        # 出场定价模式: exit_px 与当日 open/close 的接近度
        exit_row = None
        if t['exit_date'] in kl.index:
            er = kl.loc[t['exit_date']]
            d_open = abs(er['open'] / t['exit_px'] - 1)
            d_close = abs(er['close'] / t['exit_px'] - 1)
        else:
            er = None
            d_open = d_close = np.nan
        # 持有期隔夜/日内分解 (entry+1 .. exit)
        lo = pos + 1
        hi = kl.index.get_loc(t['exit_date']) if t['exit_date'] in kl.index else None
        ov_share = np.nan
        if hi is not None and hi >= lo:
            seg = kl.iloc[lo:hi + 1]
            ov = (seg['open'] / kl.iloc[lo - 1:hi]['close'].values - 1.0)
            iv = (seg['close'] / seg['open'] - 1.0)
            ov_share = np.nansum(ov) / (np.nansum(ov) + np.nansum(iv)) if (np.nansum(ov) + np.nansum(iv)) != 0 else np.nan
        rows.append(dict(
            code=t['code'], entry=t['entry_date'], exit=t['exit_date'],
            year=t['entry_date'].year, ret=t['ret'], gap_entry=gap_entry,
            d_open=d_open, d_close=d_close, ov_share=ov_share))
    df = pd.DataFrame(rows)
    print(f'可用: {len(df)}, 跳过: {skipped}')

    print('\n=== a) 入场跳空分布 ===')
    print(f"均值 {df['gap_entry'].mean()*100:+.2f}%  中位 {df['gap_entry'].median()*100:+.2f}%  "
          f"p10 {df['gap_entry'].quantile(.1)*100:+.2f}%  p90 {df['gap_entry'].quantile(.9)*100:+.2f}%")
    print('逐年均值:')
    for y, g in df.groupby('year'):
        print(f"  {y}: gap {g['gap_entry'].mean()*100:+.2f}%  n={len(g)}")

    print('\n=== b) 出场定价模式 ===')
    n_close = (df['d_close'] < 0.001).sum()
    n_open = (df['d_open'] < 0.001).sum()
    print(f'  exit_px≈close: {n_close}/{len(df)}   exit_px≈open: {n_open}/{len(df)}   '
          f'中位|Δ| open {df["d_open"].median()*100:.2f}% close {df["d_close"].median()*100:.2f}%')

    print('\n=== c) 尾盘成交反事实 (入场按前收, 免跳空) ===')
    # exit/prev_close = (exit/open_entry)×(open_entry/prev_close) = (1+ret)×(1+gap)
    df['ret_ctf'] = (1 + df['ret']) * (1 + df['gap_entry']) - 1
    d = (df['ret_ctf'] - df['ret']).mean()
    print(f'每笔平均收益变化: {d*100:+.2f}pp  (ret {df["ret"].mean()*100:+.2f}% → ctf {df["ret_ctf"].mean()*100:+.2f}%)')
    print('逐年反事实Δ(pp):')
    for y, g in df.groupby('year'):
        print(f"  {y}: {((g['ret_ctf']-g['ret']).mean())*100:+.2f}pp  "
              f"(ctf年复利 {(1+g['ret_ctf']).prod()-1:+.1%} vs 实际 {(1+g['ret']).prod()-1:+.1%})")

    print('\n=== d) 持有期隔夜占比 ===')
    m = df['ov_share'].dropna()
    print(f'隔夜收益占总收益份额: 均值 {m.mean()*100:.0f}%  中位 {m.median()*100:.0f}%  '
          f'(>50% ⇒ 隔夜为主源, 与A股文献一致)')


if __name__ == '__main__':
    main()
