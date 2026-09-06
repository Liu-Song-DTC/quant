#!/usr/bin/env python
"""交易解剖: 逐笔平仓审计的结构分析 (钱从哪赚, 从哪亏)

输入: rolling_validation_results/trade_realized.csv (逐笔: entry/exit/code/hold_days/ret/avg_cost/exit_px)
用法前必须确认该CSV属于目标run (mtime+日志"逐笔平仓审计已保存: N 笔"对得上)。
输出: 收益分布/持仓周期分桶/年度表/熊市年逐月/止损簇/盈利集中度/近0出场画像/长线赢家。
"""
import sys

import numpy as np
import pandas as pd

PATH = '/mnt/d/quant/strategy/rolling_validation_results/trade_realized.csv'


def main(path=PATH):
    df = pd.read_csv(path)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    r = df['ret']
    print(f"笔数: {len(df)} | 区间: {df['entry_date'].min().date()} → {df['exit_date'].max().date()}")
    print(f"\n=== 收益率分布 ===")
    print(r.describe().round(4))
    print(f"胜率: {(r > 0).mean()*100:.1f}% | 盈亏比: "
          f"{r[r>0].mean()/abs(r[r<0].mean()):.2f}")
    print(f"\n=== 持有天数 ===")
    print(df['hold_days'].describe().round(1))
    print("\n=== 持仓周期分桶 (sum_ret为桶内收益率之和) ===")
    bins = [0, 5, 10, 20, 30, 60, 120, 999]
    labels = ['1-5', '6-10', '11-20', '21-30', '31-60', '61-120', '121+']
    df['hb'] = pd.cut(df['hold_days'], bins=bins, labels=labels)
    g = df.groupby('hb', observed=True).agg(
        n=('ret', 'size'), winrate=('ret', lambda x: (x > 0).mean()),
        mean_ret=('ret', 'mean'), sum_ret=('ret', 'sum'))
    print(g.round(3))
    print("\n=== 年度 ===")
    df['y'] = df['exit_date'].dt.year
    print(df.groupby('y').agg(n=('ret', 'size'),
          winrate=('ret', lambda x: (x > 0).mean()),
          mean_ret=('ret', 'mean'), sum_ret=('ret', 'sum')).round(3))
    print("\n=== 熊市年逐月 (2022/2023) ===")
    for yr in (2022, 2023):
        d = df[df['exit_date'].dt.year == yr].copy()
        d['ym'] = d['exit_date'].dt.to_period('M')
        print(f"-- {yr} --")
        print(d.groupby('ym').agg(n=('ret', 'size'),
              winrate=('ret', lambda x: (x > 0).mean()),
              sum_ret=('ret', 'sum')).round(3).to_string())
    print("\n=== 负收益分布 (止损簇识别) ===")
    neg = r[r < 0]
    hist, edges = np.histogram(neg, bins=np.arange(-0.36, 0.02, 0.02))
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        if h > 0:
            print(f"  [{e0:+.2f},{e1:+.2f}): {h}")
    print(f"\n=== 近0出场 (-2%,+2%): 可能是熊市清仓切掉的新仓 ===")
    z = df[(r > -0.02) & (r < 0.02)]
    print(f"笔数: {len(z)} ({len(z)/len(df)*100:.1f}%) | 平均持有: {z['hold_days'].mean():.1f}天 | "
          f"中位持有: {z['hold_days'].median():.0f}天")
    print("近0出场逐年:")
    print(z.groupby(z['exit_date'].dt.year).size())
    print("\n=== 盈利集中度 ===")
    pos = r[r > 0]
    print(f"总盈利和: {pos.sum():.2f} | top10: {pos.nlargest(10).sum():.2f} "
          f"({pos.nlargest(10).sum()/pos.sum()*100:.1f}%) | "
          f"top20: {pos.nlargest(20).sum()/pos.sum()*100:.1f}%")
    print("\n=== 长线赢家 (hold>120天) ===")
    print(df[df['hold_days'] > 120][['entry_date', 'exit_date', 'code', 'hold_days', 'ret']]
          .round(3).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else PATH)
