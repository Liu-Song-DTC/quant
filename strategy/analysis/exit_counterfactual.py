#!/usr/bin/env python
"""出场反事实: 每次平仓后该股继续涨了多少 (出场机制留下的钱)

对trade_realized.csv的每一笔平仓(code, exit_date), 用该股qfq行情
计算出场后5/20个交易日(该股自身日历, 天然跳过停牌)的涨幅。
聚合口径:
  - 总体: 若不在exit日卖、再拿20天, 多赚/多赔多少
  - 按出场性质: 近0出场(-2%~+2%) / 止损区(<0) / 盈利出场(>0)
  - 按持有周期分桶 / 按年
用法: python analysis/exit_counterfactual.py [trade_csv]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

TRADE_PATH = '/mnt/d/quant/strategy/rolling_validation_results/trade_realized.csv'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'


def load_qfq(code):
    p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def fwd_after(df_price, exit_date, horizon):
    """出场日之后第horizon个交易日相对出场日收盘的涨幅 (用出场日后首根bar做基准更贴近卖出实际)"""
    idx = df_price['datetime'].searchsorted(pd.Timestamp(exit_date))
    if idx >= len(df_price):
        return np.nan
    base = df_price['close'].iloc[idx]
    if idx + horizon >= len(df_price) or base <= 0:
        return np.nan
    return float(df_price['close'].iloc[idx + horizon] / base - 1)


def main():
    trades = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else TRADE_PATH)
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])

    cache = {}
    rows = []
    for _, t in trades.iterrows():
        code = str(t['code']).zfill(6)
        if code not in cache:
            cache[code] = load_qfq(code)
        dfp = cache[code]
        if dfp is None:
            continue
        rows.append({
            'ret': t['ret'], 'hold_days': t['hold_days'],
            'exit_date': t['exit_date'], 'code': code,
            'fwd5': fwd_after(dfp, t['exit_date'], 5),
            'fwd20': fwd_after(dfp, t['exit_date'], 20),
        })
    d = pd.DataFrame(rows)
    d['y'] = d['exit_date'].dt.year
    d['exit_type'] = pd.cut(d['ret'], [-99, -0.02, 0.02, 99],
                            labels=['止损区(< -2%)', '近0出场(-2~+2%)', '盈利出场(>+2%)'])
    d['hb'] = pd.cut(d['hold_days'], [0, 10, 20, 60, 999],
                     labels=['1-10天', '11-20天', '21-60天', '61+天'])

    print(f"样本: {len(d)}笔 (qfq匹配{'全部' if len(d)==len(trades) else f'{len(d)}/{len(trades)}'})")
    print(f"\n=== 出场后继续持有20日的反事实 ===")
    print(f"fwd20均值: {d['fwd20'].mean()*100:+.2f}% | 中位: {d['fwd20'].median()*100:+.2f}% | "
          f"继续持有仍盈利占比: {(d['fwd20']>0).mean()*100:.1f}%")
    print(f"fwd5 均值: {d['fwd5'].mean()*100:+.2f}% | 中位: {d['fwd5'].median()*100:+.2f}%")

    print("\n=== 按出场性质 ===")
    g = d.groupby('exit_type', observed=True).agg(
        n=('fwd20', 'size'), ret=('ret', 'mean'), fwd5=('fwd5', 'mean'),
        fwd20=('fwd20', 'mean'), up20=('fwd20', lambda x: (x > 0).mean()))
    print(g.round(4))

    print("\n=== 按持有周期 ===")
    g2 = d.groupby('hb', observed=True).agg(
        n=('fwd20', 'size'), ret=('ret', 'mean'), fwd5=('fwd5', 'mean'),
        fwd20=('fwd20', 'mean'), up20=('fwd20', lambda x: (x > 0).mean()))
    print(g2.round(4))

    print("\n=== 按年 ===")
    g3 = d.groupby('y').agg(n=('fwd20', 'size'), ret=('ret', 'mean'),
                            fwd20=('fwd20', 'mean'))
    print(g3.round(4))

    # 近0出场专项: 这些"白做"的交易如果多拿20天
    z = d[d['exit_type'] == '近0出场(-2~+2%)']
    print(f"\n=== 近0出场专项 (n={len(z)}, 中位持有{d['hold_days'].median():.0f}天) ===")
    print(f"fwd5={z['fwd5'].mean()*100:+.2f}% fwd20={z['fwd20'].mean()*100:+.2f}% "
          f"20日后上涨占比{(z['fwd20']>0).mean()*100:.1f}%")
    print("近0出场逐年fwd20:")
    print(z.groupby('y')['fwd20'].agg(['size', 'mean']).round(4).to_string())

    # H2诊断: 长持仓龄队列分年 — 61+天赢家保护在设计期(2021-2024)是否也成立
    # 若设计期持续正 → 按持仓龄保护(不依赖regime); 若仅2025+正 → 假设不诚实, park
    print(f"\n=== H2诊断: 61+天持仓出场 分年fwd20 (设计期vs持有期) ===")
    l = d[d['hb'] == '61+天']
    print(l.groupby('y')['fwd20'].agg(['size', 'mean']).round(4).to_string())
    print("21-60天队列分年fwd20:")
    m2 = d[d['hb'] == '21-60天']
    print(m2.groupby('y')['fwd20'].agg(['size', 'mean']).round(4).to_string())

    # 总反事实收益: 所有出场再拿20天的收益和
    print(f"\n=== 总量 ===")
    print(f"实际已实现收益和: {d['ret'].sum():.2f}")
    print(f"若全部多拿20天再卖: {d['fwd20'].sum():.2f}")


if __name__ == '__main__':
    main()
