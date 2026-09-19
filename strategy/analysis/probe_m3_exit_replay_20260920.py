#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3探针 (2026-09-20): 逐笔出场重放审计 — 基线trade_realized逐笔 × 日K线,
重放生产trailing规则, 对照两个候选收紧变体, 计算"止损帮了还是坑了"(fwd20)。

背景(代码审计结论):
- 生产 stock_stop_loss.enabled=false → cost={} → avg_cost_check恒0 →
  成本止损/分级峰值回撤/tiered_trailing_stop(yaml)/死钱退出 全部惰性;
  profit_from_peak恒0.10 → "盈利>15%收紧"硬编码行永不触发。
- 生产活跃止损 = sig_sell / time_stop_by_bp(15d/-3%, 按bp) /
  trailing_stop_by_buy_point(bp分层: 1:0.15 2:0.12 3:0.10 default:0.08, 无盈利分级) /
  chan结构退出 / entry_reason_lost / S2恶化跟踪。
- 所以M3的"yaml tiered_trailing_stop收紧"是死旋钮; 真正可拧的 =
  trailing_stop_by_buy_point 与 盈利分级收紧(需让profit计算可用)。

变体:
  V-A: 盈利分级收紧激活(以入场价为成本代理): pnl≥0.15→trail−0.03,
       pnl≥0.30→trail=0.07, pnl≥0.50→trail=0.04 (floor 0.05)。
  V-B: bp分层全线−0.03: 1:0.12 2:0.09 3:0.07 default:0.05。

输出: 类别×n×平均实际ret×两变体sim ret×fwd5/20 → 判定M3 go/no-go。
注意(纪律): 逐笔反事实≠机制可实现(E-N12教训), 本探针只做方向性筛选。
"""
import os, sys
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES = os.path.join(BASE, 'arms_20260919', 'C1_faithful_0_55', 'pre_trade_realized.csv')
SIGNALS = os.path.join(BASE, 'arms_20260919', '_baseline_sig', 'backtest_signals.csv')
KLINE_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'

TRAIL_BP = {1: 0.15, 2: 0.12, 3: 0.10, 'default': 0.08}
TIER_A = [(0.50, 0.04), (0.30, 0.07), (0.15, 0.10)]  # profit: trail (收紧)
TRAIL_B = {1: 0.12, 2: 0.09, 3: 0.07, 'default': 0.05}

_kcache = {}


def kline(code):
    if code in _kcache:
        return _kcache[code]
    p = os.path.join(KLINE_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        _kcache[code] = None
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')['close'].sort_index()
    _kcache[code] = df
    return df


def sim_trailing(closes, trail_fn, start_i=0):
    """从start_i起按closes逐日走trailing; 返回(触发索引, 触发日close)或(None,None)。"""
    peak = closes.iloc[start_i]
    for i in range(start_i, len(closes)):
        c = closes.iloc[i]
        peak = max(peak, c)
        dd = (peak - c) / peak
        t = trail_fn(c / closes.iloc[0] - 1)  # pnl_from_entry
        if dd >= t:
            return i, c
    return None, None


def trail_fn_prod(bp, pnl_from_entry):
    return TRAIL_BP.get(bp, TRAIL_BP['default'])


def trail_fn_A(bp, pnl_from_entry):
    base = TRAIL_BP.get(bp, TRAIL_BP['default'])
    # 生产代码硬编码意图: profit_from_peak>0.15 → trail−0.03 (floor 0.05); 用入场价做profit代理
    if pnl_from_entry >= 0.15:
        return max(base - 0.03, 0.05)
    return base


def trail_fn_B(bp, pnl_from_entry):
    return TRAIL_B.get(bp, TRAIL_B['default'])


def main():
    tr = pd.read_csv(TRADES, parse_dates=['entry_date', 'exit_date'])
    print(f'逐笔平仓总数: {len(tr)}')
    # 入场bp: 信号表 entry_date×code 的 chan_buy_point (近似: 入场日信号即选股日信号)
    sig = pd.read_csv(SIGNALS, usecols=['date', 'code', 'chan_buy_point'], dtype={'code': str})
    sig['date'] = pd.to_datetime(sig['date'])
    sig['chan_buy_point'] = pd.to_numeric(sig['chan_buy_point'], errors='coerce').fillna(0).astype(int)
    sig = sig.drop_duplicates(['date', 'code'], keep='last').set_index(['date', 'code'])

    rows = []
    missing = 0
    for _, t in tr.iterrows():
        code = str(t['code']).zfill(6)
        ed, xd = t['entry_date'], t['exit_date']
        k = kline(code)
        if k is None:
            missing += 1
            continue
        bp = 0
        try:
            bp = int(sig.loc[(ed, code), 'chan_buy_point'])
        except KeyError:
            try:  # 买点为前一日决策, 填入日行缺失时回退决策日
                bp = int(sig.loc[(ed - pd.Timedelta(days=1), code), 'chan_buy_point'])
            except KeyError:
                pass
        # 重放窗口: entry_date..exit_date-1 的close (出场决策发生在exit_date-1)
        seg = k.loc[ed:xd - pd.Timedelta(days=1)]
        if len(seg) == 0:
            missing += 1
            continue
        closes = seg
        entry_close = closes.iloc[0]
        actual_ret = t['ret']
        # 生产规则 sim
        pi, pc = sim_trailing(closes, lambda p: trail_fn_prod(bp, p))
        # V-A sim
        ai, ac = sim_trailing(closes, lambda p: trail_fn_A(bp, p))
        # V-B sim
        bi, bc = sim_trailing(closes, lambda p: trail_fn_B(bp, p))
        # fwd after actual exit
        fwd = k.loc[xd:xd + pd.Timedelta(days=30)]
        fwd5 = fwd.iloc[min(4, len(fwd) - 1)] / t['exit_px'] - 1 if len(fwd) > 0 else np.nan
        fwd20 = fwd.iloc[min(19, len(fwd) - 1)] / t['exit_px'] - 1 if len(fwd) > 0 else np.nan
        rows.append({
            'entry': ed, 'exit': xd, 'code': code, 'hold': t['hold_days'],
            'actual_ret': actual_ret, 'bp': bp,
            'prod_triggered': pi is not None,
            'prod_sim_ret': (pc / entry_close - 1) if pc is not None else np.nan,
            'A_triggered': ai is not None,
            'A_sim_ret': (ac / entry_close - 1) if ac is not None else np.nan,
            'B_triggered': bi is not None,
            'B_sim_ret': (bc / entry_close - 1) if bc is not None else np.nan,
            'fwd5': fwd5, 'fwd20': fwd20,
        })
    df = pd.DataFrame(rows)
    print(f'可重放笔数: {len(df)} (缺K线/窗口: {missing})')

    # 分类: 止损类近似 = (时间止损: hold>15 & ret<-3%) | (trailing: 生产规则在窗口内触发)
    time_stop = (df['hold'] > 15) & (df['actual_ret'] < -0.03)
    trail_stop = df['prod_triggered']
    stop_class = time_stop | trail_stop
    print('\n=== 出场分类 ===')
    print(f"时间止损类(hold>15d & ret<-3%): {int(time_stop.sum())} 笔, 平均ret {df.loc[time_stop,'actual_ret'].mean()*100:.2f}%")
    print(f"trailing类(生产规则窗口内触发): {int(trail_stop.sum())} 笔, 平均ret {df.loc[trail_stop,'actual_ret'].mean()*100:.2f}%")
    print(f"止损类合计(并集): {int(stop_class.sum())} 笔")
    print(f"其余(信号/调仓/chan退出): {int((~stop_class).sum())} 笔")

    print('\n=== 止损类: 收紧变体会改变什么 ===')
    sdf = df[stop_class]
    print(f"n={len(sdf)}")
    for name, c in [('actual', 'actual_ret'), ('V-A分级收紧', 'A_sim_ret'), ('V-B全线-0.03', 'B_sim_ret')]:
        v = sdf[c]
        print(f"  {name}: 均值 {v.mean()*100:.2f}%  中位 {v.median()*100:.2f}%  负笔率 {(v<0).mean()*100:.1f}%  "
              f"sum {v.sum()*100:.1f}%")
    print(f"  V-A vs actual: sum差 {(sdf['A_sim_ret']-sdf['actual_ret']).sum()*100:+.1f}% (更负=止损更伤)")
    print(f"  V-B vs actual: sum差 {(sdf['B_sim_ret']-sdf['actual_ret']).sum()*100:+.1f}%")
    a_trig = df[stop_class & df['A_triggered']]
    print(f"  V-A新触发(生产未触发): {int((df['A_triggered']&~df['prod_triggered']).sum())} 笔, "
          f"若触发sim ret均值 {a_trig['A_sim_ret'].mean()*100:.2f}%")
    b_trig = df[stop_class & df['B_triggered']]
    print(f"  V-B新触发: {int((df['B_triggered']&~df['prod_triggered']).sum())} 笔, "
          f"若触发sim ret均值 {b_trig['B_sim_ret'].mean()*100:.2f}%")

    print('\n=== fwd20(出场后20日): 止损帮了还是坑了 ===')
    print(f"全部: mean fwd5 {df['fwd5'].mean()*100:.2f}%  fwd20 {df['fwd20'].mean()*100:.2f}%")
    print(f"止损类: mean fwd5 {sdf['fwd5'].mean()*100:.2f}%  fwd20 {sdf['fwd20'].mean()*100:.2f}% "
          f"(正=fwd恢复→止损割在坑里)")
    print(f"时间止损类: mean fwd20 {df.loc[time_stop,'fwd20'].mean()*100:.2f}%")
    print(f"trailing类: mean fwd20 {df.loc[trail_stop,'fwd20'].mean()*100:.2f}%")
    print(f"非止损类: mean fwd20 {df.loc[~stop_class,'fwd20'].mean()*100:.2f}%")

    # 年度分解(止损类)
    df['yr'] = df['exit'].dt.year
    print('\n=== 止损类年度分布 (笔数/均值ret%) ===')
    for yr, g in df[stop_class].groupby('yr'):
        print(f"  {yr}: n={len(g):3d}  ret {g['actual_ret'].mean()*100:+.2f}%  "
              f"V-AΔ {(g['A_sim_ret']-g['actual_ret']).sum()*100:+.2f}%  "
              f"V-BΔ {(g['B_sim_ret']-g['actual_ret']).sum()*100:+.2f}%")

    out = os.path.join(BASE, 'analysis', 'probe_m3_replay_20260920.csv')
    df.to_csv(out, index=False)
    print(f'\n明细已存: {out}')


if __name__ == '__main__':
    main()
