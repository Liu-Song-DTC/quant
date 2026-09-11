#!/usr/bin/env python3
"""2026-09-12 阶段7探针C: 冲高去风险机制稳健性核验 (轻量)
探针B发现 runup X=0.40族四指标全胜(曲线级)。核验:
1. 归因join修复(zfill+最近前向信号) → 崩塌腿真实factor_name
2. 事件分解: L=60 X=0.40 f=0.5 的触发事件列表(几年几次/每次NAV贡献)
3. 参数稳健性: X/L/f细网格 → 全胜区是平台还是尖峰
4. 跨状态: 同机制跑基线(prePIT0911)曲线 → 若基线态也全胜=机制不依赖gate态
结论标准: 事件≥2次分布多年+全胜区成平台+基线态也赢 → 值得写机制跑全链
"""
import pandas as pd
import numpy as np

RVR = '/mnt/d/quant/strategy/rolling_validation_results'

def load(path):
    eq = pd.read_csv(path, parse_dates=['date']).set_index('date')
    return eq.iloc[:, 0]

nav = load(f'{RVR}/equity_curve.csv')
r = nav.pct_change().fillna(0.0).values
nav_b = load(f'{RVR}/equity_curve.prePIT0911.csv')
r_b = nav_b.pct_change().fillna(0.0).values

def metrics(n):
    return (n[-1], (n[-1]/n[0]-1)*100,
            pd.Series(n).pct_change().dropna().pipe(lambda s: s.mean()/s.std()*np.sqrt(252)),
            (pd.Series(n)/pd.Series(n).cummax()-1).min()*100)

def sim(nav0, rseq, X, f, L=60, ema=True):
    n = nav0.values.copy()
    e = np.ones(len(n)); prev = 1.0
    events = []  # (start_i, end_i)
    in_ev = False; ev_start = 0
    for i in range(1, len(n)):
        target = 1.0
        if i >= L:
            tr = n[i]/n[i-L] - 1
            if tr > X: target = f
            elif tr <= X/2: target = 1.0
            else: target = e[i-1]
        e[i] = 0.3*prev + 0.7*target if ema else target
        prev = e[i]
        if e[i] < 0.99 and not in_ev:
            in_ev = True; ev_start = i
        elif e[i] >= 0.99 and in_ev:
            in_ev = False; events.append((ev_start, i))
        n[i] = n[i-1] * (1 + e[i]*rseq[i])
    if in_ev: events.append((ev_start, len(n)-1))
    return n, e, events, metrics(n)

BASE = metrics(nav.values)
BASE_B = metrics(nav_b.values)
print(f"gate基线: NAV {BASE[0]:,.0f} | {BASE[1]:.2f}% | {BASE[2]:.4f} | MDD {BASE[3]:.2f}%")
print(f"旧基线:   NAV {BASE_B[0]:,.0f} | {BASE_B[1]:.2f}% | {BASE_B[2]:.4f} | MDD {BASE_B[3]:.2f}%")

# ---------- 1) 归因修复 ----------
sig = pd.read_csv(f'{RVR}/backtest_signals.csv', parse_dates=['date'], low_memory=False)
sig['code'] = sig['code'].astype(str).str.zfill(6)
buy = sig[sig['buy'] == True][['date', 'code', 'factor_name', 'industry', 'score']].copy()
tr = pd.read_csv(f'{RVR}/trade_realized.csv', parse_dates=['entry_date', 'exit_date'])
tr['code'] = tr['code'].astype(str).str.zfill(6)
lo, hi = pd.Timestamp('2021-07-05'), pd.Timestamp('2022-12-23')
tw = tr[(tr['exit_date'] >= lo) & (tr['exit_date'] <= hi)].copy()
rows = []
for _, row in tw.iterrows():
    c = buy[(buy['code'] == row['code']) &
            (buy['date'] <= row['entry_date']) &
            (buy['date'] >= row['entry_date'] - pd.Timedelta(days=7))]
    if len(c):
        rows.append(c.sort_values('date').iloc[-1][['factor_name', 'industry']])
    else:
        rows.append(pd.Series({'factor_name': '?', 'industry': '?'}))
tw[['fn', 'ind']] = pd.DataFrame(rows).values
print(f"\n=== 1) 归因修复后 ({len(tw)}笔, '?'剩余 {(tw['fn']=='?').sum()}) ===")
g = tw.groupby('fn')['ret'].agg(['count', 'sum'])
print('窗内亏损factor_name top12:')
print(g.sort_values('sum').head(12).to_string(float_format=lambda x: f'{x*100:+.1f}'))

# ---------- 2) 事件分解 ----------
print('\n=== 2) 事件分解 L=60 X=0.40 f=0.5 ===')
n2, e2, ev2, m2 = sim(nav, r, 0.40, 0.5, 60)
print(f"模拟: NAV {m2[0]:,.0f} ({m2[0]/BASE[0]-1:+5.1%}) | {m2[1]:.2f}% | {m2[2]:.4f} | MDD {m2[3]:.2f}%")
print(f"触发事件 {len(ev2)} 次:")
for a, b in ev2:
    da, db = nav.index[a], nav.index[b]
    sim_ret = (n2[b]/n2[a] - 1)*100
    raw_ret = (nav.values[b]/nav.values[a] - 1)*100
    print(f"  {da.date()}→{db.date()} ({b-a}d) | 模拟期内 {sim_ret:+7.2f}% vs 原始 {raw_ret:+7.2f}% "
          f"| 贡献 {sim_ret-raw_ret:+6.2f}pp")

# ---------- 3) 参数稳健性 ----------
print('\n=== 3) 细网格全胜区 (gate态) ===')
wins = []
for L in (40, 50, 60, 70, 80, 90, 100, 126):
    for X in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        for f in (0.7, 0.5, 0.3, 0.0):
            _, _, _, m = sim(nav, r, X, f, L)
            if m[0] > BASE[0] and m[1] > BASE[1] and m[2] > BASE[2] and m[3] > BASE[3]:
                wins.append((L, X, f, m))
print(f"全胜参数 {len(wins)}/{8*6*4}:")
for L, X, f, m in sorted(wins, key=lambda x: -x[3][2])[:15]:
    print(f"  L={L:3d} X={X:.2f} f={f:.1f} | NAV {m[0]:,.0f} ({m[0]/BASE[0]-1:+5.1%}) {m[1]:6.2f}% {m[2]:.4f} {m[3]:6.2f}%")

# ---------- 4) 跨状态: 同机制跑旧基线 ----------
print('\n=== 4) 跨状态核验 (跑prePIT0911基线曲线) ===')
wins_b = []
for L in (40, 60, 90, 126):
    for X in (0.30, 0.35, 0.40, 0.45):
        for f in (0.5, 0.3, 0.0):
            _, _, ev, m = sim(nav_b, r_b, X, f, L)
            if m[0] > BASE_B[0] and m[1] > BASE_B[1] and m[2] > BASE_B[2] and m[3] > BASE_B[3]:
                wins_b.append((L, X, f, len(ev), m))
print(f"旧基线态全胜参数 {len(wins_b)}/{4*4*3}:")
for L, X, f, ne, m in sorted(wins_b, key=lambda x: -x[3][2])[:10]:
    print(f"  L={L:3d} X={X:.2f} f={f:.1f} 事件{ne} | NAV {m[0]:,.0f} ({m[0]/BASE_B[0]-1:+5.1%}) {m[1]:6.2f}% {m[2]:.4f} {m[3]:6.2f}%")
