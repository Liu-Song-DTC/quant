#!/usr/bin/env python3
"""E-A4 证据分析: BEAR期V反快速恢复触发点的数据画像
复刻 market_regime_detector 的 bear_risk 公式 (纯确定性, 非模型):
  drawdown = close/rolling_max(120) - 1 < -0.07
  mom_120  = close/shift(120) - 1        < -0.03
  ema_bearish = NOT(ema20>ema60) AND NOT(ema60>ema120)
  (i<60 为warm-up, 全部False)
v_recovery 候选触发: mom20>+5% 且 close/20日低点-1>+8%
输出: 每段effective-BEAR(连续<=60天, 与portfolio streak降级一致)的V反日列表,
      当日信号密度(买入信号数/当日宇宙)、fwd5/10/20指数收益、若在V反日再入场省下的踏空
"""
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/mnt/d/quant/strategy')
BASE = '/mnt/d/quant'
idx = pd.read_csv(f'{BASE}/data/stock_data/backtrader_data/sh000001_qfq.csv', parse_dates=['datetime'])
idx = idx.sort_values('datetime').reset_index(drop=True)
close = idx['close']
n = len(idx)

# ---- 复刻检测器 ----
ema20 = close.ewm(span=20).mean()
ema60 = close.ewm(span=60).mean()
ema120 = close.ewm(span=120).mean()
ema_bearish = ~(ema20 > ema60) & ~(ema60 > ema120)
rolling_max = close.rolling(window=120, min_periods=1).max()
drawdown = (close - rolling_max) / rolling_max
mom_120 = close / close.shift(120) - 1
mom_20 = close / close.shift(20) - 1
bear_risk = (drawdown < -0.07) & (mom_120 < -0.03) & ema_bearish
# 快速熊市(60日维度)
rolling_max_60 = close.rolling(window=60, min_periods=1).max()
dd60 = (close - rolling_max_60) / rolling_max_60
mom_60v = close / close.shift(60) - 1
bear_risk_fast = (dd60 < -0.05) & (mom_60v < -0.02)
bear_risk = bear_risk & (np.arange(n) >= 60)  # warm-up
bear_risk_fast = bear_risk_fast & (np.arange(n) >= 60)

# ---- v_recovery 候选 ----
low20 = close.rolling(20, min_periods=5).min()
v_recovery = (mom_20 > 0.05) & (close / low20 - 1 > 0.08)
# 熊市期外无意义 (仅在effective-BEAR日统计)

# ---- 连续BEAR天数 (镜像portfolio._bear_streak) ----
streak = np.zeros(n, dtype=int)
s = 0
for i in range(n):
    if bear_risk.iloc[i]:
        s += 1
    else:
        s = 0
    streak[i] = s
eff_bear = bear_risk & (streak <= 60)  # streak>60 → portfolio已降级FAST

# ---- 信号密度: backtest_signals.csv 按日buy计数/宇宙 ----
print("加载信号文件...", flush=True)
sig = pd.read_csv(f'{BASE}/strategy/rolling_validation_results/backtest_signals.csv',
                  usecols=['code', 'date', 'buy'], dtype={'buy': 'bool'})
sig['date'] = pd.to_datetime(sig['date'])
day_buys = sig[sig['buy']].groupby('date').size()
day_univ = sig.groupby('date')['code'].nunique()
density = (day_buys / day_univ).rename('density')

# ---- 拼接 ----
df = pd.DataFrame({
    'date': idx['datetime'], 'close': close, 'bear_risk': bear_risk.values,
    'bear_fast': bear_risk_fast.values, 'streak': streak,
    'eff_bear': eff_bear.values, 'v_rec': v_recovery.values,
})
df['density'] = df['date'].map(density)
df['fwd5'] = close.shift(-5) / close - 1
df['fwd10'] = close.shift(-10) / close - 1
df['fwd20'] = close.shift(-20) / close - 1

print(f"\n=== 分布: 全历史 {df['eff_bear'].sum()} effective-BEAR天 / {df['bear_risk'].sum()} 检测器BEAR天")
print("密度分布:")
print(f"  NORM/FAST日:  mean={df.loc[~df['eff_bear'] & ~df['bear_risk'], 'density'].mean():.4f}  "
      f"median={df.loc[~df['eff_bear'] & ~df['bear_risk'], 'density'].median():.4f}")
print(f"  有效BEAR日:   mean={df.loc[df['eff_bear'], 'density'].mean():.4f}  "
      f"median={df.loc[df['eff_bear'], 'density'].median():.4f}")

# ---- BEAR期内V反日 ----
vrec = df[df['eff_bear'] & df['v_rec']].copy()
print(f"\n=== BEAR期内V反候选日: {len(vrec)} 天 ===")
if len(vrec):
    cols = ['date', 'streak', 'density', 'fwd5', 'fwd10', 'fwd20']
    pd.set_option('display.width', 200)
    print(vrec[cols].to_string(index=False))
    print("\n按fwd20>0分组:")
    pos = vrec[vrec['fwd20'] > 0]
    neg = vrec[vrec['fwd20'] <= 0]
    print(f"  fwd20>0: {len(pos)}天, 当日密度 mean={pos['density'].mean():.4f} median={pos['density'].median():.4f}, "
          f"fwd20 mean={pos['fwd20'].mean()*100:.1f}%")
    print(f"  fwd20<=0: {len(neg)}天, 当日密度 mean={neg['density'].mean():.4f} median={neg['density'].median():.4f}, "
          f"fwd20 mean={neg['fwd20'].mean()*100:.1f}%")

# ---- 密度阈值扫描: 选V反日中 fwd20>0 的比例 vs 密度门槛 ----
if len(vrec):
    print("\n=== 密度门槛扫描 (V反日中fwd20胜率) ===")
    for th in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
        sel = vrec[vrec['density'] >= th]
        if len(sel) == 0:
            print(f"  th>={th:.2f}: 0天")
            continue
        wr = (sel['fwd20'] > 0).mean()
        print(f"  th>={th:.2f}: {len(sel)}天  胜率={wr*100:.0f}%  fwd5={sel['fwd5'].mean()*100:+.1f}%  "
              f"fwd10={sel['fwd10'].mean()*100:+.1f}%  fwd20={sel['fwd20'].mean()*100:+.1f}%")
        sel.to_csv(f'/mnt/d/quant/strategy/logs/ea4_vrec_th{int(th*100):02d}.csv', index=False)

# ---- 2022年明细 (验证E-A3的4月踏空 vs 5月负IC) ----
print("\n=== 2022 effective-BEAR各段明细 ===")
d22 = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2022-12-31')].copy()
d22['seg'] = (d22['eff_bear'] != d22['eff_bear'].shift()).cumsum()
segs = d22[d22['eff_bear']].groupby('seg').agg(
    起=('date', 'first'), 止=('date', 'last'), 天数=('date', 'count'),
    段内V反日数=('v_rec', 'sum'))
print(segs.to_string())
