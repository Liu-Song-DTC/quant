#!/usr/bin/env python3
"""买入执行日开盘状态 vs 交易结果 (2026-10-22).

买点优化续: 成交时点/限价staging 已闭, 但"T+1 开盘可观测状态"作为**入场过滤**
从未单独测过 (跳空画像当年只看均值=中性, 未按收益分桶).

只用回测已经实现的决策 (trade_realized 246 笔) + 每只票的 qfq 日线:
  gap      = open(T+1)/close(T) - 1          # 执行日集合竞价缺口 (9:25已知)
  gap_hi   = open(T+1)/high(T) - 1           # 开盘相对信号日高点
  gap_lo   = open(T+1)/low(T) - 1            # 开盘相对信号日低点
分桶看: 逐笔收益/胜率/持有天数, 以及从入场开盘起 +1/5/10/20 日原始收益 (退出前路径).
判据: 若某桶 n>=30 且收益/胜率同向大幅劣化 => 候选过滤; 再跑引擎臂验证 NAV.
"""
import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
TAG = 'base_rk1022'

tr = pd.read_csv(f'{R}/trade_realized.csv.{TAG}', dtype={'code': str})
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['exit_date'] = pd.to_datetime(tr['exit_date'])

rows = []
for code, g in tr.groupby('code'):
    f = f'{BT}/{code}_qfq.csv'
    try:
        px = pd.read_csv(f, usecols=['datetime', 'open', 'high', 'low', 'close'])
    except FileNotFoundError:
        continue
    px['datetime'] = pd.to_datetime(px['datetime'])
    px = px.sort_values('datetime').reset_index(drop=True)
    idx = {d: i for i, d in enumerate(px['datetime'])}
    for _, t in g.iterrows():
        i = idx.get(t['entry_date'])
        if i is None or i < 1:
            rows.append(dict(code=code, entry_date=t['entry_date'], ok=0))
            continue
        prev = px.iloc[i - 1]
        cur = px.iloc[i]
        o = float(cur['open'])
        rec = dict(
            code=code, entry_date=t['entry_date'], exit_date=t['exit_date'],
            hold_days=t['hold_days'], ret=t['ret'], ok=1,
            gap=o / float(prev['close']) - 1 if prev['close'] > 0 else np.nan,
            gap_hi=o / float(prev['high']) - 1 if prev['high'] > 0 else np.nan,
            gap_lo=o / float(prev['low']) - 1 if prev['low'] > 0 else np.nan,
        )
        # 路径收益: 从入场日开盘价起, 持有到 +k 个交易日收盘
        for k in (1, 5, 10, 20):
            j = i + k - 1
            rec[f'r{k}'] = float(px.iloc[j]['close']) / o - 1 if j < len(px) else np.nan
        rows.append(rec)

d = pd.DataFrame(rows)
bad = d[d.ok == 0]
d = d[d.ok == 1].copy()
print(f'交易 {len(tr)} 笔, 匹配 {len(d)}, 未匹配 {len(bad)}')
print('gap 分布: mean={:+.2%} median={:+.2%} std={:.2%} 高开占比={:.1%}'.format(
    d.gap.mean(), d.gap.median(), d.gap.std(), (d.gap > 0).mean()))


def stats(x):
    r = x.ret
    return (len(x), r.mean(), (r > 0).mean(), x.hold_days.mean(),
            x.r1.mean(), x.r5.mean(), x.r10.mean(), x.r20.mean(),
            (x.r5 > 0).mean(), (x.r10 > 0).mean(), (x.r20 > 0).mean())


hdr = f'{"bin":18s} {"n":>3s} {"ret":>8s} {"win":>6s} {"hold":>5s} ' \
      f'{"r1":>7s} {"r5":>7s} {"r10":>7s} {"r20":>7s} {"w5":>6s} {"w10":>6s} {"w20":>6s}'

print('\n=== 按 gap 分位 (实际成交笔) ===')
print(hdr)
qs = d.gap.quantile([0, .2, .4, .6, .8, 1.0]).values
d['qbin'] = pd.cut(d.gap, qs, include_lowest=True)
for b, x in d.groupby('qbin', observed=True):
    print(f'{str(b):18s} ' + ' '.join(
        [f'{stats(x)[0]:3d}', f'{stats(x)[1]:8.2%}', f'{stats(x)[2]:6.1%}',
         f'{stats(x)[3]:5.1f}', f'{stats(x)[4]:7.2%}', f'{stats(x)[5]:7.2%}',
         f'{stats(x)[6]:7.2%}', f'{stats(x)[7]:7.2%}', f'{stats(x)[8]:6.1%}',
         f'{stats(x)[9]:6.1%}', f'{stats(x)[10]:6.1%}']))

print('\n=== 固定阈值过滤 (保留/剔除两侧) ===')
print(hdr)
for thr in (-0.05, -0.03, -0.02, 0.02, 0.03, 0.05, 0.07):
    for side, lab in ((-1, f'gap<{thr:+.0%}'), (1, f'gap>{thr:+.0%}')):
        x = d[d.gap < thr] if side < 0 else d[d.gap > thr]
        if len(x) < 10:
            continue
        s = stats(x)
        print(f'{lab:18s} {s[0]:3d} {s[1]:8.2%} {s[2]:6.1%} {s[3]:5.1f} '
              f'{s[4]:7.2%} {s[5]:7.2%} {s[6]:7.2%} {s[7]:7.2%} '
              f'{s[8]:6.1%} {s[9]:6.1%} {s[10]:6.1%}')

print('\n=== 全部 (基准) ===')
s = stats(d)
print(f'{"all":18s} {s[0]:3d} {s[1]:8.2%} {s[2]:6.1%} {s[3]:5.1f} '
      f'{s[4]:7.2%} {s[5]:7.2%} {s[6]:7.2%} {s[7]:7.2%} {s[8]:6.1%} {s[9]:6.1%} {s[10]:6.1%}')

print('\n=== gap_hi (开盘相对信号日最高) 分位 ===')
print(hdr)
qs2 = d.gap_hi.quantile([0, .2, .4, .6, .8, 1.0]).values
d['qbin_hi'] = pd.cut(d.gap_hi, qs2, include_lowest=True)
for b, x in d.groupby('qbin_hi', observed=True):
    s = stats(x)
    print(f'{str(b):18s} {s[0]:3d} {s[1]:8.2%} {s[2]:6.1%} {s[3]:5.1f} '
          f'{s[4]:7.2%} {s[5]:7.2%} {s[6]:7.2%} {s[7]:7.2%} '
          f'{s[8]:6.1%} {s[9]:6.1%} {s[10]:6.1%}')

# 相关/回归
try:
    from scipy import stats as st
    for col in ('gap', 'gap_hi', 'gap_lo'):
        x = d[[col, 'ret', 'r5', 'r10']].dropna()
        rho, p = st.spearmanr(x[col], x['ret'])
        print(f'\nSpearman {col} vs ret: rho={rho:+.3f} p={p:.3f} n={len(x)}')
except Exception as e:
    print('scipy不: ', e)

d.to_csv('/mnt/d/quant/.arms/probe/entrygap_1022.csv', index=False)
print('\n← .arms/probe/entrygap_1022.csv')
