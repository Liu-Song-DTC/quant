#!/usr/bin/env python3
"""修复态首夜铁律对比: 今晚基线(todate 10-20) vs fix365臂(todate 10-19, 同代码 cb245b3).

真·铁律判据(memory fix365-arm-held): 两臂同代码同数据态, 历史段(≤10-19)净值
应逐位相同, 唯一允许差异:
  (i)  7 只重述股的 3 笔交易(fix臂曾选 601198×2, 002541×1; 供应商修正
       corporate_actions 19格 → 表驱动乘性重建重写受影响股票全史);
  (ii) 尾段追加(10-20 一天)。
判据口径(navlaw): 共有区间历史段 净值(连续) + 选股集合(离散) 逐位不变,
容差 1e-12。

用法: cd /mnt/d/quant/strategy && ../.venv/bin/python \
        /mnt/d/quant/.arms/probe/navlaw_fix_first_night.py
"""
import pandas as pd

RVD = 'rolling_validation_results'
TONIGHT = f'{RVD}/equity_curve.csv.base1020fix'
FIX = '/mnt/d/quant/.arms/armout/fix365_1020/equity_curve.csv'
TONIGHT_SEL = f'{RVD}/portfolio_selections.csv.base1020fix'
FIX_SEL = '/mnt/d/quant/.arms/armout/fix365_1020/portfolio_selections.csv'
TOL = 1e-12

a = pd.read_csv(TONIGHT)
a['date'] = pd.to_datetime(a['date'])
b = pd.read_csv(FIX)
b['date'] = pd.to_datetime(b['date'])
m = a.merge(b, on='date', suffixes=('_t', '_f'))
hist = m[m['date'] <= '2026-10-19']
print(f"今晚基线末行: {a['date'].iloc[-1].date()} NAV {a['nav'].iloc[-1]:,.2f}")
print(f"fix365臂末行: {b['date'].iloc[-1].date()} NAV {b['nav'].iloc[-1]:,.2f}")
print(f"共有历史段(≤10-19): {len(hist)} 天")

d = (hist['nav_t'] - hist['nav_f']).abs()
bad = hist[d > TOL]
print(f"净值差>1e-12: {len(bad)} 天" + (f"  首: {bad['date'].iloc[0].date()}"
      f"  Δ={d[bad.index[0]]:+.6f}  末: {bad['date'].iloc[-1].date()}" if len(bad) else ""))
if len(bad):
    print("  差异日明细(前10):")
    for _, r in bad.head(10).iterrows():
        print(f"    {r['date'].date()}  今晚 {r['nav_t']:,.2f} vs fix {r['nav_f']:,.2f}"
              f"  Δ {r['nav_t'] - r['nav_f']:+,.4f}")
tail = m[m['date'] > '2026-10-19']
print(f"尾段(>10-19): {len(tail)} 天" +
      (f"  NAV {tail['nav_t'].iloc[-1]:,.2f} ret {tail['nav_t'].iloc[-1] / hist['nav_t'].iloc[-1] - 1:+.3%}"
       if len(tail) else ""))

# 选股集合(离散)
sa = pd.read_csv(TONIGHT_SEL, dtype={'code': str})
sb = pd.read_csv(FIX_SEL, dtype={'code': str})
sa['date'] = pd.to_datetime(sa['date']).dt.strftime('%Y-%m-%d')
sb['date'] = pd.to_datetime(sb['date']).dt.strftime('%Y-%m-%d')
ka = set(zip(sa['code'], sa['date']))
kb = set(zip(sb['code'], sb['date']))
lost, new = kb - ka, ka - kb
print(f"\n选股集合: 今晚 {len(ka)} / fix {len(kb)} / 共有 {len(ka & kb)}")
for tag, s in (('fix有今晚无(历史段应=重述股)', lost), ('今晚有fix无(应=尾段10-20)', new)):
    if s:
        print(f"  {tag}: {len(s)}")
        for c, dt in sorted(s, key=lambda x: x[1])[:8]:
            print(f"    {c} @ {dt}")

# 重述股关联交易核对
RESTATED = {'600293', '601198', '300773', '601059', '000563', '002957', '002541'}
hits_f = [k for k in kb if k[0] in RESTATED and k[1] <= '2026-10-19']
hits_t = [k for k in ka if k[0] in RESTATED and k[1] <= '2026-10-19']
print(f"\n重述股历史选股: fix臂 {sorted(hits_f)} / 今晚 {sorted(hits_t)}")
