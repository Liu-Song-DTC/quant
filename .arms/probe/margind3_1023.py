#!/usr/bin/env python3
"""margin mchg5 分离的 regime/年份结构 + 臂前定性 (2026-10-23, 只读)."""
import numpy as np
import pandas as pd

AR = '/mnt/d/quant/.arms/armout'
mm = pd.read_csv(f'{AR}/margind_realized_1023.csv', dtype={'code': str})
mm['code'] = mm['code'].str.zfill(6)
mm['entry_date'] = pd.to_datetime(mm['entry_date'])
mm['sig_date'] = pd.to_datetime(mm['sig_date'])
mm['year'] = mm['entry_date'].dt.year
pool = pd.read_csv('/mnt/d/quant/.arms/pool_1022.csv', dtype={'code': str})
pool['code'] = pool['code'].str.zfill(6)
pool['date'] = pd.to_datetime(pool['date'])
mm = mm.merge(pool[['code', 'date', 'regime']], left_on=['code', 'sig_date'],
              right_on=['code', 'date'], how='left').drop(columns=['date'])

print('══ regime × mchg5>=3% ══')
for r, g in mm.groupby('regime'):
    a, b = g[g['mchg5'] >= 3], g[(g['mchg5'] < 3) | g['mchg5'].isna()]
    print(f'  {str(r):8s}: 拥挤 n={len(a):2d} tW={(a.ret>0).mean()*100 if len(a) else float("nan"):5.1f}% '
          f'均ret={a.ret.mean()*100 if len(a) else float("nan"):+6.2f} | 其余 n={len(b):3d} '
          f'tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}')

print('\n══ 年份 × 拥挤桶 均ret 全表 ══')
for y, g in mm.groupby('year'):
    a, b = g[g['mchg5'] >= 3], g[(g['mchg5'] < 3) | g['mchg5'].isna()]
    print(f'  {y}: 拥挤 n={len(a):2d} tW={(a.ret>0).mean()*100 if len(a) else float("nan"):5.1f}% '
          f'均ret={a.ret.mean()*100 if len(a) else float("nan"):+6.2f}% 均持仓{(a.hold_days.mean() if len(a) and "hold_days" in a else float("nan")):.0f}d'
          f' | 其余 n={len(b):3d} tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}%')

print('\n══ 2025 拥挤 10 笔明细 (反转源) ══')
c25 = mm[(mm.year == 2025) & (mm['mchg5'] >= 3)]
cols = [c for c in ['code', 'entry_date', 'ret', 'weight', 'mchg5', 'mchg20'] if c in mm.columns]
print(c25[cols].to_string(index=False))

print('\n══ 阈值敏感性 (tW 拥挤 vs 其余, 全样本) ══')
for th in [1, 2, 3, 5, 8]:
    a, b = mm[mm['mchg5'] >= th], mm[mm['mchg5'] < th]
    print(f'  mchg5>={th}%: n={len(a):3d} tW={(a.ret>0).mean()*100:5.1f}% 均ret={a.ret.mean()*100:+6.2f} '
          f'| 其余 tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}')

print('\n══ H1-24 vs H25-26 两段 × 阈值 ══')
for lab, g in [('2021-24', mm[mm.year <= 2024]), ('2025-26', mm[mm.year >= 2025])]:
    for th in [3, 5]:
        a, b = g[g['mchg5'] >= th], g[g['mchg5'] < th]
        print(f'  {lab} mchg5>={th}%: n={len(a):2d} tW={(a.ret>0).mean()*100 if len(a) else float("nan"):5.1f}% '
              f'均ret={a.ret.mean()*100 if len(a) else float("nan"):+6.2f} | 其余 n={len(b):3d} '
              f'tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}')
