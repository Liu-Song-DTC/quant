#!/usr/bin/env python3
"""池级 mchg5 年份结构 (含未入选候选) — 臂前最后一块拼图."""
import pandas as pd
qq = pd.read_csv('/mnt/d/quant/.arms/armout/margind_pool_1023.csv', dtype={'code': str})
qq['date'] = pd.to_datetime(qq['date'])
qq['year'] = qq['date'].dt.year
print('══ 池级 nr<=2: mchg5>=3% vs 其余, 按 year (w5/w20 胜率+均值) ══')
for y, g in qq.groupby('year'):
    a, b = g[g['mchg5'] >= 3], g[(g['mchg5'] < 3) | g['mchg5'].isna()]
    if not len(g):
        continue
    print(f'  {y}: 拥挤 n={len(a):3d} w5={(a.w5>0).mean()*100 if len(a) else float("nan"):5.1f}% '
          f'w20={(a.w20>0).mean()*100 if len(a) else float("nan"):5.1f}% '
          f'均w20={a.w20.mean()*100 if len(a) else float("nan"):+6.2f} | '
          f'其余 n={len(b):3d} w5={(b.w5>0).mean()*100 if len(b) else float("nan"):5.1f}% '
          f'w20={(b.w20>0).mean()*100 if len(b) else float("nan"):5.1f}% '
          f'均w20={b.w20.mean()*100 if len(b) else float("nan"):+6.2f} | '
          f'锚选 crowded {int(a.is_ent.sum())}/{int(g.is_ent.sum())}')
# 未入选候选单独看 (臂的增量来源)
nh = qq[~qq['is_ent']]
print('\n══ 未入选候选 (臂的增量来源) mchg5>=3% vs 其余 ══')
a, b = nh[nh['mchg5'] >= 3], nh[(nh['mchg5'] < 3) | nh['mchg5'].isna()]
print(f'  拥挤 n={len(a):3d} w5={(a.w5>0).mean()*100:5.1f}% w20={(a.w20>0).mean()*100:5.1f}% '
      f'均w20={a.w20.mean()*100:+6.2f}% | 其余 n={len(b):3d} '
      f'w5={(b.w5>0).mean()*100:5.1f}% w20={(b.w20>0).mean()*100:5.1f}% 均w20={b.w20.mean()*100:+6.2f}%')
for y, g in nh.groupby('year'):
    a, b = g[g['mchg5'] >= 3], g[(g['mchg5'] < 3) | g['mchg5'].isna()]
    if not len(g):
        continue
    print(f'  {y}: 拥挤 n={len(a):3d} w20={(a.w20>0).mean()*100 if len(a) else float("nan"):5.1f}% '
          f'均w20={a.w20.mean()*100 if len(a) else float("nan"):+6.2f} | '
          f'其余 n={len(b):3d} w20={(b.w20>0).mean()*100 if len(b) else float("nan"):5.1f}% '
          f'均w20={b.w20.mean()*100 if len(b) else float("nan"):+6.2f}')
