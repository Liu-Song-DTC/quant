"""短期胜率谷归因 (2026-10-25): T+3/T+5 的胜率塌陷是系统性的还是彩票的?

账簿 166 笔实际成交, 次日开盘入场, 按 horizon 盯市。问三件事:
  (a) 谷是**普遍性回撤**还是少数腿拖累?  → 剔除最差1/3/5笔后胜率还塌吗?
  (b) 亏损腿是不是"早夭"型?            → 亏损腿的平均谷底在第几天?
  (c) 回撤是"洗盘后修复"还是"一路下行"?  → 谷底后 T+10 是否回到入场价之上?
只读离线。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt/d/quant')
RR = ROOT / 'strategy/rolling_validation_results'
H = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40)


def main():
    t = pd.read_csv(RR / 'trade_realized.csv', dtype={'code': str})
    t['code'] = t['code'].str.zfill(6)
    t['entry_date'] = pd.to_datetime(t['entry_date'])
    t['exit_date'] = pd.to_datetime(t['exit_date'])
    q = pd.read_parquet(ROOT / 'data/research/qfq_kline.parquet',
                        columns=['code', 'date', 'open', 'close'])
    q['code'] = q['code'].astype(str).str.zfill(6)
    q['date'] = pd.to_datetime(q['date'])
    piv_c = q.pivot_table(index='date', columns='code', values='close')
    piv_o = q.pivot_table(index='date', columns='code', values='open')

    rows = []
    for _, r in t.iterrows():
        if r['code'] not in piv_c.columns:
            continue
        s, o = piv_c[r['code']], piv_o[r['code']]
        idx = s.index.searchsorted(r['entry_date'])
        if idx + 1 >= len(s):
            continue
        e = idx + 1
        if not np.isfinite(o.iloc[e]) or o.iloc[e] <= 0:
            continue
        entry = float(o.iloc[e])
        path = {}
        for h in H:
            j = e + h - 1
            path[h] = (float(s.iloc[j]) / entry - 1.0) if j < len(s) and np.isfinite(s.iloc[j]) else np.nan
        ex = s.index.searchsorted(r['exit_date'])
        path['exit'] = (float(s.iloc[ex]) / entry - 1.0) if ex < len(s) and np.isfinite(s.iloc[ex]) else np.nan
        path['hold'] = r['hold_days']
        path['ret'] = r['ret']
        path['win'] = r['ret'] > 0
        path['entry_date'] = r['entry_date']
        path['code'] = r['code']
        rows.append(path)
    df = pd.DataFrame(rows)
    print(f'[probe] 可算路径 {len(df)}/{len(t)} 笔\n')

    print('--- 盯市胜率曲线 (毛收益, 未扣费) ---')
    hdr = '  horizon:  ' + '  '.join(f'T+{h:<4d}' for h in H)
    print(hdr)
    print('  胜率   :  ' + '  '.join(f'{100*(df[h] > 0).mean():5.1f}%' for h in H))
    print('  均值   :  ' + '  '.join(f'{100*df[h].mean():+5.2f}%' for h in H))
    print('  中位   :  ' + '  '.join(f'{100*df[h].median():+5.2f}%' for h in H))
    print('  n      :  ' + '  '.join(f'{df[h].notna().sum():5d}' for h in H))

    print('\n--- (a) 谷是彩票还是普遍? 剔除最差 N 笔后的 T+3 胜率 ---')
    for k in (0, 1, 3, 5, 10):
        idx = df[3].sort_values().index[:k]
        sub = df.drop(idx)
        print(f'  剔最差{k:>2d}笔: n={len(sub):3d}  T+3胜率={100*(sub[3] > 0).mean():5.1f}%  '
              f'T+5胜率={100*(sub[5] > 0).mean():5.1f}%')
    print('  → 若剔5笔就回到50%以上 ⇒ 谷由少数腿承载(彩票); 否则是普遍性回撤')

    print('\n--- (b) 亏损腿 vs 盈利腿的路径形态 ---')
    for name, grp in (('亏损腿', df[~df['win']]), ('盈利腿', df[df['win']])):
        if not len(grp):
            continue
        print(f'  {name} n={len(grp):3d}  T+3均值={100*grp[3].mean():+6.2f}%  '
              f'T+5均值={100*grp[5].mean():+6.2f}%  T+10均值={100*grp[10].mean():+6.2f}%  '
              f'最终={100*grp["ret"].mean():+6.2f}%  持有中位={grp["hold"].median():.0f}d')

    print('\n--- (c) 谷底后是否修复: T+3 为负的样本, T+10 回到正的比例 ---')
    neg3 = df[df[3] < 0]
    if len(neg3):
        print(f'  T+3为负 n={len(neg3)}  其中最终盈利占比={100*neg3["win"].mean():.1f}%  '
              f'T+10均值={100*neg3[10].mean():+.2f}%  T+20均值='
              f'{100*neg3[20].mean():+.2f}%')
        rec = neg3[neg3[10] > 0]
        print(f'  T+10 回到入场价之上: {len(rec)}/{len(neg3)} = {100*len(rec)/len(neg3):.1f}%')
    neg5 = df[df[5] < 0]
    if len(neg5):
        print(f'  T+5为负 n={len(neg5)}  最终盈利占比={100*neg5["win"].mean():.1f}%  '
              f'T+10均值={100*neg5[10].mean():+.2f}%')
    print('\n--- 持有期分组 ---')
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 999)):
        g = df[(df['hold'] > lo) & (df['hold'] <= hi)]
        if len(g):
            print(f'  hold {lo:>3d}-{hi:<3d} n={len(g):3d} 最终wr={100*g["win"].mean():5.1f}% '
                  f'最终均值={100*g["ret"].mean():+6.2f}%  T+3均值={100*g[3].mean():+6.2f}%')
    df.to_csv(ROOT / '.arms/armout/shortwin_attrib_1025.csv', index=False)
    print('\n→ .arms/armout/shortwin_attrib_1025.csv')


if __name__ == '__main__':
    sys.exit(main())
