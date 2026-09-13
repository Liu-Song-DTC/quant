#!/usr/bin/env python3
"""P0-1 Stage2 诚实反事实结果比对: deadline影子跑 vs 锚点(泄漏口径)

锚点(2026-09-12 .venv 验证冷跑, PIT gate 后权威基线):
  NAV 1,143,938 / +357.58% / Sharpe 1.5555 / MDD 27.41%
Stage2(2026-09-13, paths.fundamental=fundamental_data_deadline, 法定截止上界):
  同代码同数据同池, 唯一差=基本面数据可用日期PIT化。

只读。比对四指标 + 逐年NAV。
执行: /mnt/d/quant/.venv/bin/python analysis/probe_pit_stage2_compare_0913.py
"""
import os
import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
BK = os.path.join(ROOT, 'strategy', 'backups', 'stage2_0913')


def load_curve(path):
    df = pd.read_csv(path, parse_dates=['date'] if 'date' in
                     pd.read_csv(path, nrows=1).columns else None)
    return df


def metrics(df, nav_col='nav'):
    nav = df[nav_col].values.astype(float)
    ret_total = nav[-1] / nav[0] - 1
    daily = np.diff(nav) / nav[:-1]
    sharpe = np.mean(daily) / max(np.std(daily), 1e-12) * np.sqrt(252)
    peak = np.maximum.accumulate(nav)
    mdd = ((peak - nav) / peak).max()
    return nav[-1], ret_total, sharpe, mdd


def main():
    print('=' * 72)
    print('P0-1 Stage2: 诚实反事实(deadline上界) vs 锚点(泄漏口径)')
    curves = {}
    for tag, path in [('anchor', os.path.join(BK, 'equity_curve.csv')),
                      ('stage2', os.path.join(BK, 'stage2_equity_curve.csv'))]:
        if not os.path.exists(path):
            print(f'[MISSING] {path}')
            return
        curves[tag] = load_curve(path)
        print(f'[{tag}] {path}: {len(curves[tag])} 行, '
              f'{curves[tag].iloc[0,0]} ~ {curves[tag].iloc[-1,0]}')

    print('\n四指标:')
    print(f'{"指标":<10}{"锚点":>14}{"Stage2":>14}{"Δ":>14}')
    ms = {}
    for tag in ('anchor', 'stage2'):
        nav, rt, sp, mdd = metrics(curves[tag])
        ms[tag] = (nav, rt, sp, mdd)
    for name, i in (('NAV', 0), ('收益%', 1), ('Sharpe', 2), ('MDD%', 3)):
        a, s = ms['anchor'][i], ms['stage2'][i]
        if i in (1, 3):
            print(f'{name:<10}{100*a:>13.2f}%{100*s:>13.2f}%{100*(s - a):>+13.2f}pp')
        else:
            print(f'{name:<10}{a:>14,.0f}{s:>14,.0f}{s - a:>+14,.0f}'
                  if i == 0 else f'{name:<10}{a:>14.4f}{s:>14.4f}{s - a:>+14.4f}')

    print('\n逐年NAV(年末):')
    for tag in ('anchor', 'stage2'):
        c = curves[tag]
        c['_y'] = pd.to_datetime(c['date']).dt.year
        print(f'  [{tag}]', {int(y): f'{g.iloc[-1]["nav"]:,.0f}'
                             for y, g in c.groupby('_y')})
    print('\n结论: Stage2(法定截止上界) NAV vs 锚点 — '
          '真实公告早于截止 → 真实泄漏影响小于此差, 区间上界。')


if __name__ == '__main__':
    main()
