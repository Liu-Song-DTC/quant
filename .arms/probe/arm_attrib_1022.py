#!/usr/bin/env python3
"""臂归因 (2026-10-22): 基准 vs 臂, 按 (date,code) 对齐入场, 补前瞻收益.

回答四问:
  1) 共享入场是否逐字相同 (胜率/收益);
  2) 基准独有 (被跳过) 入场的胜率/收益;
  3) 臂独有 (路径创造) 入场的胜率/收益;
  4) 臂独有收益是否集中在单笔 (ex-top 判据).

用法: python arm_attrib_1022.py new2 temom [more tags...]
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
import winrate_diag_1022 as W
import winrate_diag2_1022 as W2

RVD = '/mnt/d/quant/strategy/rolling_validation_results'
FWD = ['w5', 'w20', 'r5', 'r20']


def load(tag):
    eq = pd.read_csv(f'{RVD}/equity_curve.csv.{tag}', parse_dates=['date'])
    cal = list(eq['date'].sort_values())
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    s = pd.read_csv(f'{RVD}/portfolio_selections.csv.{tag}', dtype={'code': str},
                    parse_dates=['date'])
    s['code'] = s['code'].str.zfill(6)
    s['exec_date'] = s['date'].map(nxt)
    s = W.add_forward(s)
    return s


def stats(s, label):
    if len(s) == 0:
        print(f'  {label}: n=0')
        return
    w5 = s['w5'].astype(float).mean() * 100
    w20 = s['w20'].astype(float).mean() * 100
    r5 = s['r5'].mean() * 100
    r20 = s['r20'].mean() * 100
    print(f'  {label}: n={len(s):3d} w5={w5:5.1f}% w20={w20:5.1f}% '
          f'r5={r5:+.2f}% r20={r20:+.2f}% sum20={s["r20"].sum()*100:+.1f}pp')


def main():
    base_tag = sys.argv[1]
    base = load(base_tag)
    print(f'# base={base_tag} n={len(base)}')
    kb = set(zip(base['date'], base['code']))
    for tag in sys.argv[2:]:
        arm = load(tag)
        ka = set(zip(arm['date'], arm['code']))
        sh_b = base[base.apply(lambda r: (r['date'], r['code']) in ka, axis=1)]
        sh_a = arm[arm.apply(lambda r: (r['date'], r['code']) in kb, axis=1)]
        skipped = base[base.apply(lambda r: (r['date'], r['code']) not in ka, axis=1)]
        added = arm[arm.apply(lambda r: (r['date'], r['code']) not in kb, axis=1)]
        print(f'\n=== {tag} (n={len(arm)}) vs {base_tag} ===')
        stats(sh_b, '共享')
        stats(skipped, '被跳过 (base独有)')
        stats(added, '路径新增 (arm独有)')
        if len(sh_a) == len(sh_b):
            a = sh_b.sort_values(['date', 'code'])[FWD].astype(float).values
            b = sh_a.sort_values(['date', 'code'])[FWD].astype(float).values
            d = np.nanmax(np.abs(a - b))
            print(f'  共享前瞻逐字: 最大|Δ|={d:.2e} ({"一致" if d < 1e-12 else "⚠不一致"})')
        if len(skipped):
            sk = skipped.copy()
            sk = sk[sk['r20'].notna()]
            top = sk.reindex(sk['r20'].abs().sort_values(ascending=False).index).head(5)
            print('  被跳过中 |r20| 最大 5 笔: ' +
                  ', '.join(f'{r.date.date()}/{r.code} {r.r20*100:+.1f}%' for r in top.itertuples()))
        if len(added):
            ad = added[added['r20'].notna()].copy()
            tot = ad['r20'].sum()
            srt = ad.sort_values('r20', ascending=False)
            t1 = srt['r20'].iloc[0]
            print(f'  臂独有 sum20={tot*100:+.1f}pp; top1={t1*100:+.1f}% '
                  f'({t1/tot*100 if tot else np.nan:+.0f}%); '
                  f'ex-top1={tot-t1:+.1f}pp; ex-top3={tot-srt["r20"].head(3).sum():+.1f}pp')
        # 整日跳过: base 入场日里 arm 当日 0 入场
        bdays = set(base['date']); adays = set(arm['date'])
        skipped_days = sorted(bdays - adays)
        print(f'  跳过 {skipped["date"].nunique()} 笔 / {len(skipped_days)} 个整日; '
              f'新增 {len(added)} 笔 / {added["date"].nunique()} 日; '
              f'入场日 {len(bdays)} → {len(adays)}')
        if skipped_days and len(skipped_days) <= 40:
            print('  整日跳过: ' + ', '.join(str(pd.Timestamp(d).date()) for d in skipped_days))


if __name__ == '__main__':
    main()
