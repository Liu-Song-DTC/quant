#!/usr/bin/env python3
"""候选胜率过滤器的分年/分半稳定性 + 重叠 (2026-10-22, 只读)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
import winrate_diag_1022 as W
import winrate_diag2_1022 as W2

SIGROWS, SIGHDR = W2.SIGROWS, W2.SIGHDR


def attach_sig(m):
    hdr = pd.read_csv(SIGHDR, nrows=0).columns.tolist()
    sig = pd.read_csv(SIGROWS, header=None, names=hdr)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['date_s'] = pd.to_datetime(sig['date']).dt.strftime('%Y-%m-%d')
    keep = ['code', 'date_s', 'signal_level', 'trend_type', 'chan_buy_point',
            'industry', 'ml_score', 'adjusted_score', 'gate_quality']
    m2 = m.copy()
    m2['date_s'] = m2['date'].dt.strftime('%Y-%m-%d')
    m2 = m2.merge(sig[[c for c in keep if c in sig]].rename(columns={'date_s': 'date_s'}),
                  on=['code', 'date_s'], how='left')
    return m2


def cell(df, label):
    df = df[df['w5'].notna()]
    n = len(df)
    if n == 0:
        return
    tr = df['ret'].dropna()
    kt = int((tr > 0).sum())
    kw = int((df['w5'] > 0).sum())
    k20 = int((df['w20'].dropna() > 0).sum())
    n20 = int(df['w20'].notna().sum())
    lo, hi = W.wilson(kt, len(tr)) if len(tr) else (np.nan, np.nan)
    print(f'{label:34s} n={n:3d} tW={kt/len(tr)*100 if len(tr) else np.nan:5.1f}% [{lo:.0f}-{hi:.0f}] '
          f'w5={kw/n*100:5.1f}% w20={k20/n20*100 if n20 else np.nan:5.1f}% '
          f'r20={df["r20"].mean()*100:+6.2f}% mfe5={df["mfe5"].mean()*100:5.1f}% '
          f'mfe20={df["mfe20"].mean()*100:5.1f}% fast5={float((df["mfe5"] >= 0.7*df["mfe20"]).mean())*100:5.1f}%')


def main():
    m, eq = W.load_panel()
    m = W.add_forward(m)
    m = W2.add_extras(m)
    m = attach_sig(m)
    mm = m[m['w5'].notna()].copy()
    mm['year'] = mm['date'].dt.year
    mm['half'] = np.where(mm['year'] <= 2023, 'H1', 'H2')
    mm['rank_ok'] = mm['srank'] <= 2
    mm['tt2'] = mm['trend_type'] != 2
    mm['score_hi'] = mm['score'] > 0.45
    mm['sl_ok'] = mm['signal_level'] != 1
    mm['ah_ok'] = mm['industry'] != 'AH股'
    mm['fast5'] = mm['mfe5'] >= 0.7 * mm['mfe20']

    print('=' * 110)
    print('A. 候选过滤器总体 (w5 非空样本)')
    print('-' * 110)
    for name, sub in [('ALL', mm), ('rank<=2', mm[mm.rank_ok]), ('drop rank3', mm[~mm.rank_ok]),
                      ('score>0.45', mm[mm.score_hi]), ('score<=0.45', mm[~mm.score_hi]),
                      ('TT!=2', mm[mm.tt2]), ('TT==2', mm[~mm.tt2]),
                      ('SL!=1', mm[mm.sl_ok]), ('SL==1', mm[~mm.sl_ok]),
                      ('drop AH', mm[mm.ah_ok]), ('AH only', mm[~mm.ah_ok]),
                      ('rank<=2 & score>0.45', mm[mm.rank_ok & mm.score_hi]),
                      ('rank<=2 & TT!=2', mm[mm.rank_ok & mm.tt2]),
                      ('TT!=2 & score>0.45', mm[mm.tt2 & mm.score_hi]),
                      ('rank<=2 & SL!=1', mm[mm.rank_ok & mm.sl_ok]),
                      ('rank<=2 & drop AH', mm[mm.rank_ok & mm.ah_ok])]:
        cell(sub, name)

    print('\n' + '=' * 110)
    print('B. 分年 (tW / w5 / w20 / r20)')
    print('-' * 110)
    for name, col, val in [('rank<=2', 'rank_ok', True), ('score>0.45', 'score_hi', True),
                           ('TT!=2', 'tt2', True), ('drop AH', 'ah_ok', True)]:
        print(f'\n[{name}] 每年 KEEP / DROP  (n | tW% | w5% | w20% | r20%)')
        print('  year   KEEP n/tW/w5/w20/r20        DROP n/tW/w5/w20/r20')
        for y, g in mm.groupby('year'):
            k = g[g[col] == val]; d = g[g[col] != val]
            def f(x):
                if not len(x): return '   -'
                tr = x['ret'].dropna()
                return (f'{len(x):3d}/{(tr>0).mean()*100:4.1f}/{x.w5.mean()*100:4.1f}/'
                        f'{x.w20.mean()*100:4.1f}/{x.r20.mean()*100:+5.2f}')
            print(f'   {y}   {f(k)}      {f(d)}')
        kept = mm[mm[col] == val]; drop = mm[mm[col] != val]
        for tag, g in [('KEEP', kept), ('DROP', drop)]:
            tr = g['ret'].dropna()
            print(f'   {tag}: n={len(g)} tW={(tr>0).mean()*100:.1f}% w5={g.w5.mean()*100:.1f}% '
                  f'w20={g.w20.mean()*100:.1f}% r20={g.r20.mean()*100:+.2f}%')

    print('\n' + '=' * 110)
    print('C. 分半段 (2021-23 / 2024-26)')
    print('-' * 110)
    for h, g in mm.groupby('half'):
        print(f'\n[{h}]')
        for name, sub in [('ALL', g), ('rank<=2', g[g.rank_ok]), ('score>0.45', g[g.score_hi]),
                          ('TT!=2', g[g.tt2]), ('drop AH', g[g.ah_ok]),
                          ('rank<=2 & score>0.45', g[g.rank_ok & g.score_hi])]:
            cell(sub, '  ' + name)

    print('\n' + '=' * 110)
    print('D. 过滤器重叠 (247 入选)')
    print('-' * 110)
    for a in ['rank_ok', 'score_hi', 'tt2', 'sl_ok', 'ah_ok']:
        print(f'  {a:10s} 保留 {int(mm[a].sum()):3d} / {len(mm)}')
    print(pd.crosstab(mm['rank_ok'], mm['score_hi'], margins=True))
    print(pd.crosstab(mm['rank_ok'], mm['tt2'], margins=True))
    print(pd.crosstab(mm['tt2'], mm['score_hi'], margins=True))
    print('\nrank 分布 vs 分数:')
    print(mm.groupby('srank')['score'].agg(['size', 'mean', 'min', 'max']).to_string())

    print('\n' + '=' * 110)
    print('E. 快速拉升: 到第5日已实现 MFE20 的 70%+ (fast5)')
    print('-' * 110)
    cell(mm[mm.fast5], 'fast5=True')
    cell(mm[~mm.fast5], 'fast5=False')
    print('\nfast5 占比按 rank / TT / score 组:')
    for col in ['srank', 'trend_type', 'score_hi']:
        t = mm.groupby(col).agg(n=('fast5', 'size'), fast5=('fast5', 'mean'),
                                w5=('w5', 'mean'), w20=('w20', 'mean'), r20=('r20', 'mean'))
        t['fast5'] = (t['fast5'] * 100).round(1)
        t['w5'] = (t['w5'] * 100).round(1)
        t['w20'] = (t['w20'] * 100).round(1)
        t['r20'] = (t['r20'] * 100).round(2)
        print(f'\n[{col}]'); print(t.to_string())

    mm.to_pickle('/mnt/d/quant/.arms/armout/winrate_panel_1022.pkl')
    print('\n' + '=' * 110)
    print('F. 资金利用率 (regime_state 逐日)')
    print('-' * 110)
    reg = pd.read_csv(W.REG, parse_dates=['date'])
    reg['year'] = reg['date'].dt.year
    t = reg.groupby('year').agg(days=('date', 'size'), npos=('n_pos', 'mean'),
                                expo=('exposure', 'mean'), zero=('n_pos', lambda s: (s == 0).mean()))
    t['zero'] = (t['zero'] * 100).round(1); t['npos'] = t['npos'].round(2); t['expo'] = t['expo'].round(3)
    print(t.to_string())
    eq2 = pd.read_csv(W.EQ, parse_dates=['date'])
    eq2['year'] = eq2['date'].dt.year
    z = eq2.groupby('year')['daily_ret'].apply(lambda s: float((s.dropna().abs() < 1e-9).mean()) * 100)
    print('\n日收益=0 占比(%):'); print(z.round(1).to_string())
    # 持仓数 vs 目标敞口 (调仓日快照)
    print('\nexposure 分布:', reg['exposure'].describe().round(3).to_dict())


if __name__ == '__main__':
    main()
