#!/usr/bin/env python3
"""margin mchg5 分离的混杂排查 (2026-10-23, 只读) — 胜率轴首个 realized 真分离.

margind_1023 发现: 已实现账簿 mchg5>=+3% (融资拥挤) n=33 tW 75.8% vs 59.4% (Δ+16.4pp),
均 ret +6.35% vs +3.90% 双优; 池级 w20 Δ+17.5 p=0.0000. 本探针杀三个混杂假设:

  H1 价格动量代理: 融资余额升↔近期股价涨 ⇒ mchg5 可能只是动量重编码.
     → backward 5/20日收益 (qfq_kline parquet) 相关性 + 动量分层内边际.
  H2 聚簇假显著: n=33 按入场日聚簇, 行级检验高估 ⇒ 入场日簇 bootstrap.
  H3 年份/regime 集中: 33 笔若集中在一两段行情 ⇒ 情绪彩票.

用法: .venv/bin/python .arms/probe/margind2_1023.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')

PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
AR = '/mnt/d/quant/.arms/armout'
RNG = np.random.default_rng(20261023)
NBOOT = 5000


def add_backret(df, tcol):
    """backward 5/20 收盘收益 (股票自身交易日, 严格 <t)."""
    codes = ','.join(f"'{c}'" for c in df['code'].unique())
    px = pd.read_parquet(PQ, columns=['code', 'date', 'close'])
    px = px[px['code'].isin(set(df['code']))]
    px['date'] = pd.to_datetime(px['date'])
    px = px.sort_values(['code', 'date'])
    px['c5'] = px.groupby('code')['close'].shift(5)
    px['c20'] = px.groupby('code')['close'].shift(20)
    px['ret5b'] = px['close'] / px['c5'] - 1
    px['ret20b'] = px['close'] / px['c20'] - 1
    key = px.set_index(['code', 'date'])[['ret5b', 'ret20b']]
    return df.join(key, on=['code', tcol])


def clus_boot_delta(mm, mask, col='win'):
    """入场日簇 bootstrap: Δ(tW 或 均ret) 桶 vs 补集."""
    mm = mm.copy()
    mm['win'] = (mm['ret'] > 0).astype(float)
    obs = (mm[mask][col].mean() - mm[~mask][col].mean())
    days = sorted(mm['entry_date'].unique())
    grp = {d: g for d, g in mm.groupby('entry_date')}
    deltas = []
    for _ in range(NBOOT):
        pick = RNG.choice(days, size=len(days), replace=True)
        b = pd.concat([grp[d] for d in pick])
        m = mask.reindex(b.index).fillna(False)
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        deltas.append(b[m][col].mean() - b[~m][col].mean())
    deltas = np.array(deltas)
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return obs, lo, hi, p


def main():
    mm = pd.read_csv(f'{AR}/margind_realized_1023.csv', dtype={'code': str})
    mm['code'] = mm['code'].str.zfill(6)
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['sig_date'] = pd.to_datetime(mm['sig_date'])
    qq = pd.read_csv(f'{AR}/margind_pool_1023.csv', dtype={'code': str})
    qq['code'] = qq['code'].str.zfill(6)
    qq['date'] = pd.to_datetime(qq['date'])

    # 池列控制变量 (mom_60d/turnover/volume_ratio/score 等)
    pool = pd.read_csv('/mnt/d/quant/.arms/pool_1022.csv', dtype={'code': str})
    pool['code'] = pool['code'].str.zfill(6)
    pool['date'] = pd.to_datetime(pool['date'])
    keep = ['code', 'date', 'mom_60d', 'turnover', 'volume_ratio', 'score',
            'eff_score', 'daily_ret', 'regime']
    mm = mm.merge(pool[keep], left_on=['code', 'sig_date'],
                  right_on=['code', 'date'], how='left').drop(columns=['date'])

    mm = add_backret(mm, 'sig_date')
    qq = add_backret(qq, 'date')

    # ══ H1: 动量代理? ══
    print('══ H1 动量代理 ══')
    sub = mm[mm['mchg5'].notna()]
    for c in ['ret5b', 'ret20b', 'mom_60d', 'turnover', 'volume_ratio']:
        s = sub[[c, 'mchg5']].dropna()
        if len(s) > 20:
            r = s[c].corr(s['mchg5'])
            print(f'  corr(mchg5, {c:12s}) = {r:+.3f}  (n={len(s)})')
    # 动量分层内边际
    for c in ['ret5b', 'mom_60d', 'turnover']:
        s = sub[sub[c].notna()].copy()
        med = s[c].median()
        s['hi_mom'] = s[c] > med
        print(f'  ── {c} 中位分层内 mchg5>=3% vs 其余 ──')
        for lab, g in [('高', s[s.hi_mom]), ('低', s[~s.hi_mom])]:
            a, b = g[g['mchg5'] >= 3], g[g['mchg5'] < 3]
            if len(a) >= 5 and len(b) >= 5:
                print(f'    {lab}{c}: 拥挤 n={len(a):3d} tW={(a.ret>0).mean()*100:5.1f}% '
                      f'vs 其余 n={len(b):3d} tW={(b.ret>0).mean()*100:5.1f}% '
                      f'Δ={((a.ret>0).mean()-(b.ret>0).mean())*100:+6.1f}pp '
                      f'| 均ret {a.ret.mean()*100:+.2f} vs {b.ret.mean()*100:+.2f}')
    # 池级: mchg5 与 ret5b 的 w20 增量 (双特征交叉)
    print('  ── 池级 w20: mchg5 × ret20b 双分割 (n=210 覆盖) ──')
    ps = qq[qq['mchg5'].notna() & qq['ret20b'].notna()].copy()
    mm_med = ps['mchg5'].median()
    pm_med = ps['ret20b'].median()
    for lab, m5, m20 in [('margin高×mom高', ps.mchg5 > mm_med, ps.ret20b > pm_med),
                         ('margin高×mom低', ps.mchg5 > mm_med, ps.ret20b <= pm_med),
                         ('margin低×mom高', ps.mchg5 <= mm_med, ps.ret20b > pm_med),
                         ('margin低×mom低', ps.mchg5 <= mm_med, ps.ret20b <= pm_med)]:
        g = ps[m5 & m20] if 'margin高' in lab and 'mom高' in lab else ps[m5 & m20]
        print(f'    {lab:12s} n={len(g):3d} w20={(g.w20>0).mean()*100:5.1f}% '
              f'均w20={g.w20.mean()*100:+6.2f}%')

    # ══ H2: 入场日簇 bootstrap ══
    print('\n══ H2 入场日簇 bootstrap (ΔtW 与 Δ均ret) ══')
    mask = (mm['mchg5'] >= 3).reindex(mm.index).fillna(False)
    for col, lab in [('win', 'ΔtW'), ('ret', 'Δ均ret')]:
        obs, lo, hi, p = clus_boot_delta(mm, mask, col)
        unit = 'pp' if col == 'win' else 'pp'
        print(f'  {lab:8s} obs={obs*100:+6.1f}{unit}  95%CI [{lo*100:+.1f}, {hi*100:+.1f}]  p={p:.4f}')
    ndays = mm[mask].entry_date.nunique()
    print(f'  拥挤桶 {int(mask.sum())} 笔分布在 {ndays} 个入场日 (最大单日 '
          f'{mm[mask].groupby("entry_date").size().max()} 笔)')

    # ══ H3: 年份/regime 集中度 ══
    print('\n══ H3 年份×拥挤桶 ══')
    mm['year'] = mm['entry_date'].dt.year
    for y, g in mm.groupby('year'):
        a, b = g[g['mchg5'] >= 3], g[(g['mchg5'] < 3) | g['mchg5'].isna()]
        if not len(g):
            continue
        print(f'  {y}: 拥挤 {len(a):2d}/{len(g):3d} '
              f'tW {(a.ret>0).mean()*100 if len(a) else float("nan"):5.1f}% | '
              f'其余 n={len(b):3d} tW={(b.ret>0).mean()*100 if len(b) else float("nan"):5.1f}%')

    # 融券镜像 (n=19 小样本, 仅登记)
    print('\n── mrq>=0.5% 融券压力桶年份分布 (n=19, 登记) ──')
    sm = mm[mm['mrq'] >= 0.005]
    print(' ', dict(sm.groupby('year').size()))


if __name__ == '__main__':
    main()
