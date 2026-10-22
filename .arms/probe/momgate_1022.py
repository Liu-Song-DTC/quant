#!/usr/bin/env python3
"""60日动量门控反事实 (2026-10-22, 只读).

背景: shortfeat_1022 在 cap2 候选 (非持仓行序<=2) 上发现 mom_60d 高的候选
w5 胜率显著更差 (hi−lo −12.7pp, q=.023; 名次内标准化后 −13.7pp, p=.0008),
而组合层当前对 NORM 反而用 mom_adj = mom_60d*0.18 **奖励**动量.

本探针离线比较两种组合层规则 (都只改"选谁", 不改槽位/敞口):
  A) 软改: effective_score 里的 NORM 动量项换成 c*mom_60d, c ∈ {-.18,-.09,0,.09,.18}
  B) 硬跳过: 保持当前分数序, 跳过 mom_60d>=t 的候选, 由后面的候选补位
      t ∈ {.15,.20,.25,.30,.40}
逐日取前2个非持仓候选 (与 cap2 同构), 比较 w5/w20 与入场数/空槽.

注意: 这里忽略实际空槽数 (回测中持仓占用槽位), 用于规则筛选; 结论须由臂验证.
用法: python momgate_1022.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load

CAP = 2


def pick_order(cand, col='eff_score_alt'):
    return list(cand.sort_values(col, ascending=False, kind='mergesort')['code'].head(CAP))


def pick_skip(cand, t):
    out = []
    for _, r in cand.iterrows():  # cand 已按引擎行序
        if r['mom_60d'] >= t:
            continue
        out.append(r['code'])
        if len(out) >= CAP:
            break
    return out


def evaluate(p, picks_fn, label):
    rows = []
    for d, g in p.groupby('date', sort=True):
        cand = g[g['nr'].notna()].sort_values('nr')
        codes = picks_fn(cand)
        sub = cand[cand['code'].isin(codes)]
        rows.append(dict(date=d, n=len(sub),
                         w5=sub['w5'].mean(), w20=sub['w20'].mean(),
                         r5=sub['r5'].mean(), r20=sub['r20'].mean(),
                         codes=set(codes)))
    r = pd.DataFrame(rows)
    return dict(label=label, n=int(r['n'].sum()), days=len(r),
                days2=int((r['n'] == 2).sum()), days1=int((r['n'] == 1).sum()),
                days0=int((r['n'] == 0).sum()),
                w5=r['w5'].mean() * 100, w20=r['w20'].mean() * 100,
                r5=r['r5'].mean() * 100, r20=r['r20'].mean() * 100,
                _r=r)


def main():
    p = load()
    p = p[p['nr'].notna()].copy()
    # 非持仓行序与 eff_score 的一致性
    for _, g in p.groupby('date'):
        rho = g['nr'].corr(g['eff_score'], method='spearman')
        break
    print(f'非持仓候选 n={len(p)} 日={p["date"].nunique()}; '
          f'首日 nr vs eff_score rho={rho:.3f}')
    # 重建 alt 分数: 当前 eff_score 已含 mom_adj; NORM 的 mom_adj = mom_60d*0.18
    cur = np.where(p['regime'].astype(str) == 'FAST', -p['mom_60d'].abs() * 0.10,
                   p['mom_60d'] * 0.18)
    p['_momadj_cur'] = cur
    p['_base_no_mom'] = p['eff_score'] - p['_momadj_cur']

    res = []
    r0 = evaluate(p, lambda c: list(c.sort_values('eff_score', ascending=False,
                                                  kind='mergesort')['code'].head(CAP)),
                  'current(eff_score)')
    res.append(r0)

    print('\nA. 软改: NORM 动量系数 c')
    for c in (-0.18, 0.0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.60):
        alt = np.where(p['regime'].astype(str) == 'FAST',
                       -p['mom_60d'].abs() * 0.10, p['mom_60d'] * c)
        p['eff_score_alt'] = p['_base_no_mom'] + alt
        res.append(evaluate(p, lambda cc: pick_order(cc), f'mom_coef={c:+.2f}'))

    print('B. 硬跳过: 当前分数序 + mom_60d >= t 跳过')
    for t in (0.15, 0.20, 0.25, 0.30, 0.40):
        res.append(evaluate(p, lambda cc, t=t: pick_skip(cc, t), f'skip>={t:.2f}'))

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in res])
    base = df.iloc[0]
    df['Δw5'] = df['w5'] - base['w5']
    df['Δw20'] = df['w20'] - base['w20']
    df['Δn'] = df['n'] - base['n']
    print('\n' + '=' * 100)
    print(df[['label', 'n', 'Δn', 'days2', 'days1', 'days0', 'w5', 'Δw5',
              'w20', 'Δw20', 'r5', 'r20']].round(2).to_string(index=False))

    # 变动构成: 软改 c=0 相对当前的换入/换出
    alt0 = np.where(p['regime'].astype(str) == 'FAST', -p['mom_60d'].abs() * 0.10, 0.0)
    p['eff_score_alt'] = p['_base_no_mom'] + alt0
    ra = evaluate(p, lambda cc: pick_order(cc), 'c0')
    rr = r0['_r'].set_index('date')['codes']
    aa = ra['_r'].set_index('date')['codes']
    drop, add = [], []
    key = p.set_index(['date', 'code'])
    for d in rr.index:
        for code in rr[d] - aa[d]:
            drop.append(key.loc[(d, code)])
        for code in aa[d] - rr[d]:
            add.append(key.loc[(d, code)])
    drop = pd.DataFrame(drop); add = pd.DataFrame(add)
    print(f'\nc=0 相对当前: 换出 {len(drop)} 换入 {len(add)}')
    for nm, t in (('换出', drop), ('换入', add)):
        if len(t):
            print(f'  {nm}: w5={t.w5.mean()*100:.1f}% w20={t.w20.mean()*100:.1f}% '
                  f'r20={t.r20.mean()*100:+.2f}% mom60均={t.mom_60d.mean():+.3f} '
                  f'nr分布={t["nr"].value_counts().sort_index().to_dict()}')
    # 同 rank 内的换手 (剔除 rank 代理): 只统计 old/new 同为 nr=1 或同为 nr=2 的
    if len(drop) and len(add):
        do = drop.set_index('date'); ad = add.set_index('date')
        common_days = do.index.intersection(ad.index)
        print('  同日换出/换入 rank 对:')
        pairs = [(do.loc[d, 'nr'], ad.loc[d, 'nr']) for d in common_days] \
            if len(common_days) == len(drop) == len(add) else []
        if pairs:
            import collections
            print('   ', dict(collections.Counter(pairs)))
        # 按换出 rank 分组看 w5
        print('  换出按 nr: ' + ' | '.join(
            f'nr{int(k)} n={len(g)} w5={g.w5.mean()*100:.1f}% w20={g.w20.mean()*100:.1f}%'
            for k, g in drop.groupby('nr')))
        print('  换入按 nr: ' + ' | '.join(
            f'nr{int(k)} n={len(g)} w5={g.w5.mean()*100:.1f}% w20={g.w20.mean()*100:.1f}%'
            for k, g in add.groupby('nr')))

    df.drop(columns=[c for c in df.columns if c.startswith('_')]).to_csv(
        '/mnt/d/quant/.arms/armout/momgate_1022.csv', index=False)
    print('\n落盘: .arms/armout/momgate_1022.csv')


if __name__ == '__main__':
    main()
