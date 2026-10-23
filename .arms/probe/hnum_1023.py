#!/usr/bin/env python3
"""股东户数 (holdernum_det) — 真实筹码集中度数据 w5/w20 判别力 (2026-10-23, 只读).

用户 10/22 问询"筹码结构(筹码峰)" — altfeat_1022 的真筹码模型是价量重建(零信号),
本探针扫交易所直接披露的股东户数 (439k 披露, 含 PIT 公告日 hold_notice_date):
  hnum_ratio   = 最近一次披露的户数变化% (负=筹码集中, 经典多头信号)
  hnum_ratio2  = 前一次披露的变化% (持续性)
  hnum_chg3    = 最近3次累计 log 变化 (持续集中 vs 持续分散)
  hnum_age     = 距最近披露天数 (陈旧度)
  hnum_level   = log(股东户数) (规模代理, z-within 控制)
PIT: searchsorted(hold_notice_date, t, right)-1 ⇒ 公告日<=t (yjyg 同口径, 盘后生成信号)。
数据卫生: pre_holder_num<1000 (IPO 首披, ratio 400000% 类) 视为无效; ratio 截断±60%。
样本/判据: 同 altfeat (nr<=2, 日簇 bootstrap 5000 + BH, |Δw5|>=8pp & q<=.10 & w20同向;
臂化门槛 锚∩>=30)。

用法: .venv/bin/python .arms/probe/hnum_1023.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load, terciles, zwithin, ClusterBoot, bh

AD = '/mnt/d/quant/data/alternative_data'
R = '/mnt/d/quant/strategy/rolling_validation_results'


def main():
    p = load()
    q = p[(p['nr'].notna()) & (p['nr'] <= 2)].copy()
    print(f'样本: nr<=2 n={len(q)} 日={q["date"].nunique()}')

    hn = pd.read_pickle(f'{AD}/holdernum_det.pkl')
    hn['code'] = hn['code'].astype(str).str.zfill(6)
    hn['nd'] = pd.to_datetime(hn['hold_notice_date'], errors='coerce')
    hn['ed'] = pd.to_datetime(hn['end_date'], errors='coerce')
    # 数据卫生: 无效公告日/无效户数剔除; IPO 首披(pre<1000)剔除
    hn = hn[hn['nd'].notna() & hn['holder_num'].notna()]
    hn = hn[~((hn['pre_holder_num'].notna()) & (hn['pre_holder_num'] < 1000))]
    hn = hn.sort_values(['code', 'nd']).drop_duplicates(['code', 'ed'], keep='last')
    hn_map = {c: g[['nd', 'holder_num']].values for c, g in hn.groupby('code')}
    print(f'holdernum: {len(hn)} 有效披露 / {len(hn_map)} 只')

    rows = []
    for r in q.itertuples():
        rec = {}
        arr = hn_map.get(r.code)
        if arr is not None:
            idx = int(np.searchsorted(arr[:, 0], np.datetime64(r.date), side='right')) - 1
            if idx >= 0:
                nums = arr[:idx + 1, 1].astype(float)
                nds = arr[:idx + 1, 0]
                rec['hnum_age'] = float((pd.Timestamp(r.date) - pd.Timestamp(nds[idx])).days)
                if len(nums) >= 2 and nums[-2] > 0:
                    rec['hnum_ratio'] = float((nums[-1] / nums[-2] - 1) * 100)
                if len(nums) >= 3 and nums[-3] > 0:
                    rec['hnum_ratio2'] = float((nums[-2] / nums[-3] - 1) * 100)
                    rec['hnum_chg3'] = float(np.log(nums[-1] / nums[-3]) * 100)
                rec['hnum_level'] = float(np.log(max(nums[-1], 1)))
        rows.append(rec)
    ff = pd.DataFrame(rows, index=q.index)
    # 截断异常 (±60% 外的 ratio 多为股本变动/数据噪声)
    for c in ('hnum_ratio', 'hnum_ratio2'):
        if c in ff:
            ff[c] = ff[c].clip(-60, 60)
    q = pd.concat([q, ff], axis=1)

    sel = pd.read_csv(f'{R}/portfolio_selections.csv.anchor_1022', dtype={'code': str},
                      parse_dates=['date'])
    sel['code'] = sel['code'].str.zfill(6)
    akey = set(zip(sel.date, sel.code))
    q['is_ent'] = [(d, c) in akey for d, c in zip(q.date, q.code)]

    feats = ['hnum_ratio', 'hnum_ratio2', 'hnum_chg3', 'hnum_age', 'hnum_level']
    print('\n覆盖度:')
    for k in feats:
        print(f'  {k:14s} {q[k].notna().mean()*100:5.1f}%')

    cb = ClusterBoot(sorted(q['date'].unique()))
    rows = []
    print('\n== 连续特征 tercile (hi vs lo) ==')
    print(f'{"feature":14s} {"n":>4s} {"lo%":>6s} {"hi%":>6s} {"Δw5":>7s} '
          f'{"CI":>16s} {"p":>7s} {"q":>7s} {"Δw20":>7s} {"p20":>7s} {"zΔ":>6s} {"zp":>6s} {"锚∩":>4s}')
    for col in feats:
        lo_m, hi_m = terciles(q, col)
        if lo_m is None:
            continue
        n = int((lo_m.fillna(False) | hi_m.fillna(False)).sum())
        if n < 40:
            continue
        d5, l5, h5, p5 = cb.diff(q, hi_m, lo_m, 'w5')
        d20, _, _, p20 = cb.diff(q, hi_m, lo_m, 'w20')
        w5lo = q.loc[lo_m.fillna(False), 'w5'].mean() * 100
        w5hi = q.loc[hi_m.fillna(False), 'w5'].mean() * 100
        z = zwithin(q, col, 'nr')
        qz = q.assign(_z=z)
        lo_z, hi_z = terciles(qz, '_z')
        zd, zl, zh, zp = cb.diff(qz, hi_z, lo_z, 'w5') if lo_z is not None else (np.nan,) * 4
        nent = int(q[hi_m.fillna(False)].is_ent.sum() + q[lo_m.fillna(False)].is_ent.sum())
        rows.append(dict(feature=col, n=n, lo=w5lo, hi=w5hi, d5=d5, lo5=l5, hi5=h5,
                         p5=p5, d20=d20, p20=p20, zd=zd, zp=zp, nent=nent))
        print(f'{col:14s} {n:4d} {w5lo:6.1f} {w5hi:6.1f} {d5:+7.1f} '
              f'[{l5:+.1f},{h5:+.1f}] {p5:7.4f} {"":>7s} {d20:+7.1f} {p20:7.4f} '
              f'{zd:+6.1f} {zp:6.4f} {nent:4d}')
    rt = pd.DataFrame(rows)
    if len(rt):
        rt['q'] = bh(rt['p5'].values)
        print('\nBH q 值:')
        for _, x in rt.iterrows():
            print(f'  {x.feature:14s} q={x.q:.4f}')

    # 分桶细看: ratio 符号 × 新鲜度
    print('\n== hnum_ratio 分桶 (负=集中) ==')
    for lab, m in [('集中(<=-2%)', q['hnum_ratio'] <= -2),
                   ('平稳(-2~2%)', q['hnum_ratio'].abs() < 2),
                   ('分散(>=2%)', q['hnum_ratio'] >= 2)]:
        sub = q[m.fillna(False)]
        if len(sub) >= 20:
            print(f'  {lab:12s} n={len(sub):4d} w5={sub["w5"].mean()*100:5.1f} '
                  f'w20={sub["w20"].mean()*100:5.1f} r20={sub.get("r20", pd.Series(dtype=float)).mean()*100 if "r20" in sub else float("nan"):+5.1f} '
                  f'锚∩={int(sub.is_ent.sum())}')
    print('\n== hnum_ratio × 新鲜度 (age<=45d) ==')
    fresh = (q['hnum_age'] <= 45).fillna(False)
    for lab, m in [('新鲜&集中', fresh & (q['hnum_ratio'] <= -2)),
                   ('新鲜&分散', fresh & (q['hnum_ratio'] >= 2))]:
        sub = q[m.fillna(False)]
        if len(sub) >= 10:
            print(f'  {lab:12s} n={len(sub):4d} w5={sub["w5"].mean()*100:5.1f} '
                  f'w20={sub["w20"].mean()*100:5.1f} 锚∩={int(sub.is_ent.sum())}')
    q.to_csv('/mnt/d/quant/.arms/armout/hnum_1023.csv', index=False)
    print('\n落盘 .arms/armout/hnum_1023.csv')


if __name__ == '__main__':
    main()
