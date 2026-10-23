#!/usr/bin/env python3
"""个股两融明细 (margin_detail) — 最后一个未扫真实数据源 × 胜率 realized-first (2026-10-23).

margin_detail.pkl: 4.48M 行/4528 只/2021-01→2026-08-26 (冻结, 回补中), 交易所披露
逐日每股融资余额 rzye/融资买入额 rzmre/融券余额 rqye。margin_daily (市场级) 生产在用,
**个股级从未进胜率镜头** — 民俗: 融资余额持续上升=杠杆资金拥挤(多头动量确认/拥挤风险)。

特征 (PIT: 盘后披露, 严格 <t 即最新=T-1; 末次披露>10天则 NaN — 冻结段不外推):
  mchg5/mchg20 = 融资余额 5/20 交易日变化% (按股内序列位置, 非日历日)
  mlevel       = log(rzye) (规模, 池内 z-within 控制)
  mflow1       = rzmre/rzye (当日买入/余额 = 融资活跃度)
  mrq          = rqye/rzye (融券/融资 = 空头压力, rqye 缺失则 NaN)

协议: 先查 166 笔已实现账簿分离 (决定性), 池级 (nr≤2) 确认次之。
用法: .venv/bin/python .arms/probe/margind_1023.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load, ClusterBoot

AD = '/mnt/d/quant/data/alternative_data'
R = '/mnt/d/quant/strategy/rolling_validation_results'


def build_map():
    df = pd.read_pickle(f'{AD}/margin_detail.pkl')
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])
    for c in ('rzye', 'rzmre', 'rqye'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df['rzye'] > 0].sort_values(['code', 'date'])
    df = df.drop_duplicates(['code', 'date'], keep='last')
    m = {c: (g['date'].values, g['rzye'].values, g['rzmre'].values, g['rqye'].values)
         for c, g in df.groupby('code')}
    print(f'margin_detail: {len(df)} 行 / {len(m)} 只 / '
          f'{df["date"].min().date()}~{df["date"].max().date()}')
    return m


def feats_at(code, t, m):
    rec = {}
    arr = m.get(code)
    if arr is None:
        return rec
    d, rz, rm, rq = arr
    i = int(np.searchsorted(d, np.datetime64(pd.Timestamp(t)), side='left')) - 1  # 严格 <t
    if i < 1:
        return rec
    if (pd.Timestamp(t) - pd.Timestamp(d[i])).days > 10:   # 冻结段不外推
        return rec
    rz_now, rz_prev = rz[i], max(rz[i - 1], 1)
    rec['mlevel'] = float(np.log(rz_now))
    if i >= 5 and rz[i - 5] > 0:
        rec['mchg5'] = float((rz_now / rz[i - 5] - 1) * 100)
    if i >= 20 and rz[i - 20] > 0:
        rec['mchg20'] = float((rz_now / rz[i - 20] - 1) * 100)
    rec['mflow1'] = float(rm[i] / rz_now) if rm[i] == rm[i] and rm[i] > 0 else np.nan
    if rq[i] == rq[i] and rq[i] > 0:
        rec['mrq'] = float(rq[i] / rz_now)
    return rec


def main():
    m = build_map()

    # ── 镜头1: 已实现账簿 (决定性) ──
    tr = pd.read_csv(f'{R}/trade_realized.csv.anchor_1022', dtype={'code': str})
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    tr['code'] = tr['code'].str.zfill(6)
    q = pd.read_csv('/mnt/d/quant/.arms/pool_1022.csv', dtype={'code': str},
                    parse_dates=['date'])
    q['code'] = q['code'].str.zfill(6)
    qe = q[['date', 'code']].drop_duplicates().sort_values('date')
    mm = pd.merge_asof(tr.sort_values('entry_date'), qe,
                       left_on='entry_date', right_on='date', by='code',
                       direction='backward', allow_exact_matches=False,
                       tolerance=pd.Timedelta('7D')).rename(columns={'date': 'sig_date'})
    mm = mm[mm.sig_date.notna()].copy()
    ff = pd.DataFrame([feats_at(r.code, r.sig_date, m) for r in mm.itertuples()],
                      index=mm.index)
    mm = pd.concat([mm, ff], axis=1)
    print(f'\n======== 镜头1: 已实现账簿 {len(mm)} 笔 (决定性) ========')
    print(f'覆盖: mchg5 {mm["mchg5"].notna().mean()*100:.0f}% | '
          f'mchg20 {mm["mchg20"].notna().mean()*100:.0f}% | mrq {mm["mrq"].notna().mean()*100:.0f}%')
    tests = [
        ('mchg5>=+3% 拥挤', mm['mchg5'] >= 3),
        ('mchg5<=-3% 去杠杆', mm['mchg5'] <= -3),
        ('mchg20>=+8% 强拥挤', mm['mchg20'] >= 8),
        ('mchg20<=-8% 强去杠杆', mm['mchg20'] <= -8),
        ('mflow1 hi(>=3%)', mm['mflow1'] >= 0.03),
        ('mrq hi(>=0.5%)', mm['mrq'] >= 0.005),
    ]
    for lab, mask in tests:
        mask = mask.reindex(mm.index).fillna(False)
        a, b = mm[mask], mm[~mask]
        if len(a) < 8:
            print(f'  {lab:20s} n={len(a):3d} (不足<8)')
            continue
        print(f'  {lab:20s} n={len(a):3d} tW={(a.ret>0).mean()*100:5.1f}% '
              f'均ret={a.ret.mean()*100:+6.2f}% | 补集 n={len(b):3d} '
              f'tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}% '
              f'| ΔtW={((a.ret>0).mean()-(b.ret>0).mean())*100:+6.1f}pp')

    # ── 镜头2: 池级 (确认) ──
    p = load()
    qq = p[(p['nr'].notna()) & (p['nr'] <= 2)].copy()
    ff = pd.DataFrame([feats_at(r.code, r.date, m) for r in qq.itertuples()],
                      index=qq.index)
    qq = pd.concat([qq, ff], axis=1)
    sel = pd.read_csv(f'{R}/portfolio_selections.csv.anchor_1022', dtype={'code': str},
                      parse_dates=['date'])
    sel['code'] = sel['code'].str.zfill(6)
    akey = set(zip(sel.date, sel.code))
    qq['is_ent'] = [(d, c) in akey for d, c in zip(qq.date, qq.code)]
    print(f'\n======== 镜头2: 池级 nr<=2 n={len(qq)} ========')
    cb = ClusterBoot(sorted(qq['date'].unique()))
    for col, lab in [('mchg5', '融资5日变'), ('mchg20', '融资20日变'),
                     ('mflow1', '融资活跃度'), ('mrq', '融券压力')]:
        sub = qq[qq[col].notna()]
        if len(sub) < 40:
            print(f'  {lab:10s} 覆盖 {len(sub)} (<40 跳过)')
            continue
        med = sub[col].median()
        hi = (qq[col] > med).fillna(False)
        lo = (~hi) & qq[col].notna()
        d5, l5, h5, p5 = cb.diff(qq, hi, lo, 'w5')
        d20, _, _, p20 = cb.diff(qq, hi, lo, 'w20')
        nent = int(qq[hi].is_ent.sum() + qq[lo].is_ent.sum())
        print(f'  {lab:10s} n={len(sub):4d} 中位分割 w5 Δ={d5:+6.1f} '
              f'[{l5:+.1f},{h5:+.1f}] p={p5:.4f} | w20 Δ={d20:+6.1f} p={p20:.4f} '
              f'| 锚∩={nent}')
    qq.to_csv('/mnt/d/quant/.arms/armout/margind_pool_1023.csv', index=False)
    mm.to_csv('/mnt/d/quant/.arms/armout/margind_realized_1023.csv', index=False)
    print('\n落盘 armout/margind_{pool,realized}_1023.csv')


if __name__ == '__main__':
    main()
