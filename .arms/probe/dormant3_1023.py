#!/usr/bin/env python3
"""三个休眠事件文件 × 胜率 — realized-first 协议 (2026-10-23, 只读).

#353 审计的休眠文件 (无生产消费者): research_reports / blocktrades / increase_records。
协议升级 (hnum 教训): **先查已实现账簿分离** (AH 的存活模式: realized tW 20.8 vs 56.1),
无分离即死 (不再进池级w5→转换的死胡同); 有分离才做池级确认+日型臂。

特征 (PIT: 大宗/增持用严格 <t 盘后披露, 研报 <=t yjyg同口径):
  大宗(全史覆盖): bt_cnt10/bt_amt10 (10日内笔数/金额), bt_prem10 (金额加权溢价率),
                  bt_inst10 (买方含'机构专用'的金额占比)
  研报(仅2025-04后): rr_days(距最近)/rr_cnt20(20日覆盖数)/rr_buy(最近评级=买入)
  增持(全史): ir_days(距最近增持公告)/ir_rate(变动比例)

用法: .venv/bin/python .arms/probe/dormant3_1023.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load, terciles, zwithin, ClusterBoot

AD = '/mnt/d/quant/data/alternative_data'
R = '/mnt/d/quant/strategy/rolling_validation_results'


def build_maps():
    # 大宗: {code: (dates, amts, prems, inst)}
    bt = pd.read_pickle(f'{AD}/blocktrades.pkl')
    bt['code'] = bt['code'].astype(str).str.zfill(6)
    bt['td'] = pd.to_datetime(bt['TRADE_DATE'])
    bt['amt'] = pd.to_numeric(bt['DEAL_AMT'], errors='coerce').fillna(0.0)
    bt['prem'] = pd.to_numeric(bt['PREMIUM_RATIO'], errors='coerce')
    bt['inst'] = bt['BUYER_NAME'].astype(str).str.contains('机构专用').astype(float)
    bt = bt.sort_values('td')
    bt_map = {c: (g['td'].values, g['amt'].values, g['prem'].values, g['inst'].values)
              for c, g in bt.groupby('code')}
    # 研报: {code: (dates, rating)}
    rr = pd.read_pickle(f'{AD}/research_reports.pkl')
    rr['code'] = rr['code'].astype(str).str.zfill(6)
    rr['pd_'] = pd.to_datetime(rr['publishDate'], errors='coerce')
    rr = rr[rr['pd_'].notna()].sort_values('pd_')
    rr_map = {c: g[['pd_', 'sRatingName']].values for c, g in rr.groupby('code')}
    # 增持: {code: (dates, rates)}
    ir = pd.read_pickle(f'{AD}/increase_records.pkl')
    ir['code'] = ir['code'].astype(str).str.zfill(6)
    ir['et'] = pd.to_datetime(ir['eitime'], errors='coerce')
    ir = ir[ir['et'].notna()].sort_values('et')
    ir_map = {c: (g['et'].values, pd.to_numeric(g['change_rate'], errors='coerce').values)
              for c, g in ir.groupby('code')}
    return bt_map, rr_map, ir_map


def feats_at(code, t, bt_map, rr_map, ir_map):
    rec = {}
    ts = np.datetime64(t)
    arr = bt_map.get(code)
    if arr is not None:
        d, a, pr, ins = arr
        i0 = np.searchsorted(d, ts - np.timedelta64(10, 'D'), side='left')
        i1 = np.searchsorted(d, ts, side='left')          # 严格 <t
        w = a[i0:i1]
        if len(w):
            rec['bt_cnt10'] = float(len(w))
            rec['bt_amt10'] = float(w.sum())
            if w.sum() > 0:
                rec['bt_prem10'] = float(np.nansum(w * pr[i0:i1]) / w.sum())
                rec['bt_inst10'] = float(np.nansum(w * ins[i0:i1]) / w.sum())
        rec['bt_ever'] = 1.0
    r2 = rr_map.get(code)
    if r2 is not None:
        i = int(np.searchsorted(r2[:, 0], ts, side='right')) - 1   # <=t
        if i >= 0:
            rec['rr_days'] = float((pd.Timestamp(t) - pd.Timestamp(r2[i, 0])).days)
            rec['rr_buy'] = 1.0 if str(r2[i, 1]) in ('买入', '增持', '强推', '推荐') else 0.0
        i0 = np.searchsorted(r2[:, 0], ts - np.timedelta64(20, 'D'), side='left')
        rec['rr_cnt20'] = float(i - i0 + 1) if i >= 0 else 0.0
    a3 = ir_map.get(code)
    if a3 is not None:
        d3, r3 = a3
        j = int(np.searchsorted(d3, ts, side='left')) - 1        # 严格 <t
        if j >= 0:
            rec['ir_days'] = float((pd.Timestamp(t) - pd.Timestamp(d3[j])).days)
            rec['ir_rate'] = float(r3[j]) if r3[j] == r3[j] else np.nan
    return rec


def main():
    bt_map, rr_map, ir_map = build_maps()
    print(f'大宗 {len(bt_map)} 只 / 研报 {len(rr_map)} 只 / 增持 {len(ir_map)} 只')

    # ── 镜头1: 已实现账簿 (166笔) — 决定性过滤 ──
    tr = pd.read_csv(f'{R}/trade_realized.csv.anchor_1022', dtype={'code': str})
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    tr['code'] = tr['code'].str.zfill(6)
    q = pd.read_csv('/mnt/d/quant/.arms/pool_1022.csv', dtype={'code': str}, parse_dates=['date'])
    q['code'] = q['code'].str.zfill(6)
    qe = q[['date', 'code']].drop_duplicates().sort_values('date')
    m = pd.merge_asof(tr.sort_values('entry_date'), qe[['date', 'code']],
                      left_on='entry_date', right_on='date', by='code',
                      direction='backward', allow_exact_matches=False,
                      tolerance=pd.Timedelta('7D'))
    m = m.rename(columns={'date': 'sig_date'})
    print(f'已实现: {m.sig_date.notna().sum()}/{len(m)} 笔匹配信号日')
    m = m[m.sig_date.notna()].copy()
    m = pd.concat([m, pd.DataFrame([feats_at(r.code, r.sig_date, bt_map, rr_map, ir_map)
                                    for r in m.itertuples()], index=m.index)], axis=1)

    print('\n======== 镜头1: 已实现账簿 (决定性) ========')
    tests = [
        ('bt_cnt10 > 0', m.bt_cnt10.notna() & (m.bt_cnt10 > 0), '10日内有大宗'),
        ('bt_inst10 hi(>=50%)', m.bt_inst10.notna() & (m.bt_inst10 >= 0.5), '机构买方占大宗>=50%'),
        ('bt_prem10 hi(>=0)', m.bt_prem10.notna() & (m.bt_prem10 >= 0), '大宗溢价成交'),
        ('rr fresh(<=30d)', m.rr_days.notna() & (m.rr_days <= 30), '30日内有研报'),
        ('rr_cnt20 >=2', m.rr_cnt20.notna() & (m.rr_cnt20 >= 2), '20日内研报>=2篇'),
        ('ir fresh(<=60d)', m.ir_days.notna() & (m.ir_days <= 60), '60日内有增持公告'),
    ]
    for lab, mask, desc in tests:
        mm = mask.fillna(False)
        a, b = m[mm], m[~mm]
        if len(a) < 8:
            print(f'  {lab:22s} n={len(a):3d} (样本不足)'); continue
        print(f'  {lab:22s} n={len(a):3d} tW={(a.ret>0).mean()*100:5.1f}% 均ret={a.ret.mean()*100:+6.2f}% '
              f'| 补集 n={len(b):3d} tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}% '
              f'| ΔtW={((a.ret>0).mean()-(b.ret>0).mean())*100:+6.1f}pp')

    # ── 镜头2: 池级 (确认+量化) ──
    print('\n======== 镜头2: 池级 nr<=2 (确认) ========')
    p = load()
    qq = p[(p['nr'].notna()) & (p['nr'] <= 2)].copy()
    qq = pd.concat([qq, pd.DataFrame([feats_at(r.code, r.date, bt_map, rr_map, ir_map)
                                      for r in qq.itertuples()], index=qq.index)], axis=1)
    cb = ClusterBoot(sorted(qq['date'].unique()))
    for col, lab in [('bt_cnt10', '大宗笔数'), ('bt_prem10', '大宗溢价'), ('bt_inst10', '机构占比'),
                     ('rr_cnt20', '研报密度'), ('rr_days', '研报新鲜'), ('ir_days', '增持新鲜')]:
        sub = qq[qq[col].notna()]
        if len(sub) < 40:
            print(f'  {lab:10s} 覆盖 {len(sub)} (<40 跳过)'); continue
        med = sub[col].median()
        hi = qq[col] > med
        d5, l5, h5, p5 = cb.diff(qq, hi.fillna(False), ~hi.fillna(False) & qq[col].notna(), 'w5')
        d20, _, _, p20 = cb.diff(qq, hi.fillna(False), ~hi.fillna(False) & qq[col].notna(), 'w20')
        print(f'  {lab:10s} n={len(sub):4d} w5 Δ={d5:+6.1f} [{l5:+.1f},{h5:+.1f}] p={p5:.4f} | '
              f'w20 Δ={d20:+6.1f} p={p20:.4f}')
    m.to_csv('/mnt/d/quant/.arms/armout/dormant3_realized_1023.csv', index=False)
    print('\n落盘 .arms/armout/dormant3_realized_1023.csv')


if __name__ == '__main__':
    main()
