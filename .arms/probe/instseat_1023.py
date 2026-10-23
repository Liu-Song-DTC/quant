#!/usr/bin/env python3
"""龙虎榜机构席位 (dragon_tiger_inst, 逐日真实主力资金) × 胜率 — realized-first (2026-10-23, 只读).

用户 10/22 点名"主力资金流入" — mf5 价量代理已死(#390), 快照层 get_dragon_tiger_signal
只有滚动1年累计口径. 本探针用**新回补的逐日×逐股机构席位数据** (datacenter-web
RPT_ORGANIZATION_TRADE_DETAILS, 2021-01~今, 47093 股日, 不在指纹4文件内):

协议 (hnum 教训, realized-first): 先查 166 笔已实现账簿分离 (决定性),
  有分离才做池级确认; 无分离即死, 不再进 池级w5→转换 死胡同.

特征 (PIT: 上榜日<=t, 榜单~17:00发布, 信号当晚生成 — 与信号层 dt<=date 同口径;
      过闸特征再验严格 <t 稳健性):
  dt10      = (t-10,t] 有无上榜 (事件本身, 机构零参与也含)
  net10/20  = 窗口内机构净买额合计 (元)
  net_pct_t = 最近一次上榜的 机构净买额/市场总成交额 (%)
  inst_buy  = 窗口内有买方机构参与的上榜次数
  days      = 距最近上榜天数

用法: .venv/bin/python .arms/probe/instseat_1023.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load, terciles, ClusterBoot, bh

AD = '/mnt/d/quant/data/alternative_data'
R = '/mnt/d/quant/strategy/rolling_validation_results'


def build_map():
    df = pd.read_pickle(f'{AD}/dragon_tiger_inst.pkl')
    df = df.sort_values(['code', 'date'])
    # 零机构行占比 (端点是否只含机构参与的上榜)
    zero = ((df['nbuy_inst'] == 0) & (df['nsell_inst'] == 0)).mean()
    print(f'零买方&零卖方机构行: {zero*100:.1f}% '
          f'({"端点含全部上榜" if zero > 0.3 else "端点可能只含机构参与榜, 事件flag以history并集兜底"})')
    return {c: g[['date', 'net_amt', 'net_pct', 'nbuy_inst']].values
            for c, g in df.groupby('code')}, df


def feats_at(code, t, m, strict=False):
    rec = {}
    arr = m.get(code)
    if arr is None:
        return rec
    ts = pd.Timestamp(t)
    hi = np.searchsorted(arr[:, 0], np.datetime64(ts), side='right' if not strict else 'left')
    if hi == 0:
        return rec
    lo10 = np.searchsorted(arr[:, 0], np.datetime64(ts) - pd.Timedelta('10D'),
                           side='left')
    lo20 = np.searchsorted(arr[:, 0], np.datetime64(ts) - pd.Timedelta('20D'),
                           side='left')
    w10, w20 = arr[lo10:hi], arr[lo20:hi]
    rec['days'] = float((ts - pd.Timestamp(arr[hi - 1, 0])).days)
    rec['net_pct_t'] = float(arr[hi - 1, 2]) if arr[hi - 1, 2] == arr[hi - 1, 2] else np.nan
    if len(w10):
        rec['dt10'] = 1.0
        rec['net10'] = float(np.nansum(w10[:, 1]))
        rec['inst_buy'] = float((w10[:, 3] > 0).sum())
    if len(w20):
        rec['dt20'] = 1.0
        rec['net20'] = float(np.nansum(w20[:, 1]))
    return rec


def report_realized(m, strict=False):
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
    ff = pd.DataFrame([feats_at(r.code, r.sig_date, m, strict) for r in mm.itertuples()],
                      index=mm.index)
    mm = pd.concat([mm, ff], axis=1)
    print(f'\n======== 镜头1: 已实现账簿 {len(mm)} 笔 (决定性){"" if not strict else " [严格<t]"} ========')
    tests = [
        ('dt10 上榜(10d)', mm.get('dt10', pd.Series(dtype=float)) == 1,
         '10日内上过龙虎榜'),
        ('net10>0 机构净买', mm.get('net10', pd.Series(dtype=float)) > 0,
         '窗口机构净买为正'),
        ('net10<0 机构净卖', mm.get('net10', pd.Series(dtype=float)) < 0,
         '窗口机构净卖'),
        ('net_pct_t>=3%', mm.get('net_pct_t', pd.Series(dtype=float)) >= 3,
         '最近榜净买占总成交>=3%'),
        ('net_pct_t<=-3%', mm.get('net_pct_t', pd.Series(dtype=float)) <= -3,
         '最近榜净卖占总成交<=-3%'),
        ('days<=5 新鲜榜', mm.get('days', pd.Series(dtype=float)) <= 5,
         '5日内刚上榜'),
    ]
    for lab, mask, desc in tests:
        mask = mask.reindex(mm.index).fillna(False)
        a, b = mm[mask], mm[~mask]
        if len(a) < 8:
            print(f'  {lab:18s} n={len(a):3d} (样本不足<8)')
            continue
        print(f'  {lab:18s} n={len(a):3d} tW={(a.ret>0).mean()*100:5.1f}% '
              f'均ret={a.ret.mean()*100:+6.2f}% | 补集 n={len(b):3d} '
              f'tW={(b.ret>0).mean()*100:5.1f}% 均ret={b.ret.mean()*100:+6.2f}% '
              f'| ΔtW={((a.ret>0).mean()-(b.ret>0).mean())*100:+6.1f}pp')
    # 榜内三分位 (只看上过榜的)
    sub = mm[mm.get('dt10', pd.Series(dtype=float)).reindex(mm.index) == 1]
    if len(sub) >= 12 and sub['net10'].notna().sum() >= 12:
        s = sub[sub['net10'].notna()]
        q33, q66 = s['net10'].quantile([1 / 3, 2 / 3])
        lo, hi = s[s['net10'] <= q33], s[s['net10'] >= q66]
        print(f'  榜内net10三分位: 低 n={len(lo)} tW={(lo.ret>0).mean()*100:.1f}% '
              f'vs 高 n={len(hi)} tW={(hi.ret>0).mean()*100:.1f}%')
    mm.to_csv('/mnt/d/quant/.arms/armout/instseat_realized_1023.csv', index=False)
    return mm


def report_pool(m):
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
    print('覆盖: dt10 '
          f'{qq["dt10"].notna().mean()*100:.1f}% | net10 '
          f'{qq["net10"].notna().mean()*100:.1f}% | net_pct_t '
          f'{qq["net_pct_t"].notna().mean()*100:.1f}%')
    cb = ClusterBoot(sorted(qq['date'].unique()))
    # 事件桶 vs 补集
    for lab, col, cond in [('上榜10d', 'dt10', lambda s: s == 1),
                           ('机构净买(净额>0)', 'net10', lambda s: s > 0),
                           ('机构净卖(净额<0)', 'net10', lambda s: s < 0),
                           ('强净买(net_pct>=3)', 'net_pct_t', lambda s: s >= 3)]:
        if col not in qq or qq[col].notna().sum() < 15:
            print(f'  {lab:18s} n={qq[col].notna().sum() if col in qq else 0} (<15 跳过)')
            continue
        hi = (qq[col].notna() & cond(qq[col].fillna(0))).fillna(False)
        lo = (~hi) & qq[col].notna()
        if col == 'dt10':
            lo = ~hi  # 事件flag: 补集=全未上榜
        n = int(hi.sum())
        if n < 15:
            print(f'  {lab:18s} n={n} (<15 跳过)')
            continue
        d5, l5, h5, p5 = cb.diff(qq, hi, lo, 'w5')
        d20, _, _, p20 = cb.diff(qq, hi, lo, 'w20')
        nent = int(qq[hi].is_ent.sum())
        print(f'  {lab:18s} n={n:4d} w5 {qq.loc[hi,"w5"].mean()*100:5.1f}%→Δ{d5:+6.1f} '
              f'[{l5:+.1f},{h5:+.1f}] p={p5:.4f} | w20 Δ={d20:+6.1f} p={p20:.4f} '
              f'| 锚∩={nent}')
    # 连续特征: 只在上榜子集内 tercile net10
    sub = qq[qq['net10'].notna()]
    if len(sub) >= 30:
        q33, q66 = sub['net10'].quantile([1 / 3, 2 / 3])
        lo_m = (qq['net10'] <= q33).fillna(False)
        hi_m = (qq['net10'] >= q66).fillna(False)
        d5, l5, h5, p5 = cb.diff(qq, hi_m, lo_m, 'w5')
        d20, _, _, p20 = cb.diff(qq, hi_m, lo_m, 'w20')
        print(f'  榜内net10三分位    n={int((lo_m|hi_m).sum()):4d} Δw5={d5:+6.1f} '
              f'[{l5:+.1f},{h5:+.1f}] p={p5:.4f} | Δw20={d20:+6.1f} p={p20:.4f}')
    qq.to_csv('/mnt/d/quant/.arms/armout/instseat_pool_1023.csv', index=False)


def main():
    m, raw = build_map()
    print(f'dragon_tiger_inst: {len(raw)} 股日 / {len(m)} 只 / '
          f'{raw["date"].min().date()}~{raw["date"].max().date()}')
    report_realized(m)
    report_pool(m)


if __name__ == '__main__':
    main()
