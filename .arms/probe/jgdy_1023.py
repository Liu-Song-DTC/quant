#!/usr/bin/env python3
"""★机构调研(jgdy)×胜率 realized-first 探针 (2026-10-23夜, §24, 只读).

数据: akshare stock_jgdy_tj_em — 东财机构调研快照, 仅最近12个月 (2025-10-24→2026-10-23),
  21,753 事件 / ~5k 股。字段: 代码/接待机构数量/接待日期/公告日期。
  PIT时点=公告日期 (接待日期是事件日, 公告滞后1-3天)。
  ⚠快照型接口: 更早历史不可回补 ⇒ 2021-2025 池级验证不可行, 只做近12月。
  ⚠幸存者 caveat: 快照可能不含期间退市股 (少, 登记)。

协议 (realized-first, 同§19/§21):
  R1 覆盖: 全宇宙5421中多少股有事件; 事件/股分布; 机构数分布。
  R2 realized: 166笔账簿中 entry≥2025-10-24 的笔 × 入场前60天(公告日)是否有调研
     / 机构数 → tW/均ret 分离度 (n 小, 只作方向)。
  R3 池级: pool_1022 近12月行 × 同特征 → w5/w20 前瞻分离 (n 大, 主判读)。
  R4 剂量: 机构数分位 × w5/w20。
判读: 池级无分离 → 家族第8例闭合; 有分离 → 谈变现通道 (选股拉新/分数加成/权重, 先验全灭)。

用法: .venv/bin/python .arms/probe/jgdy_1023.py [--dl]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')

AR = '/mnt/d/quant/.arms/armout'
PKL = '/mnt/d/quant/data/alternative_data/jgdy_events.pkl'
R = '/mnt/d/quant/strategy/rolling_validation_results'
WSTART = '2025-10-24'  # 快照起点


def download():
    import akshare as ak
    df = ak.stock_jgdy_tj_em(date='20210101')
    out = df.rename(columns={'代码': 'code', '接待机构数量': 'n_inst',
                             '接待日期': 'visit_date', '公告日期': 'ann_date'})[
        ['code', 'n_inst', 'visit_date', 'ann_date']].copy()
    out['code'] = out['code'].astype(str).str.split('.').str[0].str.zfill(6)
    out['n_inst'] = pd.to_numeric(out['n_inst'], errors='coerce').fillna(0).astype(int)
    for c in ['visit_date', 'ann_date']:
        out[c] = pd.to_datetime(out[c], errors='coerce')
    out = out.dropna(subset=['ann_date']).sort_values('ann_date')
    out.to_pickle(PKL)
    print(f'下载 {len(out)} 事件 → {PKL}')
    return out


def main():
    import os
    if '--dl' in sys.argv or not os.path.exists(PKL):
        ev = download()
    else:
        ev = pd.read_pickle(PKL)
        print(f'缓存 {len(ev)} 事件 ({ev.ann_date.min():%Y-%m-%d} → {ev.ann_date.max():%Y-%m-%d})')

    # ── R1 覆盖 ──
    print(f'\n══ R1 覆盖 ══')
    print(f'  股数 {ev.code.nunique()} | 事件/股中位 {ev.groupby("code").size().median():.0f} '
          f'| 机构数中位 {ev.n_inst.median():.0f} 分位 '
          f'{ev.n_inst.quantile([.25, .75]).round(0).tolist()}')
    print(f'  公告滞后(公告-接待)中位 {(ev.ann_date - ev.visit_date).dt.days.median():.0f}天')

    # ── R2 realized (账簿近12月) ──
    print(f'\n══ R2 realized: 账簿 entry≥{WSTART} ══')
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['code'] = mm['code'].str.zfill(6)
    rec = mm[mm.entry_date >= WSTART].copy()
    print(f'  近12月笔数 {len(rec)}/{len(mm)}')

    def feat(code, date, win=60):
        e = ev[(ev.code == code) & (ev.ann_date <= date) &
               (ev.ann_date > date - pd.Timedelta(days=win))]
        return len(e), (e.n_inst.max() if len(e) else 0)

    f = [feat(c, d) for c, d in zip(rec.code, rec.entry_date)]
    rec['n_ev60'] = [a for a, _ in f]
    rec['inst60'] = [b for _, b in f]
    for lab, m in [('有调研(60d)', rec.n_ev60 > 0), ('机构数≥10', rec.inst60 >= 10),
                   ('机构数≥50', rec.inst60 >= 50)]:
        a, b = rec[m], rec[~m]
        if len(a) < 3 or len(b) < 3:
            print(f'  {lab}: n={len(a)} 不足')
            continue
        print(f'  {lab:12s} n={len(a):2d} tW={(a.ret > 0).mean() * 100:5.1f}% '
              f'均ret={a.ret.mean() * 100:+6.2f}% | 无 n={len(b):2d} '
              f'tW={(b.ret > 0).mean() * 100:5.1f}% 均ret={b.ret.mean() * 100:+6.2f}%')

    # ── R3 池级前瞻 (主判读) ──
    print(f'\n══ R3 池级: pool_1022 近12月 × w5/w20 ══')
    sys.argv = [sys.argv[0]]  # shortfeat_1022 解析 argv[1], 清掉 --dl
    from shortfeat_1022 import load as pool_load
    q = pool_load()
    q['code'] = q['code'].str.zfill(6)
    q['date'] = pd.to_datetime(q['date'])
    qq = q[q.date >= WSTART].copy()
    print(f'  池行 {len(qq)} ({qq.date.nunique()} 个调仓日)')

    evs = ev.sort_values('ann_date')
    f2 = [feat(c, d) for c, d in zip(qq.code, qq.date)]
    qq['n_ev60'] = [a for a, _ in f2]
    qq['inst60'] = [b for _, b in f2]
    base5 = (qq.w5 > 0).mean()
    base20 = (qq.w20 > 0).mean()
    print(f'  基线 w5 {base5 * 100:.1f}% w20 {base20 * 100:.1f}% '
          f'均w5 {qq.w5.mean() * 100:+.2f}% 均w20 {qq.w20.mean() * 100:+.2f}%')
    for lab, m in [('有调研(60d)', qq.n_ev60 > 0), ('机构数≥10', qq.inst60 >= 10),
                   ('机构数≥50', qq.inst60 >= 50), ('机构数≥100', qq.inst60 >= 100)]:
        a, b = qq[m], qq[~m]
        if len(a) < 30:
            print(f'  {lab}: n={len(a)} 不足')
            continue
        print(f'  {lab:12s} n={len(a):5d} ({len(a) / len(qq) * 100:4.1f}%) '
              f'w5={(a.w5 > 0).mean() * 100:5.1f}% 均w5={a.w5.mean() * 100:+6.2f}% '
              f'w20={(a.w20 > 0).mean() * 100:5.1f}% 均w20={a.w20.mean() * 100:+6.2f}%')

    # ── R4 剂量曲线 ──
    print(f'\n══ R4 机构数剂量 × w5/w20 ══')
    qq['dose'] = pd.cut(qq.inst60, [-1, 0, 5, 20, 50, 10 ** 6],
                        labels=['0', '1-5', '6-20', '21-50', '50+'])
    for lab, g in qq.groupby('dose', observed=True):
        if len(g) < 30:
            print(f'  {lab:5s} n={len(g):5d} 不足')
            continue
        print(f'  {lab:5s} n={len(g):5d} w5={(g.w5 > 0).mean() * 100:5.1f}% '
              f'w20={(g.w20 > 0).mean() * 100:5.1f}% 均w20={g.w20.mean() * 100:+6.2f}%')

    # R5: 分时间稳健 (前半/后半)
    mid = qq.date.quantile(0.5)
    for lab, g in [('前半', qq[qq.date <= mid]), ('后半', qq[qq.date > mid])]:
        a, b = g[g.n_ev60 > 0], g[g.n_ev60 == 0]
        if len(a) < 20:
            continue
        print(f'  {lab}: 有调研 w5 {(a.w5 > 0).mean() * 100:.1f}% vs 无 {(b.w5 > 0).mean() * 100:.1f}% '
              f'w20 {(a.w20 > 0).mean() * 100:.1f}% vs {(b.w20 > 0).mean() * 100:.1f}%')

    qq.to_csv(f'{AR}/jgdy_pool_1023.csv', index=False)
    print(f'\n落盘 {AR}/jgdy_pool_1023.csv')


if __name__ == '__main__':
    main()
