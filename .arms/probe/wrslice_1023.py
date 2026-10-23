#!/usr/bin/env python3
"""胜率轴三切片探针 (2026-10-23晚, 只读) — #395 结构定理后的剩余空间.

定理: tW 只能靠改变"哪些交易存在"来动 (纯仓位倾斜 tW 逐位不动)。
三个从未扫过的入口维度, 全部在 anchor_1022 已实现账簿 (166笔) 上:

  A. 因子家族: 买入信号由哪类因子选出 (V41回退/DYN_T纯技术/DYN_F含基本面/IND静态)
     × realized tW/均ret — 若某家族在实现层系统性坏, 家族级闸=新入口。
     (⚠ 因子选择是走样IC动态的, 这里只做 realized 描述, 非因果反事实。)
  B. 上市账龄: 首bar(qfq_kline parquet, 2016前截断) → 账龄桶 × tW。
     次新股 (无基本面/V41回退高发/波动大) 是显式未扫人口。
  C. 日历残差: 周几/旬/季末月 × tW — ahday(已采纳)之外的日历结构。
  D. 附带: 入场日 score 十分位 × tW 单调性 (阈值是否放得对)。

用法: .venv/bin/python .arms/probe/wrslice_1023.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load as pool_load

PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
R = '/mnt/d/quant/strategy/rolling_validation_results'
AR = '/mnt/d/quant/.arms/armout'


def fam_of(fn):
    """信号因子名 → 家族标签."""
    if not isinstance(fn, str) or not fn:
        return 'NA'
    if fn.startswith('V41'):
        return 'V41回退'
    if fn.startswith('DYN'):
        return 'DYN_T纯技术' if fn.endswith('_T') else 'DYN_F含基本面'
    if fn.startswith('IND'):
        return 'IND静态'
    return '其他'


def load_sig_factor(pairs):
    """分块扫 backtest_signals.csv 取 (code,date)→factor_name/score. 3.2GB 谨慎."""
    codes = set(c for c, _ in pairs)
    rows = []
    cols = ['code', 'date', 'buy', 'factor_name', 'score']
    for ch in pd.read_csv(f'{R}/backtest_signals.csv', usecols=cols,
                          dtype={'code': str}, chunksize=2_000_000):
        ch = ch[ch['code'].str.zfill(6).isin(codes)]
        if len(ch):
            rows.append(ch[cols])
    sig = pd.concat(rows, ignore_index=True)
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    return sig.drop_duplicates(['code', 'date'], keep='last')


def main():
    tr = pd.read_csv(f'{R}/trade_realized.csv.anchor_1022', dtype={'code': str})
    tr['code'] = tr['code'].str.zfill(6)
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])

    # sig_date: merge_asof 到池日期 (与 margind 同口径); w5/w20 用 shortfeat load()
    q = pool_load()
    q['code'] = q['code'].str.zfill(6)
    qe = q[['date', 'code']].drop_duplicates().sort_values('date')
    mm = pd.merge_asof(tr.sort_values('entry_date'), qe,
                       left_on='entry_date', right_on='date', by='code',
                       direction='backward', allow_exact_matches=False,
                       tolerance=pd.Timedelta('7D')).rename(columns={'date': 'sig_date'})
    mm = mm[mm.sig_date.notna()].copy()
    print(f'账簿 166→{len(mm)} 笔匹配池日期 (同 margind 口径)')
    qk = q[['date', 'code', 'w5', 'w20']]
    mm = mm.merge(qk, left_on=['code', 'sig_date'], right_on=['code', 'date'],
                  how='left').drop(columns=['date'])

    # ── A. 因子家族 ──
    sig = load_sig_factor(set(zip(mm['code'], mm['sig_date'])))
    mm = mm.merge(sig, left_on=['code', 'sig_date'], right_on=['code', 'date'],
                  how='left').drop(columns=['date'])
    mm['fam'] = mm['factor_name'].map(fam_of)
    print(f'\n══ A. 因子家族 × realized (factor_name 匹配 {mm.factor_name.notna().sum()}/{len(mm)}) ══')
    for fam, g in mm.groupby('fam'):
        if len(g) < 5:
            print(f'  {fam:10s} n={len(g):3d} (不足<5)')
            continue
        print(f'  {fam:10s} n={len(g):3d} tW={(g.ret>0).mean()*100:5.1f}% '
              f'均ret={g.ret.mean()*100:+6.2f}% 中位hold={g.hold_days.median():.0f}d '
              f'w5={(g.w5>0).mean()*100:5.1f}% w20={(g.w20>0).mean()*100:5.1f}%')

    # ── B. 上市账龄 ──
    px = pd.read_parquet(PQ, columns=['code', 'date'])
    first = pd.to_datetime(px.groupby('code')['date'].min())
    mm['first_bar'] = mm['code'].map(first)
    mm['age_days'] = (mm['entry_date'] - mm['first_bar']).dt.days
    bins = [0, 365, 730, 1460, 3000, 99_999]
    labs = ['<1y', '1-2y', '2-4y', '4-8y', '>=8y(截断)']
    mm['age_b'] = pd.cut(mm['age_days'], bins=bins, labels=labs)
    print(f'\n══ B. 上市账龄 (首bar, 2016前截断) × realized ══')
    for lab, g in mm.groupby('age_b', observed=True):
        print(f'  {str(lab):10s} n={len(g):3d} tW={(g.ret>0).mean()*100:5.1f}% '
              f'均ret={g.ret.mean()*100:+6.2f}% 均hold={g.hold_days.mean():.0f}d')

    # ── C. 日历残差 ──
    print(f'\n══ C. 日历残差 (入场日) ══')
    mm['dow'] = mm['entry_date'].dt.dayofweek
    mm['旬'] = pd.cut(mm['entry_date'].dt.day, [0, 10, 20, 32],
                      labels=['上旬', '中旬', '下旬'])
    mm['季末月'] = mm['entry_date'].dt.month.isin([3, 6, 9, 12])
    for lab, g in mm.groupby('dow'):
        print(f'  周{"一二三四五"[lab]} n={len(g):3d} tW={(g.ret>0).mean()*100:5.1f}% '
              f'均ret={g.ret.mean()*100:+6.2f}%')
    for lab, g in mm.groupby('旬', observed=True):
        print(f'  {str(lab):4s} n={len(g):3d} tW={(g.ret>0).mean()*100:5.1f}% '
              f'均ret={g.ret.mean()*100:+6.2f}%')
    for lab, g in mm.groupby('季末月'):
        print(f'  {"季末月" if lab else "非季末月"} n={len(g):3d} '
              f'tW={(g.ret>0).mean()*100:5.1f}% 均ret={g.ret.mean()*100:+6.2f}%')

    # ── D. score 十分位 ──
    if mm['score'].notna().sum() > 50:
        mm['sq'] = pd.qcut(mm['score'].rank(method='first'), 5,
                           labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高'])
        print(f'\n══ D. 入场信号score五分位 × realized ══')
        for lab, g in mm.groupby('sq', observed=True):
            print(f'  {str(lab):5s} n={len(g):3d} score中位={g.score.median():.3f} '
                  f'tW={(g.ret>0).mean()*100:5.1f}% 均ret={g.ret.mean()*100:+6.2f}%')

    # B 家族 × w5/w20 汇总 (A 家族已在上方表格带出)
    print(f'\n══ B 账龄桶 × w5/w20 ══')
    for lab, g in mm.groupby('age_b', observed=True):
        print(f'  {str(lab):10s} n={len(g):3d} w5={(g.w5>0).mean()*100:5.1f}% '
              f'w20={(g.w20>0).mean()*100:5.1f}%')

    mm.to_csv(f'{AR}/wrslice_realized_1023.csv', index=False)
    print(f'\n落盘 {AR}/wrslice_realized_1023.csv')


if __name__ == '__main__':
    main()
