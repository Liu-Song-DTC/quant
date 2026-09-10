#!/usr/bin/env python3
"""2026-09-10 探针B终验: 单季ROE推导(去YTD季节) → 干净roe_std
推导: sq_roe(Q1)=YTD1; sq_roe(Qx)=YTDx-YTD(x-1)  (x=2,3,4)
     不完整年度(缺Q1起点)按季度内可用差分尽力推导
干净因子: sq_roe 8季滚动std (min 4), 公告后携带
测度: 信号日叠加(4桶×fwd5/10/20) + 行业中性ic60 + 与fund_roe水平的相关性
结论口径: 若干净版单调性仍在 → 机制=季度平滑度真实; 若死 → 季节/水平代理
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FUND = '/mnt/d/quant/data/stock_data/fundamental_data'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'


def load_fund():
    rows = []
    for fn in sorted(os.listdir(FUND)):
        if not fn.endswith('.csv'):
            continue
        try:
            f = pd.read_csv(os.path.join(FUND, fn),
                            usecols=['股票代码', '报告期', '最新公告日期', '净资产收益率', '所处行业'])
        except Exception:
            continue
        f['code'] = f['股票代码'].astype(str).str.zfill(6)
        f['报告期'] = pd.to_datetime(f['报告期'].astype(str), format='%Y%m%d', errors='coerce')
        f['公告日'] = pd.to_datetime(f['最新公告日期'], errors='coerce')
        f['roe'] = pd.to_numeric(f['净资产收益率'], errors='coerce')
        rows.append(f[['code', '报告期', '公告日', 'roe', '所处行业']])
    return pd.concat(rows, ignore_index=True).dropna(subset=['code', '公告日'])


def build_sq_std(fund):
    """单季ROE推导 + 8季std携带 + 同期ROE水平(相关性用)"""
    carry, carry_lvl, ind_map = {}, {}, {}
    q = fund.dropna(subset=['roe']).copy()
    for code, g in q.groupby('code'):
        g = g.drop_duplicates('报告期').sort_values('报告期')
        ind_map[code] = g['所处行业'].iloc[-1] if '所处行业' in g else 'NA'
        roe = g['roe']
        sq = np.empty(len(g))
        for i in range(len(g)):
            qtr = g['报告期'].iloc[i].quarter
            if qtr == 1:
                sq[i] = roe.iloc[i]
            else:
                prev = g[g['报告期'].dt.quarter == qtr - 1]
                # 上一季度须是紧邻的上一期
                prev = prev[prev['报告期'] < g['报告期'].iloc[i]]
                if not prev.empty and (g['报告期'].iloc[i] - prev['报告期'].iloc[-1]).days < 150:
                    sq[i] = roe.iloc[i] - prev['roe'].iloc[-1]
                else:
                    sq[i] = np.nan
        s = pd.Series(sq, index=g['报告期'])
        std = s.rolling(8, min_periods=4).std()
        known = g.set_index('报告期')['公告日']
        t = pd.Series(std.values, index=known.values).dropna()
        carry[code] = t
        carry_lvl[code] = pd.Series(roe.values, index=known.values)
    return carry, carry_lvl, ind_map


def signal_overlay(carry, carry_lvl, prices, label):
    print(f'\n[Part1] {label} 信号日叠加:')
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    grp = {c: g['date'].values for c, g in sig.groupby('code')}
    rows = []
    for code, t in carry.items():
        if code not in prices or code not in grp:
            continue
        s = prices[code]
        lvl = carry_lvl[code]
        for d in grp[code]:
            past = t[t.index <= pd.Timestamp(d)]
            if past.empty:
                continue
            idx = s.index.searchsorted(pd.Timestamp(d))
            if idx + 21 > len(s.index):
                continue
            lpast = lvl[lvl.index <= pd.Timestamp(d)]
            c0 = s.iloc[idx]
            rows.append((d, past.iloc[-1], lpast.iloc[-1],
                         s.iloc[idx + 5] / c0 - 1, s.iloc[idx + 10] / c0 - 1,
                         s.iloc[idx + 20] / c0 - 1))
    r = pd.DataFrame(rows, columns=['date', 'x', 'roe_lvl', 'f5', 'f10', 'f20'])
    print(f'  配对信号: {len(r)} 条  x与roe水平Spearman={spearmanr(r.x, r.roe_lvl)[0]:+.3f}')
    for c in ('f5', 'f10', 'f20'):
        r[c + 'a'] = r[c] - r.groupby('date')[c].transform('median')
    r['qb'] = pd.qcut(r['x'].rank(method='first'), 4, labels=False)
    for q in range(4):
        sub = r[r['qb'] == q]
        print(f'  桶{q}(x[{sub.x.min():.1f},{sub.x.max():.1f}]): n={len(sub)} '
              f'fwd5a={sub.f5a.mean()*100:+.2f}% fwd10a={sub.f10a.mean()*100:+.2f}% '
              f'fwd20a={sub.f20a.mean()*100:+.2f}% 胜率={100*(sub.f20 > 0).mean():.0f}%')
    print(f'  桶0-桶3 fwd20a差: {(r[r.qb==0].f20a.mean()-r[r.qb==3].f20a.mean())*100:+.2f}pp')


def industry_neutral_ic(carry, ind_map, prices, label):
    print(f'\n[Part2] {label} 行业中性ic60:')
    months = pd.date_range('2021-01-01', '2026-08-01', freq='MS')
    rows = []
    for month in months:
        xs, ys, inds = [], [], []
        for code, t in carry.items():
            if code not in prices:
                continue
            past = t[t.index <= month]
            if past.empty:
                continue
            s = prices[code]
            idx = s.index.searchsorted(month)
            if idx + 62 > len(s.index):
                continue
            xs.append(past.iloc[-1]); ys.append(s.iloc[idx + 60] / s.iloc[idx] - 1)
            inds.append(ind_map.get(code, 'NA'))
        if len(xs) < 50:
            continue
        d = pd.DataFrame({'x': xs, 'y': ys, 'ind': inds})
        y_adj = d['y'] - d.groupby('ind')['y'].transform('median')
        rows.append((month, spearmanr(d['x'], y_adj)[0], len(xs)))
    df = pd.DataFrame(rows, columns=['month', 'ic60', 'n'])
    ir = df.ic60.mean() / df.ic60.std() if df.ic60.std() > 0 else 0
    yr = df.copy(); yr['year'] = yr.month.dt.year
    print(f'  ic60: mean={df.ic60.mean():+.4f} std={df.ic60.std():.4f} IR={ir:+.2f} '
          f'|IC|>0.05占比={100*(df.ic60.abs()>0.05).mean():.0f}% 截面n={df.n.mean():.0f}')
    print(f'  逐年: ' + '  '.join(f'{i}:{v:+.3f}' for i, v in yr.groupby("year").ic60.mean().items()))


def main():
    fund = load_fund()
    print(f'基本面行: {len(fund)}, {fund.code.nunique()} 只')
    carry, carry_lvl, ind_map = build_sq_std(fund)
    print(f'单季std携带: {len(carry)} 只')
    prices = {}
    for c in set(carry):
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        prices[c] = pd.Series(df['close'].values, index=pd.to_datetime(df['datetime']))
    print(f'价格覆盖: {len(prices)} 只')
    signal_overlay(carry, carry_lvl, prices, '单季ROE 8季std(干净版)')
    industry_neutral_ic(carry, ind_map, prices, '单季ROE 8季std')


if __name__ == '__main__':
    main()
