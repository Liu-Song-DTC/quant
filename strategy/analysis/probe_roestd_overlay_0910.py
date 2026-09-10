#!/usr/bin/env python3
"""2026-09-10 探针B决定性一步: roe_std 是否组合层可兑现
Part0: ROE季节性诊断 (若YTD口径, 季度std混季节 → 需改年度口径)
Part1: 信号日叠加 — buy信号当日roe_std四分桶 × fwd5/10/20(市场调整) + 胜率
Part2: 行业中性IC — 因子减行业中位数后 ic60 (排除行业代理嫌疑)
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


def build_roe_std(fund, annual_only=False):
    q = fund.dropna(subset=['roe']).copy()
    if annual_only:
        q = q[q['报告期'].dt.month == 12]
    q = q.sort_values(['code', '报告期'])
    carry, ind_map = {}, {}
    for code, g in q.groupby('code'):
        g = g.drop_duplicates('报告期')
        ind_map[code] = g['所处行业'].iloc[-1] if '所处行业' in g else 'NA'
        roe = g.set_index('报告期')['roe']
        std = roe.rolling(8, min_periods=4).std()
        known = g.set_index('报告期')['公告日']
        s = pd.Series(std.values, index=known.values)
        carry[code] = s.dropna()
    return carry, ind_map


def load_prices(codes):
    prices = {}
    for c in codes:
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        prices[c] = pd.Series(df['close'].values, index=pd.to_datetime(df['datetime']))
    return prices


def signal_overlay(carry, prices, label):
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
        for d in grp[code]:
            past = t[t.index <= pd.Timestamp(d)]
            if past.empty:
                continue
            idx = s.index.searchsorted(pd.Timestamp(d))
            if idx + 21 > len(s.index):
                continue
            c0 = s.iloc[idx]
            rows.append((d, past.iloc[-1],
                         s.iloc[idx + 5] / c0 - 1, s.iloc[idx + 10] / c0 - 1,
                         s.iloc[idx + 20] / c0 - 1))
    r = pd.DataFrame(rows, columns=['date', 'x', 'f5', 'f10', 'f20'])
    print(f'  配对信号: {len(r)} 条')
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
    print(f'\n[Part2] {label} 行业中性IC (因子减当日行业中位数):')
    # 每月: 行业内去中位数 -> 全截面spearman
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
        med = d.groupby('ind')['y'].transform('median')
        y_adj = d['y'] - med
        rows.append((month, spearmanr(d['x'], y_adj)[0], len(xs)))
    df = pd.DataFrame(rows, columns=['month', 'ic60', 'n'])
    ir = df.ic60.mean() / df.ic60.std() if df.ic60.std() > 0 else 0
    print(f'  ic60(行业中性): mean={df.ic60.mean():+.4f} std={df.ic60.std():.4f} IR={ir:+.2f} '
          f'|IC|>0.05占比={100*(df.ic60.abs()>0.05).mean():.0f}% 截面n={df.n.mean():.0f}')


def main():
    fund = load_fund()
    print(f'基本面行: {len(fund)}, {fund.code.nunique()} 只')
    # Part0 季节性诊断
    med = fund.groupby(fund['报告期'].dt.quarter)['roe'].median()
    print('[Part0] ROE中位数按报告期季度: ' + '  '.join(f'Q{int(q)}:{v:.2f}' for q, v in med.items()))
    q4 = fund[fund['报告期'].dt.month == 12]
    print(f'  Q4行ROE中位数={q4.roe.median():.2f} (若Q4远高于Q1-Q3 → YTD口径, 有季节伪影)')

    carry_q, ind_map = build_roe_std(fund, annual_only=False)
    carry_a, ind_map_a = build_roe_std(fund, annual_only=True)
    print(f'roe_std季度口径携带: {len(carry_q)} 只; 年度口径携带: {len(carry_a)} 只')
    prices = load_prices(set(carry_q) | set(carry_a))
    print(f'价格覆盖: {len(prices)} 只')
    signal_overlay(carry_q, prices, 'roe_std季度口径')
    industry_neutral_ic(carry_q, ind_map, prices, 'roe_std季度口径')
    signal_overlay(carry_a, prices, 'roe_std年度口径')
    industry_neutral_ic(carry_a, ind_map_a, prices, 'roe_std年度口径')


if __name__ == '__main__':
    main()
