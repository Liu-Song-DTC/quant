#!/usr/bin/env python3
"""2026-09-10 阶段3探针B: 质量类因子IC — 应计比率 + ROE 8季稳定性
数据全本地 fundamental_data CSV (每股/现金流/ROE/总资产/公告日)。
因子:
  acc = (净利润 - 经营性现金流净额) / 总资产   (年度行, 公告日后携带; 高=差质量)
  roe_std = 净资产收益率 8季度滚动std        (公告日后携带; 低=稳定=好质量)
测度: 月度截面Spearman IC vs 前向20/60日市场调整(中位数)收益, 2021-2026
      + 分桶 + 逐年。质量类慢变量, 60日为关键口径。
PIT: 因子在最新公告日期后已知, 前视无泄漏。
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FUND = '/mnt/d/quant/data/stock_data/fundamental_data'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'


def load_fund():
    rows = []
    for fn in sorted(os.listdir(FUND)):
        if not fn.endswith('.csv'):
            continue
        try:
            f = pd.read_csv(os.path.join(FUND, fn),
                            usecols=['股票代码', '报告期', '最新公告日期', '净利润-净利润',
                                     '净资产收益率', 'xjll_经营性现金流-现金流量净额',
                                     'zcfz_资产-总资产'])
        except Exception:
            continue
        f['code'] = f['股票代码'].astype(str).str.zfill(6)
        f['报告期'] = pd.to_datetime(f['报告期'].astype(str), format='%Y%m%d', errors='coerce')
        f['公告日'] = pd.to_datetime(f['最新公告日期'], errors='coerce')
        for c in ('净利润-净利润', '净资产收益率', 'xjll_经营性现金流-现金流量净额', 'zcfz_资产-总资产'):
            f[c] = pd.to_numeric(f[c], errors='coerce')
        rows.append(f[['code', '报告期', '公告日', '净利润-净利润', '净资产收益率',
                       'xjll_经营性现金流-现金流量净额', 'zcfz_资产-总资产']])
    return pd.concat(rows, ignore_index=True).dropna(subset=['code', '公告日'])


def build_factors(fund):
    # acc: 年度行
    ann = fund[fund['报告期'].dt.month == 12].copy()
    ann = ann.dropna(subset=['净利润-净利润', 'xjll_经营性现金流-现金流量净额', 'zcfz_资产-总资产'])
    ann = ann[ann['zcfz_资产-总资产'] > 0]
    ann['acc'] = ((ann['净利润-净利润'] - ann['xjll_经营性现金流-现金流量净额'])
                  / ann['zcfz_资产-总资产']).clip(-1, 1)
    acc_carry = {}
    for code, g in ann.groupby('code'):
        g = g.sort_values('公告日')
        acc_carry[code] = pd.Series(g['acc'].values, index=g['公告日'])

    # roe_std: 季度行滚动8期std
    q = fund.dropna(subset=['净资产收益率']).copy()
    q = q.sort_values(['code', '报告期'])
    roe_std_carry = {}
    for code, g in q.groupby('code'):
        g = g.drop_duplicates('报告期')
        roe = g.set_index('报告期')['净资产收益率']
        std = roe.rolling(8, min_periods=4).std()
        known = g.set_index('报告期')['公告日']
        s = pd.Series(std.values, index=known.values)
        roe_std_carry[code] = s.dropna()
    return acc_carry, roe_std_carry


def ic_table(carry, prices, months, horizons=(20, 60)):
    rows = []
    for month in months:
        xs = []
        ys = {h: [] for h in horizons}
        for code, t in carry.items():
            if code not in prices:
                continue
            past = t[t.index <= month]
            if past.empty:
                continue
            s = prices[code]
            idx = s.index.searchsorted(month)
            if idx + max(horizons) + 2 > len(s.index):
                continue
            xs.append(past.iloc[-1])
            for h in horizons:
                ys[h].append(s.iloc[idx + h] / s.iloc[idx] - 1)
        if len(xs) < 50:
            continue
        row = [month]
        for h in horizons:
            y = np.array(ys[h]) - np.median(ys[h])
            row.append(spearmanr(xs, y)[0])
        row.append(len(xs))
        rows.append(row)
    cols = ['month'] + [f'ic{h}' for h in horizons] + ['n']
    df = pd.DataFrame(rows, columns=cols)
    df['year'] = df['month'].dt.year
    return df


def report(df, label):
    print(f'  {label}:')
    for c in [x for x in df.columns if x.startswith('ic')]:
        ir = df[c].mean() / df[c].std() if df[c].std() > 0 else 0
        print(f'    {c}: mean={df[c].mean():+.4f} std={df[c].std():.4f} IR={ir:+.2f} '
              f'|IC|>0.05占比={100*(df[c].abs()>0.05).mean():.0f}% 截面n均值={df["n"].mean():.0f}')
    yr = df.groupby('year')['ic60' if 'ic60' in df.columns else 'ic20'].mean()
    print(f'    逐年(ic60): ' + '  '.join(f'{i}:{v:+.3f}' for i, v in yr.items()))


def buckets(carry, prices, months, label, horizon=60):
    rows = []
    for month in months:
        xs, codes = [], []
        for code, t in carry.items():
            if code not in prices:
                continue
            past = t[t.index <= month]
            if past.empty:
                continue
            s = prices[code]
            idx = s.index.searchsorted(month)
            if idx + horizon + 2 > len(s.index):
                continue
            xs.append(past.iloc[-1]); codes.append((code, idx))
        if len(xs) < 100:
            continue
        b = pd.qcut(pd.Series(xs).rank(method='first'), 5, labels=False)
        fwds = []
        for (code, idx), bb in zip(codes, b):
            s = prices[code]
            fwds.append((bb, s.iloc[idx + horizon] / s.iloc[idx] - 1))
        fd = pd.DataFrame(fwds, columns=['b', 'f'])
        fd['f'] = fd['f'] - fd['f'].median()
        rows.append(fd.groupby('b')['f'].mean())
    bd = pd.DataFrame(rows)
    print(f'  {label} 分桶 fwd{horizon}市场调整: ' +
          '  '.join(f'桶{int(i)}:{v*100:+.2f}%' for i, v in bd.mean().items()))


def main():
    fund = load_fund()
    print(f'基本面行: {len(fund)}, 覆盖 {fund.code.nunique()} 只')
    acc_carry, roe_std_carry = build_factors(fund)
    print(f'acc携带序列: {len(acc_carry)} 只; roe_std携带序列: {len(roe_std_carry)} 只')

    codes_need = set(acc_carry) | set(roe_std_carry)
    prices = {}
    for c in codes_need:
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        prices[c] = pd.Series(df['close'].values, index=pd.to_datetime(df['datetime']))
    print(f'价格覆盖: {len(prices)} 只')

    months = pd.date_range('2021-01-01', '2026-08-01', freq='MS')
    print('\n[acc] 应计比率 (净利润-经营现金流)/总资产, 年度, 公告后携带:')
    report(ic_table(acc_carry, prices, months), 'acc')
    buckets(acc_carry, prices, months, 'acc')
    print('\n[roe_std] ROE 8季滚动std, 公告后携带:')
    report(ic_table(roe_std_carry, prices, months), 'roe_std')
    buckets(roe_std_carry, prices, months, 'roe_std')


if __name__ == '__main__':
    main()
