#!/usr/bin/env python3
"""2026-09-10 阶段3探针A: 涨停基因族因子 IC — 模型从未见过的信息?
三问:
Q0 factor_df缓存里 limit_up_freq/consec_limit_up 是否真实非零 (申报未计算坐实?)
Q1 现役口径IC: qfq收盘收益率>=9.5%盲板 (factor_calculator.py:243-261同款)
Q2 板块正确口径IC: change_percent(原始涨跌幅) + 主板9.8/创业板科创板19.8
    (创业板2020-08-24前10%板)
因子: lu20=过去20日涨停次数; consec=当前连板数
测度: 月度截面Spearman IC vs 前向10/20日市场调整(中位数)收益, 2021-2026
      + 分桶(5桶×fwd20) + 逐年ic20
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
CACHE = '/mnt/d/quant/strategy/cache'
CHG_DATE = pd.Timestamp('2020-08-24')  # 创业板涨跌幅改革日


def q0_parquet_check():
    """factor_df parquet: 两列是否存在且非零"""
    files = sorted(glob.glob(os.path.join(CACHE, 'factor_df_*.parquet')))
    print(f'[Q0] factor_df缓存: {files}')
    if not files:
        print('  无缓存文件 -> 无法检查 (因子层可能尚未落盘或已被清理)')
        return
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('  无pyarrow')
        return
    p = files[-1]
    schema = pq.ParquetFile(p).schema_arrow
    names = set(schema.names)
    for col in ('limit_up_freq', 'consec_limit_up', 'smart_money_flow'):
        if col not in names:
            print(f'  {col}: 不在factor_df列中')
            continue
        df = pd.read_parquet(p, columns=[col])
        nz = (df[col].fillna(0) != 0).mean() * 100
        print(f'  {col}: 行数={len(df)} 非零占比={nz:.2f}% mean={df[col].mean():.4f}')


def load_prices():
    """dict[code] -> DataFrame(datetime, close, chg_raw); 排除指数/北交所"""
    prices = {}
    for p in sorted(glob.glob(os.path.join(BT, '*_qfq.csv'))):
        code = os.path.basename(p).split('_')[0]
        if not code.isdigit() or code.startswith(('399', '8', '43', '92')):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close', 'change_percent'])
        except Exception:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['chg'] = pd.to_numeric(df['change_percent'], errors='coerce')
        prices[code] = df[['datetime', 'close', 'chg']]
    return prices


def limit_mask(df, code, variant):
    """variant='q1' 现役口径(qfq收益>=9.5%); 'q2' 板块正确口径(原始涨跌幅)"""
    if variant == 'q1':
        ret = df['close'].pct_change()
        return (ret >= 0.095).astype(float)
    chg = df['chg']
    if code.startswith(('300', '301', '688')):
        before = df['datetime'] < CHG_DATE
        thr = np.where(before, 9.8, 19.8)
        thr = np.where(code.startswith('688'), 19.8, thr)
    else:
        thr = 9.8
    return (chg >= thr).astype(float)


def build_factors(prices, variant, window=20):
    """每只股票每日: lu20=窗口内涨停次数; consec=当前连板数"""
    fac = {}
    mask_pct = {}
    for code, df in prices.items():
        m = limit_mask(df, code, variant)
        m = m.fillna(0)
        lu20 = m.rolling(window).sum()
        consec = np.zeros(len(m))
        cnt = 0
        for i, v in enumerate(m.values):
            cnt = cnt + 1 if v > 0 else 0
            consec[i] = cnt
        fac[code] = pd.DataFrame({'lu20': lu20.values, 'consec': consec},
                                 index=df['datetime'])
        mask_pct[code] = (m > 0).mean() * 100
    return fac, mask_pct


def ic_table(fac, prices, months, fcol):
    rows = []
    for month in months:
        xs, y10, y20 = [], [], []
        for code, f in fac.items():
            df = prices[code]
            t = df['datetime'].values
            idx = np.searchsorted(t, np.datetime64(month))
            if idx + 21 > len(t):
                continue
            xv = f[fcol].iloc[idx]
            if pd.isna(xv):
                continue
            xs.append(xv)
            y10.append(df['close'].iloc[idx + 10] / df['close'].iloc[idx] - 1)
            y20.append(df['close'].iloc[idx + 20] / df['close'].iloc[idx] - 1)
        if len(xs) < 50:
            continue
        y10 = np.array(y10) - np.median(y10)
        y20 = np.array(y20) - np.median(y20)
        rows.append((month, spearmanr(xs, y10)[0], spearmanr(xs, y20)[0], len(xs)))
    df = pd.DataFrame(rows, columns=['month', 'ic10', 'ic20', 'n'])
    df['year'] = df['month'].dt.year
    return df


def report(df, label):
    print(f'  {label}:')
    for N in ('ic10', 'ic20'):
        ir = df[N].mean() / df[N].std() if df[N].std() > 0 else 0
        print(f'    {N}: mean={df[N].mean():+.4f} std={df[N].std():.4f} IR={ir:+.2f} '
              f'|IC|>0.05占比={100*(df[N].abs()>0.05).mean():.0f}% 截面n均值={df["n"].mean():.0f}')
    yr = df.groupby('year')['ic20'].mean()
    print(f'    ic20逐年: ' + '  '.join(f'{i}:{v:+.3f}' for i, v in yr.items()))


def buckets(fac, prices, months, fcol, label):
    """5桶×fwd20市场调整收益 (桶按当期截面rank)"""
    rows = []
    for month in months:
        xs, codes = [], []
        for code, f in fac.items():
            df = prices[code]
            t = df['datetime'].values
            idx = np.searchsorted(t, np.datetime64(month))
            if idx + 21 > len(t):
                continue
            xv = f[fcol].iloc[idx]
            if pd.isna(xv):
                continue
            xs.append(xv); codes.append((code, idx))
        if len(xs) < 100:
            continue
        b = pd.qcut(pd.Series(xs).rank(method='first'), 5, labels=False)
        fwds = []
        for (code, idx), bb in zip(codes, b):
            df = prices[code]
            fwds.append((bb, df['close'].iloc[idx + 20] / df['close'].iloc[idx] - 1))
        fd = pd.DataFrame(fwds, columns=['b', 'f20'])
        med = fd['f20'].median()
        fd['f20'] = fd['f20'] - med
        rows.append(fd.groupby('b')['f20'].mean())
    bd = pd.DataFrame(rows)
    print(f'  {label} 分桶 fwd20市场调整均值: ' +
          '  '.join(f'桶{int(i)}:{v*100:+.2f}%' for i, v in bd.mean().items()))


def main():
    q0_parquet_check()
    prices = load_prices()
    print(f'价格覆盖: {len(prices)} 只')
    months = pd.date_range('2021-01-01', '2026-08-01', freq='MS')

    for variant in ('q1', 'q2'):
        fac, mask_pct = build_factors(prices, variant)
        mp = np.mean(list(mask_pct.values()))
        print(f'\n[{variant}] 涨停日均占比(全池平均): {mp:.2f}%')
        for fcol in ('lu20', 'consec'):
            df = ic_table(fac, prices, months, fcol)
            report(df, f'{variant} {fcol}')
        buckets(fac, prices, months, 'lu20', variant)


if __name__ == '__main__':
    main()
