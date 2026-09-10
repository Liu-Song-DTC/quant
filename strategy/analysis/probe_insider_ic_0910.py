#!/usr/bin/env python3
"""2026-09-10 阶段3探针: 内部人净变动因子 IC (增持+减持合并, 公告时刻PIT)
变体: V1 net60 = 60日净变动(增-减, change_num万股 signed)
      V2 inc60 = 60日纯增持强度(仅增持且实际变动>0)
      V3 netcnt60 = 60日净事件数
市场调整: 收益减截面中位数。输出: 月度截面IC(10/20日)逐年 + 分桶 + 事件研究(大额净增持公告后收益)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INC = '/mnt/d/quant/data/alternative_data/increase_records.pkl'
RED = '/mnt/d/quant/data/alternative_data/reduction_records.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'


def load_events():
    inc = pd.read_pickle(INC)
    inc = inc[['code', 'change_num', 'eitime']].copy()
    inc['w'] = +inc['change_num']          # 万股, 正=增持
    red = pd.read_pickle(RED)
    red = red[['code', 'change_num', 'eitime']].copy()
    red['w'] = -red['change_num'].abs()    # 万股, 负=减持
    ev = pd.concat([inc, red], ignore_index=True)
    ev['eitime'] = pd.to_datetime(ev['eitime'], errors='coerce')
    ev = ev.dropna(subset=['eitime', 'w'])
    ev['code'] = ev['code'].astype(str).str.zfill(6)
    ev = ev[ev['eitime'] >= '2020-07-01']
    # 增持公告内实际变动为负的记录: 在V2变体里要排除, 这里标记
    inc_raw = pd.read_pickle(INC)
    inc_raw['eitime'] = pd.to_datetime(inc_raw['eitime'], errors='coerce')
    inc_pos = inc_raw[inc_raw['change_num'] > 0].copy()
    inc_pos['code'] = inc_pos['code'].astype(str).str.zfill(6)
    return ev, inc_pos[['code', 'change_num', 'eitime']]


def load_prices(codes):
    cal = None
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
        if cal is None:
            cal = prices[c].index
    return prices, cal


def build_factor(ev, cal, window='60D'):
    """每只股票在每个交易日的窗口内净变动(事件按公告日eitime计入)"""
    fac = {}
    full_days = pd.date_range(cal.min(), cal.max(), freq='D')
    for code, g in ev.groupby('code'):
        g = g.sort_values('eitime')
        t = g.set_index('eitime')['w']
        # 同日多事件: 按日求和; 上采样到全日历; 窗口滚动和; 对齐交易日
        daily = t.resample('D').sum().reindex(full_days).fillna(0)
        daily_roll = daily.rolling(window, min_periods=1).sum()
        fac[code] = daily_roll.reindex(cal, method='ffill').fillna(0)
    return fac


def ic_table(fac, prices, cal, months):
    rows = []
    for month in months:
        xs, y10, y20 = [], [], []
        for code, v in fac.items():
            if code not in prices:
                continue
            s = prices[code]
            idx = s.index.searchsorted(month)
            if idx + 21 > len(s.index):
                continue
            t_date = s.index[idx]
            if t_date not in v.index:
                continue
            xv = v.loc[t_date]
            if pd.isna(xv):
                continue
            xs.append(xv)
            y10.append(s.iloc[idx + 10] / s.iloc[idx] - 1)
            y20.append(s.iloc[idx + 20] / s.iloc[idx] - 1)
        if len(xs) < 30:
            continue
        # 市场调整(中位数)
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


def event_study(ev, prices, cal, thresh=100):
    """大额净增持事件(单条>thresh万股) 公告后市场调整收益"""
    big = ev[ev['w'] > thresh].copy()
    vals5, vals10, vals20 = [], [], []
    for _, r in big.iterrows():
        if r['code'] not in prices:
            continue
        s = prices[r['code']]
        idx = s.index.searchsorted(r['eitime'])
        if idx + 1 + 20 > len(s.index):
            continue
        start = s.iloc[idx + 1]
        v5 = s.iloc[idx + 5] / start - 1 if idx + 6 <= len(s.index) else np.nan
        v10 = s.iloc[idx + 10] / start - 1 if idx + 11 <= len(s.index) else np.nan
        v20 = s.iloc[idx + 20] / start - 1
        vals5.append(v5); vals10.append(v10); vals20.append(v20)
    print(f'  大额净增持(>{thresh}万股, n={len(big)}): '
          f'next5d={np.nanmean(vals5)*100:+.2f}% next10d={np.nanmean(vals10)*100:+.2f}% next20d={np.nanmean(vals20)*100:+.2f}%')


def main():
    ev, inc_pos = load_events()
    print(f'净事件流(2020-07后): {len(ev)} 条, {ev.code.nunique()} 只')
    print(f'  其中增持正变动: {len(inc_pos)} 条')
    prices, cal = load_prices(sorted(ev.code.unique()))
    print(f'价格覆盖: {len(prices)} 只, 交易日 {len(cal)}')
    ev = ev[ev.code.isin(prices.keys())]
    inc_pos = inc_pos[inc_pos.code.isin(prices.keys())]
    inc_pos['w'] = inc_pos['change_num']

    months = pd.date_range('2021-01-01', '2026-08-01', freq='MS')

    print('\n[V1] net60 = 60日净变动(增-减, 万股):')
    f1 = build_factor(ev, cal)
    report(ic_table(f1, prices, cal, months), 'net60')

    print('\n[V2] inc60 = 60日纯增持强度(仅实际变动为正的增持):')
    f2 = build_factor(inc_pos, cal)
    report(ic_table(f2, prices, cal, months), 'inc60')

    print('\n[事件研究] 单条大额净增持:')
    event_study(ev, prices, cal, 100)
    event_study(ev, prices, cal, 500)


if __name__ == '__main__':
    main()
