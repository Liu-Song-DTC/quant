#!/usr/bin/env python3
"""2026-09-10 阶段4终裁二: factor_df不可见列前6候选的信号日叠加
候选: max_ret_20/volatility/idiosyncratic_volatility/vol_confirm/
      price_volume_corr_20/limit_pullback_score/rsi_vol_combo
方法: factor_df面板(隔日采样)merge_asof(≤4d)到信号日 → 四分桶 × fwd5/10/20
      市场调整(信号人群当日中位数) + 胜率 — 与既有叠加协议一致
"""
import os
import numpy as np
import pandas as pd

PKL = '/mnt/d/quant/strategy/cache/factor_df_2757s_808d_3851bd96.parquet'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
FACS = ['max_ret_20', 'volatility', 'idiosyncratic_volatility', 'vol_confirm',
        'price_volume_corr_20', 'limit_pullback_score', 'rsi_vol_combo']


def main():
    fd = pd.read_parquet(PKL, columns=['code', 'date'] + FACS)
    fd['date'] = pd.to_datetime(fd['date'])
    fd = fd.sort_values('date')

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01'].sort_values('date')
    print(f'buy信号 {len(sig)} 条', flush=True)

    merged = pd.merge_asof(sig, fd, on='date', by='code',
                           direction='backward', tolerance=pd.Timedelta('4d'))
    merged = merged.dropna(subset=FACS, how='all')
    print(f'面板join成功: {merged.notna().sum().iloc[0] if False else len(merged)} 条', flush=True)

    # fwd收益: 仅信号股
    codes = sorted(merged.code.unique())
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
    print(f'价格覆盖: {len(prices)} 只', flush=True)

    rows = []
    for code, g in merged.groupby('code'):
        if code not in prices:
            continue
        s = prices[code]
        for _, r in g.iterrows():
            d = r['date']
            idx = s.index.searchsorted(d)
            if idx >= len(s.index) or idx + 21 > len(s.index):
                continue
            c0 = s.iloc[idx]
            vals = [r[f] for f in FACS]
            rows.append((d, *vals, s.iloc[idx + 5] / c0 - 1,
                         s.iloc[idx + 10] / c0 - 1, s.iloc[idx + 20] / c0 - 1))
    df = pd.DataFrame(rows, columns=['date'] + FACS + ['f5', 'f10', 'f20'])
    print(f'叠加配对: {len(df)} 条', flush=True)
    for c in ('f5', 'f10', 'f20'):
        df[c + 'a'] = df[c] - df.groupby('date')[c].transform('median')

    for fac in FACS:
        sub = df.dropna(subset=[fac]).copy()
        if len(sub) < 10000:
            print(f'\n[{fac}] 有效n不足: {len(sub)}')
            continue
        sub['qb'] = pd.qcut(sub[fac].rank(method='first'), 4, labels=False)
        print(f'\n[{fac}] n={len(sub)}:')
        for q in range(4):
            s = sub[sub['qb'] == q]
            print(f'  桶{q}(x[{s[fac].min():.3f},{s[fac].max():.3f}]): n={len(s)} '
                  f'fwd5a={s.f5a.mean()*100:+.2f}% fwd10a={s.f10a.mean()*100:+.2f}% '
                  f'fwd20a={s.f20a.mean()*100:+.2f}% 胜率={100*(s.f20 > 0).mean():.0f}%')
        print(f'  桶0-桶3 fwd20a差: {(sub[sub.qb==0].f20a.mean()-sub[sub.qb==3].f20a.mean())*100:+.2f}pp')


if __name__ == '__main__':
    main()
