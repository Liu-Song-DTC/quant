#!/usr/bin/env python3
"""2026-09-10 探针A最后一块: 涨停次日回踩信号质量 (前一日涨停, 当日未涨停的buy)
对比: 前一日涨停组 vs 无涨停组 的 fwd5/fwd20(原始+市场调整) + 逐年 + 买点类分布
结论直接回答: 9,247条涨停次日信号是否应该被过滤/降权
"""
import os
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
CHG_DATE = pd.Timestamp('2020-08-24')


def main():
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'chan_buy_point', 'factor_name'],
                      dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    print(f'buy信号 {len(sig)} 条', flush=True)

    rows = []
    for code, g in sig.groupby('code'):
        p = os.path.join(BT, f'{code}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close', 'change_percent'])
        except Exception:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        chg = pd.to_numeric(df['change_percent'], errors='coerce')
        if code.startswith(('300', '301', '688')):
            before = df['datetime'] < CHG_DATE
            thr = np.where(before, 9.8, 19.8)
            thr = np.where(code.startswith('688'), 19.8, thr)
        else:
            thr = 9.8
        m = (chg >= thr).fillna(0).astype(float)
        m_prev = m.shift(1).fillna(0)
        t = df['datetime'].values
        idxs = np.searchsorted(t, g['date'].values.astype('datetime64[D]'))
        for i, idx in zip(g.index, idxs):
            if idx >= len(t) or idx + 21 > len(t):
                continue
            if abs((pd.Timestamp(t[idx]) - g.loc[i, 'date']).days) > 10:
                continue
            c0 = df['close'].iloc[idx]
            rows.append((g.loc[i, 'code'], g.loc[i, 'date'], int(m_prev.iloc[idx]),
                         g.loc[i, 'chan_buy_point'], g.loc[i, 'factor_name'],
                         df['close'].iloc[idx + 5] / c0 - 1,
                         df['close'].iloc[idx + 20] / c0 - 1))
    r = pd.DataFrame(rows, columns=['code', 'date', 'prev_lu', 'bp', 'fac', 'f5', 'f20'])
    print(f'配对: {len(r)} 条', flush=True)
    med5 = r.groupby('date')['f5'].transform('median')
    med20 = r.groupby('date')['f20'].transform('median')
    r['f5a'] = r['f5'] - med5
    r['f20a'] = r['f20'] - med20

    for b, label in ((1, '前一日涨停(次日回踩)'), (0, '前一日未涨停')):
        sub = r[r['prev_lu'] == b]
        print(f'\n{label}: n={len(sub)} ({100*len(sub)/len(r):.1f}%) '
              f'fwd5={sub.f5.mean()*100:+.2f}% fwd20={sub.f20.mean()*100:+.2f}% '
              f'市场调整fwd5={sub.f5a.mean()*100:+.2f}% fwd20={sub.f20a.mean()*100:+.2f}% '
              f'fwd20胜率={100*(sub.f20 > 0).mean():.0f}%')
        if b == 1:
            print('  买点类分布: ' + '  '.join(f'{k}:{int(v)}' for k, v in
                  sub['bp'].astype(str).value_counts().head(6).items()))
    print('\n逐年 fwd20市场调整 (前一日涨停 vs 未涨停):')
    for y in sorted(r.date.dt.year.unique()):
        sub = r[r.date.dt.year == y]
        if len(sub) < 50:
            continue
        hi = sub[sub.prev_lu == 1].f20a.mean()
        lo = sub[sub.prev_lu == 0].f20a.mean()
        n1 = (sub.prev_lu == 1).sum()
        print(f'  {y}: 涨停次日:{hi*100:+.2f}% (n={n1})  无涨停:{lo*100:+.2f}%  差={100*(hi-lo):+.2f}pp')


if __name__ == '__main__':
    main()
