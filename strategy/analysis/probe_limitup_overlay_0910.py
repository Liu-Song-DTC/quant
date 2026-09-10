#!/usr/bin/env python3
"""2026-09-10 探针A后续: 连板数 × 自身buy信号叠加 (负因子是否与系统买点重叠)
数据: 9/9态基线 backtest_signals.csv (buy=True) + qfq change_percent
口径: 信号日consec(板块正确连板数, 同probe_limitup_ic_0910.py q2)
测度: 信号日fwd5/fwd20原始收益 + 当日截面中位数调整, 按consec桶(0/1/2+)
      + 逐年(consec>=1的稳定性) + lu20四分桶
"""
import os
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
CHG_DATE = pd.Timestamp('2020-08-24')


def load_prices(codes):
    prices = {}
    for c in codes:
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close', 'change_percent'])
        except Exception:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['chg'] = pd.to_numeric(df['change_percent'], errors='coerce')
        prices[c] = df[['datetime', 'close', 'chg']]
    return prices


def compute_factors(df, code):
    """返回 DataFrame: datetime-index, consec(连板数), lu20(20日涨停次数)"""
    chg = df['chg']
    if code.startswith(('300', '301', '688')):
        before = df['datetime'] < CHG_DATE
        thr = np.where(before, 9.8, 19.8)
        thr = np.where(code.startswith('688'), 19.8, thr)
    else:
        thr = 9.8
    m = (chg >= thr).fillna(0).astype(float)
    consec = np.zeros(len(m))
    cnt = 0
    for i, v in enumerate(m.values):
        cnt = cnt + 1 if v > 0 else 0
        consec[i] = cnt
    return pd.DataFrame({'consec': consec, 'lu20': m.rolling(20).sum().values},
                        index=df['datetime'])


def main():
    print('读buy信号 (1.6GB, usecols)...', flush=True)
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'])
    sig = sig[sig['buy'] == True].copy()  # noqa: E712
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    print(f'buy信号: {len(sig)} 条, {sig.code.nunique()} 只', flush=True)

    prices = load_prices(sorted(sig.code.unique()))
    print(f'价格覆盖: {len(prices)} 只', flush=True)

    rows = []
    for code, g in sig.groupby('code'):
        if code not in prices:
            continue
        df = prices[code]
        f = compute_factors(df, code)
        t = df['datetime'].values
        idxs = np.searchsorted(t, g['date'].values.astype('datetime64[D]'))
        for i, idx in zip(g.index, idxs):
            if idx >= len(t) or idx + 21 > len(t):
                continue
            d = g.loc[i, 'date']
            td = t[idx]
            if abs((pd.Timestamp(td) - d).days) > 10:  # 停牌导致错位
                continue
            c0 = df['close'].iloc[idx]
            rows.append((code, d, f['consec'].iloc[idx], f['lu20'].iloc[idx],
                         df['close'].iloc[idx + 5] / c0 - 1,
                         df['close'].iloc[idx + 20] / c0 - 1))
    r = pd.DataFrame(rows, columns=['code', 'date', 'consec', 'lu20', 'f5', 'f20'])
    print(f'配对成功: {len(r)} 条 ({100*len(r)/len(sig):.0f}% 信号)', flush=True)

    # 当日截面中位数调整
    med5 = r.groupby('date')['f5'].transform('median')
    med20 = r.groupby('date')['f20'].transform('median')
    r['f5a'] = r['f5'] - med5
    r['f20a'] = r['f20'] - med20

    print('\n[consec 桶] 信号日连板数 × 前向收益:')
    for b, label in ((0, '0(无连板)'), (1, '1'), (2, '2+')):
        sub = r[r['consec'] == b] if b < 2 else r[r['consec'] >= 2]
        print(f'  consec={label}: n={len(sub)} ({100*len(sub)/len(r):.1f}%) '
              f'fwd5={sub.f5.mean()*100:+.2f}% fwd20={sub.f20.mean()*100:+.2f}% '
              f'市场调整 fwd5={sub.f5a.mean()*100:+.2f}% fwd20={sub.f20a.mean()*100:+.2f}% '
              f'fwd20胜率={100*(sub.f20>0).mean():.0f}%')
    # 逐年稳定性 (consec>=1 vs 0 的f20差)
    print('\n[逐年] fwd20市场调整: consec>=1 vs consec=0 (差):')
    for y in sorted(r.date.dt.year.unique()):
        sub = r[r.date.dt.year == y]
        if len(sub) < 50:
            continue
        hi = sub[sub.consec >= 1].f20a.mean()
        lo = sub[sub.consec == 0].f20a.mean()
        n1 = (sub.consec >= 1).sum()
        print(f'  {y}: >=1:{hi*100:+.2f}% (n={n1})  =0:{lo*100:+.2f}%  差={100*(hi-lo):+.2f}pp')

    print('\n[lu20 四分桶] 信号日20日涨停次数 × fwd20市场调整:')
    r['qb'] = pd.qcut(r['lu20'].rank(method='first'), 4, labels=False)
    for q in range(4):
        sub = r[r['qb'] == q]
        print(f'  桶{q}: lu20[{sub.lu20.min():.1f},{sub.lu20.max():.1f}] n={len(sub)} '
              f'fwd20a={sub.f20a.mean()*100:+.2f}% fwd20胜率={100*(sub.f20>0).mean():.0f}%')


if __name__ == '__main__':
    main()
