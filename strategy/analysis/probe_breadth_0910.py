#!/usr/bin/env python3
"""2026-09-10 探针C: 市场宽度 (上涨家数占比) — 信号日timing因子
宽度是全市场同日同值 → 截面IC无意义, 只测timing:
信号日宽度/宽度ma5/ma20 分桶(按日期四分) × buy信号fwd5/10/20市场调整 + 逐年
若桶间平坦 → regime层已消费timing信息, 方向关闭
"""
import os
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'


def build_breadth():
    """全池日频: 上涨家数占比 (close>prev_close)"""
    ups, tots = {}, {}
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        code = fn.split('_')[0]
        if not code.isdigit() or code.startswith(('399', '8', '43', '92')):
            continue
        try:
            df = pd.read_csv(os.path.join(BT, fn), usecols=['datetime', 'close'])
        except Exception:
            continue
        t = pd.to_datetime(df['datetime']).values
        up = (df['close'].pct_change() > 0).values
        for i in range(1, len(t)):
            d = t[i]
            ups[d] = ups.get(d, 0) + (1 if up[i] else 0)
            tots[d] = tots.get(d, 0) + 1
    days = sorted(set(ups) | set(tots))
    return pd.Series({d: ups.get(d, 0) / max(tots.get(d, 0), 1) for d in days}).sort_index()


def main():
    print('构建市场宽度序列...', flush=True)
    b = build_breadth()
    print(f'宽度序列: {len(b)} 交易日, 均值={b.mean():.3f}', flush=True)
    b5 = b.rolling(5).mean()
    b20 = b.rolling(20).mean()

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    print(f'buy信号 {len(sig)} 条', flush=True)

    # 价格加载(只为前向收益)
    codes = sorted(sig.code.unique())
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

    for name, x in (('breadth', b), ('breadth_ma5', b5), ('breadth_ma20', b20)):
        rows = []
        for code, g in sig.groupby('code'):
            if code not in prices:
                continue
            s = prices[code]
            for d in g['date'].values:
                if d not in x.index or pd.isna(x.loc[d]):
                    continue
                idx = s.index.searchsorted(d)
                if idx + 21 > len(s.index):
                    continue
                c0 = s.iloc[idx]
                rows.append((d, x.loc[d], s.iloc[idx + 5] / c0 - 1,
                             s.iloc[idx + 10] / c0 - 1, s.iloc[idx + 20] / c0 - 1))
        r = pd.DataFrame(rows, columns=['date', 'x', 'f5', 'f10', 'f20'])
        for c in ('f5', 'f10', 'f20'):
            r[c + 'a'] = r[c] - r.groupby('date')[c].transform('median')
        r['qb'] = pd.qcut(r['x'].rank(method='first'), 4, labels=False)
        print(f'\n[{name}] 信号日四分桶 (n={len(r)}):')
        for q in range(4):
            sub = r[r['qb'] == q]
            print(f'  桶{q}(x[{sub.x.min():.3f},{sub.x.max():.3f}]): n={len(sub)} '
                  f'fwd5a={sub.f5a.mean()*100:+.2f}% fwd10a={sub.f10a.mean()*100:+.2f}% '
                  f'fwd20a={sub.f20a.mean()*100:+.2f}% 胜率={100*(sub.f20 > 0).mean():.0f}%')
        print(f'  桶0-桶3 fwd20a差: {(r[r.qb==0].f20a.mean()-r[r.qb==3].f20a.mean())*100:+.2f}pp')
        # 逐年: 高宽度(桶3) vs 低宽度(桶0)
        print('  逐年 fwd20a(高宽-低宽): ' + '  '.join(
            f'{y}:{100*(r[(r.date.dt.year==y)&(r.qb==3)].f20a.mean()-r[(r.date.dt.year==y)&(r.qb==0)].f20a.mean()):+.2f}pp'
            for y in sorted(r.date.dt.year.unique()) if (r.date.dt.year == y).sum() > 50))


if __name__ == '__main__':
    main()
