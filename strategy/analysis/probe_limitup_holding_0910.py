#!/usr/bin/env python3
"""2026-09-10 探针A收尾: ①信号日涨跌幅分布(验证consec=0非bug) ②持仓侧连板事件
Part1: buy信号当日/前一日原始涨跌幅分布 — 系统是否结构性不碰涨停日/涨停次日
Part2: trade_realized 506笔 — 持仓期内max连板数分布 × realized收益;
       持仓内连板事件研究: 持仓票consec>=2日之后5/10/20日收益(未出场假设)
"""
import os
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
REAL = '/mnt/d/quant/strategy/rolling_validation_results/trade_realized.csv'
CHG_DATE = pd.Timestamp('2020-08-24')


def load_qfq(code):
    p = os.path.join(BT, f'{code}_qfq.csv')
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, usecols=['datetime', 'close', 'change_percent'])
    except Exception:
        return None
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['chg'] = pd.to_numeric(df['change_percent'], errors='coerce')
    return df


def limit_mask(df, code):
    chg = df['chg']
    if code.startswith(('300', '301', '688')):
        before = df['datetime'] < CHG_DATE
        thr = np.where(before, 9.8, 19.8)
        thr = np.where(code.startswith('688'), 19.8, thr)
    else:
        thr = 9.8
    return (chg >= thr).fillna(0).astype(float)


def streak_from_mask(m):
    consec = np.zeros(len(m))
    cnt = 0
    for i, v in enumerate(m.values):
        cnt = cnt + 1 if v > 0 else 0
        consec[i] = cnt
    return consec


def part1_signals():
    print('[Part1] 信号日涨跌幅分布验证')
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    print(f'  buy信号 {len(sig)} 条')
    same_lu = prev_lu = same_big = 0
    for code, g in sig.groupby('code'):
        df = load_qfq(code)
        if df is None:
            continue
        m = limit_mask(df, code)
        m_prev = m.shift(1).fillna(0)
        chg = df['chg'].fillna(0)
        t = df['datetime'].values
        idxs = np.searchsorted(t, g['date'].values.astype('datetime64[D]'))
        for i, idx in zip(g.index, idxs):
            if idx >= len(t):
                continue
            if abs((pd.Timestamp(t[idx]) - g.loc[i, 'date']).days) > 10:
                continue
            if m.iloc[idx] > 0:
                same_lu += 1
            if m_prev.iloc[idx] > 0:
                prev_lu += 1
            if chg.iloc[idx] >= 5:
                same_big += 1
    print(f'  信号日当日涨停: {same_lu} 条  前一日涨停: {prev_lu} 条  当日涨>=5%: {same_big} 条')
    print(f'  -> 若当日/前一日涨停均≈0: 缠论买点结构性避开涨停日及次日 (consec=0非bug)')


def part2_holding():
    print('\n[Part2] 持仓侧连板: 506笔realized')
    tr = pd.read_csv(REAL)
    tr['code'] = tr['code'].astype(str).str.zfill(6)
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    max_cons, n_days_hi, ev5, ev10, ev20 = [], [], [], [], []
    maxc_bucket = []
    for _, r in tr.iterrows():
        df = load_qfq(r['code'])
        if df is None:
            max_cons.append(np.nan); continue
        t = df['datetime'].values
        i0 = np.searchsorted(t, np.datetime64(r['entry_date']))
        i1 = np.searchsorted(t, np.datetime64(r['exit_date']))
        if i0 >= len(t):
            max_cons.append(np.nan); continue
        i1 = min(i1, len(t) - 1)
        seg = df.iloc[i0:i1 + 1]
        if len(seg) < 2:
            max_cons.append(np.nan); continue
        m = limit_mask(seg, r['code'])
        c = streak_from_mask(m)
        max_cons.append(c.max())
        maxc_bucket.append(r['ret'])
        # 持仓内连板事件研究: consec从1->2的日(第2板), 之后5/10/20日收益
        for j in range(1, len(c)):
            if c[j] == 2 and c[j - 1] == 1:
                if j + 21 <= len(seg):
                    c0 = seg['close'].iloc[j]
                    ev5.append(seg['close'].iloc[j + 5] / c0 - 1)
                    ev10.append(seg['close'].iloc[j + 10] / c0 - 1)
                    ev20.append(seg['close'].iloc[j + 20] / c0 - 1)
        n_days_hi.append((c >= 2).sum())
    tr['max_consec'] = max_cons
    tr['days_hi'] = n_days_hi
    ok = tr.dropna(subset=['max_consec'])
    print(f'  持仓期max连板分布: ' +
          '  '.join(f'{int(k)}板:{int((ok.max_consec == k).sum())}' for k in sorted(ok.max_consec.unique())))
    for b, label in ((0, 'max=0(全程无板)'), (1, 'max=1'), (2, 'max>=2')):
        sub = ok[ok.max_consec == b] if b < 2 else ok[ok.max_consec >= b]
        print(f'  {label}: n={len(sub)} ({100*len(sub)/len(ok):.0f}%) realized均值={sub.ret.mean()*100:+.2f}% '
              f'中位={sub.ret.median()*100:+.2f}% 胜率={100*(sub.ret > 0).mean():.0f}%')
    print(f'  持仓内consec>=2总天数: {ok.days_hi.sum()} (占{ok.days_hi.sum() + (ok.max_consec >= 0).sum()}持仓日)')
    if ev20:
        print(f'  持仓票第2板事件: n={len(ev20)} 后5日={np.mean(ev5)*100:+.2f}% '
              f'后10日={np.mean(ev10)*100:+.2f}% 后20日={np.mean(ev20)*100:+.2f}% '
              f'(后20日胜率{100*np.mean(np.array(ev20) > 0):.0f}%)')
    else:
        print('  持仓内第2板事件: n=0')


if __name__ == '__main__':
    part1_signals()
    part2_holding()
