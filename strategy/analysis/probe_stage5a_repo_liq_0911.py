#!/usr/bin/env python3
"""2026-09-11 阶段5a+5d: 回购事件 + 池流动性 信号日叠加联合探针
共享步骤: buy信号 → fwd5/10/20 + fwd20a(信号人群当日中位数调整) 计算一次。
A. 回购: buy ≤30d内有回购公告(NOTICEDATE) → 分桶; 大额(金额上限>5亿 或 占总股本>2%)
B. 流动性: buy日 20日均成交额(shift1, 与执行层tradable矩阵同口径) 分桶
   [500万-1千万) / [1-3千万) / [3千万+) → 看500万阈值收紧是否有利
事件研究另跑(probe_stage5a_repo_event): 回购公告后漂移, 不与信号人群绑定。
"""
import os
import numpy as np
import pandas as pd

PKL_REPO = '/mnt/d/quant/data/alternative_data/repurchase_plans.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'


def load_buys():
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01'].sort_values('date')
    # 北交所排除(用户纪律)
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    print(f'buy信号 {len(sig)} 条', flush=True)
    return sig


def fwd_returns(sig):
    """fwd5/10/20 + fwd20a(当日中位数调整)。复用既有叠加协议。"""
    codes = sorted(sig.code.unique())
    prices = {}
    vols = {}
    for c in codes:
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close', 'volume'])
        except Exception:
            continue
        prices[c] = pd.Series(df['close'].values, index=pd.to_datetime(df['datetime']))
        vols[c] = pd.Series(df['volume'].values, index=pd.to_datetime(df['datetime']))
    print(f'价格覆盖: {len(prices)} 只', flush=True)

    rows = []
    for code, g in sig.groupby('code'):
        if code not in prices:
            continue
        s = prices[code]
        v = vols[code]
        for _, r in g.iterrows():
            d = r['date']
            idx = s.index.searchsorted(d)
            if idx >= len(s.index) or idx + 21 > len(s.index):
                continue
            c0 = s.iloc[idx]
            # 20日均成交额(shift1, 与bt_execution tradable矩阵同口径): close*volume*100
            if idx >= 20:
                v20 = (s.iloc[max(0, idx - 20):idx] * v.iloc[max(0, idx - 20):idx] * 100.0).mean()
            else:
                v20 = np.nan
            rows.append((code, d, v20,
                         s.iloc[idx + 5] / c0 - 1,
                         s.iloc[idx + 10] / c0 - 1,
                         s.iloc[idx + 20] / c0 - 1))
    df = pd.DataFrame(rows, columns=['code', 'date', 'avg_val20', 'f5', 'f10', 'f20'])
    print(f'叠加配对: {len(df)} 条', flush=True)
    for c in ('f5', 'f10', 'f20'):
        df[c + 'a'] = df[c] - df.groupby('date')[c].transform('median')
    return df


def repo_flag(df):
    """A. 回购公告(≤30d)分桶。"""
    plans = pd.read_pickle(PKL_REPO)
    plans = plans[plans['NOTICEDATE'] >= '2021-01-01'].copy()
    plans['code'] = plans['code'].astype(str).str.zfill(6)
    plans = plans[~plans['code'].str.startswith(('4', '8', '92'))]
    plans = plans.sort_values('NOTICEDATE')
    print(f'回购事件(2021+池内): {len(plans)} 条', flush=True)

    ev = plans[['code', 'NOTICEDATE', 'JESX', 'ZSZSX']].dropna(subset=['NOTICEDATE'])
    df = df.sort_values('date')
    m = pd.merge_asof(df, ev.rename(columns={'NOTICEDATE': 'ev_date'}), left_on='date',
                      right_on='ev_date', by='code', direction='backward',
                      tolerance=pd.Timedelta('30d'))
    m['has_repo'] = m['ev_date'].notna()
    m['big_repo'] = m['has_repo'] & ((m['JESX'] > 5e8) | (m['ZSZSX'] > 2.0))

    print('\n[A] 回购公告(≤30d)叠加:')
    report_bucket(m, 'has_repo', '无回购', '有回购(≤30d)')
    print('\n[A2] 大额回购(≤30d, 金额>5亿 或 占比>2%):')
    report_bucket(m, 'big_repo', '非大额/无', '大额回购')
    # 逐年稳定性(有回购桶 vs 无回购桶 差)
    if m['has_repo'].sum() > 1000:
        print('\n[A] 有回购桶 fwd20a 逐年(桶内均值, 差=有-无):')
        for y in sorted(m.date.dt.year.unique()):
            sub = m[m.date.dt.year == y]
            a = sub[sub.has_repo].f20a.mean()
            b = sub[~sub.has_repo].f20a.mean()
            print(f'  {y}: 有回购n={sub.has_repo.sum():5d} {a*100:+.2f}%  无 {b*100:+.2f}%  差 {(a-b)*100:+.2f}pp')


def liq_buckets(df):
    """B. 20日均成交额分桶(shift1口径)。"""
    sub = df.dropna(subset=['avg_val20'])
    print(f'\n[B] 流动性分桶(20日均成交额, n={len(sub)}):')
    bins = [0, 1e7, 3e7, np.inf]
    labels = ['500万-1千万', '1-3千万', '3千万+']
    sub['lb'] = pd.cut(sub['avg_val20'], bins=bins, labels=labels)
    for lb in labels:
        s = sub[sub.lb == lb]
        print(f'  {lb:>12s}: n={len(s):6d} fwd5a={s.f5a.mean()*100:+.2f}% '
              f'fwd10a={s.f10a.mean()*100:+.2f}% fwd20a={s.f20a.mean()*100:+.2f}% '
              f'胜率={100*(s.f20 > 0).mean():.0f}%')
    print('\n[B] 逐年 低流动桶(500万-1千万) vs 高流动桶(3千万+) fwd20a差:')
    for y in sorted(sub.date.dt.year.unique()):
        s = sub[sub.date.dt.year == y]
        lo = s[s.lb == labels[0]].f20a.mean()
        hi = s[s.lb == labels[2]].f20a.mean()
        print(f'  {y}: 低 {lo*100:+.2f}%  高 {hi*100:+.2f}%  差(低-高) {(lo-hi)*100:+.2f}pp '
              f'(低n={len(s[s.lb==labels[0]])})')


def report_bucket(m, flag, name0, name1):
    a = m[m[flag]].f20a
    b = m[~m[flag]].f20a
    print(f'  {name1}: n={len(a):6d} fwd5a={m[m[flag]].f5a.mean()*100:+.2f}% '
          f'fwd10a={m[m[flag]].f10a.mean()*100:+.2f}% fwd20a={a.mean()*100:+.2f}% '
          f'胜率={100*(m[m[flag]].f20 > 0).mean():.0f}%')
    print(f'  {name0}: n={len(b):6d} fwd20a={b.mean()*100:+.2f}% '
          f'胜率={100*(m[~m[flag]].f20 > 0).mean():.0f}%')
    print(f'  差({name1} - {name0}): {(a.mean()-b.mean())*100:+.2f}pp')


def main():
    sig = load_buys()
    df = fwd_returns(sig)
    repo_flag(df)
    liq_buckets(df)


if __name__ == '__main__':
    main()
