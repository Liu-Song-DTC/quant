#!/usr/bin/env python3
"""2026-09-11 阶段5a-E: 回购×近门候选人探针 (E-K1模式复检)
问题: 评分系统对回购人群是否标定系统性偏低? (E-K1结论: bp2类五档flat→类级加成)
方法: 取 buy=False 且 score∈[-0.15, 0) 的"差一点买入"行(buy_threshold=0.0),
      叠加回购flag(≤30d, 金额分桶) → fwd20a。
      若 回购flag近门行 显著优于 无flag近门行, 且接近/超过 buy人群,
      → 类级score加成有据(E-K1模式第2例), 反之=评分系统已正确标定。
PIT: 公告日<=信号日(盘后公告当日可见于盘后信号引擎), 另按日龄分桶。
"""
import os
import numpy as np
import pandas as pd

PKL_REPO = '/mnt/d/quant/data/alternative_data/repurchase_plans.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'


def load_rows():
    """near-miss行(score∈[-0.15,0)) + 全部buy行(对比基准)。"""
    cols = ['code', 'date', 'buy', 'sell', 'score', 'chan_buy_point']
    sig = pd.read_csv(SIG, usecols=cols, dtype={'code': str})
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    # 北交所排除(用户纪律)
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    nm = sig[(sig.buy == False) & (sig.score >= -0.15) & (sig.score < 0.0)].copy()
    buys = sig[sig.buy == True].copy()
    print(f'near-miss行(score∈[-0.15,0)): {len(nm)}, buy行: {len(buys)}', flush=True)
    return nm, buys


def fwd_returns(df, tag):
    """fwd5/10/20 + fwd20a(当日中位数调整)。"""
    codes = sorted(df.code.unique())
    prices = {}
    for c in codes:
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        prices[c] = pd.Series(d['close'].values, index=pd.to_datetime(d['datetime']))
    print(f'[{tag}] 价格覆盖: {len(prices)} 只', flush=True)

    rows = []
    for code, g in df.groupby('code'):
        if code not in prices:
            continue
        s = prices[code]
        for _, r in g.iterrows():
            d = r['date']
            i = s.index.searchsorted(d)
            if i >= len(s.index) or i + 21 > len(s.index):
                continue
            c0 = s.iloc[i]
            rows.append((code, d, r['score'],
                         s.iloc[i + 5] / c0 - 1,
                         s.iloc[i + 10] / c0 - 1,
                         s.iloc[i + 20] / c0 - 1))
    out = pd.DataFrame(rows, columns=['code', 'date', 'score', 'f5', 'f10', 'f20'])
    for c in ('f5', 'f10', 'f20'):
        out[c + 'a'] = out[c] - out.groupby('date')[c].transform('median')
    print(f'[{tag}] 叠加配对: {len(out)} 条', flush=True)
    return out


def attach_repo(df):
    plans = pd.read_pickle(PKL_REPO)
    plans = plans[plans['NOTICEDATE'] >= '2020-12-01'].copy()
    plans['code'] = plans['code'].astype(str).str.zfill(6)
    plans = plans[~plans['code'].str.startswith(('4', '8', '92'))]
    plans = plans.sort_values('NOTICEDATE')
    ev = plans[['code', 'NOTICEDATE', 'JESX']].dropna(subset=['NOTICEDATE'])
    df = df.sort_values('date')
    m = pd.merge_asof(df, ev.rename(columns={'NOTICEDATE': 'ev_date'}), left_on='date',
                      right_on='ev_date', by='code', direction='backward',
                      tolerance=pd.Timedelta('30d'))
    m['has_repo'] = m['ev_date'].notna()
    m['days_since'] = (m['date'] - m['ev_date']).dt.days
    m['small_repo'] = m['has_repo'] & (m['JESX'] < 1e8)
    m['big_repo'] = m['has_repo'] & (m['JESX'] >= 1e8)
    return m


def rep(tag, sub):
    print(f'  [{tag}] n={len(sub):6d} fwd5a={sub.f5a.mean()*100:+.2f}% '
          f'fwd20a={sub.f20a.mean()*100:+.2f}% 胜率={100*(sub.f20 > 0).mean():.0f}%')


def main():
    nm, buys = load_rows()
    nm = fwd_returns(nm, 'near-miss')
    nm = attach_repo(nm)

    print('\n[1] near-miss 人群: 回购flag 分桶')
    rep('无回购', nm[~nm.has_repo])
    rep('有回购(≤30d)', nm[nm.has_repo])
    rep('  小回购<1亿', nm[nm.small_repo])
    rep('  大回购>=1亿', nm[nm.big_repo])
    print(f'  差(有-无): {(nm[nm.has_repo].f20a.mean()-nm[~nm.has_repo].f20a.mean())*100:+.2f}pp')
    print(f'  差(小-无): {(nm[nm.small_repo].f20a.mean()-nm[~nm.has_repo].f20a.mean())*100:+.2f}pp')

    print('\n[2] near-miss 人群: score档 × 回购 (标定是否偏低)')
    nm['sb'] = pd.cut(nm['score'], [-0.15, -0.10, -0.05, 0.0],
                      labels=['[-0.15,-0.10)', '[-0.10,-0.05)', '[-0.05,0)'])
    for b in ['[-0.15,-0.10)', '[-0.10,-0.05)', '[-0.05,0)']:
        for fl, tag in [(nm.has_repo, '有回购'), (~nm.has_repo, '无回购')]:
            s = nm[(nm.sb == b) & fl]
            rep(f'score{b} {tag}', s)

    print('\n[3] near-miss 有回购桶 逐年 (fwd20a)')
    for y in sorted(nm.date.dt.year.unique()):
        sub = nm[nm.date.dt.year == y]
        a = sub[sub.has_repo].f20a.mean()
        b = sub[~sub.has_repo].f20a.mean()
        print(f'  {y}: 有回购n={sub.has_repo.sum():5d} {a*100:+.2f}%  无 {b*100:+.2f}%  '
              f'差 {(a-b)*100:+.2f}pp')

    print('\n[4] 对照: buy人群 回购flag 分桶 (同口径, 与repo_liq可互验)')
    buys = fwd_returns(buys, 'buy')
    buys = attach_repo(buys)
    rep('buy·无回购', buys[~buys.has_repo])
    rep('buy·有回购', buys[buys.has_repo])
    rep('  buy·小回购<1亿', buys[buys.small_repo])
    print(f'  差(有-无): {(buys[buys.has_repo].f20a.mean()-buys[~buys.has_repo].f20a.mean())*100:+.2f}pp')

    print('\n[5] 近门回购人群 与 买入线: 若近门回购行 fwd20a 高于 buy人群, 加成可推进买入')
    rep('near-miss·小回购', nm[nm.small_repo])
    rep('buy人群·全部', buys)


if __name__ == '__main__':
    main()
