#!/usr/bin/env python3
"""2026-09-11 阶段5a: 回购公告事件漂移研究(人群无关, 不需要信号文件)
事件=NOTICEDATE(回购公告日, 2021+, 池内, 北交所排除)。
fwd5/10/20d 原始收益 + 指数调整(sh000001同期)。分桶: 金额上限/占总股本/进度/年份。
对比线: bp2类 realized mean5 +3.02%/hit1 81.8%; 增持事件 +1.56%/20d(否决)。
"""
import os
import numpy as np
import pandas as pd

PKL_REPO = '/mnt/d/quant/data/alternative_data/repurchase_plans.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
IDX = '/mnt/d/quant/data/stock_data/backtrader_data/sh000001_qfq.csv'


def main():
    plans = pd.read_pickle(PKL_REPO)
    plans = plans[plans['NOTICEDATE'] >= '2021-01-01'].copy()
    plans['code'] = plans['code'].astype(str).str.zfill(6)
    plans = plans[~plans['code'].str.startswith(('4', '8', '92'))]
    plans = plans.dropna(subset=['NOTICEDATE'])
    plans = plans.sort_values('NOTICEDATE')
    print(f'回购事件(2021+池内): {len(plans)} 条, 涉及 {plans.code.nunique()} 只', flush=True)

    # 指数基准
    idx = pd.read_csv(IDX, usecols=['datetime', 'close'], parse_dates=['datetime'])
    idx = pd.Series(idx['close'].values, index=idx['datetime'])

    codes = sorted(plans.code.unique())
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
    for _, r in plans.iterrows():
        c, d = r['code'], r['NOTICEDATE']
        if c not in prices:
            continue
        s = prices[c]
        i = s.index.searchsorted(d)
        if i >= len(s.index) or i + 21 > len(s.index):
            continue
        c0 = s.iloc[i]
        # 指数同期窗口
        ii = idx.index.searchsorted(d)
        if ii >= len(idx.index) or ii + 21 > len(idx.index):
            continue
        i0 = idx.iloc[ii]
        obj = str(r['REPUROBJECTIVE'] or '')
        purpose = '注销' if '注销' in obj else ('激励' if ('激励' in obj or '员工持股' in obj) else '其他')
        rows.append((c, d, r['JESX'], r['ZSZSX'], r['REPURPROGRESS'], r['SHARETYPE'], purpose,
                     s.iloc[i + 5] / c0 - 1, s.iloc[i + 10] / c0 - 1,
                     s.iloc[i + 20] / c0 - 1,
                     idx.iloc[ii + 5] / i0 - 1, idx.iloc[ii + 10] / i0 - 1,
                     idx.iloc[ii + 20] / i0 - 1))
    ev = pd.DataFrame(rows, columns=['code', 'date', 'JESX', 'ZSZSX', 'progress',
                                     'sharetype', 'purpose',
                                     'f5', 'f10', 'f20', 'i5', 'i10', 'i20'])
    for c in ('5', '10', '20'):
        ev['f' + c + 'a'] = ev['f' + c] - ev['i' + c]
    print(f'事件配对: {len(ev)} 条\n', flush=True)

    def rep(tag, sub):
        print(f'[{tag}] n={len(sub)}: fwd5a={sub.f5a.mean()*100:+.2f}% '
              f'fwd10a={sub.f10a.mean()*100:+.2f}% fwd20a={sub.f20a.mean()*100:+.2f}% '
              f'20日胜率={100*(sub.f20a > 0).mean():.0f}% 原始fwd20={sub.f20.mean()*100:+.2f}%')

    rep('全部', ev)
    ev['amt_b'] = pd.cut(ev['JESX'], [-np.inf, 1e8, 5e8, np.inf], labels=['<1亿', '1-5亿', '>5亿'])
    for b in ['<1亿', '1-5亿', '>5亿']:
        rep(f'金额{b}', ev[ev.amt_b == b])
    ev['pct_b'] = pd.cut(ev['ZSZSX'], [-np.inf, 0.5, 2, np.inf], labels=['<0.5%', '0.5-2%', '>2%'])
    for b in ['<0.5%', '0.5-2%', '>2%']:
        rep(f'占总股本{b}', ev[ev.pct_b == b])
    rep('进度=006(完成)', ev[ev.progress == '006'])
    for p in ['注销', '激励', '其他']:
        rep(f'用途={p}', ev[ev.purpose == p])
    print('\n逐年 fwd20a(指数调整):')
    for y in sorted(ev.date.dt.year.unique()):
        sub = ev[ev.date.dt.year == y]
        print(f'  {y}: n={len(sub):4d} fwd20a={sub.f20a.mean()*100:+.2f}% '
              f'胜率={100*(sub.f20a > 0).mean():.0f}%')
    print('\n逐年 金额<1亿桶 fwd20a:')
    for y in sorted(ev.date.dt.year.unique()):
        sub = ev[(ev.date.dt.year == y) & (ev.amt_b == '<1亿')]
        if len(sub) == 0:
            continue
        print(f'  {y}: n={len(sub):4d} fwd20a={sub.f20a.mean()*100:+.2f}% '
              f'胜率={100*(sub.f20a > 0).mean():.0f}%')


if __name__ == '__main__':
    main()
