#!/usr/bin/env python3
"""2026-09-11 阶段5b: 大宗交易折溢价事件漂移研究(人群无关)
事件=交易日有1笔以上大宗成交 (按code×日聚合: 加权折溢价/总额/笔数)。
fwd5/10/20d 原始 + 指数调整。分桶: 溢价/折价0-3%/3-8%/>8%深折价 × 金额档。
席位标签: 买方/卖方含"机构专用"营业部。
"""
import os
import numpy as np
import pandas as pd

PKL_BT = '/mnt/d/quant/data/alternative_data/blocktrades.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
IDX = '/mnt/d/quant/data/stock_data/backtrader_data/sh000001_qfq.csv'


def main():
    raw = pd.read_pickle(PKL_BT)
    raw['code'] = raw['code'].astype(str).str.zfill(6)
    raw = raw[~raw['code'].str.startswith(('4', '8', '92'))]
    raw = raw[raw['TRADE_DATE'] >= '2021-01-01'].copy()
    print(f'大宗原始: {len(raw)} 条 (2021+)', flush=True)

    # 按 code×日 聚合 (折溢价用DEAL/CLOSE-1直接算, DISCOUNT_RATIO字段不可靠)
    raw['is_inst_buy'] = raw['BUYER_NAME'].fillna('').str.contains('机构专用')
    raw['is_inst_sell'] = raw['SELLER_NAME'].fillna('').str.contains('机构专用')
    raw['ratio'] = raw['DEAL_PRICE'] / raw['CLOSE_PRICE'] - 1.0
    g = raw.groupby(['code', 'TRADE_DATE'], as_index=False).agg(
        n=('DEAL_AMT', 'size'),
        tot_amt=('DEAL_AMT', 'sum'),
        w_ratio=('ratio', lambda s: np.average(s, weights=raw.loc[s.index, 'DEAL_AMT'].fillna(0) + 1)),
        inst_buy=('is_inst_buy', 'any'),
        inst_sell=('is_inst_sell', 'any'),
    )
    print(f'聚合事件日: {len(g)} 个', flush=True)

    # 价格
    codes = sorted(g.code.unique())
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
    idx = pd.read_csv(IDX, usecols=['datetime', 'close'], parse_dates=['datetime'])
    idx = pd.Series(idx['close'].values, index=idx['datetime'])
    print(f'价格覆盖: {len(prices)} 只', flush=True)

    rows = []
    for _, r in g.iterrows():
        c, d = r['code'], r['TRADE_DATE']
        if c not in prices:
            continue
        s = prices[c]
        i = s.index.searchsorted(d)
        if i >= len(s.index) or i + 21 > len(s.index):
            continue
        c0 = s.iloc[i]
        ii = idx.index.searchsorted(d)
        if ii >= len(idx.index) or ii + 21 > len(idx.index):
            continue
        i0 = idx.iloc[ii]
        rows.append((c, d, r['n'], r['tot_amt'], r['w_ratio'],
                     r['inst_buy'], r['inst_sell'],
                     s.iloc[i + 5] / c0 - 1, s.iloc[i + 10] / c0 - 1,
                     s.iloc[i + 20] / c0 - 1,
                     idx.iloc[ii + 5] / i0 - 1, idx.iloc[ii + 20] / i0 - 1))
    ev = pd.DataFrame(rows, columns=['code', 'date', 'n', 'tot_amt', 'w_ratio',
                                     'inst_buy', 'inst_sell',
                                     'f5', 'f10', 'f20', 'i5', 'i20'])
    ev['f5a'] = ev['f5'] - ev['i5']
    ev['f20a'] = ev['f20'] - ev['i20']
    print(f'事件配对: {len(ev)} 条\n', flush=True)

    def rep(tag, sub):
        print(f'[{tag}] n={len(sub)}: fwd5a={sub.f5a.mean()*100:+.2f}% '
              f'fwd20a={sub.f20a.mean()*100:+.2f}% '
              f'20日胜率={100*(sub.f20a > 0).mean():.0f}%')

    rep('全部', ev)
    ev['b'] = '溢价>0%'
    ev.loc[ev.w_ratio <= 0, 'b'] = '平价'
    ev.loc[ev.w_ratio < 0, 'b'] = '折价<8%'
    ev.loc[ev.w_ratio < -0.08, 'b'] = '深折价<-8%'
    for b in ['溢价>0%', '平价', '折价<8%', '深折价<-8%']:
        rep(f'折溢={b}', ev[ev.b == b])
    ev['amt_b'] = pd.cut(ev['tot_amt'], [0, 3e7, 1e8, np.inf], labels=['<3千万', '3千万-1亿', '>1亿'])
    for b in ['<3千万', '3千万-1亿', '>1亿']:
        rep(f'成交额{b}', ev[ev.amt_b == b])
    rep('买方机构专用', ev[ev.inst_buy])
    rep('卖方机构专用', ev[ev.inst_sell])
    print('\n逐年 fwd20a 深折价桶 vs 溢价桶:')
    for y in sorted(ev.date.dt.year.unique()):
        sub = ev[ev.date.dt.year == y]
        deep = sub[sub.b == '深折价<-8%'].f20a.mean()
        prem = sub[sub.b == '溢价>0%'].f20a.mean()
        print(f'  {y}: 深折价 {deep*100:+.2f}% (n={len(sub[sub.b=="深折价<-8%"])})  '
              f'溢价 {prem*100:+.2f}% (n={len(sub[sub.b=="溢价>0%"])})  '
              f'差 {(prem-deep)*100:+.2f}pp')


if __name__ == '__main__':
    main()
