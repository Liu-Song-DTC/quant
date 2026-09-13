#!/usr/bin/env python3
"""P0-1 Stage2 信号层差异诊断: 锚点(泄漏) vs Stage2(截止上界) 买入信号

对比 backtest_signals.csv 两份:
  - 买入(code,date)集合的重叠率 / 净新增 / 净消失
  - 按年度分解
  - score 值分布差异 (fund相关列的间接影响)
只读。串行执行(bt跑完后)。
"""
import os
import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
BK = os.path.join(ROOT, 'strategy', 'backups', 'stage2_0913')
ANCHOR = os.path.join(BK, 'backtest_signals.csv')
STAGE2 = os.path.join(BK, 'stage2_backtest_signals.csv')


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    return df


def main():
    print('=' * 72)
    print('P0-1 Stage2 信号差异诊断')
    a = load(ANCHOR)
    s = load(STAGE2)
    print(f'锚点: {len(a)} 行, Stage2: {len(s)} 行')

    # 买入行集合 (buy列非0/正)
    buy_col = 'buy' if 'buy' in a.columns else 'signal'
    for df, tag in ((a, 'anchor'), (s, 'stage2')):
        buys = df[df[buy_col] > 0]
        print(f'[{tag}] 买入信号: {len(buys)} 条, 涉及 '
              f'{buys["code"].nunique()} 只')

    ba = a[a[buy_col] > 0][['code', 'date']].copy()
    bs = s[s[buy_col] > 0][['code', 'date']].copy()
    ba['_k'] = ba['code'].astype(str).str.zfill(6) + '@' + ba['date'].dt.strftime('%Y%m%d')
    bs['_k'] = bs['code'].astype(str).str.zfill(6) + '@' + bs['date'].dt.strftime('%Y%m%d')
    ka, ks = set(ba['_k']), set(bs['_k'])
    inter = ka & ks
    only_a, only_s = ka - ks, ks - ka
    print(f'\n买入(code,date)集合: 锚点 {len(ka):,} | Stage2 {len(ks):,} | '
          f'重叠 {len(inter):,} ({100*len(inter)/max(len(ka),1):.1f}% of 锚点)')
    print(f'锚点独有 {len(only_a):,} ({100*len(only_a)/max(len(ka),1):.1f}%) | '
          f'Stage2独有 {len(only_s):,} ({100*len(only_s)/max(len(ks),1):.1f}%)')

    print('\n按年度 (锚点买入 → Stage2保留率):')
    ba['_y'] = ba['date'].dt.year
    bs['_y'] = bs['date'].dt.year
    for y in sorted(ba['_y'].unique()):
        ka_y = set(ba[ba['_y'] == y]['_k'])
        ks_y = set(bs[bs['_y'] == y]['_k'])
        inter_y = ka_y & ks_y
        print(f'  {y}: 锚点 {len(ka_y):,} 条 → 保留 {len(inter_y):,} '
              f'({100*len(inter_y)/max(len(ka_y),1):.1f}%), '
              f'Stage2新增 {len(ks_y - ka_y):,}')

    # 平均买入分 (若buy>0行有score列)
    if 'score' in a.columns:
        for df, tag in ((a, 'anchor'), (s, 'stage2')):
            b = df[df[buy_col] > 0]['score']
            print(f'[{tag}] 买入分: mean={b.mean():.4f} std={b.std():.4f} '
                  f'p50={b.median():.4f} p10={b.quantile(.1):.4f} p90={b.quantile(.9):.4f}')

    # 持有行总数对比
    for df, tag in ((a, 'anchor'), (s, 'stage2')):
        n_buy = (df[buy_col] > 0).sum()
        n_sell = (df[buy_col] < 0).sum() if (df[buy_col] < 0).any() else 0
        print(f'[{tag}] buy>0: {n_buy:,}  sell<0: {n_sell:,}  零/中性: '
              f'{(df[buy_col] == 0).sum():,}')


if __name__ == '__main__':
    main()
