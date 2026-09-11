#!/usr/bin/env python3
"""2026-09-11 阶段6c: PIT gate上线前影响预测量化 (基线信号CSV静态扫描)

方法: 逐行检查 buy 行的 industry 是否属于"当日尚未成立"的概念
(industry ∈ concept_inception 且 date < inception)。
与阶段5g同口径(2021 18.04%/2022 17.47%/2023 8.84%)可比。
注意: 该口径只测"行业归属会被gate改变"的行; gate后新归属由P0循环
下一个非gate概念/回退决定, 由全链跑的季度诊断计数补充。
"""
import pickle
import pandas as pd

ROOT = '/mnt/d/quant'
RVR = f'{ROOT}/strategy/rolling_validation_results'
inc = pickle.load(open(f'{ROOT}/data/concept_inception.pkl', 'rb'))
inc = {k: pd.Timestamp(v) for k, v in inc.items()}

cols = ['date', 'code', 'buy', 'industry']
df = pd.read_csv(f'{RVR}/backtest_signals.csv', usecols=cols)
print(f'信号行: {len(df):,}')
buy = df[df['buy'] == True].copy()
print(f'buy行: {len(buy):,}')

buy['d'] = pd.to_datetime(buy['date'])
buy['incep'] = buy['industry'].map(inc)
gated = buy[buy['d'] < buy['incep']]
print(f'gate命中buy行: {len(gated):,} ({100*len(gated)/len(buy):.2f}%)')

print('\n逐年 buy行 gate命中率 (vs 阶段5g 2021 18.04%/2022 17.47%/2023 8.84%):')
yr = buy.groupby(buy['d'].dt.year).size()
gy = gated.groupby(gated['d'].dt.year).size()
for y in sorted(yr.index):
    print(f'  {y}: {gy.get(y,0):,}/{yr[y]:,} = {100*gy.get(y,0)/yr[y]:.2f}%')

print('\ngate命中最多的20个概念 (概念 | gate日期 | 行数):')
vc = gated['industry'].value_counts().head(20)
for c, n in vc.items():
    print(f'  {c} | {inc[c].date()} | {n:,}')

print('\ngate命中的概念数:', gated['industry'].nunique())
# gate命中行里行业会变的, 占该行业全部buy行的比例 (top概念)
print('\n命中概念被gate削掉的行占比 (gate行/该行业buy行):')
tot = buy['industry'].value_counts()
for c, n in vc.head(10).items():
    print(f'  {c}: {100*n/tot[c]:.1f}% ({n:,}/{tot[c]:,})')
