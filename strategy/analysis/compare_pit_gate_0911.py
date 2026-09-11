#!/usr/bin/env python3
"""2026-09-11 阶段6c: PIT gate全链跑 vs 基线 对比脚本
基线(磁盘.prePIT0911): 979,885/291.95%/1.4094/18.33%
对比维度: 四指标+逐年收益 / 信号差异定位(gated行之外应逐位一致) / gate命中统计
"""
import pickle
import pandas as pd

ROOT = '/mnt/d/quant'
RVR = f'{ROOT}/strategy/rolling_validation_results'

def load_metrics(path):
    eq = pd.read_csv(path)
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.set_index('date')
    nav = eq.iloc[-1].iloc[0]
    total_ret = nav / 200000 - 1  # 初值200k? 用首值算
    total_ret = nav / eq.iloc[0].iloc[0] - 1
    eq['ret'] = eq.iloc[:, 0].pct_change()
    sharpe = eq['ret'].mean() / eq['ret'].std() * (252 ** 0.5)
    cummax = eq.iloc[:, 0].cummax()
    mdd = ((eq.iloc[:, 0] - cummax) / cummax).min()
    return dict(nav=nav, ret=total_ret * 100, sharpe=sharpe, mdd=mdd * 100,
                first=eq.index[0].date(), last=eq.index[-1].date())

base = load_metrics(f'{RVR}/equity_curve.prePIT0911.csv')
new = load_metrics(f'{RVR}/equity_curve.csv')
print('=== 四指标 ===')
print(f"指标        基线        PIT gate    差")
for k, lab in [('nav', '净值'), ('ret', '收益%'), ('sharpe', 'Sharpe'), ('mdd', 'MDD%')]:
    d = new[k] - base[k]
    # mdd 为负值(-18.33), 恶化=更负: d<0 才是 WIN
    flag = ' WIN' if (k != 'mdd' and d > 0) or (k == 'mdd' and d > 0) else ''
    print(f"{lab:<6} {base[k]:>11,.2f} {new[k]:>11,.2f} {d:>+11,.2f}{flag}")
print(f"窗口 {base['first']} ~ {base['last']}")

print('\n=== 逐年收益 ===')
be = pd.read_csv(f'{RVR}/equity_curve.prePIT0911.csv', parse_dates=['date']).set_index('date').iloc[:, 0]
ne = pd.read_csv(f'{RVR}/equity_curve.csv', parse_dates=['date']).set_index('date').iloc[:, 0]
b_y = be.resample('YE').last().pct_change()
n_y = ne.resample('YE').last().pct_change()
n_y.iloc[0] = ne.resample('YE').last().iloc[0] / 200000 - 1
b_y.iloc[0] = be.resample('YE').last().iloc[0] / 200000 - 1
for y in b_y.index:
    print(f"  {y.year}: 基线{b_y[y]*100:+7.2f}%  gate{n_y[y]*100:+7.2f}%  差{(n_y[y]-b_y[y])*100:+7.2f}pp")

print('\n=== 信号CSV对比 (date,code,industry) ===')
cols = ['date', 'code', 'industry']
sb = pd.read_csv(f'{RVR}/backtest_signals.prePIT0911.csv', usecols=cols)
sn = pd.read_csv(f'{RVR}/backtest_signals.csv', usecols=cols)
sb['k'] = sb['date'].astype(str) + '_' + sb['code'].astype(str) + '_' + sb['industry']
sn['k'] = sn['date'].astype(str) + '_' + sn['code'].astype(str) + '_' + sn['industry']
bset, nset = set(sb['k']), set(sn['k'])
print(f"基线行 {len(sb):,} | gate行 {len(sn):,} | 仅基线 {len(bset-nset):,} | 仅gate {len(nset-bset):,}")

inc = pickle.load(open(f'{ROOT}/data/concept_inception.pkl', 'rb'))
inc = {k: pd.Timestamp(v) for k, v in inc.items()}
sb2 = sb[sb['k'].isin(bset - nset)].copy()
sb2['d'] = pd.to_datetime(sb2['date'])
sb2['incep'] = sb2['industry'].map(inc)
explain = (sb2['d'] < sb2['incep']).sum()
print(f"仅基线行中gate可解释: {explain:,}/{len(sb2):,} ({100*explain/max(len(sb2),1):.1f}%)")
print('\n仅基线行的行业分布top10:')
print(sb2['industry'].value_counts().head(10).to_string())
print('\n仅gate行(新增归属)的行业分布top10:')
sn2 = sn[sn['k'].isin(nset - bset)]
print(sn2['industry'].value_counts().head(10).to_string())
