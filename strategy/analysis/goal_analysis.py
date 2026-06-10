"""从收益率/正确率/夏普率三个验收目标出发，逐模块深度分析选股全链路"""
import pandas as pd
import numpy as np

sig = pd.read_csv('strategy/rolling_validation_results/backtest_signals.csv')
val = pd.read_csv('strategy/rolling_validation_results/validation_results.csv')
port = pd.read_csv('strategy/rolling_validation_results/portfolio_selections.csv')

# Use val's buy/sell (validation data), merge sig fields with suffix
sig_cols = ['code','date','score','factor_name','gate_quality',
    'chan_buy_point','signal_level','trend_type','mom_60d','dist_ma60',
    'risk_vol','exhaustion_risk','daily_return','volume_ratio',
    'factor_quality','chan_divergence_strength']
m = val.merge(sig[sig_cols], on=['code','date'], how='left', suffixes=('_val',''))
# val already has 'buy' column — use it
buys = m[m['buy']==True]

# ===================================================================
# GOAL 1: 收益率
# ===================================================================
print('='*70)
print('GOAL 1: 收益率分析 — 每层对正收益的贡献')
print('='*70)

print('\n--- 全市场基线 ---')
all_mr = m['future_ret'].mean()
print('所有信号 MeanRet = {:.4%}'.format(all_mr))
print('所有信号 MedianRet = {:.4%}'.format(m['future_ret'].median()))

print('\n--- Layer1 因子引擎 → 买入信号 ---')
buy_mr = buys['future_ret'].mean()
non_mr = m[~m['buy']]['future_ret'].mean()
print('买入 MeanRet = {:.4%} (N={:,})'.format(buy_mr, len(buys)))
print('非买入 MeanRet = {:.4%}'.format(non_mr))
print('因子层提升: {:.2f}%'.format((buy_mr - all_mr)*100))

print('\n--- Layer2 门控分层 ---')
hi = buys[buys['gate_quality']>1.5]
lo = buys[(buys['gate_quality']>0.85)&(buys['gate_quality']<=1.5)]
print('GQ>1.5:  MR={:.4%} N={:,}'.format(hi['future_ret'].mean(), len(hi)))
print('GQ0.85-1.5: MR={:.4%} N={:,}'.format(lo['future_ret'].mean(), len(lo)))

print('\n--- Layer3 信号组合收益矩阵 ---')
best = []
for tt in [-2,0,1,2]:
    for bp in [0,1,2,3]:
        b = buys[(buys['trend_type']==tt)&(buys['chan_buy_point']==bp)]
        if len(b)>30:
            best.append((tt, bp, b['future_ret'].mean(), (b['future_ret']>0).mean(), len(b)))
best.sort(key=lambda x: -x[2])
print('  {"TT":>3} {"BP":>3}  {"MR":>8}  {"Acc":>7}  {"N":>6}')
for tt, bp, mr, acc, n in best[:15]:
    print('  {:>3}  {:>3}  {:>7.3%}  {:>6.1%}  {:>6,}'.format(tt, bp, mr, acc, n))

print('\n--- Layer4 组合层 ---')
port_m = port.merge(val, on=['code','date'], how='left')
port_mr = port_m['future_ret'].mean()
port_acc = (port_m['future_ret']>0).mean()
print('入选组合 MeanRet = {:.4%} (N={})'.format(port_mr, len(port_m)))
print('入选组合 Acc = {:.1%}'.format(port_acc))
if len(port_m)>10:
    sharpe = port_m['future_ret'].mean()/max(port_m['future_ret'].std(),1e-10)*np.sqrt(252)
    print('入选组合 年化Sharpe = {:.2f}'.format(sharpe))

# ===================================================================
# GOAL 2: 正确率
# ===================================================================
print('\n' + '='*70)
print('GOAL 2: 正确率分析 — 各维度对Acc的区分力')
print('='*70)

print('\n--- 单维度Acc区分力 ---')
dims = [
    ('trend_type=2', m['trend_type']==2),
    ('trend_type=-2', m['trend_type']==-2),
    ('trend_type>=1', m['trend_type']>=1),
    ('GQ>1.5', m['gate_quality']>1.5),
    ('GQ<1.0', m['gate_quality']<1.0),
    ('score>P90', m['score']>m['score'].quantile(0.9)),
    ('score<P10', m['score']<m['score'].quantile(0.1)),
    ('mom_60d<0', m['mom_60d']<0),
    ('mom_60d>0.3', m['mom_60d']>0.3),
    ('exhaustion<0.1', m['exhaustion_risk']<0.1),
    ('exhaustion>0.3', m['exhaustion_risk']>0.3),
    ('signal_level>=2', m['signal_level']>=2),
    ('signal_level<=-1', m['signal_level']<=-1),
    ('daily_ret>0', m['daily_return']>0),
    ('daily_ret<-0.03', m['daily_return']<-0.03),
]
for name, mask in dims:
    s = m[mask]
    if len(s)>100:
        print('  {:25s}: Acc={:.1%} MR={:+.4%} N={:,}'.format(name, (s['future_ret']>0).mean(), s['future_ret'].mean(), len(s)))

print('\n--- 多维度组合Acc ---')
combos = [
    ('trend>=1+GQ>1.5', (m['trend_type']>=1)&(m['gate_quality']>1.5)),
    ('trend>=1+mom<0', (m['trend_type']>=1)&(m['mom_60d']<0)),
    ('trend>=1+exh<0.1', (m['trend_type']>=1)&(m['exhaustion_risk']<0.1)),
    ('GQ>1.5+mom<0', (m['gate_quality']>1.5)&(m['mom_60d']<0)),
    ('GQ>1.5+exh<0.1', (m['gate_quality']>1.5)&(m['exhaustion_risk']<0.1)),
    ('trend=2+GQ>1.5+mom<0', (m['trend_type']==2)&(m['gate_quality']>1.5)&(m['mom_60d']<0)),
    ('trend>=1+GQ>1.5+exh<0.1+mom<0', (m['trend_type']>=1)&(m['gate_quality']>1.5)&(m['exhaustion_risk']<0.1)&(m['mom_60d']<0)),
]
for name, mask in combos:
    s = m[mask]
    if len(s)>100:
        s_buy = s[s['buy']==True]
        print('  {:35s}: Acc={:.1%} MR={:+.4%} BuyRate={:.1%} N={:,}'.format(name, (s['future_ret']>0).mean(), s['future_ret'].mean(), s['buy'].mean(), len(s)))

# ===================================================================
# GOAL 3: 夏普率
# ===================================================================
print('\n' + '='*70)
print('GOAL 3: 夏普率分析 — 收益稳定性与IC持续性')
print('='*70)

m['month'] = pd.to_datetime(m['date']).dt.to_period('M')
monthly = []
for month, grp in m.groupby('month'):
    valid = grp[['score','future_ret']].dropna()
    if len(valid)>500:
        ic = valid['score'].corr(valid['future_ret'], method='spearman')
        mr = valid['future_ret'].mean()
        std = valid['future_ret'].std()
        monthly.append({'month':str(month), 'IC':ic, 'MeanRet':mr, 'StdRet':std, 'N':len(valid)})

ms = pd.DataFrame(monthly)
print('\nIC月度稳定性:')
print('  IC均值={:.4f}  IC标准差={:.4f}'.format(ms['IC'].mean(), ms['IC'].std()))
print('  IC>0月占比={:.0%}'.format((ms['IC']>0).mean()))
print('  IR(IC mean/std)={:.2f}'.format(ms['IC'].mean()/max(ms['IC'].std(),1e-10)))
print('  最佳IC={:.4f} ({})'.format(ms['IC'].max(), ms.loc[ms['IC'].idxmax(),'month']))
print('  最差IC={:.4f} ({})'.format(ms['IC'].min(), ms.loc[ms['IC'].idxmin(),'month']))

print('\n收益月度稳定性:')
print('  MeanRet均值={:.4%}  MeanRet标准差={:.4%}'.format(ms['MeanRet'].mean(), ms['MeanRet'].std()))
print('  MeanRet>0月占比={:.0%}'.format((ms['MeanRet']>0).mean()))
print('  月收益Sharpe={:.2f}'.format(ms['MeanRet'].mean()/max(ms['MeanRet'].std(),1e-10)))

# What would improve Sharpe most?
print('\n--- Sharpe提升路径模拟 ---')
# Path A: increase accuracy from 50.75% to 55%
buy_acc = (buys['future_ret']>0).mean()
buy_mean = buys['future_ret'].mean()
buy_std = buys['future_ret'].std()
print('当前Buy: Acc={:.1%} Mean={:.4%} Std={:.4%} Sharpe={:.2f}'.format(
    buy_acc, buy_mean, buy_std, buy_mean/max(buy_std,1e-10)*np.sqrt(252)))

# Path B: trend>=1 only (filter out trend=-2)
trend_filtered = buys[buys['trend_type']>=1]
print('过滤trend<1: Acc={:.1%} Mean={:.4%} Std={:.4%} Sharpe={:.2f} N={:,}'.format(
    (trend_filtered['future_ret']>0).mean(),
    trend_filtered['future_ret'].mean(),
    trend_filtered['future_ret'].std(),
    trend_filtered['future_ret'].mean()/max(trend_filtered['future_ret'].std(),1e-10)*np.sqrt(252),
    len(trend_filtered)))

# Path C: trend>=1 + GQ>1.5 + mom<0
sharpe_filtered = buys[(buys['trend_type']>=1)&(buys['gate_quality']>1.5)&(buys['mom_60d']<0)]
print('trend>=1+GQ>1.5+mom<0: Acc={:.1%} Mean={:.4%} Std={:.4%} Sharpe={:.2f} N={:,}'.format(
    (sharpe_filtered['future_ret']>0).mean(),
    sharpe_filtered['future_ret'].mean(),
    sharpe_filtered['future_ret'].std(),
    sharpe_filtered['future_ret'].mean()/max(sharpe_filtered['future_ret'].std(),1e-10)*np.sqrt(252),
    len(sharpe_filtered)))

# Path D: trend>=1 + GQ>1.5 + exh<0.1
sharpe2 = buys[(buys['trend_type']>=1)&(buys['gate_quality']>1.5)&(buys['exhaustion_risk']<0.1)]
print('trend>=1+GQ>1.5+exh<0.1: Acc={:.1%} Mean={:.4%} Std={:.4%} Sharpe={:.2f} N={:,}'.format(
    (sharpe2['future_ret']>0).mean(),
    sharpe2['future_ret'].mean(),
    sharpe2['future_ret'].std(),
    sharpe2['future_ret'].mean()/max(sharpe2['future_ret'].std(),1e-10)*np.sqrt(252),
    len(sharpe2)))

print('\n=== 关键结论 ===')
print('1. 当前Buy Acc={:.1%} — 接近抛硬币, 正确率提升空间最大'.format(buy_acc))
print('2. 最强单维度Acc提升: trend_type>=1 (Acc={:.1%} vs baseline {:.1%})'.format(
    (m[m['trend_type']>=1]['future_ret']>0).mean(), (m['future_ret']>0).mean()))
print('3. 最强组合: trend>=1+GQ>1.5+exh<0.1 → Acc={:.1%} Sharpe={:.2f}'.format(
    (sharpe2['future_ret']>0).mean(),
    sharpe2['future_ret'].mean()/max(sharpe2['future_ret'].std(),1e-10)*np.sqrt(252)))
