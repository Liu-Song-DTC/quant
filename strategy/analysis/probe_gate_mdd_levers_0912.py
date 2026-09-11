#!/usr/bin/env python3
"""2026-09-12 阶段7探针B: MDD攻击杠杆扫描 (轻量, 不跑全链)
1. 窗内亏损按factor_name归因 (真实计算路径, 避开CSV industry展示artifact)
2. 熔断时间线重建: CLB连亏3日减半 + HDS -20%触发 (何时触发/恢复)
3. 曲线级敞口反事实: 在gate净值曲线上叠加敞口去风险机制, 线性重建四指标
   - S1 冲高去风险: 尾部收益>X% → 敞口×f, 回落至X/2恢复
   - S2 回撤闸: NAV自峰回撤>T → 敞口×f, 恢复至T/2
   - 找四指标全胜基线(1,143,938/357.58%/1.5555/27.41%)的参数区; 无全胜则看最优权衡
   注: 线性近似(敞口缩放日收益), 忽略再平衡摩擦→偏乐观; 仅判"交易空间存在与否"
"""
import pandas as pd
import numpy as np

RVR = '/mnt/d/quant/strategy/rolling_validation_results'
eq = pd.read_csv(f'{RVR}/equity_curve.csv', parse_dates=['date']).set_index('date')
nav = eq.iloc[:, 0]
r = nav.pct_change().fillna(0.0).values
dates = nav.index
rng = np.random.default_rng(0)  # 无用, 占位

def metrics(n):
    ret = (n[-1] / n[0] - 1) * 100
    rr = pd.Series(n).pct_change().dropna()
    sharpe = rr.mean() / rr.std() * np.sqrt(252)
    mdd = (pd.Series(n) / pd.Series(n).cummax() - 1).min() * 100
    return n[-1], ret, sharpe, mdd

BASE = metrics(nav.values)
print(f"基线(gate态): NAV {BASE[0]:,.0f} | 收益 {BASE[1]:.2f}% | Sharpe {BASE[2]:.4f} | MDD {BASE[3]:.2f}%")

# ---------- 1) factor_name归因 (窗内平仓交易, 真实计算路径) ----------
sig = pd.read_csv(f'{RVR}/backtest_signals.csv', parse_dates=['date'], low_memory=False)
sig['code'] = sig['code'].astype(str)
buy = sig[sig['buy'] == True].copy()
tr = pd.read_csv(f'{RVR}/trade_realized.csv', parse_dates=['entry_date', 'exit_date'])
tr['code'] = tr['code'].astype(str)
lo, hi = pd.Timestamp('2021-07-05'), pd.Timestamp('2022-12-23')
tw = tr[(tr['exit_date'] >= lo) & (tr['exit_date'] <= hi)].copy()
fn_map = {}
for _, row in tw.iterrows():
    rows = buy[(buy['date'] == row['entry_date']) & (buy['code'] == row['code'])]
    fn_map[(row['entry_date'], row['code'])] = rows.iloc[0]['factor_name'] if len(rows) else '?'
tw['fn'] = [fn_map[(d, c)] for d, c in zip(tw['entry_date'], tw['code'])]
print(f"\n=== 1) 窗内亏损 factor_name归因 ({len(tw)}笔, 合计{tw['ret'].sum()*100:+.1f}%) ===")
g = tw.groupby('fn')['ret'].agg(['count', 'sum'])
print(g.sort_values('sum').head(12).to_string())
print('\n亏损侧(<0) factor_name:')
los = tw[tw['ret'] < 0]
print(los.groupby('fn')['ret'].agg(['count', 'sum']).sort_values('sum').head(10).to_string(float_format=lambda x: f'{x*100:+.1f}'))

# 两段崩塌腿详情
print('\n2021-12入场(2022-01崩塌)交易:')
leg = tw[(tw['entry_date'] >= '2021-12-01') & (tw['entry_date'] <= '2021-12-31')]
for _, row in leg.iterrows():
    print(f"  {row['entry_date'].date()}→{row['exit_date'].date()} {row['code']} {row['ret']*100:+6.1f}% {row['fn'][:55]}")
print('2021-07入场(顶点崩塌)交易:')
leg2 = tw[(tw['entry_date'] >= '2021-06-20') & (tw['entry_date'] <= '2021-08-05')]
for _, row in leg2.iterrows():
    print(f"  {row['entry_date'].date()}→{row['exit_date'].date()} {row['code']} {row['ret']*100:+6.1f}% {row['fn'][:55]}")

# ---------- 2) 熔断时间线 ----------
print('\n=== 2) 熔断触发重建 ===')
dr = pd.Series(r, index=dates)
clb_loss = dr < -0.005
clb_win = dr > 0.005
streak = 0
clb_on = []
for d, w in zip(clb_loss, clb_win):
    if d: streak += 1
    elif w: streak = 0
    else: streak = 0
    clb_on.append(streak >= 3)
clb_on = pd.Series(clb_on, index=dates)
clb_dates = []
in_trig = False
for d, on in clb_on.items():
    if on and not in_trig:
        clb_dates.append(d); in_trig = True
    elif not on:
        in_trig = False
win = dr > 0.005
n_trig = 0
for d in clb_dates:
    if d <= pd.Timestamp('2022-12-23') and d >= pd.Timestamp('2021-07-05'):
        n_trig += 1
print(f'CLB(连亏3日≤-0.5%)窗内触发 {n_trig} 次: ' + ' '.join(str(d.date()) for d in clb_dates if lo <= d <= hi))
dd = nav / nav.cummax() - 1
cross = dd[dd < -0.20]
hds_date = cross.index[0] if len(cross) else None
if hds_date is not None:
    print(f'HDS(-20%回撤)首次触发: {hds_date.date()} 当日DD {dd[hds_date]*100:.1f}% | 峰值日 {nav[:hds_date].idxmax().date()}')
print(f'窗内DD最差日: {dd[lo:hi].idxmin().date()} {dd[lo:hi].min()*100:.2f}%')

# ---------- 3) 曲线级反事实扫描 ----------
print('\n=== 3) 敞口反事实扫描 (线性重建) ===')
def sim(trailing_trig, red, mode, look=60, ema=True):
    """mode='runup': trailing收益>trig→敞口red; <trig/2恢复
       mode='dd':   自峰回撤>trig→敞口red; 恢复到trig/2"""
    n = nav.values.copy()
    e = np.ones(len(n))
    peak = n[0]
    prev = 1.0
    for i in range(len(n)):
        target = 1.0
        if mode == 'runup':
            if i >= look:
                tr_ret = n[i] / n[i - look] - 1
                if tr_ret > trailing_trig:
                    target = red
                elif tr_ret <= trailing_trig / 2:
                    target = 1.0
                else:
                    target = e[i - 1]  # 滞回区保持
        else:
            peak = max(peak, n[i - 1] if i else n[0])
            d = n[i] / peak - 1
            if d < -trailing_trig:
                target = red
            elif d > -trailing_trig / 2:
                target = 1.0
            else:
                target = e[i - 1]
        e[i] = 0.3 * prev + 0.7 * target if ema else target
        prev = e[i]
        n[i] = n[i - 1] * (1 + e[i] * r[i]) if i else n[i]
    return metrics(n)

results = []
for mode, grid in [('runup', [(L, X, f) for L in (40, 60, 90, 126) for X in (0.15, 0.20, 0.25, 0.30, 0.40) for f in (0.5, 0.3, 0.0)]),
                   ('dd', [(0, T, f) for T in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25) for f in (0.5, 0.3)])]:
    for L, X, f in grid:
        m = sim(X, f, mode, look=L)
        tag = f'{mode} L={L} X={X:.2f} f={f:.1f}'
        results.append((tag, m))

win_all = [r for r in results if r[1][0] > BASE[0] and r[1][1] > BASE[1] and r[1][2] > BASE[2] and r[1][3] > BASE[3]]
print(f'四指标全胜参数 {len(win_all)} 个:')
for tag, m in sorted(win_all, key=lambda x: -x[1][2])[:12]:
    print(f"  {tag:34s} NAV {m[0]:,.0f} ({m[0]/BASE[0]-1:+5.1%}) 收益 {m[1]:6.2f}%  Sharpe {m[2]:.4f}  MDD {m[3]:6.2f}%")
if not win_all:
    print('无四指标全胜参数。MDD≤18.33%(原基线水平)的最优Sharpe:')
    better_mdd = [r for r in results if r[1][3] > -18.4]
    for tag, m in sorted(better_mdd, key=lambda x: -x[1][2])[:8]:
        print(f"  {tag:34s} NAV {m[0]:,.0f} ({m[0]/BASE[0]-1:+5.1%}) 收益 {m[1]:6.2f}%  Sharpe {m[2]:.4f}  MDD {m[3]:6.2f}%")
    print('\nMDD<20%的最优Sharpe:')
    for tag, m in sorted([r for r in results if r[1][3] > -20.1], key=lambda x: -x[1][2])[:5]:
        print(f"  {tag:34s} NAV {m[0]:,.0f} ({m[0]/BASE[0]-1:+5.1%}) 收益 {m[1]:6.2f}%  Sharpe {m[2]:.4f}  MDD {m[3]:6.2f}%")
