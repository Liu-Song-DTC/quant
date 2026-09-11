#!/usr/bin/env python3
"""2026-09-12 阶段7探针A: gate态2021-07→2022-12 MDD 27.41%解剖
新权威基线 1,143,938/357.58%/1.5555/27.41%。探针目标(全轻量, 不跑全链):
1. MDD窗口精确定位 + 窗内月度净值分解
2. 窗内平仓交易解剖: 行业/持仓天数/入场月份/入场score
3. 盈亏两侧事前特征对比 (找稳定可事前实现的判别特征)
4. gate vs 基线 交易流diff: 诚实重分配交易的PnL
5. 2021峰腿(1/4→7/5)增量解剖: gate的+19.18pp从哪来
"""
import pandas as pd
import numpy as np

ROOT = '/mnt/d/quant'
RVR = f'{ROOT}/strategy/rolling_validation_results'

# ---------- 0) 加载 ----------
eq = pd.read_csv(f'{RVR}/equity_curve.csv', parse_dates=['date']).set_index('date')
eq = eq.iloc[:, 0]  # nav
eq_b = pd.read_csv(f'{RVR}/equity_curve.prePIT0911.csv', parse_dates=['date']).set_index('date').iloc[:, 0]
sig = pd.read_csv(f'{RVR}/backtest_signals.csv', parse_dates=['date'])
sig_b = pd.read_csv(f'{RVR}/backtest_signals.prePIT0911.csv', parse_dates=['date'])
tr = pd.read_csv(f'{RVR}/trade_realized.csv', parse_dates=['entry_date', 'exit_date'])
tr_b = pd.read_csv(f'{RVR}/trade_realized.prePIT0911.csv', parse_dates=['entry_date', 'exit_date'])
print('signals列:', list(sig.columns))

# ---------- 1) MDD窗口精确定位 ----------
cummax = eq.cummax()
dd = (eq - cummax) / cummax
trough_d = dd.idxmin()
peak_d = eq.loc[:trough_d].idxmax()
print(f"\n=== 1) gate态MDD ===\n峰 {peak_d.date()} {eq[peak_d]:,.0f} → 谷 {trough_d.date()} {eq[trough_d]:,.0f} "
      f"({dd[trough_d]*100:.2f}%)")
# 基线同窗
w_bb = (eq_b.loc[:trough_d] / eq_b.loc[:trough_d].cummax() - 1).min()
print(f"基线同窗 {w_bb*100:.2f}% | 基线自身MDD {((eq_b/eq_b.cummax()-1).min()*100):.2f}% @ {((eq_b/eq_b.cummax()-1).idxmin()).date()}")

# 窗内月度分解 (峰月→谷月)
m = eq.loc[peak_d:trough_d]
print('\n窗内月度收益%:')
mr = m.resample('ME').last().pct_change() * 100
mr.iloc[0] = (m.resample('ME').last().iloc[0] / eq[peak_d] - 1) * 100
for d, v in mr.items():
    print(f"  {d.strftime('%Y-%m')} {v:+7.2f}")

# 窗内子腿: 找出所有局部谷
sub = []
running_peak = peak_d
running_val = eq[peak_d]
for d in m.index:
    if eq[d] < running_val * 0.94:  # 每跌6%记一腿
        sub.append((running_peak, d, (eq[d] / eq[running_peak] - 1) * 100))
        running_peak, running_val = d, eq[d]
    elif eq[d] > running_val:
        running_peak, running_val = d, eq[d]
if sub:
    print(f'窗内跌幅≥6%腿 {len(sub)}: ' + ' | '.join(
        f'{a.date()}→{b.date()} {c:+.1f}%' for a, b, c in sub))

# ---------- 2) 窗内平仓交易解剖 ----------
w_lo, w_hi = peak_d, trough_d
def trades_in(tr, lo, hi):
    t = tr[(tr['exit_date'] >= lo) & (tr['exit_date'] <= hi)].copy()
    # 入场score + industry (入场日signals buy行)
    ind = {}
    sc = {}
    buy = sig[sig['signal'] == 'buy'] if 'signal' in sig.columns else sig
    for _, r in t.iterrows():
        d = r['entry_date']
        rows = buy[(buy['date'] == d) & (buy['code'] == r['code'])]
        if len(rows):
            ind[(d, r['code'])] = rows.iloc[0]['industry']
            sc[(d, r['code'])] = rows.iloc[0]['score']
    t['industry'] = [ind.get((d, c), '?') for d, c in zip(t['entry_date'], t['code'])]
    t['score'] = [sc.get((d, c), np.nan) for d, c in zip(t['entry_date'], t['code'])]
    return t

tw = trades_in(tr, w_lo, w_hi)
print(f"\n=== 2) 窗内平仓交易 {len(tw)}笔 ===")
print(f"合计ret {tw['ret'].sum()*100:+.1f}% | 胜率 {(tw['ret']>0).mean()*100:.1f}% | "
      f"平均 {tw['ret'].mean()*100:+.2f}% | 中位hold {tw['hold_days'].median():.0f}d")
print('\n按行业top10 (笔数, 合计ret%):')
g = tw.groupby('industry')['ret'].agg(['count', 'sum'])
print(g.sort_values('sum').head(10).to_string())
print('\n按持仓天桶:')
tw['hb'] = pd.cut(tw['hold_days'], [0, 5, 10, 20, 40, 80, 999], labels=['1-5', '6-10', '11-20', '21-40', '41-80', '81+'])
print(tw.groupby('hb', observed=True)['ret'].agg(['count', 'mean']).to_string(float_format=lambda x: f'{x*100:+.1f}'))
print('\n按入场月份:')
tw['em'] = tw['entry_date'].dt.strftime('%Y-%m')
print(tw.groupby('em')['ret'].agg(['count', 'mean']).to_string(float_format=lambda x: f'{x*100:+.1f}'))

# ---------- 3) 盈亏两侧事前特征 ----------
print('\n=== 3) 盈亏两侧事前特征 ===')
win = tw[tw['ret'] > 0]
los = tw[tw['ret'] <= 0]
for lab, s in [('赢', win), ('亏', los)]:
    print(f"{lab} {len(s)}笔: hold中位 {s['hold_days'].median():.0f}d | score均值 {s['score'].mean():.3f} | "
          f"ret均值 {s['ret'].mean()*100:+.2f}%")
print('\n亏损侧行业分布:')
print(los.groupby('industry')['ret'].agg(['count', 'sum']).sort_values('sum').head(8).to_string(float_format=lambda x: f'{x*100:+.1f}'))

# ---------- 4) gate vs 基线交易流diff (2021全年+2022全年) ----------
print('\n=== 4) gate vs 基线交易流diff (2021-01~2022-12) ===')
lo, hi = pd.Timestamp('2021-01-01'), pd.Timestamp('2022-12-31')
tk = lambda t: set(zip(t['entry_date'], t['exit_date'], t['code']))
tg = tr[(tr['exit_date'] >= lo) & (tr['exit_date'] <= hi)]
tb = tr_b[(tr_b['exit_date'] >= lo) & (tr_b['exit_date'] <= hi)]
kg, kb = tk(tg), tk(tb)
only_g = tg[[k in (kg - kb) for k in zip(tg['entry_date'], tg['exit_date'], tg['code'])]]
only_b = tb[[k in (kb - kg) for k in zip(tb['entry_date'], tb['exit_date'], tb['code'])]]
print(f"gate {len(tg)}笔 | 基线 {len(tb)}笔 | 仅gate {len(only_g)} | 仅基线 {len(only_b)}")
print(f"仅gate笔 PnL合计 {only_g['ret'].sum()*100:+.1f}% | 仅基线笔 PnL合计 {only_b['ret'].sum()*100:+.1f}%")
og = trades_in(only_g, lo, hi)
print('仅gate笔行业分布:')
print(og.groupby('industry')['ret'].agg(['count', 'sum']).sort_values('sum').head(8).to_string(float_format=lambda x: f'{x*100:+.1f}'))

# ---------- 5) 2021峰腿增量: gate的+19.18pp从哪来 ----------
print('\n=== 5) 2021峰腿(1/4→7/5) 交易对比 ===')
lo5, hi5 = pd.Timestamp('2021-01-01'), pd.Timestamp('2021-07-05')
t5g = tr[(tr['exit_date'] >= lo5) & (tr['exit_date'] <= hi5)]
t5b = tr_b[(tr_b['exit_date'] >= lo5) & (tr_b['exit_date'] <= hi5)]
print(f"gate {len(t5g)}笔 合计 {t5g['ret'].sum()*100:+.1f}% | 基线 {len(t5b)}笔 合计 {t5b['ret'].sum()*100:+.1f}%")
g5 = trades_in(t5g, lo5, hi5)
print('gate峰腿行业分布:')
print(g5.groupby('industry')['ret'].agg(['count', 'sum']).sort_values('sum').head(8).to_string(float_format=lambda x: f'{x*100:+.1f}'))
print(f'\n2021峰时点净值: gate {eq[hi5]:,.0f} vs 基线 {eq_b[hi5]:,.0f}')
