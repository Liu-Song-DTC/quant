#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Alpha 2.0 探针批次1 (2026-09-17): 0d冲击成本审计 + N2隔夜分解 +
C2 bp2加成敏感性 + C1 ML blend IC曲线 + N1周内效应。
全部无回测、只读生产产物+panel+K线。产物: rolling_validation_results/probe_optbatch1_20260917.pkl
"""
import os, sys, pickle, glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
SNAP = '/home/liusong/quant_archive_0915/rvd_signals_snapshots/backtest_signals.csv'
PARQ = os.path.join(BASE, 'cache/factor_df_2674s_810d_df8acbb5.parquet')
KLINE_DIR = os.path.join(BASE, '..', 'data', 'stock_data', 'backtrader_data')
OUT = {}
rng = np.random.default_rng(42)

t0 = pd.Timestamp.now()

# ---------- 0. 加载面板(fwd10) ----------
panel = pd.read_parquet(PARQ, columns=['code', 'date', 'future_ret'])
panel['date'] = pd.to_datetime(panel['date'])
panel['code'] = panel['code'].astype(str).str.zfill(6)
print(f"[panel] {len(panel)} rows, {panel.date.min().date()}~{panel.date.max().date()}", flush=True)

# ---------- 1. 加载信号CSV(只取所需列) ----------
sig_cols = ['code', 'date', 'buy', 'sell', 'score', 'adjusted_score', 'ml_score',
            'chan_buy_point', 'chan_sell_point', 'signal_level']
sig = pd.read_csv(SNAP, usecols=sig_cols, dtype={'code': str})
sig['code'] = sig['code'].str.zfill(6)
sig['date'] = pd.to_datetime(sig['date'])
buy_rows = sig[sig['buy'] == 1].copy()
print(f"[sig] {len(sig)} rows, buy {len(buy_rows)}", flush=True)

# 与panel精确join(code,date) — 面板日覆盖~47%无偏
buy_j = buy_rows.merge(panel, on=['code', 'date'], how='inner')
print(f"[join] buy rows on panel days: {len(buy_j)} ({len(buy_j)/max(len(buy_rows),1)*100:.0f}%)", flush=True)

def ic_series(df, val_col, ret_col='future_ret'):
    return df.groupby('date').apply(
        lambda g: g[[val_col, ret_col]].corr(method='spearman').iloc[0, 1]
        if len(g) >= 5 else np.nan, include_groups=False)

# ==================== C2: bp2加成敏感性 ====================
bp2m = (buy_rows['chan_buy_point'] == 2) & (buy_rows['chan_sell_point'] == 0)
bp0m = (buy_rows['chan_buy_point'] == 0) & (buy_rows['chan_sell_point'] == 0)
bp2 = buy_rows[bp2m]; bp0 = buy_rows[bp0m]
bp2_j = buy_j[(buy_j['chan_buy_point'] == 2) & (buy_j['chan_sell_point'] == 0)]
bp0_j = buy_j[(buy_j['chan_buy_point'] == 0) & (buy_j['chan_sell_point'] == 0)]

def bucket_stats(df):
    if len(df) == 0:
        return dict(n=0)
    f = df['future_ret']
    return dict(n=len(df), mean5=f.mean() * 100, hit1=(f > 0).mean() * 100,
                p50=np.median(f) * 100)

out_c2 = {}
out_c2['bp2'] = bucket_stats(bp2_j)
out_c2['bp0'] = bucket_stats(bp0_j)
out_c2['pool_allbuy'] = bucket_stats(buy_j)
# 年度稳定性
out_c2['bp2_by_year'] = {y: bucket_stats(g) for y, g in bp2_j.groupby(bp2_j['date'].dt.year)}
out_c2['bp2_by_year_f'] = {}
for y, g in bp2_j.groupby(bp2_j['date'].dt.year):
    out_c2['bp2_by_year_f'][y] = bucket_stats(g)
# score五档区分度(与E-K1证据一致?)
if len(bp2_j) >= 20:
    qs = pd.qcut(bp2_j['score'], 5, labels=False, duplicates='drop')
    out_c2['score_quintiles'] = {q: bucket_stats(bp2_j[qs == q]) for q in sorted(qs.dropna().unique())}
# bp2占buy比 + adjusted_score分位(加成后)
out_c2['bp2_share_buys'] = len(bp2) / max(len(buy_rows), 1)
buy_daily_adj = buy_rows.set_index('date')['adjusted_score']
out_c2['bp2_adj_pct_rank'] = float((bp2.set_index('date')['adjusted_score'] > 0).mean()) if len(bp2) else np.nan
print(f"[C2] bp2 n={len(bp2_j)} mean5={out_c2['bp2']['mean5']:.2f}% hit1={out_c2['bp2']['hit1']:.1f}% | "
      f"bp0 n={len(bp0_j)} mean5={out_c2['bp0']['mean5']:.2f}% | pool n={len(buy_j)} mean5={out_c2['pool_allbuy']['mean5']:.2f}%", flush=True)

# ==================== C1: ML blend权重IC曲线 ====================
mlm = np.abs(buy_j['ml_score']) > 0.01
sub = buy_j[mlm].copy()
z = np.tanh(sub['ml_score'].to_numpy() * 3)
a0 = sub['adjusted_score'].to_numpy()
w0 = 0.4
score_clean = (a0 - w0 * z) / (1 - w0)
out_c1 = {}
for w in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
    adj_w = (1 - w) * score_clean + w * z
    sub2 = sub.assign(adj_w=adj_w)
    ic = ic_series(sub2, 'adj_w')
    out_c1[w] = dict(ic_mean=ic.mean(), ic_ir=ic.mean() / max(ic.std(), 1e-9), n_days=int(ic.notna().sum()), n=len(sub2))
    print(f"[C1] w={w:.2f} IC={out_c1[w]['ic_mean']:+.4f} IR={out_c1[w]['ic_ir']:+.2f} days={out_c1[w]['n_days']}", flush=True)
# 基线(0.4生产)对照
ic_base = ic_series(sub, 'adjusted_score')
out_c1['base_ic'] = dict(ic_mean=ic_base.mean(), n_days=int(ic_base.notna().sum()))
print(f"[C1] base(0.4生产) IC={out_c1['base_ic']['ic_mean']:+.4f}", flush=True)

# ==================== N1: 周内效应 ====================
out_n1 = {}
out_n1['by_weekday'] = {int(d): bucket_stats(g) for d, g in buy_j.groupby(buy_j['date'].dt.dayofweek)}
for d, g in buy_j.groupby(buy_j['date'].dt.dayofweek):
    out_n1['by_weekday'][int(d)] = bucket_stats(g)
print(f"[N1] weekday mean5: " + ' '.join(f"{d}={'%.2f'%out_n1['by_weekday'].get(d,{}).get('mean5',0)}%" for d in range(5)), flush=True)
# 年度交互: 周五买入是否各年稳定
fri = buy_j[buy_j['date'].dt.dayofweek == 4]
out_n1['fri_by_year'] = {y: bucket_stats(g) for y, g in fri.groupby(fri['date'].dt.year)}

# ==================== 0d: 冲击成本审计 ====================
eq = pd.read_csv(os.path.join(RVD, 'equity_curve.csv'))
eq.columns = [c.strip() for c in eq.columns]
dcol = [c for c in eq.columns if 'date' in c.lower() or 'time' in c.lower()][0]
vcol = [c for c in eq.columns if 'equity' in c.lower() or 'nav' in c.lower() or 'value' in c.lower()][0]
eq[dcol] = pd.to_datetime(eq[dcol])
eq = eq[[dcol, vcol]].rename(columns={dcol: 'date', vcol: 'nav'}).drop_duplicates('date').set_index('date')['nav']
sel = pd.read_csv(os.path.join(RVD, 'portfolio_selections.csv'), dtype={'code': str})
sel['code'] = sel['code'].str.zfill(6)
sel['date'] = pd.to_datetime(sel['date'])
sel = sel.merge(eq.rename('nav').reset_index(), on='date', how='left')
sel['notional'] = sel['weight'] * sel['nav']

# 逐股票ADV20(用当日往前20日成交额均值) — 只读持仓过的股票
codes_need = sel['code'].unique()
adv_cache = {}
for code in codes_need:
    fp = os.path.join(KLINE_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        continue
    try:
        k = pd.read_csv(fp, usecols=['datetime', 'close', 'volume'])
        k['datetime'] = pd.to_datetime(k['datetime'])
        k['turnover'] = k['close'] * k['volume']
        k = k.sort_values('datetime').set_index('datetime')['turnover']
        adv_cache[code] = k.rolling(20, min_periods=5).mean().shift(1)
    except Exception:
        pass
print(f"[0d] kline ADV cache: {len(adv_cache)}/{len(codes_need)}", flush=True)

rows = []
for code, adv in adv_cache.items():
    s = sel[sel['code'] == code]
    for _, r in s.iterrows():
        a = adv.get(r['date'])
        if a is not None and a > 0 and r['notional'] > 0:
            rows.append((r['date'], r['notional'] / a))
impact = pd.DataFrame(rows, columns=['date', 'adv_ratio']).set_index('date')
out_0d = dict(
    n=len(impact),
    p50=float(impact['adv_ratio'].median() * 100),
    p90=float(impact['adv_ratio'].quantile(0.9) * 100),
    p99=float(impact['adv_ratio'].quantile(0.99) * 100),
    pct_gt_1pct=float((impact['adv_ratio'] > 0.01).mean() * 100),
    pct_gt_5pct=float((impact['adv_ratio'] > 0.05).mean() * 100),
    max_=float(impact['adv_ratio'].max() * 100),
)
print(f"[0d] impact: n={out_0d['n']} p50={out_0d['p50']:.1f}% p90={out_0d['p90']:.1f}% p99={out_0d['p99']:.1f}% "
      f">1%={out_0d['pct_gt_1pct']:.1f}% >5%={out_0d['pct_gt_5pct']:.2f}% max={out_0d['max_']:.1f}%", flush=True)

# ==================== N2: 隔夜分解(close-fill vs open-fill) ====================
tr = pd.read_csv(os.path.join(RVD, 'trade_realized.csv'), dtype={'code': str})
tr['code'] = tr['code'].str.zfill(6)
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['exit_date'] = pd.to_datetime(tr['exit_date'])

gap_entries, gap_exits = [], []
for code, g in tr.groupby('code'):
    fp = os.path.join(KLINE_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        continue
    try:
        k = pd.read_csv(fp, usecols=['datetime', 'open', 'close'])
        k['datetime'] = pd.to_datetime(k['datetime'])
        k = k.sort_values('datetime').set_index('datetime')
    except Exception:
        continue
    o, c = k['open'], k['close']
    for _, r in g.iterrows():
        # 信号日=entry前一交易日(信号收盘出, 次日开盘成交)
        # close-fill替代 = 信号日收盘价成交
        prev = o.index[o.index < r['entry_date']]
        if len(prev) == 0:
            continue
        sig_day = prev[-1]
        if sig_day in o.index:
            gap_entries.append((r['entry_date'], c[sig_day] / o[r['entry_date']] - 1))
        # 卖出: 信号日=exit前一交易日
        prev2 = o.index[o.index < r['exit_date']]
        if len(prev2) == 0:
            continue
        sig_day2 = prev2[-1]
        if sig_day2 in o.index:
            gap_exits.append((r['exit_date'], c[sig_day2] / o[r['exit_date']] - 1))

ge = pd.DataFrame(gap_entries, columns=['date', 'gap']).set_index('date')
gx = pd.DataFrame(gap_exits, columns=['date', 'gap']).set_index('date')
out_n2 = dict(
    entry_gap_mean=float(ge['gap'].mean() * 100),
    entry_gap_sum=float(np.log1p(ge['gap']).sum() * 100),
    entry_gap_n=len(ge),
    entry_gap_pct_pos=float((ge['gap'] > 0).mean() * 100),
    exit_gap_mean=float(gx['gap'].mean() * 100),
    exit_gap_sum=float(np.log1p(gx['gap']).sum() * 100),
    exit_gap_n=len(gx),
    exit_gap_pct_pos=float((gx['gap'] > 0).mean() * 100),
)
out_n2['entry_by_year'] = {y: dict(mean=float(g['gap'].mean() * 100), n=len(g))
                           for y, g in ge.groupby(ge.index.year)}
out_n2['exit_by_year'] = {y: dict(mean=float(g['gap'].mean() * 100), n=len(g))
                          for y, g in gx.groupby(gx.index.year)}
# 解读: 若entry_gap_sum为负 → 现制(次日开盘买)在买入侧省下隔夜跌幅 → 尾盘买=亏;
#       若为正 → 尾盘买更优(信号日隔夜常跳高)。
print(f"[N2] entry gap: mean={out_n2['entry_gap_mean']:+.2f}% sum(log)={out_n2['entry_gap_sum']:+.1f}% "
      f"pos={out_n2['entry_gap_pct_pos']:.0f}% n={out_n2['entry_gap_n']}", flush=True)
print(f"[N2] exit  gap: mean={out_n2['exit_gap_mean']:+.2f}% sum(log)={out_n2['exit_gap_sum']:+.1f}% "
      f"pos={out_n2['exit_gap_pct_pos']:.0f}% n={out_n2['exit_gap_n']}", flush=True)
print(f"[N2] entry by year: " + ' '.join(f"{y}:{'%.2f'%v['mean']}%" for y, v in sorted(out_n2['entry_by_year'].items())), flush=True)

OUT = dict(c2=out_c2, c1=out_c1, n1=out_n1, d0=out_0d, n2=out_n2)
with open(os.path.join(RVD, 'probe_optbatch1_20260917.pkl'), 'wb') as f:
    pickle.dump(OUT, f)
print(f"=== 完成 {(pd.Timestamp.now()-t0).total_seconds():.0f}s → {os.path.join(RVD,'probe_optbatch1_20260917.pkl')} ===")
