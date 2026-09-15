"""P1-3b: GRU得分在生产buy信号子集上的信号层IC (2026-09-15)

问题: GRU池IC(测试期+0.062)是否到达信号层? 在生产backtest_signals.csv的
  buy=True子集(2025-2026)上, GRU rank是否对future_ret有序? 与incumbent对照。
覆盖说明: 生产signals为逐日; 因子面板(parquet)为~2日采样且future_ret止于
  2026-08-31(fwd10尾部), 故GRU预测只落在面板日期 — 覆盖≈50%buy信号,
  该子集与奇偶交易日无关, 是无偏抽样, IC评估有效。
判据: buy子集月度IC(GRU) |mean|>0.03 且 正率≥60% → 排队E-seq1机制实验;
  残差检验: 剥离incumbent adjusted_score后GRU残差IC>0.02 → 增量确认;
  否则降级为"截面强≠组合可用"第六例。
产出: rolling_validation_results/gru_signal_layer_probe.pkl
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as _st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
SIG = os.path.join(OUT, 'backtest_signals.csv')
GRU = os.path.join(OUT, 'gru_preds_2025_2026.pkl')
PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'


def monthly_ic(df, col, y='future_ret', label='', show_months=False):
    rows = []
    for m, g in df.groupby(pd.to_datetime(df.date).dt.to_period('M')):
        if len(g) < 20:
            continue
        rows.append((str(m), len(g), _st.spearmanr(g[col], g[y])[0]))
    r = pd.DataFrame(rows, columns=['month', 'n', 'ic'])
    print(f'  [{label}] n月={len(r)} IC={r.ic.mean():+.4f} IR={r.ic.mean()/r.ic.std():+.2f} '
          f'正率={100*(r.ic>0).mean():.0f}% n总={df[col].notna().sum()}', flush=True)
    if show_months:
        for _, row in r.iterrows():
            print(f'    {row.month}: n={row.n} IC={row.ic:+.4f}', flush=True)
    return r


def residual_ic(df, pred_col, base_cols, y='future_ret', label=''):
    """月内 y 对 base_cols 回归取残差, 看 pred 对残差的IC"""
    rows = []
    for mo, g in df.groupby(pd.to_datetime(df.date).dt.to_period('M')):
        if len(g) < 20:
            continue
        yv = g[y].values.astype(float)
        X = np.column_stack([np.ones(len(yv))] + [g[c].values.astype(float) for c in base_cols])
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        rows.append((str(mo), _st.spearmanr(g[pred_col].values, resid)[0]))
    r = pd.DataFrame(rows, columns=['month', 'ic'])
    print(f'  [{label}] n月={len(r)} IC={r.ic.mean():+.4f} '
          f'IR={r.ic.mean()/r.ic.std():+.2f} 正率={100*(r.ic>0).mean():.0f}%', flush=True)
    for _, row in r.iterrows():
        print(f'    {row.month}: IC={row.ic:+.4f}', flush=True)
    return r


def quintiles(df, col, label):
    """月内按col五分位 → future_ret均值/胜率"""
    d = df.copy()
    d['q'] = d.groupby(pd.to_datetime(d.date).dt.to_period('M'))[col].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
    print(f'  [{label} 月内五分位 → fwd10均值/胜率]', flush=True)
    res = {}
    for q in range(5):
        sub = d[d.q == q]
        res[q] = (len(sub), sub.future_ret.mean(), (sub.future_ret > 0).mean())
        print(f'    Q{q+1}: n={len(sub):>7} 均值={100*sub.future_ret.mean():+.2f}% '
              f'胜率={100*(sub.future_ret>0).mean():.0f}%', flush=True)
    return res


def main():
    # --- 读取: 生产signals(新schema 40列, 无future_ret) ---
    sig = pd.read_csv(SIG, parse_dates=['date'], dtype={'code': str}, low_memory=False)
    buys = sig[(sig.buy == True) & (sig.date >= '2025-01-01')].copy()
    buys['code'] = buys['code'].str.zfill(6)
    print(f'生产buy信号 2025+ : {len(buys)} 条, {buys.code.nunique()} 只, '
          f'{buys.date.min().date()}~{buys.date.max().date()}', flush=True)

    # --- GRU预测(只落面板日期) ---
    g = pd.read_pickle(GRU)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['date'] = pd.to_datetime(g['date'])
    m = buys.merge(g[['code', 'date', 'pred_mean', 'pred_std']], on=['code', 'date'], how='inner')
    print(f'GRU预测覆盖 buy信号: {len(m)}/{len(buys)} = {100*len(m)/len(buys):.0f}% '
          f'(面板~2日采样, 无偏)', flush=True)

    # --- future_ret 从因子面板补入(生产signals无此列) ---
    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'future_ret'])
    lab = t.to_pandas()
    lab['code'] = lab['code'].astype(str).str.zfill(6)
    lab['date'] = pd.to_datetime(lab['date'])
    m = m.merge(lab, on=['code', 'date'], how='inner')
    m = m.dropna(subset=['future_ret', 'pred_mean'])
    m = m[m['adjusted_score'].notna() & m['score'].notna()]
    print(f'有效 {len(m)} 条, {m.date.min().date()}~{m.date.max().date()}, '
          f'{m.code.nunique()} 只', flush=True)

    # --- 信号层月度IC ---
    print('\n=== 信号层IC: buy子集 (2025-2026, 月度Spearman) ===', flush=True)
    ic_gru = monthly_ic(m, 'pred_mean', label='GRU@buy子集', show_months=True)
    ic_adj = monthly_ic(m, 'adjusted_score', label='incumbent adjusted_score@buy子集')
    ic_raw = monthly_ic(m, 'score', label='incumbent 原始score@buy子集')
    ic_ml = monthly_ic(m, 'ml_score', label='incumbent ml_score@buy子集')

    # --- 年度拆分: 2025 vs 2026 ---
    print('\n=== 年度拆分 (GRU) ===', flush=True)
    for yr in [2025, 2026]:
        sub = m[m.date.dt.year == yr]
        monthly_ic(sub, 'pred_mean', label=f'GRU@{yr}')
        monthly_ic(sub, 'adjusted_score', label=f'adj@{yr}')

    # --- 残差: 剥离adjusted_score后GRU增量 ---
    print('\n=== 信号层残差: 剥离adjusted_score后 GRU ===', flush=True)
    rr = residual_ic(m, 'pred_mean', ['adjusted_score'], label='GRU vs adj残差')
    print('=== 信号层残差: 剥离 adjusted_score+ml_score 后 GRU ===', flush=True)
    rr2 = residual_ic(m, 'pred_mean', ['adjusted_score', 'ml_score'], label='GRU vs adj+ml残差')

    # --- 分桶 ---
    print('\n=== 分桶 (月内五分位) ===', flush=True)
    q_gru = quintiles(m, 'pred_mean', 'GRU')
    q_adj = quintiles(m, 'adjusted_score', 'adjusted_score(incumbent)')

    # --- 组合信号: adjusted_score + GRU排名等权z ---
    print('\n=== blend: z(adjusted_score) + z(GRU) ===', flush=True)
    bd = m.copy()
    bd['z_adj'] = bd.groupby(bd.date.dt.to_period('M'))['adjusted_score'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))
    bd['z_gru'] = bd.groupby(bd.date.dt.to_period('M'))['pred_mean'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))
    bd['blend'] = bd['z_adj'] + bd['z_gru']
    ic_blend = monthly_ic(bd, 'blend', label='blend(adj+GRU)@buy子集', show_months=True)

    # --- 双变量对比: 月内同时rank adj与gru的增量(交叉分桶) ---
    print('\n=== 交叉分桶: adj五分位×GRU五分位 → fwd10均值% ===', flush=True)
    bd['qa'] = bd.groupby(bd.date.dt.to_period('M'))['adjusted_score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
    bd['qg'] = bd.groupby(bd.date.dt.to_period('M'))['pred_mean'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
    pivot = bd.pivot_table(index='qa', columns='qg', values='future_ret', aggfunc='mean')
    print(pivot.to_string(float_format=lambda v: f'{100*v:+.2f}%'))

    pd.to_pickle({'merged': m, 'ic_gru': ic_gru, 'ic_adj': ic_adj, 'ic_raw': ic_raw,
                  'ic_ml': ic_ml, 'ic_blend': ic_blend, 'resid_adj': rr, 'resid_adj_ml': rr2,
                  'q_gru': q_gru, 'q_adj': q_adj},
                 os.path.join(OUT, 'gru_signal_layer_probe.pkl'))
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
