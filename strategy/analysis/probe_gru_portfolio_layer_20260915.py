"""E-seq1b: 组合层诊断 — GRU blend为何在信号层强(+0.052)组合层却灾难(-333k)? (2026-09-15)

假设: (a) 组合只吃buy候选池的top-k极端尾部, GRU在尾部增量≈噪声;
      (b) 全行注入(含持仓行)扰乱出场判定 → churn/MDD;
      (c) w=0.32注入方差≈score方差 → 排序翻转过度。
判据: 在"组合实际候选池"(每调仓日buy信号集)和"实际选中集"上,
      IC(incumbent score, fwd10) vs IC(blend, fwd10), top-k配对均值差。
产出: rolling_validation_results/gru_portfolio_layer_probe.pkl
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq
from scipy import stats as _st

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
WS = [0.08, 0.16, 0.32]


def main():
    sig = pd.read_csv(os.path.join(OUT, 'backtest_signals.csv'),
                      dtype={'code': str}, low_memory=False,
                      usecols=['code', 'date', 'buy', 'score'])
    sig['date'] = pd.to_datetime(sig['date'])
    sig['code'] = sig['code'].str.zfill(6)
    g = pd.read_pickle(os.path.join(OUT, 'gru_panelchain2_daily.pkl'))
    gdf = pd.DataFrame({'code': pd.Series(g['code']).astype(str).str.zfill(6),
                        'date': pd.to_datetime(g['date']), 'z': np.clip(g['z'], -3, 3)})
    sel = pd.read_csv(os.path.join(OUT, 'portfolio_selections.csv'),
                      dtype={'code': str}, low_memory=False)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['code'] = sel['code'].str.zfill(6)
    # 选中集的入场事件(每股首次出现日近似入场日)
    first_entry = sel.groupby('code')['date'].min()
    entry_rows = sel.drop_duplicates('code')[['code']].copy()
    entry_rows['entry_date'] = first_entry.values

    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'future_ret'])
    lab = t.to_pandas()
    lab['code'] = lab['code'].astype(str).str.zfill(6)
    lab['date'] = pd.to_datetime(lab['date'])

    # 1) 实际选中入场股的 z 是否预示好坏? (入场日与未来10日收益)
    er = entry_rows.merge(lab, left_on=['code', 'entry_date'], right_on=['code', 'date'],
                          how='left').dropna(subset=['future_ret'])
    er = er.merge(gdf, left_on=['code', 'entry_date'], right_on=['code', 'date'],
                  how='left', suffixes=('', '_g'))
    z_ok = er.dropna(subset=['z'])
    rho_sel = _st.spearmanr(z_ok.z, z_ok.future_ret)[0] if len(z_ok) > 50 else np.nan
    print(f'[选中入场] n={len(z_ok)} rho(z, fwd10)={rho_sel:+.4f}', flush=True)

    # 2) 每调仓日候选池(buy信号)上的IC对比 + top-k配对
    buys = sig[(sig.buy == True) & (sig.date >= '2025-01-01')].merge(
        gdf, on=['code', 'date'], how='left').merge(lab, on=['code', 'date'], how='left')
    buys = buys.dropna(subset=['future_ret'])
    # 调仓日 = portfolio_selections的日期
    rebal_days = sel['date'].unique()
    rows = []
    for d in rebal_days:
        sub = buys[buys.date == d]
        if len(sub) < 5:
            continue
        rho_inc = _st.spearmanr(sub.score, sub.future_ret)[0]
        k = min(8, len(sub))
        for w in WS:
            sub['bl'] = sub.score + w * sub.z
            rho_bl = _st.spearmanr(sub.bl, sub.future_ret)[0]
            top_inc = sub.nlargest(k, 'score').future_ret.mean()
            top_bl = sub.nlargest(k, 'bl').future_ret.mean()
            rows.append({'date': d, 'w': w, 'n': len(sub),
                         'ic_inc': rho_inc, 'ic_blend': rho_bl, 'ic_delta': rho_bl - rho_inc,
                         'topk_inc': top_inc, 'topk_bl': top_bl,
                         'topk_delta': top_bl - top_inc})
    R = pd.DataFrame(rows)
    print('\n[候选池诊断] 按w分组 (调仓日配对):', flush=True)
    for w in WS:
        rw = R[R.w == w]
        print(f'  w={w}: 日数={len(rw)} IC_inc={rw.ic_inc.mean():+.4f} '
              f'IC_blend={rw.ic_blend.mean():+.4f} IC_delta={rw.ic_delta.mean():+.4f} '
              f'(正率{100*(rw.ic_delta>0).mean():.0f}%) | topk_delta={rw.topk_delta.mean()*100:+.2f}pp '
              f'(正率{100*(rw.topk_delta>0).mean():.0f}%)', flush=True)
    # 3) 2026单独
    R26 = R[R.date >= '2026-01-01']
    print('\n[2026子集]:', flush=True)
    for w in WS:
        rw = R26[R26.w == w]
        if len(rw):
            print(f'  w={w}: IC_delta={rw.ic_delta.mean():+.4f} topk_delta={rw.topk_delta.mean()*100:+.2f}pp '
                  f'(正率{100*(rw.topk_delta>0).mean():.0f}%)', flush=True)
    pd.to_pickle(R, os.path.join(OUT, 'gru_portfolio_layer_probe.pkl'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
