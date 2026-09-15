"""P1-3c决定性复核: 面板分辨率GRU的诚实早停版 (2026-09-15)

背景: 探针面板模型buy子集IC+0.060(2025-26), 但日频重建(v1随机val/v2时点val)
均仅+0.009~+0.011 — 信号在日频分辨率消失。两个候选解释:
  (A) 信号真实存在于面板分辨率(2日采样, 30行≈60日历日lookback) → 日频需加长窗口
  (D) 探针结果被早停val=2025的选择偏置夸大(2025选择+2026邻接自相关) → 方向关闭
本实验=决定性: 面板分辨率(与探针同构建/同切分), 唯一变量=早停val从2025改为
2024-10-16~2024-12-17面板日(时点val, 无2025+窥视)。3种子。
判据: 诚实版2026测试期|IC|≥0.04且buy子集IC≥0.03 → 信号真实(A), 日频加窗重建;
      否则(D)成立, P1-3关闭, 探针IC归因于选择偏置。
产出: rolling_validation_results/gru_panel_honest.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from probe_sequence_deep_20260915 import build_features, SeqNet, train_model, monthly_ic

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
SEEDS = [42, 7, 2024]
VAL_LO = np.datetime64('2024-10-16')
VAL_HI = np.datetime64('2024-12-17')


def buy_subset_ic(D, C, pred, y, label):
    """面板日buy子集IC (与信号层探针同判据)"""
    from scipy import stats as _st
    sig = pd.read_csv(os.path.join(OUT, 'backtest_signals.csv'),
                      dtype={'code': str}, low_memory=False, usecols=['code', 'date', 'buy'])
    sig['date'] = pd.to_datetime(sig['date'])
    sig['code'] = sig['code'].str.zfill(6)
    b = sig[(sig.buy == True) & (sig.date >= '2025-01-01')]
    d = pd.DataFrame({'code': pd.Series(C).astype(str).str.zfill(6),
                      'date': pd.to_datetime(D), 'p': pred, 'y': y})
    m = b.merge(d, on=['code', 'date'], how='inner').dropna(subset=['y'])
    rows = []
    for mo, sub in m.groupby(m.date.dt.to_period('M')):
        if len(sub) < 20:
            continue
        rows.append((str(mo), _st.spearmanr(sub.p, sub.y)[0]))
    r = pd.DataFrame(rows, columns=['m', 'ic'])
    print(f'  [{label}] buy子集 n={len(m)} IC={r.ic.mean():+.4f} '
          f'IR={r.ic.mean()/r.ic.std():+.2f} 正率={100*(r.ic>0).mean():.0f}%', flush=True)
    for _, row in r.iterrows():
        print(f'    {row.m}: IC={row.ic:+.4f}', flush=True)
    return r


def main():
    torch.set_num_threads(10)
    t0 = time.time()
    print('=== 面板特征构建 (探针同款) ===', flush=True)
    Xz, X, y, D, C, tr, va, te = build_features()
    print(f'构建 {time.time()-t0:.0f}s', flush=True)

    tr_idx = np.where(tr)[0]
    val_sel = (D[tr_idx] > VAL_LO) & (D[tr_idx] <= VAL_HI)
    print(f'时点val(面板日): {val_sel.sum()} 行, train {int((~val_sel).sum())}', flush=True)
    Xt_t = torch.from_numpy(Xz[tr_idx[~val_sel]]); yt_t = torch.from_numpy(y[tr_idx[~val_sel]])
    Xv_t = torch.from_numpy(Xz[tr_idx[val_sel]]); yv_t = torch.from_numpy(y[tr_idx[val_sel]])

    all_idx = np.where(va | te)[0]
    Xa_t = torch.from_numpy(Xz[all_idx])
    seed_preds = {}
    for s in SEEDS:
        m = train_model(SeqNet(6, 32, 'gru'), Xt_t, yt_t, Xv_t, yv_t, s)
        with torch.no_grad():
            p = m(Xa_t).numpy()
        seed_preds[f'seed{s}'] = p
        va_mask_all = va[all_idx]
        te_mask_all = te[all_idx]
        monthly_ic(D[all_idx][va_mask_all], p[va_mask_all], y[all_idx][va_mask_all], f'seed{s}@val2025')
        monthly_ic(D[all_idx][te_mask_all], p[te_mask_all], y[all_idx][te_mask_all], f'seed{s}@test2026')
        print(f'  seed{s} 完成 {time.time()-t0:.0f}s', flush=True)

    p = np.stack([seed_preds[f'seed{s}'] for s in SEEDS])
    pred_mean = p.mean(0)
    d = pd.DataFrame({'code': pd.Series(C[all_idx]).astype(str).str.zfill(6),
                      'date': pd.to_datetime(D[all_idx]), 'y': y[all_idx], 'p': pred_mean})
    d['z'] = d.groupby('date')['p'].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))

    print('\n=== 3种子均值: pool IC ===', flush=True)
    va_mask_all = va[all_idx]
    te_mask_all = te[all_idx]
    monthly_ic(D[all_idx][va_mask_all], pred_mean[va_mask_all], y[all_idx][va_mask_all], 'honest@val2025')
    monthly_ic(D[all_idx][te_mask_all], pred_mean[te_mask_all], y[all_idx][te_mask_all], 'honest@test2026')

    print('\n=== 3种子均值: buy子集IC ===', flush=True)
    for yr, mm in [(2025, d.date.dt.year == 2025), (2026, d.date.dt.year == 2026)]:
        sub = d[mm]
        buy_subset_ic(sub.date.values, sub.code.values, sub.p.values, sub.y.values,
                      f'honest@{yr}')

    pd.to_pickle({'code': C[all_idx], 'date': pd.to_datetime(D[all_idx]), 'y': y[all_idx],
                  'pred_mean': pred_mean, 'z': d['z'].values,
                  'preds_per_seed': seed_preds},
                 os.path.join(OUT, 'gru_panel_honest.pkl'))
    print(f'DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
