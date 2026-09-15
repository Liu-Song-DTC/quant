"""E-seq1a前置v2: 逐日GRU — 时点val重训 (2026-09-15)

v1诊断: 随机5%训练留出做早停val → 日频模型buy子集IC仅+0.009(同117k面板日buy行,
面板模型+0.060)。主嫌疑=checkpoint选择: 面板探针用2025做早停val(对2025-26有选择
偏置); 随机留出val对"未来泛化"无选择压力。机制诚实协议=时点val: 用预测窗口前
最近一段(2024-10-15~2024-12-17, 标签全落2024内)做早停, 无2025+窥视。
其余与v1逐位同协议(训练2021-2024, 3种子GRU32, epochs40/patience5), 单一变量=val。
产出: rolling_validation_results/gru_daily_preds_v2.pkl (含逐种子预测)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probe_sequence_deep_20260915 import SeqNet, train_model, monthly_ic
from probe_gru_daily_build_20260915 import build_daily

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
SEEDS = [42, 7, 2024]
VAL_LO = np.datetime64('2024-10-15')
VAL_HI = np.datetime64('2024-12-17')


def main():
    torch.set_num_threads(10)
    t0 = time.time()
    print('=== 逐日特征构建 (复用v1) ===', flush=True)
    Xz, y, D, C, tr, pr = build_daily()
    print(f'特征构建 {time.time()-t0:.0f}s', flush=True)

    tr_idx = np.where(tr)[0]
    val_sel = (D[tr_idx] > VAL_LO) & (D[tr_idx] <= VAL_HI)
    print(f'时点val: {val_sel.sum()} 行 ({pd.Timestamp(D[tr_idx][val_sel].min()).date()}~'
          f'{pd.Timestamp(D[tr_idx][val_sel].max()).date()}), train {int((~val_sel).sum())}', flush=True)
    Xt_t = torch.from_numpy(Xz[tr_idx[~val_sel]]); yt_t = torch.from_numpy(y[tr_idx[~val_sel]])
    Xv_t = torch.from_numpy(Xz[tr_idx[val_sel]]); yv_t = torch.from_numpy(y[tr_idx[val_sel]])
    Xp_t = torch.from_numpy(Xz[pr])

    print('=== 3种子GRU32训练 (时点val早停) ===', flush=True)
    preds, models = [], {}
    for s in SEEDS:
        m = train_model(SeqNet(6, 32, 'gru'), Xt_t, yt_t, Xv_t, yv_t, s)
        with torch.no_grad():
            preds.append(m(Xp_t).numpy())
        models[f'seed{s}'] = {k: v.clone().cpu() for k, v in m.state_dict().items()}
        print(f'  seed{s} 完成 {time.time()-t0:.0f}s', flush=True)
    p = np.stack(preds)
    pred_mean = p.mean(0)
    df = pd.DataFrame({'code': C[pr], 'date': pd.to_datetime(D[pr]),
                       'y': y[pr], 'pred_mean': pred_mean})
    df['gru_z'] = df.groupby('date')['pred_mean'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))

    print('=== 复核: 月度IC (pool) ===', flush=True)
    for yr in [2025, 2026]:
        sub = df[df.date.dt.year == yr]
        monthly_ic(sub['date'].values, sub['pred_mean'].values, sub['y'].values, f'dailyV2@{yr}')

    print('=== 复核: 与v1预测一致性 ===', flush=True)
    g1 = pd.read_pickle(os.path.join(OUT, 'gru_daily_preds.pkl'))
    g1df = pd.DataFrame({'code': pd.Series(g1['code']).astype(str).str.zfill(6),
                         'date': pd.to_datetime(g1['date']), 'z1': g1['gru_z']})
    mrg = df.merge(g1df, on=['code', 'date'], how='inner')
    from scipy import stats as _st
    print(f'  rho(v2 z, v1 z)={_st.spearmanr(mrg.gru_z, mrg.z1)[0]:+.4f} ({len(mrg)}行)', flush=True)

    pd.to_pickle({'code': C[pr], 'date': pd.to_datetime(D[pr]), 'y': y[pr],
                  'pred_mean': pred_mean, 'gru_z': df['gru_z'].values,
                  'preds_per_seed': {f'seed{s}': pp for s, pp in zip(SEEDS, preds)},
                  'models': models},
                 os.path.join(OUT, 'gru_daily_preds_v2.pkl'))
    print(f'DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
