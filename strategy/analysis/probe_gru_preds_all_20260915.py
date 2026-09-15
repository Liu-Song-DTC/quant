"""P1-3b前置: GRU32三种子全日期(2025+)预测落盘, 供信号层探针用 (2026-09-15)

复用深探针的特征构建与训练协议(train≤2024-12-17, 3种子), 预测val+test全部日期。
产出: rolling_validation_results/gru_preds_2025_2026.pkl {code, date, y, pred_mean, pred_std}
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from probe_sequence_deep_20260915 import build_features, SeqNet, train_model

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
SEEDS = [42, 7, 2024]


def main():
    torch.set_num_threads(10)
    Xz, X, y, D, C, tr, va, te = build_features()
    Xt_t = torch.from_numpy(Xz[tr]); yt_t = torch.from_numpy(y[tr])
    Xv_t = torch.from_numpy(Xz[va]); yv_t = torch.from_numpy(y[va])
    Xe_t = torch.from_numpy(Xz[te])
    all_idx = np.where(va | te)[0]
    Xa_t = torch.from_numpy(Xz[all_idx])

    preds = []
    for s in SEEDS:
        m = train_model(SeqNet(6, 32, 'gru'), Xt_t, yt_t, Xv_t, yv_t, s)
        with torch.no_grad():
            preds.append(m(Xa_t).numpy())
    p = np.stack(preds)
    print(f'预测 {p.shape[1]} 行 (val+test), 日期 {D[all_idx].min()}~{D[all_idx].max()}', flush=True)

    df = pd.DataFrame({
        'code': C[all_idx],
        'date': pd.to_datetime(D[all_idx]),
        'y': y[all_idx],
        'pred_mean': p.mean(0),
        'pred_std': p.std(0),
    })
    df.to_pickle(os.path.join(OUT, 'gru_preds_2025_2026.pkl'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
