"""E-seq1a前置: 逐日GRU预测构建 (2026-09-15)

背景: 信号层探针过门(GRU@buy子集 IC+0.0598/残差+0.0588/与adj正交ρ0.077),
  排队E-seq1机制实验。探针用因子面板(~2日采样)日期; 生产signals逐日,
  机制接入需全交易日预测。
本脚本: 特征30d×6自qfq CSV逐交易日构建, 标签=CSV fwd10 close/close
  (identity烟测已证与parquet.future_ret一致), 训练2021-01~2024-12-17
  (fwd10 purge), 3种子GRU32, 预测2025-01-02~2026-09-14, 逐日截面z。
复核: 与面板日预测(pkl)一致性 + 测试/验证期月度IC对照探针水平。
产出: rolling_validation_results/gru_daily_preds.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from probe_sequence_deep_20260915 import SeqNet, train_model, monthly_ic

PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
CSV_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
OUT = '/mnt/d/quant/strategy/rolling_validation_results'
LEN = 30
FEATS = ['ret1', 'vol_ratio', 'amplitude', 'rng_pos', 'ret5', 'mom20']
SEEDS = [42, 7, 2024]
TRAIN_START = np.datetime64('2021-01-01')
TRAIN_END = np.datetime64('2024-12-17')      # fwd10 purge
PRED_START = np.datetime64('2025-01-01')


def build_daily():
    t = pq.read_table(PQ_PATH, columns=['code'])
    codes = sorted(set(t.to_pandas()['code'].astype(str).str.zfill(6)))
    xs, ys, ds, cs = [], [], [], []
    for i, c in enumerate(codes):
        p = os.path.join(CSV_DIR, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            k = pd.read_csv(p, parse_dates=['datetime'], usecols=[
                'datetime', 'open', 'high', 'low', 'close', 'volume']).set_index('datetime')
        except Exception:
            continue
        k = k[~k.index.duplicated(keep='last')].sort_index()
        k['ret1'] = k['close'].pct_change()
        k['vol_ratio'] = k['volume'] / k['volume'].rolling(20).mean()
        k['amplitude'] = (k['high'] - k['low']) / k['close']
        k['rng_pos'] = ((k['close'] - k['low']) / (k['high'] - k['low']).replace(0, np.nan)).fillna(0.5)
        k['ret5'] = k['close'].pct_change(5)
        k['mom20'] = k['close'] / k['close'].shift(20) - 1
        k['vol_ratio'] = k['vol_ratio'].fillna(1.0)
        k = k[k.index >= TRAIN_START]  # 2021+ 序列起点(特征已含历史窗口)
        fe = k[FEATS].values.astype(np.float32)
        # 标签: fwd10 close/close (与parquet future_ret identity一致)
        lab = (k['close'].shift(-10) / k['close'] - 1).values.astype(np.float32)
        n = len(k)
        if n < LEN + 12:
            continue
        idx = np.arange(LEN - 1, n)
        X = np.stack([fe[j - LEN + 1: j + 1] for j in idx])
        yv = lab[idx]
        xs.append(X); ys.append(yv)
        ds.append(k.index[idx])
        cs.append(np.repeat(c, len(idx)))
        if (i + 1) % 400 == 0:
            print(f'  {i+1}/{len(codes)} 只, 累计 {sum(len(x) for x in xs)/1e6:.2f}M 样本', flush=True)
    X = np.concatenate(xs); y = np.concatenate(ys)
    D = np.concatenate(ds); C = np.concatenate(cs)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=(1, 2))
    X, y, D, C = X[ok], y[ok], D[ok], C[ok]
    tr = (D >= TRAIN_START) & (D <= TRAIN_END)
    pr = D >= PRED_START
    mu = X[tr].reshape(-1, X.shape[2]).mean(0)
    sd = X[tr].reshape(-1, X.shape[2]).std(0)
    Xz = (X - mu) / np.where(sd < 1e-8, 1, sd)
    print(f'样本 {len(y)}, train={tr.sum()} pred2025+={pr.sum()}', flush=True)
    return Xz, y, D, C, tr, pr


def main():
    torch.set_num_threads(10)
    t0 = time.time()
    print('=== 逐日特征构建 ===', flush=True)
    Xz, y, D, C, tr, pr = build_daily()
    print(f'特征构建 {time.time()-t0:.0f}s', flush=True)
    Xp_t = torch.from_numpy(Xz[pr])

    print('=== 3种子GRU32训练 (train 2021-2024) ===', flush=True)
    # 早停val = 训练期固定种子5%留出 (机制模型不得窥视2025+, 探针用2025做早停val是
    # 探针协议而非机制协议)
    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    val_mask = rng.random(len(tr_idx)) < 0.05
    Xt_t = torch.from_numpy(Xz[tr_idx[~val_mask]]); yt_t = torch.from_numpy(y[tr_idx[~val_mask]])
    Xv_t = torch.from_numpy(Xz[tr_idx[val_mask]]); yv_t = torch.from_numpy(y[tr_idx[val_mask]])
    models, preds = {}, []
    for s in SEEDS:
        m = train_model(SeqNet(6, 32, 'gru'), Xt_t, yt_t, Xv_t, yv_t, s)
        with torch.no_grad():
            preds.append(m(Xp_t).numpy())
        models[f'seed{s}'] = {k: v.clone().cpu() for k, v in m.state_dict().items()}
        print(f'  seed{s} 完成 {time.time()-t0:.0f}s', flush=True)
    p = np.stack(preds)
    pred_mean = p.mean(0)

    # 逐日截面z (全截面, 机制实盘同款: 收盘后截面可知)
    df = pd.DataFrame({'code': C[pr], 'date': pd.to_datetime(D[pr]),
                       'y': y[pr], 'pred_mean': pred_mean})
    df['gru_z'] = df.groupby('date')['pred_mean'].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12))

    print('=== 复核1: 月度IC (pred_mean vs fwd10) ===', flush=True)
    for yr in [2025, 2026]:
        sub = df[df.date.dt.year == yr]
        monthly_ic(sub['date'].values, sub['pred_mean'].values, sub['y'].values, f'daily@{yr}')

    print('=== 复核2: 与面板日预测一致性 ===', flush=True)
    g = pd.read_pickle(os.path.join(OUT, 'gru_preds_2025_2026.pkl'))
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['date'] = pd.to_datetime(g['date'])
    mrg = df[['code', 'date', 'pred_mean']].merge(
        g[['code', 'date', 'pred_mean']], on=['code', 'date'], how='inner',
        suffixes=('_daily', '_panel'))
    from scipy import stats as _st
    print(f'  共享日期行 {len(mrg)}: rho={_st.spearmanr(mrg.pred_mean_daily, mrg.pred_mean_panel)[0]:+.4f}')

    print('=== 复核3: 测试期IC (2026逐月) ===', flush=True)
    te = df[df.date >= '2026-01-01']
    monthly_ic(te['date'].values, te['pred_mean'].values, te['y'].values, 'daily@test', verbose=True)

    pd.to_pickle({'code': C[pr], 'date': pd.to_datetime(D[pr]), 'y': y[pr],
                  'pred_mean': pred_mean, 'gru_z': df['gru_z'].values,
                  'pred_std': p.std(0), 'models': models},
                 os.path.join(OUT, 'gru_daily_preds.pkl'))
    print(f'DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
