"""P1-3深探针: 序列深度模型 消融+种子稳定性+XGB残差 (2026-09-15)

前置事实(identity烟测已核): parquet.future_ret = **fwd10** close/close (非fwd5).
预探针过门: GRU32 @2026测试 IC+0.084 IR+1.16 8/8月正, 压过点特征岭回归(+0.011).
本深探针回答三个问题:
  Q1 消融: 是"序列"还是"非线性"? Ridge6点 / Ridge60(Ridge最后10日展平) / Ridge180
     (30日展平) / MLP点(非线性无序列) / GRU32 / GRU64 / LSTM32
  Q2 稳定性: GRU32 × 3种子(42/7/2024), 测试IC均值±std
  Q3 决定性: 对生产特征工程XGB-lite(60 rank列, 同切分)的**残差IC** — 序列信号是否
     已被现有特征捕获? (缺口4同款设计: 残差IC≈0则关闭)
purge修正: train D ≤ 2024-12-17 (fwd10标签完整落2024内, 消除边界重叠)
判据: 测试期(2026, 8月) |IC|>0.03 且 正率≥5/8 且 残差IC(XGB后)>0.02 → 排队机制实验;
     否则 P1-3 关闭。
产出: rolling_validation_results/sequence_deep_probe.pkl
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

PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
CSV_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
OUT = '/mnt/d/quant/strategy/rolling_validation_results'
LEN = 30
FEATS = ['ret1', 'vol_ratio', 'amplitude', 'rng_pos', 'ret5', 'mom20']
SEEDS = [42, 7, 2024]
TRAIN_END = np.datetime64('2024-12-17')      # fwd10 purge: 标签落2024-12-31内
VAL = (np.datetime64('2025-01-01'), np.datetime64('2026-01-01'))
TE = np.datetime64('2026-01-01')


def build_features():
    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'future_ret'])
    lab = t.to_pandas()
    lab['code'] = lab['code'].astype(str).str.zfill(6)
    lab['date'] = pd.to_datetime(lab['date'])
    codes = sorted(lab['code'].unique())
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
        fe = k[FEATS]
        sub = lab[lab['code'] == c].set_index('date')['future_ret']
        common = fe.index.intersection(sub.index)
        if len(common) < LEN + 2:
            continue
        arr = fe.loc[common].values.astype(np.float32)
        yv = sub.loc[common].values.astype(np.float32)
        n = len(common)
        idx = np.arange(LEN - 1, n)
        X = np.stack([arr[j - LEN + 1: j + 1] for j in idx])
        xs.append(X); ys.append(yv[idx]); ds.append(common[idx])
        cs.append(np.repeat(c, len(idx)))
    X = np.concatenate(xs); y = np.concatenate(ys)
    D = np.concatenate(ds); C = np.concatenate(cs)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=(1, 2))
    X, y, D, C = X[ok], y[ok], D[ok], C[ok]
    tr = D <= TRAIN_END
    va = (D >= VAL[0]) & (D < VAL[1])
    te = D >= TE
    mu = X[tr].reshape(-1, X.shape[2]).mean(0)
    sd = X[tr].reshape(-1, X.shape[2]).std(0)
    Xz = (X - mu) / np.where(sd < 1e-8, 1, sd)
    print(f'样本 {len(y)}, train={tr.sum()} val={va.sum()} test={te.sum()}', flush=True)
    return Xz, X, y, D, C, tr, va, te


class SeqNet(nn.Module):
    def __init__(self, nf, nh, cell='gru'):
        super().__init__()
        if cell == 'gru':
            self.rnn = nn.GRU(nf, nh, batch_first=True)
        else:
            self.rnn = nn.LSTM(nf, nh, batch_first=True)
        self.out = nn.Linear(nh, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        if isinstance(h, tuple):
            h = h[0]
        return self.out(h[:, -1]).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, nf, nh=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, nh), nn.ReLU(),
                                 nn.Linear(nh, nh), nn.ReLU(),
                                 nn.Linear(nh, 1))

    def forward(self, x):
        return self.net(x[:, -1]).squeeze(-1)


def train_model(model, Xt, yt, Xv, yv, seed, lr=1e-3, epochs=40, bs=2048, patience=5):
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(Xt)
    best_val, best_state, pat = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), yv).item()
        if vl < best_val - 1e-6:
            best_val, pat = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def monthly_ic(dates, pred, y, label, verbose=True):
    from scipy import stats as _st
    df = pd.DataFrame({'d': pd.to_datetime(dates), 'p': pred, 'y': y})
    rows = []
    for m, g in df.groupby(df.d.dt.to_period('M')):
        if len(g) < 40:
            continue
        rows.append((str(m), len(g), _st.spearmanr(g['p'], g['y'])[0]))
    r = pd.DataFrame(rows, columns=['month', 'n', 'ic'])
    print(f'  [{label}] n月={len(r)} IC={r.ic.mean():+.4f} IR={r.ic.mean()/r.ic.std():+.2f} '
          f'正率={100*(r.ic>0).mean():.0f}%', flush=True)
    if verbose:
        for _, row in r.iterrows():
            print(f'    {row.month}: n={row.n} IC={row.ic:+.4f}', flush=True)
    return r


def main():
    torch.set_num_threads(10)
    print('=== 特征构建 ===', flush=True)
    Xz, X, y, D, C, tr, va, te = build_features()
    Xt, Xv, Xe = Xz[tr], Xz[va], Xz[te]
    yt, yv, ye = y[tr], y[va], y[te]
    Xt_t = torch.from_numpy(Xt); yt_t = torch.from_numpy(yt)
    Xv_t = torch.from_numpy(Xv); yv_t = torch.from_numpy(yv)
    Xe_t = torch.from_numpy(Xe)

    from sklearn.linear_model import Ridge
    print('\n=== Q1 消融 ===', flush=True)
    # Ridge6: 点特征线性
    r6 = Ridge(alpha=100.0).fit(Xz[tr][:, -1], yt)
    monthly_ic(D[te], r6.predict(Xz[te][:, -1]), ye, 'Ridge6@test')
    # Ridge60: 最后10日展平线性
    w10 = Xz[tr][:, -10:].reshape(len(Xz[tr]), -1)
    r60 = Ridge(alpha=100.0).fit(w10, yt)
    monthly_ic(D[te], r60.predict(Xz[te][:, -10:].reshape(len(Xz[te]), -1)), ye, 'Ridge60(近10d)@test')
    # Ridge180: 30日全展平线性
    r180 = Ridge(alpha=300.0).fit(Xz[tr].reshape(len(Xz[tr]), -1), yt)
    monthly_ic(D[te], r180.predict(Xz[te].reshape(len(Xz[te]), -1)), ye, 'Ridge180(30d)@test')

    results = {}
    t0 = time.time()
    # MLP点(非线性无序列)
    mlp = train_model(MLP(6), Xt_t, yt_t, Xv_t, yv_t, 42)
    with torch.no_grad():
        pe = mlp(Xe_t).numpy()
    monthly_ic(D[te], pe, ye, 'MLP点@test')
    results['mlp'] = pe
    # GRU32 × 3种子
    g32 = []
    for s in SEEDS:
        m = train_model(SeqNet(6, 32, 'gru'), Xt_t, yt_t, Xv_t, yv_t, s)
        with torch.no_grad():
            pe = m(Xe_t).numpy()
        g32.append(pe)
        monthly_ic(D[te], pe, ye, f'GRU32 s{s}@test')
        print(f'    (累计{time.time()-t0:.0f}s)', flush=True)
    pg = np.mean(g32, axis=0)
    monthly_ic(D[te], pg, ye, 'GRU32 3种子均值@test')
    results['gru32_mean'] = pg
    # GRU64
    m = train_model(SeqNet(6, 64, 'gru'), Xt_t, yt_t, Xv_t, yv_t, 42)
    with torch.no_grad():
        pe = m(Xe_t).numpy()
    monthly_ic(D[te], pe, ye, 'GRU64@test')
    results['gru64'] = pe
    # LSTM32
    m = train_model(SeqNet(6, 32, 'lstm'), Xt_t, yt_t, Xv_t, yv_t, 42)
    with torch.no_grad():
        pe = m(Xe_t).numpy()
    monthly_ic(D[te], pe, ye, 'LSTM32@test')
    results['lstm32'] = pe

    # 3种子预测逐月IC分布(稳定性)
    print('\n=== Q2 种子稳定性: GRU32三种子逐月IC ===', flush=True)
    from scipy import stats as _st
    dfe = pd.DataFrame({'d': pd.to_datetime(D[te]), 'y': ye})
    for s, p in zip(SEEDS, g32):
        dfe[f'p{s}'] = p
    for m, g in dfe.groupby(dfe.d.dt.to_period('M')):
        if len(g) < 40:
            continue
        ics = [_st.spearmanr(g[f'p{s}'], g['y'])[0] for s in SEEDS]
        print(f'  {m}: ICs={["%+.4f" % v for v in ics]} std={np.std(ics):.4f}', flush=True)

    print('\n=== Q3 XGB-lite残差检验 (60 rank列, 同切分) ===', flush=True)
    t = pq.read_table(PQ_PATH)
    df = t.to_pandas()
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])
    rank_cols = [c for c in df.columns if c.endswith('_rank')]
    print(f'  rank列 {len(rank_cols)} 个', flush=True)
    key = pd.DataFrame({'code': C, 'date': pd.to_datetime(D)})
    mrg = key.merge(df[['code', 'date', 'future_ret'] + rank_cols], on=['code', 'date'], how='inner')
    assert len(mrg) == len(key), f'合并丢失 {len(key)-len(mrg)} 行'
    Fx = mrg[rank_cols].values.astype(np.float32)
    import xgboost as xgb
    t0 = time.time()
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.6, tree_method='hist',
                             n_jobs=10, random_state=42)
    model.fit(Fx[tr], yt)
    print(f'  XGB训练 {time.time()-t0:.0f}s', flush=True)
    pv = model.predict(Fx[va]); pe = model.predict(Fx[te])
    monthly_ic(D[va], pv, yv, 'XGB-lite@val')
    monthly_ic(D[te], pe, ye, 'XGB-lite@test')
    # 残差IC: GRU32均值预测 vs (y - XGB预测)
    from scipy import stats as _st
    resid = ye - pe
    dfe = pd.DataFrame({'d': pd.to_datetime(D[te]), 'pg': pg, 'r': resid})
    rows = []
    for m, g in dfe.groupby(dfe.d.dt.to_period('M')):
        if len(g) < 40:
            continue
        rows.append((str(m), _st.spearmanr(g['pg'], g['r'])[0]))
    rr = pd.DataFrame(rows, columns=['month', 'ic'])
    print(f'  [GRU vs XGB残差@test] n月={len(rr)} IC={rr.ic.mean():+.4f} '
          f'IR={rr.ic.mean()/rr.ic.std():+.2f} 正率={100*(rr.ic>0).mean():.0f}%', flush=True)
    for _, row in rr.iterrows():
        print(f'    {row.month}: IC={row.ic:+.4f}', flush=True)
    # blend 0.5/0.5
    blend = (pg - pg.mean()) / pg.std() + (pe - pe.mean()) / pe.std()
    monthly_ic(D[te], blend, ye, 'blend(GRU+XGB)@test')

    pd.to_pickle({'D': D, 'C': C, 'y': y, 'tr': tr, 'va': va, 'te': te,
                  'gru32_mean': pg, 'xgb_test': pe, 'resid_ic': rr},
                 os.path.join(OUT, 'sequence_deep_probe.pkl'))
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
