"""P1-3: 序列深度模型预探针 (2026-09-15)

问题: 原始OHLCV序列(30日窗口)是否含有现有131列特征工程未捕获的信息?
  这是"底层分析"最后方向: 深模型若连预探针门都过不了, P1-3方向关闭。

方法:
  1. 全池2689只 qfq CSV → 6个原始特征序列 (ret1/vol_ratio/amplitude/rng_pos/ret5/mom20)
  2. 标签 = parquet.future_ret (fwd5, ML同款), PIT干净: 序列止于t, 标签t→t+5
  3. 时间切分(铁律): 训练≤2024-12-24(purge: 标签完整落设计期), 验证=2025, 测试=2026
  4. 模型: GRU(hidden=32, 1层) CPU可训; 对照1=岭回归点特征(同6特征仅t日, 无序列);
     对照2=mom20单特征(朴素地板)
  5. 测度: 月度截面Spearman IC(pred, future_ret)
判据(测试期2026, n≥8月): |IC|>0.03 且 正率≥5/8 → 有信号值得排队深探;
  且需明显优于岭回归点特征对照(否则序列本身无增量)。不过 → 关闭P1-3。
产出: rolling_validation_results/sequence_probe.pkl
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
LEN = 30          # 序列窗口(交易日)
SEED = 42
FEATS = ['ret1', 'vol_ratio', 'amplitude', 'rng_pos', 'ret5', 'mom20']


def load_pool_labels():
    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'future_ret'])
    df = t.to_pandas()
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])
    return df


def build_features():
    """每只股: qfq CSV → 6特征序列; 与标签inner merge后构造(LEN, F)样本."""
    lab = load_pool_labels()
    codes = sorted(lab['code'].unique())
    xs, ys, ds, cs = [], [], [], []
    t0 = time.time()
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
        # 一字板(high==low)日: 位置中性0.5, 不丢样本(涨停板是强信号, 丢=偏置)
        k['rng_pos'] = ((k['close'] - k['low']) / (k['high'] - k['low']).replace(0, np.nan)).fillna(0.5)
        k['ret5'] = k['close'].pct_change(5)
        k['mom20'] = k['close'] / k['close'].shift(20) - 1
        k['vol_ratio'] = k['vol_ratio'].fillna(1.0)
        fe = k[FEATS]
        # 与标签对齐: 标签在t, 序列取t-LEN+1..t
        sub = lab[lab['code'] == c].set_index('date')['future_ret']
        common = fe.index.intersection(sub.index)
        if len(common) < LEN + 2:
            continue
        arr = fe.loc[common].values.astype(np.float32)
        yv = sub.loc[common].values.astype(np.float32)
        n = len(common)
        idx = np.arange(LEN - 1, n)
        X = np.stack([arr[j - LEN + 1: j + 1] for j in idx])  # (n-LEN+1, LEN, F)
        xs.append(X); ys.append(yv[idx]); ds.append(common[idx])
        cs.append(np.repeat(c, len(idx)))
        if (i + 1) % 500 == 0:
            print(f'  特征构建 {i+1}/{len(codes)} {time.time()-t0:.0f}s', flush=True)
    X = np.concatenate(xs); y = np.concatenate(ys)
    D = np.concatenate(ds); C = np.concatenate(cs)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=(1, 2))
    print(f'剔除NaN样本 {1 - ok.mean():.1%} (尾部标签+异常)', flush=True)
    return X[ok], y[ok], D[ok], C[ok]


def split(D, y):
    tr = D < np.datetime64('2024-12-25')      # purge: fwd5标签落2024内
    va = (D >= np.datetime64('2025-01-01')) & (D < np.datetime64('2026-01-01'))
    te = D >= np.datetime64('2026-01-01')
    return tr, va, te


def zscore(X, mu, sd):
    return (X - mu) / np.where(sd < 1e-8, 1, sd)


class GRU1(nn.Module):
    def __init__(self, nf, nh=32):
        super().__init__()
        self.gru = nn.GRU(nf, nh, batch_first=True)
        self.out = nn.Linear(nh, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h[:, -1]).squeeze(-1)


def monthly_ic(dates, pred, y, label):
    """月度截面Spearman IC. 返回(mean_ic, 正率, 月数, 各月)."""
    from scipy import stats as _st
    df = pd.DataFrame({'d': pd.to_datetime(dates), 'p': pred, 'y': y})
    rows = []
    for m, g in df.groupby(df.d.dt.to_period('M')):
        if len(g) < 40:
            continue
        ic = _st.spearmanr(g['p'], g['y'])[0]
        rows.append((str(m), len(g), ic))
    r = pd.DataFrame(rows, columns=['month', 'n', 'ic'])
    print(f'  [{label}] n月={len(r)} IC均值={r.ic.mean():+.4f} IR={r.ic.mean()/r.ic.std():+.2f} '
          f'正率={100*(r.ic>0).mean():.0f}%', flush=True)
    for _, row in r.iterrows():
        print(f'    {row.month}: n={row.n} IC={row.ic:+.4f}', flush=True)
    return r


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(10)
    print('=== 特征构建 ===', flush=True)
    X, y, D, C = build_features()
    print(f'样本 {len(y)} 行, 股票 {len(np.unique(C))} 只, 日期 {D.min()}~{D.max()}', flush=True)
    tr, va, te = split(D, y)
    print(f'切分: train={tr.sum()} val={va.sum()} test={te.sum()}', flush=True)

    mu = X[tr].reshape(-1, X.shape[2]).mean(0)
    sd = X[tr].reshape(-1, X.shape[2]).std(0)
    Xz = zscore(X, mu, sd)
    Xt, Xv, Xe = Xz[tr], Xz[va], Xz[te]
    yt, yv, ye = y[tr], y[va], y[te]

    # 对照2: mom20朴素地板
    print('\n=== 对照2: mom20朴素 ===', flush=True)
    pred2_tr = X[tr][:, -1, 5]; pred2_va = X[va][:, -1, 5]; pred2_te = X[te][:, -1, 5]
    monthly_ic(D[va], pred2_va, yv, 'mom20@val')
    monthly_ic(D[te], pred2_te, ye, 'mom20@test')

    # 对照1: 岭回归点特征(仅t日6特征, 无序列)
    print('\n=== 对照1: 岭回归点特征 ===', flush=True)
    from sklearn.linear_model import Ridge
    Pt, Pv, Pe = Xz[tr][:, -1], Xz[va][:, -1], Xz[te][:, -1]
    ridge = Ridge(alpha=100.0).fit(Pt, yt)
    monthly_ic(D[va], ridge.predict(Pv), yv, 'ridge@val')
    monthly_ic(D[te], ridge.predict(Pe), ye, 'ridge@test')

    # GRU
    print('\n=== GRU(hidden=32) 训练 ===', flush=True)
    model = GRU1(X.shape[2], 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xt_t = torch.from_numpy(Xt); yt_t = torch.from_numpy(yt)
    Xv_t = torch.from_numpy(Xv); yv_t = torch.from_numpy(yv)
    Xe_t = torch.from_numpy(Xe)
    n = len(Xt_t); bs = 2048
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for ep in range(40):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xt_t[b])
            loss = lossf(pred, yt_t[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv_t), yv_t).item()
        print(f'  ep{ep+1} train_mse={tot/n:.5f} val_mse={vl:.5f} {time.time()-t0:.0f}s', flush=True)
        if vl < best_val - 1e-6:
            best_val, patience = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                print(f'早停 ep{ep+1}', flush=True)
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pg_v = model(Xv_t).numpy()
        pg_e = model(Xe_t).numpy()
    print('\n=== GRU 预测IC ===', flush=True)
    r_va = monthly_ic(D[va], pg_v, yv, 'GRU@val')
    r_te = monthly_ic(D[te], pg_e, ye, 'GRU@test')

    pd.to_pickle({'D': D, 'C': C, 'y': y, 'pred_val': pg_v, 'pred_test': pg_e,
                  'ic_val': r_va, 'ic_test': r_te},
                 os.path.join(OUT, 'sequence_probe.pkl'))
    # 判据
    ic_te = r_te.ic
    gate = (abs(ic_te.mean()) > 0.03) and ((ic_te > 0).sum() >= max(5, int(0.6 * len(ic_te))))
    print(f'\n判据: |IC|>0.03 且 正率≥5/8 → {"过门(排队深探)" if gate else "不过门 → P1-3关闭"}',
          flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
