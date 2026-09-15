"""P1-3d: 面板训练模型×日频stride-2窗口 = 机制就绪配置验证 (2026-09-15)

P1-3c定案: 诚实(时点val)面板分辨率GRU buy子集IC 2026 +0.070 — 信号真实存在;
日频LEN30重建失败(+0.011)根因=上下文长度(30交易日≈42日历日 vs 面板30行≈60
交易日)。机制就绪配置 = 日频stride-2窗口(30行×2交易日=60交易日跨度, 每交易日t
以自身parity链[t-58..t]结尾) + 面板训练模型。
判据: 全日频buy子集IC≥0.03(且parity分解不塌) → 直接接入E-seq1a烟测;
      否则 LEN=60日频重训兜底。
产出: rolling_validation_results/gru_stride2_daily.pkl (含模型权重)
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from probe_sequence_deep_20260915 import build_features, SeqNet, train_model
from probe_sequence_deep_20260915 import monthly_ic

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
CSV_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
LEN = 30
FEATS = ['ret1', 'vol_ratio', 'amplitude', 'rng_pos', 'ret5', 'mom20']
SEEDS = [42, 7, 2024]
VAL_LO = np.datetime64('2024-10-16')
VAL_HI = np.datetime64('2024-12-17')
TRAIN_START = np.datetime64('2021-01-01')


def train_panel_models():
    """P1-3c同款: 面板训练+时点val早停, 保存权重"""
    Xz, X, y, D, C, tr, va, te = build_features()
    tr_idx = np.where(tr)[0]
    val_sel = (D[tr_idx] > VAL_LO) & (D[tr_idx] <= VAL_HI)
    Xt_t = torch.from_numpy(Xz[tr_idx[~val_sel]]); yt_t = torch.from_numpy(y[tr_idx[~val_sel]])
    Xv_t = torch.from_numpy(Xz[tr_idx[val_sel]]); yv_t = torch.from_numpy(y[tr_idx[val_sel]])
    models = {}
    for s in SEEDS:
        m = train_model(SeqNet(6, 32, 'gru'), Xt_t, yt_t, Xv_t, yv_t, s)
        models[f'seed{s}'] = {k: v.clone().cpu() for k, v in m.state_dict().items()}
        print(f'  面板seed{s}训练完成', flush=True)
    return models


def build_stride2_daily(models):
    """日频stride-2窗口: 每交易日t用链[t-58, t-56, ..., t] (30行), 2025+预测"""
    import pyarrow.parquet as pq
    t = pq.read_table(PQ_PATH, columns=['code'])
    codes = sorted(set(t.to_pandas()['code'].astype(str).str.zfill(6)))
    nf = len(FEATS)
    # 训练期特征统计 (逐特征值, 与面板X[tr]统计同构)
    s1 = np.zeros(nf); s2 = np.zeros(nf); cnt = 0
    pred_rows, codes_out, dates_out, labels_out = [], [], [], []
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
        fe = k[FEATS].values.astype(np.float32)
        dates = k.index.values
        tr_mask = (dates >= TRAIN_START) & (dates <= np.datetime64('2024-12-17'))
        ftr = fe[tr_mask]
        ok = np.isfinite(ftr).all(axis=1)
        s1 += ftr[ok].sum(0); s2 += (ftr[ok] ** 2).sum(0); cnt += ok.sum()
        # 预测窗口: 2025+ 全交易日, stride-2链
        pr_mask = dates >= np.datetime64('2025-01-01')
        idx = np.where(pr_mask)[0]
        for j in idx:
            lo = j - 2 * (LEN - 1)
            if lo < 0:
                continue
            w = fe[lo:j + 1:2]
            if len(w) != LEN or not np.isfinite(w).all():
                continue
            pred_rows.append(w)
            codes_out.append(c)
            dates_out.append(dates[j])
            labels_out.append((k['close'].iloc[j + 10] / k['close'].iloc[j] - 1) if j + 10 < len(k) else np.nan)
        if (i + 1) % 600 == 0:
            print(f'  {i+1}/{len(codes)} 只', flush=True)
    mu = (s1 / cnt).astype(np.float32)
    sd = np.sqrt(s2 / cnt - mu.astype(np.float64) ** 2).astype(np.float32)
    Xw = np.stack(pred_rows)
    Xz = ((Xw - mu) / np.where(sd < 1e-8, 1, sd)).astype(np.float32)
    Xp_t = torch.from_numpy(Xz)
    print(f'stride-2窗口: {len(Xw)} 行, 预测中...', flush=True)
    preds = []
    for s in SEEDS:
        m = SeqNet(6, 32, 'gru')
        m.load_state_dict(models[f'seed{s}'])
        m.eval()
        with torch.no_grad():
            preds.append(m(Xp_t).numpy())
    p = np.stack(preds)
    df = pd.DataFrame({'code': pd.Series(codes_out).astype(str).str.zfill(6),
                       'date': pd.to_datetime(dates_out),
                       'y': np.asarray(labels_out, dtype=np.float32),
                       'pred_mean': p.mean(0)})
    df['z'] = df.groupby('date')['pred_mean'].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return df


def buy_ic(df, label):
    from scipy import stats as _st
    sig = pd.read_csv(os.path.join(OUT, 'backtest_signals.csv'),
                      dtype={'code': str}, low_memory=False, usecols=['code', 'date', 'buy'])
    sig['date'] = pd.to_datetime(sig['date'])
    sig['code'] = sig['code'].str.zfill(6)
    b = sig[(sig.buy == True) & (sig.date >= '2025-01-01')]
    m = b.merge(df[['code', 'date', 'z', 'y']], on=['code', 'date'], how='inner').dropna(subset=['y'])
    rows = []
    for mo, sub in m.groupby(m.date.dt.to_period('M')):
        if len(sub) < 20:
            continue
        rows.append((str(mo), _st.spearmanr(sub.z, sub.y)[0]))
    r = pd.DataFrame(rows, columns=['m', 'ic'])
    print(f'  [{label}] buy子集 n={len(m)} IC={r.ic.mean():+.4f} '
          f'IR={r.ic.mean()/r.ic.std():+.2f} 正率={100*(r.ic>0).mean():.0f}%', flush=True)
    for _, row in r.iterrows():
        print(f'    {row.m}: IC={row.ic:+.4f}', flush=True)
    return r, m


def main():
    torch.set_num_threads(10)
    t0 = time.time()
    print('=== 面板训练3种子 (时点val) ===', flush=True)
    models = train_panel_models()
    pd.to_pickle({'models': models}, os.path.join(OUT, 'gru_panel_models.pkl'))
    print(f'训练 {time.time()-t0:.0f}s', flush=True)
    print('=== 日频stride-2构建+预测 ===', flush=True)
    df = build_stride2_daily(models)
    print(f'构建+预测 {time.time()-t0:.0f}s', flush=True)

    print('\n=== pool IC (日频stride-2) ===', flush=True)
    for yr in [2025, 2026]:
        sub = df[(df.date.dt.year == yr) & df.y.notna()]
        monthly_ic(sub['date'].values, sub['pred_mean'].values, sub['y'].values, f'stride2@{yr}')

    print('\n=== buy子集IC: 全日频 ===', flush=True)
    r_all, m_all = buy_ic(df, 'stride2全日频')
    print('=== buy子集IC: 面板日 ===', flush=True)
    lab = pd.read_pickle(os.path.join(OUT, 'gru_panel_honest.pkl'))
    panel_dates = set(pd.to_datetime(lab['date']).unique())
    sub = df[df.date.isin(panel_dates)]
    r_p, m_p = buy_ic(sub, 'stride2面板日')
    print('=== buy子集IC: 非面板日(odd phase) ===', flush=True)
    sub2 = df[~df.date.isin(panel_dates)]
    r_o, m_o = buy_ic(sub2, 'stride2非面板日')

    pd.to_pickle({'code': df.code.values, 'date': df.date.values, 'y': df.y.values,
                  'pred_mean': df.pred_mean.values, 'z': df.z.values, 'models': models},
                 os.path.join(OUT, 'gru_stride2_daily.pkl'))
    print(f'DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
