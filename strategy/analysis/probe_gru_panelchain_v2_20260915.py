"""P1-3f: 训练链逐位复刻×日频全覆盖 — 机制终版配置 (2026-09-15)

P1-3e失败根因: searchsorted(pos)-1把链行错位一天, 与build_features训练链
(fe.loc[common].values 面板日精确行)结构断裂 — 与诚实面板预测同(code,date)处
ρ=-0.01/+0.19。本版逐位复刻: common=面板日∩CSV日, arr=面板日精确特征行,
链=arr[kk-29..kk] (kk=最后面板日≤t的位置)。面板日链与训练链完全一致,
非面板日滞后≤1交易日(仍PIT)。stats=训练链窗口加权(X[tr].reshape同构)。
身份烟测: 面板日pred_mean vs gru_panel_honest.pkl 应ρ≥0.99。
判据: buy子集全日频IC≥0.03且非面板日≥0.02 → E-seq1a烟测; 否则LEN=60兜底。
产出: rolling_validation_results/gru_panelchain2_daily.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pyarrow.parquet as pq

from probe_sequence_deep_20260915 import SeqNet, monthly_ic, build_features

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
CSV_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
LEN = 30
FEATS = ['ret1', 'vol_ratio', 'amplitude', 'rng_pos', 'ret5', 'mom20']
SEEDS = [42, 7, 2024]
TRAIN_END = np.datetime64('2024-12-17')
PRED_START = np.datetime64('2025-01-01')


def load_panel():
    """面板: {code: dates数组} + 个股future_ret (fwd10标签, 与build_features同源)"""
    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'future_ret'])
    df = t.to_pandas()
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])
    g = df.groupby('code')
    return {c: (g.date.values, g.future_ret.values) for c, g in df.groupby('code')}


def stock_features(c):
    p = os.path.join(CSV_DIR, f'{c}_qfq.csv')
    k = pd.read_csv(p, parse_dates=['datetime'], usecols=[
        'datetime', 'open', 'high', 'low', 'close', 'volume']).set_index('datetime')
    k = k[~k.index.duplicated(keep='last')].sort_index()
    k['ret1'] = k['close'].pct_change()
    k['vol_ratio'] = k['volume'] / k['volume'].rolling(20).mean()
    k['amplitude'] = (k['high'] - k['low']) / k['close']
    k['rng_pos'] = ((k['close'] - k['low']) / (k['high'] - k['low']).replace(0, np.nan)).fillna(0.5)
    k['ret5'] = k['close'].pct_change(5)
    k['mom20'] = k['close'] / k['close'].shift(20) - 1
    k['vol_ratio'] = k['vol_ratio'].fillna(1.0)
    return k


def main():
    torch.set_num_threads(10)
    t0 = time.time()
    panel = load_panel()
    models = pd.read_pickle(os.path.join(OUT, 'gru_panel_models.pkl'))['models']
    codes = sorted(panel.keys())
    nf = len(FEATS)

    # pass1: mu/sd = build_features的X[tr]统计, 保持float32累加语义与训练z完全一致
    # (根因: float32对36.6M值求和有~13%精度损失, mu_vol_ratio 0.9196≠真值1.039;
    #  模型是在这组"错误"统计上学的 — 机制必须复用同一变换, 不是修统计)
    _Xz, X, _y, _D, _C, tr, _va, _te = build_features()
    mu = X[tr].reshape(-1, X.shape[2]).mean(0).astype(np.float32)
    sd = X[tr].reshape(-1, X.shape[2]).std(0).astype(np.float32)
    print(f'build_features float32语义 stats: mu={mu.round(3)} sd={sd.round(3)}', flush=True)

    # pass2: 每日t → 链=arr[kk-29..kk], kk=最后面板日≤t
    codes_out, dates_out, labels_out = [], [], []
    rows_out, panel_pos_out = [], []
    for i, c in enumerate(codes):
        pd_, yv = panel[c]
        k = stock_features(c)
        fe = k[FEATS].values.astype(np.float32)
        dates = k.index.values
        close = k['close'].values
        pos = np.searchsorted(dates, pd_)
        ok = (pos < len(dates)) & (dates[pos] == pd_)
        pd2, pos2 = pd_[ok], pos[ok]
        arr = fe[pos2]
        n = len(pd2)
        is_panel_day = np.zeros(len(dates), bool)
        is_panel_day[pos2] = True
        t_idx = np.where(dates >= PRED_START)[0]
        for j in t_idx:
            kk = np.searchsorted(pd2, dates[j], side='right') - 1
            if kk < LEN - 1:
                continue
            w = arr[kk - LEN + 1: kk + 1]
            if not np.isfinite(w).all():
                continue
            rows_out.append(w)
            codes_out.append(c)
            dates_out.append(dates[j])
            labels_out.append((close[j + 10] / close[j] - 1) if j + 10 < len(dates) else np.nan)
            panel_pos_out.append(is_panel_day[j])
        if (i + 1) % 600 == 0:
            print(f'  {i+1}/{len(codes)} 只, {len(rows_out)} 行', flush=True)
    Xw = np.stack(rows_out)
    Xz = ((Xw - mu) / np.where(sd < 1e-8, 1, sd)).astype(np.float32)
    Xp_t = torch.from_numpy(Xz)
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
                       'pred_mean': p.mean(0),
                       'is_panel_day': np.asarray(panel_pos_out)})

    print('\n=== 身份烟测: 面板日pred vs gru_panel_honest.pkl ===', flush=True)
    from scipy import stats as _st
    h = pd.read_pickle(os.path.join(OUT, 'gru_panel_honest.pkl'))
    dh = pd.DataFrame({'code': pd.Series(h['code']).astype(str).str.zfill(6),
                       'date': pd.to_datetime(h['date']), 'p': h['pred_mean']})
    sub_p = df[df.is_panel_day]
    m = sub_p.merge(dh, on=['code', 'date'], how='inner')
    print(f'  共享行 {len(m)}, rho={_st.spearmanr(m.pred_mean, m.p)[0]:+.4f}, '
          f'mae={np.abs(m.pred_mean - m.p).mean():.4f}', flush=True)

    print('\n=== pool IC ===', flush=True)
    for yr in [2025, 2026]:
        s = df[(df.date.dt.year == yr) & df.y.notna()]
        monthly_ic(s['date'].values, s['pred_mean'].values, s['y'].values, f'panelchain2@{yr}')

    print('\n=== buy子集IC (p原始值, 与诚实探针同判据) ===', flush=True)
    sig = pd.read_csv(os.path.join(OUT, 'backtest_signals.csv'),
                      dtype={'code': str}, low_memory=False, usecols=['code', 'date', 'buy'])
    sig['date'] = pd.to_datetime(sig['date'])
    sig['code'] = sig['code'].str.zfill(6)
    b = sig[(sig.buy == True) & (sig.date >= '2025-01-01')]
    for label, mask in [('全日频', np.ones(len(df), bool)),
                        ('面板日', df.is_panel_day.values),
                        ('非面板日', ~df.is_panel_day.values)]:
        s = df[mask]
        mm = b.merge(s[['code', 'date', 'pred_mean', 'y']], on=['code', 'date'],
                     how='inner').dropna(subset=['y'])
        rows = []
        for mo, g in mm.groupby(mm.date.dt.to_period('M')):
            if len(g) < 20:
                continue
            rows.append(_st.spearmanr(g.pred_mean, g.y)[0])
        r = pd.Series(rows)
        print(f'  [{label}] n={len(mm)} IC={r.mean():+.4f} IR={r.mean()/r.std():+.2f} '
              f'正率={100*(r>0).mean():.0f}%', flush=True)

    df['z'] = df.groupby('date')['pred_mean'].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    pd.to_pickle({'code': df.code.values, 'date': df.date.values, 'y': df.y.values,
                  'pred_mean': df.pred_mean.values, 'z': df.z.values,
                  'is_panel_day': df.is_panel_day.values},
                 os.path.join(OUT, 'gru_panelchain2_daily.pkl'))
    print(f'DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
