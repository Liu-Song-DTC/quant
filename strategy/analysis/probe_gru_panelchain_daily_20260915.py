"""P1-3e: 面板日历链×日频全覆盖 — 机制终版配置验证 (2026-09-15)

P1-3d发现: 规则stride-2链在面板日buy IC仅+0.033(训练链+0.070) — 链结构失配
(训练链含个股停牌产生的4日缺口, 规则链没有)。本版=预测链与训练链同构:
每交易日t, 链=该股最后30个面板日(≤t)的特征行 — 面板日t时与训练链逐位一致,
非面板日滞后≤1个交易日(仍PIT: 只用≤t信息)。
模型: gru_panel_models.pkl (P1-3c诚实训练, 时点val)。
判据: 全日频buy子集IC≥0.03且非面板日子集≥0.02 → 接入E-seq1a烟测。
产出: rolling_validation_results/gru_panelchain_daily.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pyarrow.parquet as pq

from probe_sequence_deep_20260915 import SeqNet, monthly_ic

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
CSV_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
LEN = 30
FEATS = ['ret1', 'vol_ratio', 'amplitude', 'rng_pos', 'ret5', 'mom20']
SEEDS = [42, 7, 2024]
TRAIN_LO = np.datetime64('2021-01-01')
TRAIN_HI = np.datetime64('2024-12-17')
PRED_START = np.datetime64('2025-01-01')


def load_panel_dates():
    t = pq.read_table(PQ_PATH, columns=['code', 'date'])
    df = t.to_pandas()
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])
    return {c: g['date'].values for c, g in df.groupby('code')}


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
    panel_dates = load_panel_dates()
    models = pd.read_pickle(os.path.join(OUT, 'gru_panel_models.pkl'))['models']
    codes = sorted(panel_dates.keys())
    nf = len(FEATS)

    # pass1: 训练期面板行特征统计 (与面板训练X[tr]统计同构)
    s1 = np.zeros(nf); s2 = np.zeros(nf); cnt = 0
    for i, c in enumerate(codes):
        k = stock_features(c)
        fe = k[FEATS].values.astype(np.float32)
        dates = k.index.values
        pd_ = panel_dates[c]
        pos = np.searchsorted(dates, pd_)
        valid = (pos > 0) & (pos < len(dates)) & (pd_ >= TRAIN_LO) & (pd_ <= TRAIN_HI)
        rows = fe[pos[valid]]
        ok = np.isfinite(rows).all(axis=1)
        s1 += rows[ok].sum(0); s2 += (rows[ok] ** 2).sum(0); cnt += ok.sum()
    mu = (s1 / cnt).astype(np.float32)
    sd = np.sqrt(s2 / cnt - mu.astype(np.float64) ** 2).astype(np.float32)
    print(f'训练统计 cnt={cnt} mu={mu.round(3)} sd={sd.round(3)}', flush=True)

    # pass2: 每日t → 该股最后30个面板日≤t → 预测
    codes_out, dates_out, labels_out = [], [], []
    rows_out = []
    for i, c in enumerate(codes):
        k = stock_features(c)
        fe = k[FEATS].values.astype(np.float32)
        dates = k.index.values
        close = k['close'].values
        pd_ = panel_dates[c]
        pos = np.searchsorted(dates, pd_)
        valid = (pos > 0) & (pos <= len(dates))
        pd_ = pd_[valid]; pos = pos[valid] - 1  # 面板日的特征行位置
        # 每日t: last panel ≤ t
        t_idx = np.where(dates >= PRED_START)[0]
        for j in t_idx:
            kk = np.searchsorted(pd_, dates[j], side='right') - 1
            if kk < LEN - 1:
                continue
            chain_pos = pos[kk - LEN + 1: kk + 1]
            w = fe[chain_pos]
            if not np.isfinite(w).all():
                continue
            rows_out.append(w)
            codes_out.append(c)
            dates_out.append(dates[j])
            labels_out.append((close[j + 10] / close[j] - 1) if j + 10 < len(dates) else np.nan)
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
                       'pred_mean': p.mean(0)})
    df['z'] = df.groupby('date')['pred_mean'].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))

    print('\n=== pool IC ===', flush=True)
    for yr in [2025, 2026]:
        sub = df[(df.date.dt.year == yr) & df.y.notna()]
        monthly_ic(sub['date'].values, sub['pred_mean'].values, sub['y'].values, f'panelchain@{yr}')

    print('\n=== buy子集IC ===', flush=True)
    from scipy import stats as _st
    sig = pd.read_csv(os.path.join(OUT, 'backtest_signals.csv'),
                      dtype={'code': str}, low_memory=False, usecols=['code', 'date', 'buy'])
    sig['date'] = pd.to_datetime(sig['date'])
    sig['code'] = sig['code'].str.zfill(6)
    b = sig[(sig.buy == True) & (sig.date >= '2025-01-01')]
    panel_day_set = set(pd.to_datetime(np.concatenate(list(panel_dates.values()))))
    for label, mask in [('全日频', np.ones(len(df), bool)),
                        ('面板日', df.date.isin(panel_day_set).values),
                        ('非面板日', ~df.date.isin(panel_day_set).values)]:
        sub = df[mask]
        m = b.merge(sub[['code', 'date', 'z', 'y']], on=['code', 'date'], how='inner').dropna(subset=['y'])
        rows = []
        for mo, g in m.groupby(m.date.dt.to_period('M')):
            if len(g) < 20:
                continue
            rows.append(_st.spearmanr(g.z, g.y)[0])
        r = pd.Series(rows)
        print(f'  [{label}] n={len(m)} IC={r.mean():+.4f} IR={r.mean()/r.std():+.2f} '
              f'正率={100*(r>0).mean():.0f}%', flush=True)

    pd.to_pickle({'code': df.code.values, 'date': df.date.values, 'y': df.y.values,
                  'pred_mean': df.pred_mean.values, 'z': df.z.values},
                 os.path.join(OUT, 'gru_panelchain_daily.pkl'))
    print(f'DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
