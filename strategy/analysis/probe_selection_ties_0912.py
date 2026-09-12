#!/usr/bin/env python3
"""2026-09-12 chunk1特征选择near-tie检测 + 扰动电池: 找IC_A(0.0796) vs IC_B(0.0809)的机制

背景: 训练已被证明确定性函数(17位+模型sha256), A/B输入bit级全同却IC差2e-3。
假说: (1) top-8特征选择存在near-tie → 1e-15级扰动翻转特征集 → 混沌放大;
      (2) 训练窗/purge边界差1天 → 数据子集差 → 模型差;
      (3) regime±1 → regime特征激活 → 模型差。
本探针: A部分打印chunk1的top-12特征分数(看#8/#9 gap);
        B部分对chunk1跑扰动电池, 每变体打印IC(%.6f+17位), 与0.0796/0.0809比对。
只读。串行。执行: cd strategy && python analysis/probe_selection_ties_0912.py > logs/probe_selection_ties_0912.log 2>&1
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.ml_predictor import MLFactorPredictor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, '..', 'data')
BT_DATA = os.path.join(DATA_ROOT, 'stock_data', 'backtrader_data')
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')

IC_A_C1 = 0.0796
IC_B_C1 = 0.0809


def build_calendar(config):
    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    idx = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    return sorted(set(early) | set(calendar))


def get_chunk1(factor_df, config, all_dates):
    ml_config = config.config.get('ml', {})
    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))
    _fp_days = int(config.get('dynamic_factor.forward_period', 10))
    _all_dates = sorted(factor_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    first_pred_date = _pred_dates[0]
    _d0_pos = int(np.searchsorted(np.asarray(all_dates), pd.Timestamp(first_pred_date)))
    purge_end = all_dates[max(0, _d0_pos - _fp_days)]
    train_start = first_pred_date - pd.Timedelta(days=_train_window)
    return first_pred_date, purge_end, train_start, all_dates, _d0_pos


def selection_scores(df, features, top_n=12):
    """复刻_select_top_features的分数, 返回[(feature, score, ic_mean, ic_sharpe)]按分数降序"""
    dates = sorted(df['date'].unique())
    n_windows = min(5, len(dates))
    window_size = len(dates) // n_windows
    scores = []
    for f in features:
        if f not in df.columns:
            continue
        window_ics = []
        for w in range(n_windows):
            w_start = w * window_size
            w_end = (w + 1) * window_size if w < n_windows - 1 else len(dates)
            w_dates = set(dates[w_start:w_end])
            valid = df[df['date'].isin(w_dates)][[f, 'future_ret']].dropna()
            if len(valid) >= 30:
                ic = valid[f].corr(valid['future_ret'], method='spearman')
                window_ics.append(ic)
        if len(window_ics) >= 3:
            ic_mean = np.mean(window_ics)
            ic_std = np.std(window_ics) + 1e-10
            ic_sharpe = ic_mean / ic_std
            score = ic_sharpe * 0.7 + abs(ic_mean) * 0.3
            scores.append((f, score, ic_mean, ic_sharpe))
    scores.sort(key=lambda x: -x[1])
    return scores


def main():
    t0 = time.time()
    config = load_config()
    factor_df = pd.read_parquet(CACHE)
    if factor_df['code'].dtype != object:
        factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)
    print(f'[load] {len(factor_df)} 行 ({time.time()-t0:.0f}s)')
    all_dates = build_calendar(config)
    first_pred_date, purge_end, train_start, all_dates, d0_pos = get_chunk1(
        factor_df, config, all_dates)

    # 复刻train()的前处理: sort + zscore + rank标签
    exclude_set = {'code', 'date', 'future_ret', 'industry'}
    numeric_cols = [c for c in factor_df.columns if c not in exclude_set
                    and factor_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    meta_cols = [c for c in ['code', 'date', 'future_ret', 'industry'] if c in factor_df.columns]
    df = factor_df[meta_cols + numeric_cols].copy()
    df = df.sort_values(['date', 'code'], kind='stable').reset_index(drop=True)
    train_mask = (df['date'] >= train_start) & (df['date'] < purge_end)
    df = df[train_mask].reset_index(drop=True)
    df = MLFactorPredictor._cross_sectional_zscore(df, numeric_cols)
    df['future_ret_rank'] = df.groupby('date')['future_ret'].rank(pct=True) - 0.5
    df['future_ret'] = df['future_ret_rank']
    df.drop(columns=['future_ret_rank'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['future_ret'])

    # 白名单过滤(与prepare_features一致)
    whitelist = config.config.get('ml', {}).get('feature_whitelist', None)
    base_features = [c for c in df.columns if c not in exclude_set
                     and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    if whitelist:
        base_features = [f for f in base_features if f in whitelist]
    print(f'[sel] base特征数={len(base_features)} whitelist={"有" if whitelist else "无"}')
    print(f'[win] chunk1: pred首日={first_pred_date.date()} purge={purge_end.date()} '
          f'train={df["date"].min().date()}~{df["date"].max().date()} 样本={len(df)}')

    scores = selection_scores(df, base_features)
    for i, (f, s, m, sh) in enumerate(scores[:12]):
        print(f'  #{i+1} {f}: score={s:.10f} ic_mean={m:+.6f} ic_sharpe={sh:+.4f}')
    if len(scores) >= 9:
        gap = scores[7][1] - scores[8][1]
        print(f'[tie] #8={scores[7][0]}({scores[7][1]:.10f}) #9={scores[8][0]}'
              f'({scores[8][1]:.10f}) gap={gap:.3g}')
        if gap < 1e-6:
            print('[tie] >> gap<1e-6: 特征选择对微小扰动混沌敏感!')
        else:
            print('[tie] >> gap明显: 选择稳定, 分歧机制不在此层')

    # === 扰动电池 ===
    print('\n=== 扰动电池 (chunk1, 目标: A=0.0796 / B=0.0809) ===')
    variants = []
    # 基础变体: 标准参数
    variants.append(('v0_基线', dict(purge_shift=0, train_shift=0, regime=0)))
    variants.append(('v1_purge-1', dict(purge_shift=-1, train_shift=0, regime=0)))
    variants.append(('v2_purge+1', dict(purge_shift=1, train_shift=0, regime=0)))
    variants.append(('v3_train-1d', dict(purge_shift=0, train_shift=-1, regime=0)))
    variants.append(('v4_train+1d', dict(purge_shift=0, train_shift=1, regime=0)))
    variants.append(('v5_regime+1', dict(purge_shift=0, train_shift=0, regime=1)))
    variants.append(('v6_regime-1', dict(purge_shift=0, train_shift=0, regime=-1)))

    for label, kw in variants:
        t1 = time.time()
        p_end = all_dates[max(0, d0_pos - 10 + kw['purge_shift'])]
        t_start = train_start + pd.Timedelta(days=kw['train_shift'])
        _mask = (factor_df['date'] >= t_start) & (factor_df['date'] < p_end)
        _df = factor_df[_mask].copy()
        p = MLFactorPredictor(config.config)
        ic = p.train(_df, regime_info={'regime': kw['regime']})
        tag = 'A!' if abs(ic - IC_A_C1) < 5e-5 else ('B!' if abs(ic - IC_B_C1) < 5e-5 else '')
        print(f'  {label}: IC={ic:.17g} 样本={len(_df)} {tag} ({time.time()-t1:.0f}s)')
    print(f'总耗时 {(time.time()-t0)/60:.1f}min')


if __name__ == '__main__':
    main()
