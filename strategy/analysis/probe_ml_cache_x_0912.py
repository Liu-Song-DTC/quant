#!/usr/bin/env python3
"""2026-09-12 指定缓存ML复跑探针: 对任意809d缓存跑真实ML滚动循环(23 chunk), 与IC_A/IC_B比对

背景: 三缓存(8c19c0a8/41343d0a/d814a206)131列值级全同, 但9/11加载8c19c0a8两次=IC_A,
今天加载d814a206多次=IC_B。本探针今天重跑8c19c0a8:
  =IC_A → 文件本体存在看不见的差异(转parquet内部结构深挖)
  =IC_B → 值级相同的文件今天全给IC_B, 分歧=9/11进程环境vs今天(环境态)
用法: cd strategy && python analysis/probe_ml_cache_x_0912.py <cache名无后缀> > logs/probe_ml_cache_x_<名>_0912.log 2>&1
只读。串行执行。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, '..', 'data')
BT_DATA = os.path.join(DATA_ROOT, 'stock_data', 'backtrader_data')

LOGGED_COUNTS = [203233, 259483, 316659, 375408, 436968, 467410, 476509, 492344,
                 508497, 523040, 535951, 550620, 566159, 576517, 588214, 596622,
                 607303, 613222, 620568, 629284, 634172, 635561, 638726]
IC_A = [0.0796, 0.1146, 0.0882, 0.0451, 0.0815, 0.0426, 0.0842, 0.0992, 0.0977,
        0.1186, 0.0650, 0.1367, 0.0987, 0.1096, 0.1109, 0.1324, 0.1016, 0.1109,
        0.1100, 0.1130, 0.0937, 0.0888, 0.0136]
IC_B = [0.0809, 0.1166, 0.0865, 0.0464, 0.0826, 0.0436, 0.0833, 0.0985, 0.0993,
        0.1184, 0.0648, 0.1374, 0.0995, 0.1111, 0.1113, 0.1319, 0.1010, 0.1106,
        0.1100, 0.1135, 0.0921, 0.0893, 0.0141]


def main():
    t0 = time.time()
    cache_name = sys.argv[1] if len(sys.argv) > 1 else '8c19c0a8'
    CACHE = os.path.join(BASE, 'cache', f'factor_df_2718s_809d_{cache_name}.parquet')
    if not os.path.exists(CACHE):
        print(f'[fatal] 缓存不存在: {CACHE}')
        sys.exit(2)
    print(f'[{cache_name}] 缓存: {CACHE} ({os.path.getsize(CACHE)/1e9:.2f}GB)')

    from core.ml_predictor import MLFactorPredictor
    from core.strategy import Strategy

    config = load_config()
    ml_config = config.config.get('ml', {})
    factor_df = pd.read_parquet(CACHE)
    if factor_df['code'].dtype != object:
        factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)
    print(f'[{cache_name}] factor_df {len(factor_df)} 行, '
          f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()}, "
          f"dtype={factor_df['date'].dtype} ({time.time()-t0:.0f}s)")

    # 密集日历 (与v2探针一致, 样本数23/23校验兜底)
    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    idx = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    all_dates = sorted(set(early) | set(calendar))

    # regime map (历史已证median全0, 保留一致性)
    todate = config.get('backtest.todate', None)
    idx_df = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    if fromdate is not None:
        idx_df = idx_df[idx_df['datetime'] >= fromdate]
    if todate is not None:
        idx_df = idx_df[idx_df['datetime'] <= pd.Timestamp(todate)]
    small_cap_df = pd.read_csv(os.path.join(BT_DATA, 'sh000852_qfq.csv'), parse_dates=['datetime'])
    growth_df = pd.read_csv(os.path.join(BT_DATA, '399006_qfq.csv'), parse_dates=['datetime'])
    strategy = Strategy(init_cash=1000000)
    strategy.generate_market_regime(idx_df, small_cap_df=small_cap_df, growth_df=growth_df)
    regime_df = strategy.index_data
    _regime_map = {}
    if regime_df is not None and 'regime' in regime_df.columns:
        for _idx, _row in regime_df.iterrows():
            k = _idx.strftime('%Y-%m-%d') if isinstance(_idx, pd.Timestamp) else str(_idx)
            _regime_map[k] = int(_row['regime'])

    # === 复刻bt_execution ML滚动循环 ===
    _ml_df = factor_df
    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))
    _all_dates = sorted(_ml_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    assert len(_pred_dates) == 690, f'pred_dates={len(_pred_dates)} != 690'
    chunk_starts = list(range(0, len(_pred_dates), _retrain_freq))
    _val_ics, _last_good, _reused, _consec = [], None, 0, 0
    _min_val_ic = ml_config.get('min_val_ic', 0.03)
    _max_reuse = int(ml_config.get('max_consecutive_reuse', 3))
    _fp_days = int(config.get('dynamic_factor.forward_period', 10))
    chunk_counts, chunk_ics = [], []

    for chunk_idx, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + _retrain_freq, len(_pred_dates))
        chunk_dates = _pred_dates[chunk_start:chunk_end]
        first_pred_date = chunk_dates[0]
        _d0_ts = pd.Timestamp(first_pred_date)
        _d0_pos = int(np.searchsorted(np.asarray(all_dates), _d0_ts))
        _purge_end = all_dates[max(0, _d0_pos - _fp_days)]
        train_start = first_pred_date - pd.Timedelta(days=_train_window)
        train_mask = (_ml_df['date'] >= train_start) & (_ml_df['date'] < _purge_end)
        train_df = _ml_df[train_mask]
        chunk_counts.append(len(train_df))
        if len(train_df) < 50000:
            chunk_ics.append(None)
            continue
        ml_predictor = MLFactorPredictor(config.config)
        _train_regime = 0
        if _regime_map:
            _train_regs = [_regime_map.get(d.strftime('%Y-%m-%d'), 0)
                           for d in train_df['date'].unique()]
            _train_regime = int(np.median(_train_regs)) if _train_regs else 0
        val_ic = ml_predictor.train(train_df, regime_info={'regime': _train_regime})
        chunk_ics.append(val_ic)
        if val_ic is None or val_ic < _min_val_ic:
            if _last_good is None:
                continue
            if _consec < _max_reuse:
                ml_predictor = _last_good
                _reused += 1
                _consec += 1
            elif val_ic is not None and val_ic > 0:
                _last_good = ml_predictor
                _consec = 0
            else:
                ml_predictor = _last_good
                _reused += 1
                _consec += 1
        else:
            _val_ics.append(val_ic)
            _last_good = ml_predictor
            _consec = 0

    print(f'[{cache_name}] 有效={len(_val_ics)} 复用={_reused} avg={np.mean(_val_ics):.4f} '
          f'耗时={(time.time()-t0)/60:.1f}min')
    print(f'[{cache_name}] 样本数命中: {chunk_counts == LOGGED_COUNTS}')
    print(f'[{cache_name}] IC: ' + ' '.join(f'{x:.6f}' if x is not None else 'None'
                                            for x in chunk_ics))
    for tag, ref in (('A(pit)', IC_A), ('B(082730)', IC_B)):
        diffs = [abs(x - r) for x, r in zip(chunk_ics, ref) if x is not None]
        n_close = sum(1 for x, r in zip(chunk_ics, ref)
                      if x is not None and abs(x - r) < 1e-6)
        print(f'[{cache_name}] vs {tag}: max|Δ|={max(diffs):.6f} 一致chunk={n_close}/23')
    print(f'[{cache_name}] 判定: ' + ('=IC_A' if all(
        abs(x - r) < 1e-6 for x, r in zip(chunk_ics, IC_A) if x is not None) else
        ('=IC_B' if all(abs(x - r) < 1e-6 for x, r in zip(chunk_ics, IC_B) if x is not None)
         else '=第三家族')))


if __name__ == '__main__':
    main()
