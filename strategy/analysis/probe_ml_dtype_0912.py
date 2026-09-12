#!/usr/bin/env python3
"""2026-09-12 ML日期dtype探针v3: [us](parquet加载) vs [ns](A现算内存态) 同进程双pass

背景: 证据链闭环后A(pit) vs B(082730)的IC分歧只剩一个可测机制:
  A的df日期列=datetime64[ns] (read_csv parse_dates, 现算内存态)
  B的df日期列=datetime64[us] (read_parquet, pyarrow往返时ns→us)
  值完全相等(午夜日期), 但若任一代码路径对分辨率敏感(排序/分箱/int64化/
  pandas内部datetime groupby), 训练结果会不同。

pass1: 加载缓存原样([us]) → ML全23chunk → 预期=IC_B (跨进程确定性第2证)
pass2: 同一df, date列astype→[ns] → ML全23chunk → 若=IC_A → 机制实锤=date分辨率
       若=IC_B → date分辨率无关, 转fresh路径镜像探针(A进程态最后一个嫌疑)

只读探针: 不写任何回测产物。
执行: cd strategy && python analysis/probe_ml_dtype_0912.py > logs/probe_ml_dtype_0912.log 2>&1
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.ml_predictor import MLFactorPredictor
from core.strategy import Strategy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, '..', 'data')
BT_DATA = os.path.join(DATA_ROOT, 'stock_data', 'backtrader_data')
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')

LOGGED_COUNTS = [203233, 259483, 316659, 375408, 436968, 467410, 476509, 492344,
                 508497, 523040, 535951, 550620, 566159, 576517, 588214, 596622,
                 607303, 613222, 620568, 629284, 634172, 635561, 638726]
IC_A = [0.0796, 0.1146, 0.0882, 0.0451, 0.0815, 0.0426, 0.0842, 0.0992, 0.0977,
        0.1186, 0.0650, 0.1367, 0.0987, 0.1096, 0.1109, 0.1324, 0.1016, 0.1109,
        0.1100, 0.1130, 0.0937, 0.0888, 0.0136]
IC_B = [0.0809, 0.1166, 0.0865, 0.0464, 0.0826, 0.0436, 0.0833, 0.0985, 0.0993,
        0.1184, 0.0648, 0.1374, 0.0995, 0.1111, 0.1113, 0.1319, 0.1010, 0.1106,
        0.1100, 0.1135, 0.0921, 0.0893, 0.0141]


def build_calendar():
    config = load_config()
    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    idx = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    return sorted(set(early) | set(calendar)), fromdate


def run_ml_loop(factor_df, label, config, all_dates):
    """复刻bt_execution ML滚动循环, 返回IC序列"""
    t0 = time.time()
    ml_config = config.config.get('ml', {})
    _ml_df = factor_df
    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))

    _all_dates = sorted(_ml_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    print(f"[{label}] date.dtype={factor_df['date'].dtype} pred_dates={len(_pred_dates)}")

    chunk_starts = list(range(0, len(_pred_dates), _retrain_freq))
    _val_ics = []
    _last_good_predictor = None
    _reused = 0
    _consecutive_reuse = 0
    _min_val_ic = ml_config.get('min_val_ic', 0.03)
    _max_reuse = int(ml_config.get('max_consecutive_reuse', 3))
    _fp_days = int(config.get('dynamic_factor.forward_period', 10))
    chunk_counts = []
    chunk_ics = []

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
        val_ic = ml_predictor.train(train_df, regime_info={'regime': 0})
        chunk_ics.append(val_ic)
        if val_ic is None or val_ic < _min_val_ic:
            if _last_good_predictor is None:
                continue
            if _consecutive_reuse < _max_reuse:
                ml_predictor = _last_good_predictor
                _reused += 1
                _consecutive_reuse += 1
            elif val_ic is not None and val_ic > 0:
                _last_good_predictor = ml_predictor
                _consecutive_reuse = 0
            else:
                ml_predictor = _last_good_predictor
                _reused += 1
                _consecutive_reuse += 1
        else:
            _val_ics.append(val_ic)
            _last_good_predictor = ml_predictor
            _consecutive_reuse = 0

    print(f"[{label}] 有效={len(_val_ics)} 复用={_reused} "
          f"avg_IC={np.mean(_val_ics):.4f} 耗时={(time.time()-t0)/60:.1f}min")
    print(f"[{label}] 样本数命中日志: {chunk_counts == LOGGED_COUNTS}")
    print(f"[{label}] IC: " + ' '.join(f"{x:.6f}" if x is not None else 'None' for x in chunk_ics))
    for tag, ref in (('A(pit)', IC_A), ('B(082730)', IC_B)):
        diffs = [abs(x - r) for x, r in zip(chunk_ics, ref) if x is not None]
        n_close = sum(1 for x, r in zip(chunk_ics, ref) if x is not None and abs(x - r) < 1e-6)
        print(f"[{label}] vs {tag}: max|Δ|={max(diffs):.6f} 一致chunk数={n_close}/23")
    return chunk_ics


def main():
    t0 = time.time()
    config = load_config()
    factor_df = pd.read_parquet(CACHE)
    if factor_df['code'].dtype != object:
        factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)
    print(f"[load] {len(factor_df)} 行, date.dtype={factor_df['date'].dtype}")
    all_dates, fromdate = build_calendar()
    print(f"[cal] 密集日历 {len(all_dates)} 天")

    # pass1: 原样 [us]
    ics1 = run_ml_loop(factor_df, 'pass1[us]', config, all_dates)

    # pass2: 转 [ns] 模拟A现算内存态
    factor_ns = factor_df.copy()
    factor_ns['date'] = factor_ns['date'].astype('datetime64[ns]')
    print(f"\n[cast] date -> {factor_ns['date'].dtype}")
    ics2 = run_ml_loop(factor_ns, 'pass2[ns]', config, all_dates)

    d12 = [abs(a - b) for a, b in zip(ics1, ics2) if a is not None and b is not None]
    print(f"\n[判定] pass1 vs pass2: max|Δ|={max(d12):.6f}")
    if max(d12) > 1e-9:
        print(">> date分辨率[us]vs[ns]改变ML结果 — 分歧机制实锤, 下一步定位敏感行")
    else:
        print(">> date分辨率无关 — 两pass一致, 转fresh路径镜像探针(进程态)")
    print(f"总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    main()
