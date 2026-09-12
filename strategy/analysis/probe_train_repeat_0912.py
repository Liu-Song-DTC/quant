#!/usr/bin/env python3
"""2026-09-12 同进程重复训练探针: chunk10(d814a206) 默认参数3连训 + n_jobs=1/20对照

背景: bit级100%同输入下, 8c19c0a8今天=IC_B±5e-5, d814a206多次=IC_B精确 —
同bit不同结果=训练真非确定性。本探针判定:
  1. 同进程同参数3连训: 散布=0 → 进程内确定; 散布>0 → xgboost并行层/浮点归约有噪声
  2. n_jobs=1 vs 4 vs 20: 结果随线程数变 → 并行归约顺序是噪声源;
     不随 → 噪声源在别处(需继续深挖)
判定口径: IC打印到%.17g, 模型JSON sha256逐字节比对。

只读(模型dump到/tmp)。串行。执行: cd strategy && python analysis/probe_train_repeat_0912.py > logs/probe_train_repeat_0912.log 2>&1
"""
import os
import re
import sys
import time
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.ml_predictor import MLFactorPredictor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, '..', 'data')
BT_DATA = os.path.join(DATA_ROOT, 'stock_data', 'backtrader_data')
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')


def main():
    t0 = time.time()
    config = load_config()
    ml_config = config.config.get('ml', {})
    factor_df = pd.read_parquet(CACHE)
    if factor_df['code'].dtype != object:
        factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)
    print(f'[load] {len(factor_df)} 行 ({time.time()-t0:.0f}s)')

    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    idx = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    all_dates = sorted(set(early) | set(calendar))

    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))
    _fp_days = int(config.get('dynamic_factor.forward_period', 10))
    _all_dates = sorted(factor_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    assert len(_pred_dates) == 690, f'pred_dates={len(_pred_dates)}'
    chunk_start = 10 * _retrain_freq
    first_pred_date = _pred_dates[chunk_start]
    _d0_pos = int(np.searchsorted(np.asarray(all_dates), pd.Timestamp(first_pred_date)))
    _purge_end = all_dates[max(0, _d0_pos - _fp_days)]
    train_start = first_pred_date - pd.Timedelta(days=_train_window)
    train_mask = (factor_df['date'] >= train_start) & (factor_df['date'] < _purge_end)
    train_df = factor_df[train_mask].copy()
    print(f'[chunk10] pred首日={first_pred_date.date()} purge_end={_purge_end.date()} '
          f'train={train_df["date"].min().date()}~{train_df["date"].max().date()} '
          f'样本={len(train_df)} (日志=523040)')

    tmp = tempfile.mkdtemp(prefix='mlrep_')
    runs = [('rep1', None), ('rep2', None), ('rep3', None), ('n_jobs1', 1), ('n_jobs20', 20)]
    results = {}
    for label, nj in runs:
        t1 = time.time()
        p = MLFactorPredictor(config.config)
        if nj is not None:
            p.xgb_params['n_jobs'] = nj
        ic = p.train(train_df, regime_info={'regime': 0})
        nthread = None
        if p.model is not None:
            m = re.search(r'"nthread"\s*:\s*"?(\d+)"?',
                          p.model.get_booster().save_config())
            nthread = m.group(1) if m else None
        sha = ''
        if hasattr(p, '_ensemble_models'):
            digests = []
            for i, _m in enumerate(p._ensemble_models):
                fp = os.path.join(tmp, f'{label}.{i}.json')
                _m.save_model(fp)
                digests.append(hashlib.sha256(open(fp, 'rb').read()).hexdigest()[:16])
            sha = ' '.join(digests)
        results[label] = ic
        print(f'[{label}] IC={ic:.17g} nthread={nthread} sha={sha} '
              f'({time.time()-t1:.0f}s)')
    print('[cmp] 3连训默认 pairwise max|Δ|: '
          f'{max(abs(results["rep1"]-results["rep2"]), abs(results["rep1"]-results["rep3"]), abs(results["rep2"]-results["rep3"])):.3g}')
    print(f'[cmp] rep1 vs n_jobs1: {abs(results["rep1"]-results["n_jobs1"]):.3g}')
    print(f'[cmp] rep1 vs n_jobs20: {abs(results["rep1"]-results["n_jobs20"]):.3g}')
    print(f'总耗时 {(time.time()-t0)/60:.1f}min')


if __name__ == '__main__':
    main()
