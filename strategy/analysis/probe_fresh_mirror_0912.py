#!/usr/bin/env python3
"""2026-09-12 fresh路径镜像探针: 完整复刻A(9/11夜)的进程画像 — 现算因子+同进程ML

背景: dtype探针若判date分辨率无关, 则A vs B的IC分歧只剩最后一个可测机制:
  A的进程先做了fresh因子计算(fork Pool+CSV+中性化+排名)再同进程ML;
  B/probe = 干净进程直接加载parquet再ML。两者输入值bit级一致(往返保真已验证),
  若fresh进程画像能复现IC_A → 进程态/对象构造路径是真凶; 若=IC_B → A不可复现。

本探针:
  1. 复刻bt_execution的输入构建(股票池/基本面/all_dates)
  2. 移开现有缓存强制fresh现算 (~10min, 原缓存跑完即恢复, pristine另有备份)
  3. fresh返回的df vs pristine缓存 逐列值比对 (fresh路径确定性测试)
  4. 同进程跑ML全23chunk → IC序列 vs IC_A/IC_B

只读除缓存挪移: 不碰rolling_validation_results。串行执行, 勿与其他重任务并行。
执行: cd strategy && python analysis/probe_fresh_mirror_0912.py > logs/probe_fresh_mirror_0912.log 2>&1
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bt_execution as bte
from core.ml_predictor import MLFactorPredictor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')
PRISTINE = os.path.join(BASE, 'cache', 'd814a206.pristine_0912bak.parquet')

LOGGED_COUNTS = [203233, 259483, 316659, 375408, 436968, 467410, 476509, 492344,
                 508497, 523040, 535951, 550620, 566159, 576517, 588214, 596622,
                 607303, 613222, 620568, 629284, 634172, 635561, 638726]
IC_A = [0.0796, 0.1146, 0.0882, 0.0451, 0.0815, 0.0426, 0.0842, 0.0992, 0.0977,
        0.1186, 0.0650, 0.1367, 0.0987, 0.1096, 0.1109, 0.1324, 0.1016, 0.1109,
        0.1100, 0.1130, 0.0937, 0.0888, 0.0136]
IC_B = [0.0809, 0.1166, 0.0865, 0.0464, 0.0826, 0.0436, 0.0833, 0.0985, 0.0993,
        0.1184, 0.0648, 0.1374, 0.0995, 0.1111, 0.1113, 0.1319, 0.1010, 0.1106,
        0.1100, 0.1135, 0.0921, 0.0893, 0.0141]


def build_stock_file_map():
    """复刻add_data_and_signal线451-483: 扫描+qfq/hfq+股票池过滤"""
    stock_file_map = {}
    for item in os.listdir(bte.DATA_PATH):
        if item.startswith('._'):
            continue
        filepath = bte.DATA_PATH + item
        if item.endswith('_qfq.csv') or item.endswith('_hfq.csv'):
            stock_file_map[item[:-8]] = filepath
    if bte.config.get('stock_pool.enabled', True):
        stock_pool = bte.get_stock_pool(todate=bte._pool_todate(),
                                        bse_exclude=bte.config.get('stock_pool.bse_exclude', True))
        pool_codes = stock_pool | {'sh000001', '000001', 'sh000852', '399006'}
        stock_file_map = {k: v for k, v in stock_file_map.items() if k in pool_codes}
    return stock_file_map


def build_all_dates():
    """复刻add_data_and_signal线532-537+581+584-588: sh000001日历[FROMDATE,TODATE] + raw早段"""
    fromdate = pd.Timestamp(bte.config.get('backtest.fromdate', '2021-01-01'))
    todate = bte.config.get('backtest.todate', None)
    idx = pd.read_csv(os.path.join(bte.DATA_PATH, 'sh000001_qfq.csv'),
                      parse_dates=['datetime'])
    if fromdate is not None:
        idx = idx[idx['datetime'] >= fromdate]
    if todate is not None:
        idx = idx[idx['datetime'] <= pd.Timestamp(todate)]
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(os.path.dirname(bte.DATA_PATH.rstrip('/')), 'raw_data',
                            '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    return sorted(set(early) | set(calendar))


def run_ml_loop(factor_df, label, config, all_dates):
    """复刻bt_execution ML滚动循环(线677-800): pred_dates取自df唯一日期,
    purge定位用外层密集日历all_dates(线723 — 与v2/v3探针一致, 样本数23/23校验)"""
    t0 = time.time()
    ml_config = config.config.get('ml', {})
    _ml_df = factor_df
    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))
    _all_dates = sorted(_ml_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    print(f"[{label}] df唯一日期={len(_all_dates)} pred_dates={len(_pred_dates)} "
          f"(日志=690), 外层日历={len(all_dates)}")

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
        val_ic = ml_predictor.train(train_df, regime_info={'regime': 0})
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

    print(f"[{label}] 有效={len(_val_ics)} 复用={_reused} avg_IC={np.mean(_val_ics):.4f} "
          f"耗时={(time.time()-t0)/60:.1f}min")
    print(f"[{label}] 样本数: " + ' '.join(map(str, chunk_counts)))
    print(f"[{label}] 样本数命中日志: {chunk_counts == LOGGED_COUNTS}")
    print(f"[{label}] IC: " + ' '.join(f"{x:.6f}" if x is not None else 'None' for x in chunk_ics))
    for tag, ref in (('A(pit)', IC_A), ('B(082730)', IC_B)):
        diffs = [abs(x - r) for x, r in zip(chunk_ics, ref) if x is not None]
        print(f"[{label}] vs {tag}: max|Δ|={max(diffs):.6f}")
    return chunk_ics


def main():
    t0 = time.time()
    config = bte.config
    stock_file_map = build_stock_file_map()
    print(f"[map] {len(stock_file_map)} 只 (日志=2718)")

    # 基本面 (复刻main() 1917-1935)
    stock_codes = []
    for f in os.listdir(bte.DATA_PATH):
        if f.startswith('._'):
            continue
        if (f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv'):
            stock_codes.append(f.replace('_qfq.csv', ''))
        elif (f.endswith('_hfq.csv') and f != 'sh000001_hfq.csv'):
            stock_codes.append(f.replace('_hfq.csv', ''))
    if config.get('stock_pool.enabled', True):
        stock_pool = bte.get_stock_pool(todate=bte._pool_todate(),
                                        bse_exclude=config.get('stock_pool.bse_exclude', True))
        stock_codes = [c for c in stock_codes if c in stock_pool]
    fundamental_data = bte.FundamentalData(bte.FUNDAMENTAL_PATH, stock_codes)
    print(f"[fund] {len(stock_codes)} 只 (日志=2717)")

    all_dates = build_all_dates()
    print(f"[cal] all_dates {len(all_dates)} 天 ({all_dates[0].date()}~{all_dates[-1].date()})")

    # 强制fresh路径: 移开现有缓存(指纹未变会命中cache-hit分支)。pristine已备份。
    ASIDE = CACHE + '.freshforced_0912'
    if os.path.exists(ASIDE):
        os.remove(ASIDE)
    if os.path.exists(CACHE):
        os.rename(CACHE, ASIDE)
        print(f"[cache] 移开现有缓存 -> {os.path.basename(ASIDE)} (强制现算)")

    try:
        # fresh现算 (结束后会写新d814a206)
        factor_df, industry_codes, all_dates = bte.prepare_factor_data(
            stock_file_map, fundamental_data, bte.INDUSTRY_KEYWORDS, all_dates, bte.NUM_WORKERS)
    finally:
        # 无论成败, 恢复原缓存到原路径; fresh新产物挪到.fresh_0912供比对
        FRESH = CACHE + '.fresh_0912'
        if os.path.exists(FRESH):
            os.remove(FRESH)
        if os.path.exists(CACHE):
            os.rename(CACHE, FRESH)
        if os.path.exists(ASIDE):
            os.rename(ASIDE, CACHE)
            print(f"[cache] 已恢复原缓存到 {os.path.basename(CACHE)}; "
                  f"fresh产物在 {os.path.basename(FRESH)}")
    print(f"[fresh] {len(factor_df)} 行 × {factor_df.shape[1]} 列, "
          f"date.dtype={factor_df['date'].dtype}")
    print(f"[idx] fresh内存态: {type(factor_df.index).__name__} "
          f"unique={factor_df.index.is_unique} "
          f"monotonic={factor_df.index.is_monotonic_increasing} "
          f"head={factor_df.index[:8].tolist()}")

    # fresh vs pristine 逐列值比对 (fresh路径运行间确定性; 值数组比对避免index对齐)
    if os.path.exists(PRISTINE):
        p = pd.read_parquet(PRISTINE)
        n_dt = n_val = 0
        for c in factor_df.columns:
            x, y = factor_df[c], p[c]
            if x.dtype != y.dtype:
                n_dt += 1
                print(f"  [DTYPE!] {c}: fresh={x.dtype} pristine={y.dtype}")
            if x.dtype == object or isinstance(x.dtype, pd.StringDtype):
                xv = x.astype(object).values
                yv = y.astype(object).values
            else:
                xv = x.values
                yv = y.values
            same = (xv == yv) | (pd.isna(xv) & pd.isna(yv))
            bad = int((~same).sum())
            if bad:
                n_val += 1
                print(f"  [VALUE!] {c}: {bad}/{len(x)} 位置不一致")
        print(f"[diff] fresh vs pristine: {n_dt} 列dtype变, {n_val} 列值变")
        del p
    else:
        print("[diff] pristine备份缺失, 跳过值比对")

    # 同进程ML (外层日历=prepare_factor_data返回的all_dates, 与bt_execution线723一致)
    run_ml_loop(factor_df, 'fresh-ML', config, all_dates)
    print(f"总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    main()
