#!/usr/bin/env python3
"""2026-09-12 ML确定性探针: 复现冷跑分叉归因 (轻量, 不跑全链)

背景: 9/12冷跑 733,005 vs 基线 1,143,938 (-36%)。run_diff已排除数据/代码/因子缓存
(双跑同载 factor_df_2718s_809d_d814a206.parquet), 信号CSV首个差异=2021-04-06
ml_score列, factor_value/industry全同 → 嫌疑=ML训练非确定性。

探针: 复刻 bt_execution 的ML滚动循环chunk 0 (同一缓存df + 两融接入 + 同一窗口切片),
同一输入 train()×2 → 比对 验证IC 和 预测值。若两遍不同 → ML层运行间非确定实锤,
下一步在 n_jobs/线程归约/tree_method 上定位修复。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.ml_predictor import MLFactorPredictor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')


def prepare_ml_df(factor_df, config):
    """复刻bt_execution的两融特征接入 (与冷跑同一路径)"""
    ml_config = config.config.get('ml', {})
    _ml_df = factor_df
    if config.get('margin_features.enabled', False):
        from core.alternative_data import get_provider
        _mfeat = get_provider().get_margin_feature_frame()
        if _mfeat is not None and len(_mfeat):
            _mfeat = _mfeat.copy()
            _fdf = factor_df.sort_values('date')
            _mfeat['date'] = pd.to_datetime(_mfeat['date']).astype(_fdf['date'].dtype)
            _fdf['code'] = _fdf['code'].astype(str)
            _mfeat['code'] = _mfeat['code'].astype(str)
            _ml_df = pd.merge_asof(_fdf, _mfeat, on='date', by='code',
                                   direction='backward', allow_exact_matches=False)
            _wl = ml_config.get('feature_whitelist')
            if _wl is not None:
                for _c in ('rz_chg5', 'rz_chg20', 'rz_buy_ratio', 'rqyl_chg5'):
                    if _c not in _wl:
                        _wl.append(_c)
    return _ml_df


def run_ml_once(config, factor_df):
    """复刻ML滚动循环的chunk 0: 训练一次+预测chunk日期, 返回(ic, preds_df)"""
    ml_config = config.config.get('ml', {})
    _ml_df = prepare_ml_df(factor_df, config)

    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))

    _all_dates = sorted(_ml_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    chunk_dates = _pred_dates[:_retrain_freq]
    first_pred_date = chunk_dates[0]

    _fp_days = int(config.get('dynamic_factor.forward_period', 10))
    _d0_pos = int(np.searchsorted(np.asarray(_all_dates), pd.Timestamp(first_pred_date)))
    _purge_end = _all_dates[max(0, _d0_pos - _fp_days)]
    train_start = first_pred_date - pd.Timedelta(days=_train_window)
    train_mask = (_ml_df['date'] >= train_start) & (_ml_df['date'] < _purge_end)
    train_df = _ml_df[train_mask]
    print(f"  chunk0: train_start={train_start.date()} purge_end={_purge_end.date()} "
          f"first_pred={first_pred_date.date()} 训练行数={len(train_df)}")

    # 复刻bt_execution的regime中位数
    pred_chunk = _ml_df[_ml_df['date'].isin(chunk_dates)]

    p1 = MLFactorPredictor(config.config)
    ic1 = p1.train(train_df, regime_info={'regime': 0})
    preds1 = p1.predict(pred_chunk, regime_info={'regime': 0})
    return ic1, preds1, chunk_dates


def main():
    config = load_config()
    print(f"缓存: {CACHE} ({os.path.getsize(CACHE)/1e9:.2f}GB)")
    factor_df = pd.read_parquet(CACHE)
    print(f"factor_df: {len(factor_df)} 行, {factor_df['code'].nunique()} 只, "
          f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()}\n")

    print("=== 第1遍 ===")
    ic1, preds1, dates = run_ml_once(config, factor_df)
    print(f"  验证IC = {ic1:.10f}")
    print("\n=== 第2遍 (同一输入) ===")
    ic2, preds2, _ = run_ml_once(config, factor_df)
    print(f"  验证IC = {ic2:.10f}")

    print(f"\n=== 判定 ===")
    print(f"验证IC: {ic1:.10f} vs {ic2:.10f} 差 {abs(ic1-ic2):.2e}")
    d = np.abs(preds1 - preds2)
    print(f"预测: n={len(d)} max|Δ|={d.max():.6f} mean|Δ|={d.mean():.6f} "
          f"相对max={(d.max()/(np.abs(preds1).max()+1e-12)):.2%}")
    if ic1 != ic2 or d.max() > 1e-12:
        print(">> ML层运行间非确定: 实锤 (同一缓存+同一代码+固定种子, 输出不同)")
    else:
        print(">> chunk0可复现 — 非确定性在别处(后续chunk复用链/预测路径), 扩大探针范围")


if __name__ == '__main__':
    main()
