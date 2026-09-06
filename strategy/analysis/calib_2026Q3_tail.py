#!/usr/bin/env python3
"""
2026Q3 补标定脚本 (2026-09-06)

背景: quarterly_factors/ 只标定到 2026Q2, 而回测/实盘已进入 2026Q3 —
signal_engine._resolve_factor_config 对 2026Q3 返回 None → 全部回退全局yaml权重。
本脚本按原滚动设计补上 2026Q3: 窗口=2021Q3~2026Q2 (20个季度, 止于季度前一天, PIT干净)。

与原 quarterly_calibrate.py 的差异:
- 因子值只算尾部 (2020-06-01 起, 覆盖250日lookback+窗口+20日forward余量),
  不重算 2011-2025 全量 (原full_factor_data.parquet缓存不存在, 全量太贵)
- 只标定 2026Q3 一个季度, 不重写其余 22 个文件 (保持其mtime/内容不变)
- index.yaml 仅追加 2026Q3 条目
"""
import sys
import os
import gc
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.offline_calibration import (
    prepare_calibration_data, compute_factor_data,
    calibrate_industry_regime, select_best_factors,
)
from core.config_loader import load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'config', 'quarterly_factors')


def main():
    config = load_config()

    # === Phase 1: 尾部因子值计算 ===
    # 窗口 2021-07-01~2026-06-30; lookback=250交易日 → 起始前推~12个月;
    # forward_period=20 → 结束后推~1个月 (factor_dates = all_dates[lookback:-forward_period])
    full_start = pd.Timestamp('2020-06-01')
    full_end = pd.Timestamp('2026-08-31')

    print(f"Phase 1: 尾部因子值计算 {full_start.date()} ~ {full_end.date()}")
    stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates, concept_map = \
        prepare_calibration_data(start_date=full_start, end_date=full_end)
    print(f"  股票: {len(stock_codes)}, 交易日: {len(all_dates)}")

    factor_df = compute_factor_data(
        stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
        concept_map=concept_map)
    if factor_df.empty:
        print("错误: 无因子数据")
        return
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    print(f"因子数据: {len(factor_df)} 行, {factor_df['code'].nunique()} 只股票, "
          f"{factor_df['date'].min().date()} ~ {factor_df['date'].max().date()}")
    del stock_file_map, regime_lookup, all_dates
    gc.collect()

    # === Phase 2: 仅标定 2026Q3 ===
    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    candidate_factors = raw_config.get('backtest_factors', [])
    print(f"候选因子: {len(candidate_factors)} 个")

    q_id = '2026Q3'
    calib_start = pd.Timestamp('2021-07-01')   # 20季度窗口起点
    calib_end = pd.Timestamp('2026-06-30')     # 季度前一天
    print(f"\n[{q_id}] 标定窗口: {calib_start.date()} ~ {calib_end.date()}")

    window_df = factor_df[
        (factor_df['date'] >= calib_start) & (factor_df['date'] <= calib_end)
    ].copy()
    if len(window_df) < 10000:
        print(f"  跳过: 标定窗口数据不足 ({len(window_df)} 行)")
        return
    print(f"  数据: {len(window_df)} 行, {window_df['code'].nunique()} 只股票, "
          f"{window_df['date'].nunique()} 天")

    calibration_results = calibrate_industry_regime(
        window_df, candidate_factors, concept_map=concept_map)
    if len(calibration_results) == 0:
        print("  跳过: 无有效行业标定结果")
        return

    industry_config = select_best_factors(
        calibration_results, window_df, concept_map=concept_map)

    n_neutral = sum(1 for v in industry_config.values() if 'factors' in v)
    n_bull = sum(1 for v in industry_config.values() if 'bull_factors' in v)
    n_bear = sum(1 for v in industry_config.values() if 'bear_factors' in v)
    print(f"  industries={len(industry_config)}: neutral={n_neutral}, bull={n_bull}, bear={n_bear}")

    # 保存 2026Q3.yaml (复用原脚本的序列化格式)
    from analysis.quarterly_calibrate import save_quarter_config
    save_quarter_config(industry_config, q_id, OUTPUT_DIR)
    print(f"已保存: {os.path.join(OUTPUT_DIR, q_id + '.yaml')}")

    # index.yaml 追加 2026Q3 (保留现有条目)
    index_path = os.path.join(OUTPUT_DIR, 'index.yaml')
    with open(index_path, 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)
    index['quarters'][q_id] = {
        'start': '2026-07-01',
        'end': '2026-09-30',
        'file': f'{q_id}.yaml',
    }
    with open(index_path, 'w', encoding='utf-8') as f:
        yaml.dump(index, f, allow_unicode=True, sort_keys=False)
    print(f"季度索引已更新: {index_path} ({len(index['quarters'])} 个季度)")


if __name__ == '__main__':
    main()
