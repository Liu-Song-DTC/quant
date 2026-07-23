#!/usr/bin/env python3
"""
季度滚动标定脚本

每季度用前20个季度（5年）数据标定因子，窗口随季度前滚。
因子值全量计算一次，每季度切片跑IC+选择。

输出：
- config/quarterly_factors/2021Q1.yaml ... 2026Q2.yaml  — 每季度因子配置
- config/quarterly_factors/index.yaml  — 季度→日期范围映射
"""

import sys
import os
import gc
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.offline_calibration import (
    prepare_calibration_data, compute_factor_data,
    calibrate_industry_regime, select_best_factors, _downcast_df
)
from core.config_loader import load_config


def get_quarter(date):
    """返回日期所在季度: '2021Q1'"""
    return f"{date.year}Q{(date.month - 1) // 3 + 1}"


def quarter_date_range(quarter_id):
    """返回季度的起止日期"""
    year = int(quarter_id[:4])
    q = int(quarter_id[5])
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start = pd.Timestamp(f"{year}-{start_month:02d}-01")
    # 季度最后一天
    if end_month == 12:
        end = pd.Timestamp(f"{year}-12-31")
    else:
        end = pd.Timestamp(f"{year}-{end_month + 1:02d}-01") - pd.Timedelta(days=1)
    return start, end


def get_calibration_window(backtest_quarter_id, n_quarters=20):
    """返回用于标定backtest_quarter的20季度窗口"""
    year = int(backtest_quarter_id[:4])
    q = int(backtest_quarter_id[5])
    # 回退n_quarters个季度（1季度 = 3个月，n_quarters*3 = 回到5年前同一季度）
    total_months_back = n_quarters * 3
    end_month = (q - 1) * 3 + 3  # 回测季度开始前的最后一个月
    end_year = year
    if end_month > 12:  # Q4 → 12月
        pass  # Q4: end_month=12, start_month=10, fine
    # 标定窗口结束于回测季度前一天
    if q == 1:
        end_date = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=1)
    else:
        end_date = pd.Timestamp(f"{year}-{(q-1)*3+1:02d}-01") - pd.Timedelta(days=1)
    # 标定窗口起始 = end_date - n_quarters*3个月
    start_year = end_date.year - 5
    start_q = q  # 同季度，5年前
    start_date = pd.Timestamp(f"{start_year}-{(start_q-1)*3+1:02d}-01")
    return start_date, end_date


def _to_native(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    import numpy as np
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def save_quarter_config(industry_config, quarter_id, output_dir):
    """保存单季度标定结果"""
    q_path = os.path.join(output_dir, f"{quarter_id}.yaml")
    native_config = _to_native(industry_config)
    with open(q_path, 'w', encoding='utf-8') as f:
        yaml.dump({'industry_factors': native_config}, f,
                  allow_unicode=True, default_flow_style=False, sort_keys=False)
    return q_path


def main():
    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'config', 'quarterly_factors')
    os.makedirs(output_dir, exist_ok=True)

    # 读取候选因子
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    candidate_factors = raw_config.get('backtest_factors', [])
    print(f"候选因子: {len(candidate_factors)} 个")

    # === Phase 1: 全量因子值计算（瓶颈，只跑一次）===
    # 标定窗口最早 2016Q1，最晚结束于 2025Q4。需要 forward_period 余量
    forward_period = config.get('dynamic_factor.forward_period', 20)
    lookback = config.get('industry_factor_config.lookback_days', 250)

    full_start = pd.Timestamp('2011-01-01')  # 给 lookback 足够的暖启动
    full_end = pd.Timestamp('2025-12-31')

    cache_path = os.path.join(base_dir, 'cache', 'full_factor_data.parquet')
    if os.path.exists(cache_path):
        print(f"加载缓存: {cache_path}")
        factor_df = pd.read_parquet(cache_path)
        print(f"因子数据: {len(factor_df)} 行, {factor_df['code'].nunique()} 只股票")
        # 需要 concept_map 用于后续标定
        _, _, _, _, _, concept_map = prepare_calibration_data(
            start_date=pd.Timestamp('2016-01-01'), end_date=pd.Timestamp('2020-12-31'))
    else:
        print("Phase 1: 全量因子值计算 (2011-2025)")
        stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates, concept_map = \
            prepare_calibration_data(start_date=full_start, end_date=full_end)

        factor_df = compute_factor_data(
            stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
            concept_map=concept_map)

        if factor_df.empty:
            print("错误: 无因子数据")
            return

        # 缓存
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        factor_df.to_parquet(cache_path, compression='zstd')
        print(f"因子数据已缓存: {cache_path}")

    # === Phase 2: 季度滚动标定 ===
    # 确定回测季度范围（从因子数据可以覆盖的日期）
    if 'date' not in factor_df.columns:
        print("错误: factor_df无date列")
        return
    factor_df['date'] = pd.to_datetime(factor_df['date'])

    backtest_start = pd.Timestamp('2021-01-01')
    backtest_end = pd.Timestamp('2026-06-30')
    # 避免forward_period导致的未来数据泄漏：标定窗口不能在factor_df最后日期-forward_period之后
    max_calib_end = factor_df['date'].max() - pd.Timedelta(days=forward_period + 5)

    all_quarters = []
    d = backtest_start
    while d <= backtest_end:
        q = get_quarter(d)
        if q not in all_quarters:
            all_quarters.append(q)
        # 跳到下一季度
        month = (d.month - 1) // 3 * 3 + 4
        if month > 12:
            d = pd.Timestamp(f"{d.year + 1}-01-01")
        else:
            d = pd.Timestamp(f"{d.year}-{month:02d}-01")

    print(f"回测季度: {len(all_quarters)} — {all_quarters[0]} ~ {all_quarters[-1]}")

    quarter_index = {}

    for q_id in all_quarters:
        calib_start, calib_end = get_calibration_window(q_id, n_quarters=20)
        if calib_end > max_calib_end:
            calib_end = max_calib_end

        print(f"\n[{q_id}] 标定窗口: {calib_start.date()} ~ {calib_end.date()}")

        # 切片因子数据
        window_df = factor_df[
            (factor_df['date'] >= calib_start) & (factor_df['date'] <= calib_end)
        ].copy()
        if len(window_df) < 10000:
            print(f"  跳过: 标定窗口数据不足 ({len(window_df)} 行)")
            continue

        pct_of_total = len(window_df) / max(len(factor_df), 1) * 100
        print(f"  数据: {len(window_df)} 行 ({pct_of_total:.1f}%), "
              f"{window_df['code'].nunique()} 只股票, {window_df['date'].nunique()} 天")

        # 标定
        calibration_results = calibrate_industry_regime(
            window_df, candidate_factors, concept_map=concept_map)

        n_industries = len(calibration_results)
        if n_industries == 0:
            print(f"  跳过: 无有效行业标定结果")
            continue

        industry_config = select_best_factors(
            calibration_results, window_df, concept_map=concept_map)

        # 统计
        n_with_neutral = sum(1 for v in industry_config.values() if 'factors' in v)
        n_with_bull = sum(1 for v in industry_config.values() if 'bull_factors' in v)
        n_with_bear = sum(1 for v in industry_config.values() if 'bear_factors' in v)
        print(f"  industries={len(industry_config)}: neutral={n_with_neutral}, "
              f"bull={n_with_bull}, bear={n_with_bear}")

        save_quarter_config(industry_config, q_id, output_dir)

        q_start, q_end = quarter_date_range(q_id)
        quarter_index[q_id] = {
            'start': str(q_start.date()),
            'end': str(q_end.date()),
            'file': f'{q_id}.yaml',
        }

        # 清理内存
        del window_df
        gc.collect()

    # 写入 index.yaml
    index_path = os.path.join(output_dir, 'index.yaml')
    with open(index_path, 'w', encoding='utf-8') as f:
        yaml.dump({'quarters': quarter_index}, f, allow_unicode=True, sort_keys=False)
    print(f"\n季度索引已保存: {index_path} ({len(quarter_index)} 个季度)")

    print("\n=== 季度滚动标定完成 ===")


if __name__ == '__main__':
    main()
