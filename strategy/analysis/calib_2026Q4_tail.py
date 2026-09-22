#!/usr/bin/env python3
"""
2026Q4 补标定脚本 (2026-09-20 预写, **2026-09-30收盘数据落地后执行**)

镜像 calib_2026Q3_tail.py: 窗口=2021Q4~2026Q3 (20个季度, 止于季度前一天, PIT干净)。
先决: 用户9/30收盘下载+refresh_all完成 (V2 runbook Phase 0 通过) —
      factor_dates = all_dates[lookback:-forward_period], 需要全9/30数据。

与Q3脚本同构:
- 因子值只算尾部 (2020-09-01 起, 覆盖250日lookback+窗口+20日forward余量)
- 只标定 2026Q4 一个季度, 不重写其余 23 个文件 (保持其mtime/内容不变)
- index.yaml 仅追加 2026Q4 条目

执行顺序纪律 (见 v2_oos_runbook_0930.py Phase 5):
① 先跑 V2 OOS审计 (runbook Phase 1-4, 只读) + 9/30全链 (用户批准后);
② 再跑本脚本 (写 quarterly_factors/2026Q4.yaml + index.yaml);
③ 写盘后必须再跑一次全链裁决 — quarterly_factors/*.yaml 在信号指纹内
   (E-D2教训: 季度权重强绑定信号内容), 信号必重生成, 四指标 vs 9/30基线对账。

执行: cd strategy && /mnt/d/quant/.venv/bin/python analysis/calib_2026Q4_tail.py
"""
import sys
import os
import gc
import pickle
from collections import defaultdict
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
    # 窗口 2021-10-01~2026-09-30; lookback=250交易日 → 起始前推~12个月;
    # forward_period=20 → 结束后推~1个月 (factor_dates = all_dates[lookback:-forward_period])
    full_start = pd.Timestamp('2020-09-01')
    full_end = pd.Timestamp('2026-11-30')

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

    # === Phase 2: 仅标定 2026Q4 ===
    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    candidate_factors = raw_config.get('backtest_factors', [])
    print(f"候选因子: {len(candidate_factors)} 个")

    q_id = '2026Q4'
    calib_start = pd.Timestamp('2021-10-01')   # 20季度窗口起点
    calib_end = pd.Timestamp('2026-09-30')     # 季度前一天
    print(f"\n[{q_id}] 标定窗口: {calib_start.date()} ~ {calib_end.date()}")

    window_df = factor_df[
        (factor_df['date'] >= calib_start) & (factor_df['date'] <= calib_end)
    ].copy()
    if len(window_df) < 10000:
        print(f"  跳过: 标定窗口数据不足 ({len(window_df)} 行)")
        return
    print(f"  数据: {len(window_df)} 行, {window_df['code'].nunique()} 只股票, "
          f"{window_df['date'].nunique()} 天")

    # PIT gate (9/22覆盖预检落地): 标定估计样本与生产消费对齐 — 生产signal_engine
    # 只消费概念inception之后的归属行; 标定无gate时新概念权重被前视行稀释
    # (中报类4概念gate_share仅0.040, 96%样本在概念成立之前)。
    incep_path = os.path.join(BASE_DIR, '..', 'data', 'concept_inception.pkl')
    concept_inception = None
    if os.path.exists(incep_path):
        with open(incep_path, 'rb') as f:
            raw_inc = pickle.load(f)
        concept_inception = {k: pd.Timestamp(v) for k, v in raw_inc.items()}
        print(f"  PIT gate: concept_inception {len(concept_inception)}概念已加载")

    calibration_results = calibrate_industry_regime(
        window_df, candidate_factors, concept_map=concept_map,
        concept_inception=concept_inception)
    if len(calibration_results) == 0:
        print("  跳过: 无有效行业标定结果")
        return

    industry_config = select_best_factors(
        calibration_results, window_df, concept_map=concept_map,
        concept_inception=concept_inception)

    # 薄样本回退2026Q3权重 (9/22覆盖预检裁决): gated天数<60的概念重标定
    # 方差高且无OOS余量 — 用Q3权重(经审计§5的23日OOS验证)替代重标定。
    GATED_DAY_MIN = 60
    if concept_inception:
        codes_of = defaultdict(list)
        for code, cs in (concept_map or {}).items():
            for c in cs:
                codes_of[c].append(code)
        q3_path = os.path.join(OUTPUT_DIR, '2026Q3.yaml')
        q3_cfg = yaml.safe_load(open(q3_path, encoding='utf-8'))['industry_factors'] \
            if os.path.exists(q3_path) else {}
        copied, no_q3 = [], []
        for c in list(industry_config):
            inc = concept_inception.get(c)
            if inc is None:
                continue
            codes = codes_of.get(c)
            if not codes:
                continue
            sub = window_df[window_df['code'].isin(codes)]
            gd = sub[sub['date'] >= inc]['date'].nunique()
            if gd < GATED_DAY_MIN:
                if c in q3_cfg:
                    industry_config[c] = q3_cfg[c]
                    copied.append((c, gd))
                else:
                    no_q3.append((c, gd))
        if copied:
            print(f"  薄样本回退Q3权重 ({GATED_DAY_MIN}天内): "
                  f"{len(copied)}概念 {copied}")
        if no_q3:
            print(f"  薄样本但Q3无配置, 保留gated重标定: {no_q3}")

    n_neutral = sum(1 for v in industry_config.values() if 'factors' in v)
    n_bull = sum(1 for v in industry_config.values() if 'bull_factors' in v)
    n_bear = sum(1 for v in industry_config.values() if 'bear_factors' in v)
    print(f"  industries={len(industry_config)}: neutral={n_neutral}, bull={n_bull}, bear={n_bear}")

    # 保存 2026Q4.yaml (复用原脚本的序列化格式)
    from analysis.quarterly_calibrate import save_quarter_config
    save_quarter_config(industry_config, q_id, OUTPUT_DIR)
    print(f"已保存: {os.path.join(OUTPUT_DIR, q_id + '.yaml')}")

    # index.yaml 追加 2026Q4 (保留现有条目)
    index_path = os.path.join(OUTPUT_DIR, 'index.yaml')
    with open(index_path, 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)
    index['quarters'][q_id] = {
        'start': '2026-10-01',
        'end': '2026-12-31',
        'file': f'{q_id}.yaml',
    }
    with open(index_path, 'w', encoding='utf-8') as f:
        yaml.dump(index, f, allow_unicode=True, sort_keys=False)
    print(f"季度索引已更新: {index_path} ({len(index['quarters'])} 个季度)")


if __name__ == '__main__':
    main()
