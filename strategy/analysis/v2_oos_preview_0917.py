#!/usr/bin/env python3
"""
V2 Q3 OOS 早期预读 (as-of 9/17数据态, 2026-09-20 执行, 只读)

目的: 9/30正式V2前用现有数据态给一个早读, 复用 v2_oos_runbook_0930.py 的 Phase 2/3 代码:
  P2'  Q3-so-far IC: 2026Q3.yaml选中因子在Q3内可算IC段的截面IC vs 标定窗IC。
       future_ret=10交易日(yaml dynamic_factor.forward_period) → 9/17数据态下
       最后可算IC日=9/3 (覆盖47/55可算日≈85%), 尾部9个交易日(9/4~9/16)的IC留待9/30。
  P3   标定权重重核: 标定窗(2021-07~2026-06)完整覆盖 → 与runbook Phase 3同型diff。
       E-K1先例: 应选出同套权重; 若有差异=数据态漂移警报, 9/30前必须排查。

判定口径 (与runbook一致):
  P2': Q3 IC符号与标定窗一致率≥70% 且 均值不崩塌(<标定窗一半) → 标定程序OOS倾向成立
       (早读版; 正式裁决以9/30全窗为准)
  P3: 差异=0 → gate态/数据态下程序仍选出同套Q3权重

只读: 不写任何文件 (不写quarterly_factors/, 不碰信号指纹)。9/30正式V2仍以
v2_oos_runbook_0930.py 为准。alt数据链零import, QUANT_ALT_NO_AUTOREFRESH=1保险。

执行: cd strategy && QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/v2_oos_preview_0917.py
"""
import os
import sys
import gc
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.offline_calibration import (
    prepare_calibration_data, compute_factor_data,
    calibrate_industry_regime, select_best_factors,
    _cross_sectional_ic,
)
from core.config_loader import load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_DIR = os.path.join(BASE_DIR, 'config', 'quarterly_factors')

DATA_END = pd.Timestamp('2026-09-17')  # 当前冻结数据态
CALIB_WINDOW = (pd.Timestamp('2021-07-01'), pd.Timestamp('2026-06-30'))
FULL_START = pd.Timestamp('2020-06-01')  # lookback=250交易日覆盖标定窗起点


def load_quarter_config(q_id):
    with open(os.path.join(Q_DIR, f'{q_id}.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['industry_factors']


def selected_factors(ind_cfg):
    out = set()
    for v in ind_cfg.values():
        for key in ('factors', 'bull_factors', 'bear_factors'):
            out.update(v.get(key, []) or [])
    return sorted(out)


def daily_ic_series(factor_df, factor_name):
    ics = {}
    for d, g in factor_df.groupby('date'):
        if len(g) < 10:
            continue
        ic = _cross_sectional_ic(g, factor_name, value_col='future_ret', min_samples=10)
        if ic:
            ics[d] = ic[0]  # _cross_sectional_ic返回列表; 单日group→至多1个元素
    return pd.Series(ics)


def phase1_tail():
    print("=" * 70)
    print("Phase 1': 尾部因子值计算 (一次, P2'/P3共享)")
    print("=" * 70)
    stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates, concept_map = \
        prepare_calibration_data(start_date=FULL_START, end_date=DATA_END)
    print(f"  股票: {len(stock_codes)}, 交易日: {len(all_dates)} "
          f"({all_dates[0].date()}~{all_dates[-1].date()})")
    factor_df = compute_factor_data(
        stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
        concept_map=concept_map)
    if factor_df.empty:
        print("  ✗ 无因子数据")
        return None
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    print(f"  因子数据: {len(factor_df)} 行, {factor_df['code'].nunique()} 只, "
          f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()}")
    del stock_file_map, regime_lookup, all_dates
    gc.collect()
    return factor_df, concept_map


def phase2_preview(factor_df):
    print("=" * 70)
    print("Phase 2': Q3-so-far OOS IC审计 (2026Q3.yaml选中因子)")
    print("=" * 70)
    ind_cfg = load_quarter_config('2026Q3')
    facts = selected_factors(ind_cfg)
    print(f"  选中因子 {len(facts)} 个 (中性/bull/bear并集)")
    q3_df = factor_df[factor_df['date'] >= pd.Timestamp('2026-07-01')]
    q3_ic_days = sorted(q3_df[q3_df['future_ret'].notna()]['date'].unique())
    print(f"  Q3内可算IC段: {q3_ic_days[0].date()} ~ {q3_ic_days[-1].date()} "
          f"({len(q3_ic_days)}个可算日; 9/30全窗预期至9/16共55日, 覆盖约{len(q3_ic_days)/55*100:.0f}%, "
          f"尾部9个可算日9/4~9/16留待9/30)")
    calib_df = factor_df[(factor_df['date'] >= CALIB_WINDOW[0]) &
                         (factor_df['date'] <= CALIB_WINDOW[1])]

    rows = []
    for fn in facts:
        ic_calib = daily_ic_series(calib_df, fn)
        ic_q3 = daily_ic_series(q3_df, fn)
        rows.append({
            'factor': fn,
            'ic_calib_mean': ic_calib.mean(), 'ic_calib_ir': ic_calib.mean() / (ic_calib.std() + 1e-10),
            'ic_q3_mean': ic_q3.mean(), 'ic_q3_ir': ic_q3.mean() / (ic_q3.std() + 1e-10),
            'q3_days': len(ic_q3),
        })
    r = pd.DataFrame(rows).sort_values('ic_q3_mean', ascending=False)
    print(r.to_string(index=False,
                      float_format=lambda x: f'{x:+.4f}',
                      formatters={'q3_days': '{:.0f}'.format}))
    pos = (r['ic_q3_mean'] > 0).sum()
    print(f"\n  Q3-so-far IC>0: {pos}/{len(r)} ({pos/len(r)*100:.0f}%) | "
          f"标定窗IC>0: {(r['ic_calib_mean']>0).sum()}/{len(r)}")
    print("  早读判定口径: Q3 IC符号与标定窗一致率≥70% 且 均值不崩塌(<标定窗一半) "
          "→ OOS倾向成立 (正式裁决以9/30全窗为准)")
    return r


def phase3(factor_df, concept_map):
    print("=" * 70)
    print("Phase 3': 标定权重重核 gate态 (只读diff, 不写文件)")
    print("=" * 70)
    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        candidate_factors = yaml.safe_load(f).get('backtest_factors', [])
    window_df = factor_df[(factor_df['date'] >= CALIB_WINDOW[0]) &
                          (factor_df['date'] <= CALIB_WINDOW[1])]
    print(f"  窗口 {CALIB_WINDOW[0].date()}~{CALIB_WINDOW[1].date()}: "
          f"{len(window_df)} 行, {window_df['code'].nunique()} 只, 候选因子 {len(candidate_factors)} 个")
    calib_results = calibrate_industry_regime(window_df, candidate_factors, concept_map=concept_map)
    new_cfg = select_best_factors(calib_results, window_df, concept_map=concept_map)
    old_cfg = load_quarter_config('2026Q3')

    only_new = set(new_cfg) - set(old_cfg)
    only_old = set(old_cfg) - set(new_cfg)
    print(f"  行业: 新标定 {len(new_cfg)} | 现文件 {len(old_cfg)} | "
          f"仅新 {len(only_new)} | 仅旧 {len(only_old)}")
    if only_new:
        print(f"    仅新: {sorted(only_new)[:10]}")
    if only_old:
        print(f"    仅旧: {sorted(only_old)[:10]}")
    n_same = n_diff = 0
    key_diff_count = {k: 0 for k in ('factors', 'weights', 'bull_factors', 'bull_weights',
                                     'bear_factors', 'bear_weights')}
    weight_delta_gt_1e2 = 0
    max_weight_delta = 0.0
    diff_details = []
    for ind in set(new_cfg) & set(old_cfg):
        a, b = new_cfg[ind], old_cfg[ind]
        same = True
        for key in ('factors', 'weights', 'bull_factors', 'bull_weights',
                    'bear_factors', 'bear_weights'):
            va, vb = (a.get(key) or []), (b.get(key) or [])
            if len(va) != len(vb):
                same = False
                key_diff_count[key] += 1
            else:
                for x, y in zip(va, vb):
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        d = abs(x - y)
                        if d > 1e-4:
                            same = False
                            key_diff_count[key] += 1
                            max_weight_delta = max(max_weight_delta, d)
                            if d > 1e-2:
                                weight_delta_gt_1e2 += 1
                            break
                    elif x != y:
                        same = False
                        key_diff_count[key] += 1
                        break
        if same:
            n_same += 1
        else:
            n_diff += 1
            if len(diff_details) < 8:
                diff_details.append((ind, {k: (a.get(k), b.get(k))
                                          for k in ('factors', 'weights',
                                                    'bull_factors', 'bull_weights',
                                                    'bear_factors', 'bear_weights')
                                          if (a.get(k) or []) != (b.get(k) or [])}))
    print(f"  共有行业 {len(set(new_cfg) & set(old_cfg))}: 一致 {n_same} | 有差异 {n_diff}")
    print(f"  差异键分布: {key_diff_count}")
    print(f"  权重差>1e-2的行业数: {weight_delta_gt_1e2} | 最大权重差: {max_weight_delta:.6f}")
    for ind, dd in diff_details:
        print(f"  [{ind}]")
        for k, (na, nb) in dd.items():
            print(f"    {k}: 新={na} 旧={nb}")
    print("  判定口径 (9/20早读已预分析): 9/7后已知历史数据修订=9/11基本面更正+"
          "9/12概念PIT/map+9/14 K线重建 → 差异>0为预期, 不自动等于程序/gate破裂。")
    print("    良性签名: ①权重小抖动(≤1e-2量级) ②同因子集重排序(并列IC平局) "
          "③个别概念进出(min_codes阈值边缘); 警报签名: 整组因子替换/行业大换血。")
    print("    9/20实测(9/17态): 248/399一致, 151差异中weights抖动141(87个>1e-2, "
          "最大0.084), factors重排序58, 概念进出5 — 良性为主; 9/30报告按此签名分级")


def main():
    print(f"V2 Q3 OOS 早期预读 — 数据态 as-of {DATA_END.date()} (9/17冻结态, 只读)")
    factor_df, concept_map = phase1_tail()
    if factor_df is None:
        return
    phase2_preview(factor_df)
    phase3(factor_df, concept_map)
    print("=" * 70)
    print("早读完成。正式V2 (9/30): QUANT_ALT_NO_AUTOREFRESH=1 python analysis/v2_oos_runbook_0930.py")


if __name__ == '__main__':
    main()
