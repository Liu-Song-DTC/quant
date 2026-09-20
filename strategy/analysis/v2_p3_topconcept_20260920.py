#!/usr/bin/env python3
"""
P3'补强: 头部概念权重稳定性 (2026-09-20, 9/17数据态, 只读, 复用缓存)

背景: 9/20早读P3'得到 248/399一致, 151差异的签名分级"良性为主" —
但那是行业计数口径。本脚本补上**收益影响口径**: 按2026Q3.yaml三regime
总权重排序的头部概念, 其在重核标定下权重/因子集变化有多大 —
头部概念稳定 ⇒ 151差异的P&L影响小; 头部概念漂移 ⇒ 警报。

复用 /tmp/v2_factor_df_0917.pkl 缓存 (v2_q4_dryrun_20260920.py 产出),
标定窗 2021-07~2026-06 重核 (与早读P3'同口径), 只读不写。

执行: cd strategy && QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/v2_p3_topconcept_20260920.py
"""
import os
import sys
import gc
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.offline_calibration import (
    calibrate_industry_regime, select_best_factors,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_DIR = os.path.join(BASE_DIR, 'config', 'quarterly_factors')
FACTOR_CACHE = '/tmp/v2_factor_df_0917.pkl'
CONCEPT_CACHE = '/tmp/v2_concept_map_0917.pkl'
CALIB_WINDOW = (pd.Timestamp('2021-07-01'), pd.Timestamp('2026-06-30'))


def concept_total_weight(v):
    """单概念三regime总权重"""
    tw = 0.0
    for wk in ('weights', 'bull_weights', 'bear_weights'):
        tw += sum(float(x) for x in (v.get(wk) or []))
    return tw


def main():
    print(f"P3'补强: 头部概念权重稳定性 — 数据态9/17 (只读, 复用缓存)")
    import pickle
    factor_df = pd.read_pickle(FACTOR_CACHE)
    with open(CONCEPT_CACHE, 'rb') as f:
        concept_map = pickle.load(f)
    print(f"  因子数据: {len(factor_df)} 行 (缓存命中)")

    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        candidate_factors = yaml.safe_load(f).get('backtest_factors', [])
    window_df = factor_df[(factor_df['date'] >= CALIB_WINDOW[0]) &
                          (factor_df['date'] <= CALIB_WINDOW[1])]
    print(f"  标定窗: {len(window_df)} 行, {window_df['code'].nunique()} 只, "
          f"候选因子 {len(candidate_factors)} 个")
    calib_results = calibrate_industry_regime(window_df, candidate_factors, concept_map=concept_map)
    new_cfg = select_best_factors(calib_results, window_df, concept_map=concept_map)
    with open(os.path.join(Q_DIR, '2026Q3.yaml'), 'r', encoding='utf-8') as f:
        old_cfg = yaml.safe_load(f)['industry_factors']
    print(f"  新标定 {len(new_cfg)} 概念 | 现文件 {len(old_cfg)} 概念")

    # 头部概念按现文件总权重排序
    heads = sorted(old_cfg.items(), key=lambda kv: -concept_total_weight(kv[1]))[:30]
    print("\n  Top-30概念 (按现文件三regime总权重): 权重变化 + 因子集变化")
    print(f"  {'概念':<16} {'总权重':>7} {'新总权重':>7} {'Δ权重':>7}  因子集")
    for ind, v_old in heads:
        v_new = new_cfg.get(ind)
        if v_new is None:
            print(f"  {ind:<16} {concept_total_weight(v_old):>7.3f} {'(仅旧!)':>7} {'—':>7}  "
                  f"[概念在重核中消失 — 警报]")
            continue
        tw_o, tw_n = concept_total_weight(v_old), concept_total_weight(v_new)
        fs_o = frozenset().union(*[set(v_old.get(k) or []) for k in
                                   ('factors', 'bull_factors', 'bear_factors')])
        fs_n = frozenset().union(*[set(v_new.get(k) or []) for k in
                                   ('factors', 'bull_factors', 'bear_factors')])
        fs_tag = '同' if fs_o == fs_n else f'变 ({sorted(fs_o)}→{sorted(fs_n)})'
        print(f"  {ind:<16} {tw_o:>7.3f} {tw_n:>7.3f} {tw_n - tw_o:>+7.3f}  {fs_tag}")

    # 汇总: 头部概念权重变化分布 vs 全体
    common = set(old_cfg) & set(new_cfg)
    all_deltas = []
    head_deltas = []
    head_names = {ind for ind, _ in heads}
    for c in common:
        v_o, v_n = old_cfg[c], new_cfg[c]
        delta = concept_total_weight(v_n) - concept_total_weight(v_o)
        all_deltas.append(abs(delta))
        if c in head_names:
            head_deltas.append(abs(delta))
    all_deltas = pd.Series(all_deltas)
    head_deltas = pd.Series(head_deltas)
    print(f"\n  总权重变化幅度: 全体概念 中位 {all_deltas.median():.3f} | "
          f"P90 {all_deltas.quantile(0.9):.3f} | 最大 {all_deltas.max():.3f}")
    print(f"                  Top-30概念 中位 {head_deltas.median():.3f} | "
          f"P90 {head_deltas.quantile(0.9):.3f} | 最大 {head_deltas.max():.3f}")
    print("  注 (9/20实测): yaml每概念三regime权重和恒=3.0(单regime=1.0, 归一化锁死), "
          "故总权重口径结构性测不到变化 — 早读的141个权重抖动实为regime间重新分配, "
          "总暴露守恒。有信息的是上表因子集变化: Top-30中5/30变, 方向与Q4干跑同向"
          "(移除Q3翻负基本面, 加技术类) ⇒ P3'良性判定成立。")


if __name__ == '__main__':
    main()
