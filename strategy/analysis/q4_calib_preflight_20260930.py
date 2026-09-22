#!/usr/bin/env python3
"""Q4标定预检 (2026-09-22预写, 9/30 calib_2026Q4_tail.py写盘后、全链裁决前执行)

依据 9/22 评分配置审计(yaml_census_audit_20260922.md)的焦点纪律:
  季度标定真实增益 = 新概念覆盖(仅季度概念季度权重IC +0.2423 vs 默认+0.0089)
  + 熊分支bear_factors + 概念解析gating; 存量交集概念的中性/牛市权重重标定
  不被消费(P0中性/牛市读全局配置)。
预检不裁决(裁决=全链四指标冷跑), 只提供三张表:
  1) 覆盖结构: Q4新概念数 / 交集数 / 全局独有数 — 焦点纪律执行度
  2) 覆盖缺口: 2026Q3有买入信号但 全局∪Q4 均无配置的概念 → Q4将落P3兜底
  3) 熊分支刷新量: 交集概念 Q4 bear_factors vs 全局 factors 差异幅度
     (交集概念的季度配置只经熊分支被消费 — 这是唯一值得刷新的存量面)
只读: 不写任何配置文件。执行: .venv/bin/python analysis/q4_calib_preflight_20260930.py
"""
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(BASE_DIR, 'config', 'quarterly_factors')
GLOBAL_Y = os.path.join(BASE_DIR, 'config', 'factor_config.yaml')
VAL_CSV = os.path.join(BASE_DIR, 'rolling_validation_results', 'validation_results.csv')
MAP_P = os.path.join(BASE_DIR, '..', 'data', 'stock_concept_map.pkl')
INCEP_P = os.path.join(BASE_DIR, '..', 'data', 'concept_inception.pkl')


def load_concepts():
    with open(os.path.normpath(MAP_P), 'rb') as f:
        raw = pickle.load(f)
    cmap = {}
    for code, cs in raw.items():
        filtered = [c for c in cs if not any(
            kw in c for kw in ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔',
                               'MSCI', '创业板综', '机构重仓', 'QFII', '破增发',
                               '破发股', '昨日高', '中证500', '深成500', '中盘股',
                               '小盘股', '央国企改革', '西部大开发', '年报预增',
                               '专精特新', '上证380', 'HS300', '微盘股', '百元股',
                               '大盘股', '小盘成长', '小盘价值', '转债标的', '长江三角',
                               '深圳特区', '破净股', '创投'])]
        if filtered:
            cmap[code] = filtered
    if os.path.exists(os.path.normpath(INCEP_P)):
        with open(os.path.normpath(INCEP_P), 'rb') as f:
            raw_i = pickle.load(f)
        incep = {k: pd.Timestamp(v) for k, v in raw_i.items()}
    else:
        incep = {}
    return cmap, incep


def lists_of(cfg, k):
    e = cfg.get(k, {})
    return (set(e.get('factors', [])), set(e.get('bull_factors', [])),
            set(e.get('bear_factors', [])))


def main():
    # 1) Q4文件存在性
    q4_path = os.path.join(QD, '2026Q4.yaml')
    if not os.path.exists(q4_path):
        print('✗ 2026Q4.yaml 不存在 — 先执行 calib_2026Q4_tail.py')
        sys.exit(1)
    q4 = yaml.safe_load(open(q4_path))['industry_factors']
    g = yaml.safe_load(open(GLOBAL_Y))['industry_factors']
    gk, q4k = set(g), set(q4)
    print(f'Q4文件: {len(q4k)} 概念; 全局: {len(gk)}')
    new_c = q4k - gk
    switch = q4k & gk
    print(f'\n=== 表1 覆盖结构 (焦点纪律) ===')
    print(f'新概念(仅Q4, 唯一进P1通道消费的增量): {len(new_c)}')
    print(f'交集概念(仅bear分支消费Q4权重, 中性/牛市仍读全局): {len(switch)}')
    print(f'全局独有(不刷新): {len(gk - q4k)}')
    print(f'新概念样例: {sorted(new_c)[:8]}')

    # 2) 缺口: Q3 buy信号的概念 vs 全局∪Q4
    cmap, incep = load_concepts()
    covered = gk | q4k
    if os.path.exists(VAL_CSV):
        v = pd.read_csv(VAL_CSV, low_memory=False, dtype={'code': str})
        v = v[(v['buy'] == 1) & (v['date'] >= '2026-07-01')].copy()
        gaps = {}
        for ind, n in v['industry'].value_counts().items():
            if ind not in covered:
                gaps[ind] = int(n)
        print(f'\n=== 表2 覆盖缺口 (2026Q3 buys, 概念不在 全局∪Q4 → 落P3兜底) ===')
        if gaps:
            print(f'{len(gaps)} 个缺口概念, 合计 {sum(gaps.values())} buys '
                  f'({100*sum(gaps.values())/max(len(v),1):.1f}% of Q3 buys):')
            for ind, n in sorted(gaps.items(), key=lambda x: -x[1])[:15]:
                print(f'  {ind}: {n}')
        else:
            print('无缺口 — Q3所有buy概念在 全局∪Q4 中均有配置')

    # 3) 交集概念 bear刷新量
    print(f'\n=== 表3 交集概念 bear刷新 (唯一被消费的存量刷新面) ===')
    diff_n = same_n = 0
    wdiff = []
    for k in sorted(switch):
        g_neu, g_bull, g_bear = lists_of(g, k)
        q_neu, q_bull, q_bear = lists_of(q4, k)
        if q_bear != g_bear:
            diff_n += 1
        else:
            same_n += 1
        if g_bear and q_bear and q_bear == g_bear:
            gw = dict(zip(g[k].get('bear_factors', g[k].get('factors', [])),
                          g[k].get('bear_weights', g[k].get('weights', []))))
            qw = dict(zip(q4[k]['bear_factors'], q4[k]['bear_weights']))
            d = sum(abs(qw.get(f, 0) - gw.get(f, 0)) for f in set(gw) | set(qw))
            wdiff.append(d)
    print(f'bear因子集不同: {diff_n} / 相同: {same_n}')
    if wdiff:
        print(f'同集bear权重平均|Δ|={np.mean(wdiff):.4f} (n={len(wdiff)}) — '
              f'熊分支权重刷新幅度')

    # 4) 校验: 新概念的因子必须全部在因子宇宙内 (否则评分时_safe_get_arr返回0=静默退化)
    raw_config = yaml.safe_load(open(GLOBAL_Y))
    universe = set(raw_config.get('backtest_factors', []))
    if universe:
        bad = {}
        for k in sorted(new_c):
            for f in q4[k].get('factors', []):
                if f not in universe:
                    bad.setdefault(k, []).append(f)
        print(f'\n=== 表4 因子宇宙校验 (新概念因子不在全局因子宇宙) ===')
        if bad:
            for k, fs in bad.items():
                print(f'  {k}: {fs}')
        else:
            print('全部通过')
    print('\n预检完成 — 裁决权仍属全链四指标冷跑(runbook C)。')


if __name__ == '__main__':
    main()
