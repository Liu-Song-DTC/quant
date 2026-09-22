#!/usr/bin/env python3
"""P3兜底覆盖缺口探针 (2026-09-22, 只读, 零生产写入)

preflight烟测表2发现: 2026Q3 buy信号中40个industry标签(申万行业名)不在
全局377∪季度402概念配置内 → 落P3硬编码4因子兜底, 占Q3 buys 2.5%。
task#195已证P3信号≈零执行(score百分位中位42.6 vs 被执行96.5) — 但**为什么
不执行**未分解: P3默认因子打分垃圾 vs 信号本身差。本探针裁定是否值得给这
40个标签配权重(候选臂: 季度标定扩到申万标签 → 全链冷跑):

  A) 缺口buy的fwd均值 vs 有配置buy — 缺口信号本身有没有肉;
  B) 缺口buy的score分布 vs 有配置buy — P3打分是否被压(若fwd好但score低
     = 打分层埋没, 配权重=真headroom);
  C) 缺口内 Spearman(score,fwd) vs 有配置组 — P3打分信息量;
  D) 执行集与缺口的交集 — 确认零执行。

零写入: 仅stdout + /tmp/probe_p3_gap_20260922.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL_Y = os.path.join(BASE_DIR, 'config', 'factor_config.yaml')
Q3_Y = os.path.join(BASE_DIR, 'config', 'quarterly_factors', '2026Q3.yaml')
VAL_CSV = os.path.join(BASE_DIR, 'rolling_validation_results', 'validation_results.csv')
SEL_CSV = os.path.join(BASE_DIR, 'rolling_validation_results', 'portfolio_selections.csv')


def main():
    g = yaml.safe_load(open(GLOBAL_Y, encoding='utf-8'))['industry_factors']
    q3 = yaml.safe_load(open(Q3_Y, encoding='utf-8'))['industry_factors']
    covered = set(g) | set(q3)
    print(f'覆盖配置: 全局{len(g)} ∪ 2026Q3{len(q3)} = {len(covered)}键')

    v = pd.read_csv(VAL_CSV, usecols=['date', 'code', 'industry', 'score',
                                      'buy', 'future_ret'], low_memory=False,
                    dtype={'code': str})
    v['date'] = pd.to_datetime(v['date'])
    q3b = v[(v['date'] >= '2026-07-01') & (v['buy'] == 1)].copy()
    print(f'2026Q3 buys: {len(q3b)}行, 标签数 {q3b["industry"].nunique()}')

    q3b['gap'] = ~q3b['industry'].isin(covered)
    gapb = q3b[q3b['gap']]
    okb = q3b[~q3b['gap']]
    print(f'\n=== A) 缺口规模 ===')
    print(f'缺口: {len(gapb)}行 ({100*len(gapb)/len(q3b):.1f}%), '
          f'{gapb["industry"].nunique()}标签')
    print(f'缺口标签top10: {gapb["industry"].value_counts().head(10).to_dict()}')

    print(f'\n=== B) fwd与score对比 (缺口 vs 有配置) ===')
    for nm, df in [('缺口', gapb), ('有配置', okb)]:
        print(f'{nm}: n={len(df)}, fwd均值{df["future_ret"].mean():+.5f} '
              f'(中位{df["future_ret"].median():+.5f}), '
              f'score均值{df["score"].mean():.4f} (中位{df["score"].median():.4f}), '
              f'score<0 {100*(df["score"]<0).mean():.1f}%')
    # 中性化fwd差 (按日期): 缺口行fwd - 当日全buy均值
    day_fwd = q3b.groupby('date')['future_ret'].mean()
    gap_ex = gapb['future_ret'] - gapb['date'].map(day_fwd)
    ok_ex = okb['future_ret'] - okb['date'].map(day_fwd)
    print(f'日期中性化fwd: 缺口{np.nanmean(gap_ex):+.5f} vs '
          f'有配置{np.nanmean(ok_ex):+.5f}')

    print(f'\n=== C) score→fwd信息量 (Spearman, 组内) ===')
    for nm, df in [('缺口', gapb), ('有配置', okb)]:
        dd = df.dropna(subset=['score', 'future_ret'])
        if len(dd) > 30:
            rho = dd['score'].corr(dd['future_ret'], method='spearman')
            print(f'{nm}: n={len(dd)}, Spearman(score,fwd)={rho:+.4f}')
        else:
            print(f'{nm}: n={len(dd)} 不足')

    print(f'\n=== D) 执行集交集 ===')
    sel = pd.read_csv(SEL_CSV, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])
    m = sel.merge(v[['date', 'code', 'industry', 'buy']],
                  on=['date', 'code'], how='inner')
    m = m[m['buy'] == 1]
    print(f'执行buy匹配: {len(m)}行, 其中缺口industry: '
          f'{int((~m["industry"].isin(covered)).sum())}行')

    # 缺口标签的申万性: 这些标签是否真是申万行业名 (无概念风格后缀)
    lbls = sorted(x for x in gapb['industry'].dropna().unique() if isinstance(x, str))
    print(f'\n缺口标签样例: {lbls[:12]}')
    gapb.to_csv('/tmp/probe_p3_gap_20260922.csv', index=False)
    print('\n细节 → /tmp/probe_p3_gap_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
