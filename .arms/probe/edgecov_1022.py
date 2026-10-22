#!/usr/bin/env python3
"""#368 (边发现算法退化) 的"活表面"复核 (2026-10-22).

问题: `discover_chain_edges` 的 strength = clip((corr−0.6)*2.5, .05,.30) ⇒ 强边
全部饱和到 0.30, 排序退化为字典序. 但"缺陷存在"≠"有影响" — 先量覆盖面:
  - full_sample: 有多少条边/多少 leader/多少 follower, follower 概念有多少成分股;
  - as-of: 在若干日历月末各拟合一次, 看边与 follower 覆盖.
判据: follower 覆盖 0 只股票 ⇒ 通道无活表面, 不修 (同 chan 12 列"三重门全拦").

运行: cd strategy && ../.venv/bin/python ../.arms/probe/edgecov_1022.py
"""
import sys
import numpy as np

sys.path.insert(0, '.')
from core.concept_heat import ConceptHeatCalculator  # noqa: E402


def summarize(calc, edges, label):
    followers = [f for v in edges.values() for f, _ in v]
    strengths = [s for v in edges.values() for _, s in v]
    cov = set()
    for f in set(followers):
        cov.update(calc._concept_stocks.get(f, []))
    print(f'{label}: 边={sum(len(v) for v in edges.values()):2d} '
          f'leaders={len(edges):2d} followers={len(set(followers)):2d} '
          f'饱和(0.30)={sum(1 for s in strengths if s >= 0.2999)}/{len(strengths)} '
          f'follower覆盖股票(静态映射)={len(cov)}')
    if followers:
        print('   followers:', sorted(set(followers))[:12])
    return len(cov)


def main():
    c = ConceptHeatCalculator()
    c.load()
    print(f'stock_concepts={len(c._stock_concepts)} concept_hist={len(c._concept_hist)}')
    print('\n=== full_sample (对照臂: 非 PIT) ===')
    n_full = summarize(c, c._full_sample_edges(), 'full_sample')
    print('\n=== as-of (生产口径: 决策日 → 最近日历月末) ===')
    covs = {}
    for d in ('2023-01-31', '2023-06-30', '2024-06-28', '2024-12-31',
              '2025-06-30', '2025-12-31', '2026-06-30', '2026-09-30', '2026-10-20'):
        covs[d] = summarize(c, c._edges_asof(d), d)
    print(f'\n判定: full_sample 覆盖 {n_full} 股; as-of 非零覆盖月数 '
          f'{sum(1 for v in covs.values() if v)}/{len(covs)}.')
    print('若 full_sample 覆盖 0 (follower 概念无成分股) 且 as-of 几乎全 0 ⇒ 无活表面.')


if __name__ == '__main__':
    main()
