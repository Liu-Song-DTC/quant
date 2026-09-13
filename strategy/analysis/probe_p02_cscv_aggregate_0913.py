#!/usr/bin/env python3
"""P0-2 全实验史聚合过拟合度量 (CSCV家族法, Bailey et al. 2017)

设计 (证据约束下的最严口径):
  - 行 = 同一数据态、同一日期跨度内的候选臂 (绝对块收益, 态内可比);
  - 列 = 等行数时间块 (S偶), 块指标 = log(末/首) 对数收益;
  - 每家族独立CSCV (跨态绝对NAV不可比, 不可混入同一矩阵);
  - PBO必须与同(N,S)零分布带比对 (cscv.cscv_null, 固定M天然离散度);
  - 除PBO外报告: 采纳臂的IS冠军份额、采纳臂OOS秩CDF、IS冠军OOS秩均值。

家族 (证据: 文件名+行数+期末NAV+commit四指标):
  F1 E-K1 bracket (todate 9/2, 9/6态, commit 429194c): EK1a~e, 采纳=EK1c(0.45, 1,060,089)
  F2 E-N5 bracket (todate 9/7, 9/8态, commit da9d888): hard/bearhard/soft35, 采纳=bearhard
     (hard单态NAV更高但9/4态对应配置灾难 → 双态铁律否决, 解释节注明)
  F3 PIT gate对 (span 9/10, 9/12态, commit 939267a): prePIT0911 vs 现基线, 采纳=现基线
  F4 数据管对口 (span 9/10): 锚点(泄漏) vs Stage2(法定截止) — 管道选择特殊案例
  F5 排除: A1armB/A2pct20/0c_v3b/bak_be085 (9/9-9/10态混合实验, 臂角色不可证, 混入=污染)

覆盖缺口论证: 无曲线档案的否决臂 (E-N8/E-N12灾难等) 从来不是全样本IS冠军,
  其缺席不偏置PBO (PBO只关心IS冠军的OOS命运)。已全部入mdd_tolerance_meta注册表。

执行: /mnt/d/quant/.venv/bin/python analysis/probe_p02_cscv_aggregate_0913.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscv  # noqa: E402

RVR = '/mnt/d/quant/strategy/rolling_validation_results'
BK = '/mnt/d/quant/strategy/backups/stage2_0913'

FAMILIES = [
    {
        'name': 'F1 E-K1 bracket (9/6态, span9/2, 采纳=EK1c)',
        'arms': [('EK1a', f'{RVR}/equity_curve.EK1a.csv'),
                 ('EK1b', f'{RVR}/equity_curve.EK1b.csv'),
                 ('EK1c*', f'{RVR}/equity_curve.EK1c.csv'),
                 ('EK1d', f'{RVR}/equity_curve.EK1d.csv'),
                 ('EK1e', f'{RVR}/equity_curve.EK1e.csv')],
        'adopted': 'EK1c*',
    },
    {
        'name': 'F2 E-N5 bracket (9/8态, span9/7, 采纳=bearhard)',
        'arms': [('EN5hard', f'{RVR}/equity_curve.csv.EN5hard_0908'),
                 ('EN5bearhard*', f'{RVR}/equity_curve.csv.EN5bearhard_0908'),
                 ('EN5soft35', f'{RVR}/equity_curve.csv.EN5soft35_0908')],
        'adopted': 'EN5bearhard*',
    },
    {
        'name': 'F3 PIT gate对 (9/12态, span9/10, 采纳=现基线)',
        'arms': [('prePIT(基)', f'{RVR}/equity_curve.prePIT0911.csv'),
                 ('PITgate*', f'{RVR}/equity_curve.csv')],
        'adopted': 'PITgate*',
    },
    {
        'name': 'F4 数据管对口 (泄漏锚点 vs 法定截止Stage2)',
        'arms': [('anchor(泄漏)', f'{BK}/equity_curve.csv'),
                 ('stage2(截止)', f'{BK}/stage2_equity_curve.csv')],
        'adopted': None,  # 管道选择, 非策略臂 — 解释节注明
    },
]


def load_nav(path):
    df = pd.read_csv(path)
    nav_col = 'nav' if 'nav' in df.columns else df.columns[1]
    dates = pd.to_datetime(df.iloc[:, 0])
    nav = df[nav_col].values.astype(float)
    return dates, nav


def block_matrix(navs, S):
    """等行数块, 块指标 = log(末/首)。navs: (N, T)。返回 (N, S)。"""
    N, T = navs.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    M = np.zeros((N, S))
    for j in range(S):
        seg = navs[:, edges[j]:edges[j + 1]]
        M[:, j] = np.log(seg[:, -1] / seg[:, 0])
    return M


def run_family(fam, S_list=(16, 14, 12), K_null=30):
    print('=' * 74)
    print(fam['name'])
    navs, spans, labels = [], [], []
    for lab, path in fam['arms']:
        d, nav = load_nav(path)
        navs.append(nav)
        labels.append(lab)
        spans.append(len(nav))
    T_min = min(spans)
    navs = np.array([n[:T_min] for n in navs])
    N = len(labels)
    print(f'  臂: {labels} | 公共窗口 {T_min}行 '
          f'| 期末NAV {[f"{n[-1]:,.0f}" for n in navs]}')
    for S in S_list:
        M = block_matrix(navs, S)
        r = cscv.cscv_pbo(M, n_combos=None, seed=42)  # 全枚举 (≤100k)
        null = cscv.cscv_null(N=N, S=S, K=K_null, n_combos=2000, seed=0)
        band = (np.percentile(null, 5), np.percentile(null, 95))
        adopted = fam['adopted']
        print(f'  S={S}: PBO={r["pbo"]*100:.1f}%  零带5-95分位'
              f'[{band[0]*100:.1f},{band[1]*100:.1f}]% '
              f'(K={K_null}噪声)  IS冠军分布: ' +
              ', '.join(f'{labels[i]}({r["is_best_hist"][i]})'
                        for i in np.argsort(-r["is_best_hist"])[:N]))
        if adopted is not None:
            k = labels.index(adopted)
            share = r['is_best_hist'][k] / r['n_combos']
            # 采纳臂被选为IS冠军时其OOS秩 (独立循环重算, 与cscv_pbo同一约定)
            ranks_cond = []
            import math
            from itertools import combinations
            for combo in combinations(range(S), S // 2):
                iset = np.zeros(S, dtype=bool)
                iset[list(combo)] = True
                if np.argmax(M[:, iset].sum(1)) != k:
                    continue
                oos = M[:, ~iset].sum(1)
                from scipy.stats import rankdata
                ranks_cond.append(int(rankdata(oos)[k]))
            if ranks_cond:
                print(f'  采纳臂"{adopted}": IS冠军份额={share*100:.1f}%, '
                      f'条件OOS秩 均值{np.mean(ranks_cond):.2f}/N={N} '
                      f'(≥N/2即差于中位)')
            else:
                print(f'  采纳臂"{adopted}": 从未是IS冠军 (IS选择始终指向他臂)')
    return


def main():
    print('P0-2 全实验史CSCV聚合过拟合度量 (2026-09-13)')
    for fam in FAMILIES:
        run_family(fam)
    print('=' * 74)
    print('判定参考: PBO高于零带95分位 → 该家族选择过程过拟合; '
          'PBO≈0且低于零带5分位 → 冠军OOS稳定; 落在零带内 → 无异于噪声选择。')
    print('F4语义: 管道选择非策略臂, "冠军"=锚点(泄漏) — 高PBO意味着泄漏增益'
          '仅由少数时段驱动, 低PBO意味着全时段系统性的泄漏虚增。')


if __name__ == '__main__':
    main()
