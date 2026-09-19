#!/usr/bin/env python3
"""P0-2 聚合过拟合度量 — 2026-09-20扩展 (批次1/9-15态/0f后校准期家族)

覆盖9/13原报告(probe_p02_cscv_aggregate_0913.py, F1-F4)之后的新实验史:
  F5  批次1 (9/14区间态, 基线868,611, span1382, 全否决): 0d/C2_035/C2_055/
      C5_buffer_005/010/C8_005/015 + 基线 = 8臂。注意C5_buffer_005在此态
      +106k(3-1 MDD败) → 不对称变体C5c最终在0f后态采纳。
  F6  9/15态高分支 (基线1,728,548, span1383, 采纳=dragon_tiger OFF诚实修正):
      0g_dt_off(1,671,592) — 2臂对, 诚实修正非收益臂。
  F7  0f后校准期 (9/17态+0f日历池, 信号fp 00b4bdd7|0, span1385, 采纳=C5c=732,689):
      C5六臂族(A_repro=G复现/ref/dd05/dd08/fastoff/C5c*) + C1七臂(030/050/055/
      52_0/57_0/faithful0.55) + 批次2十一臂 + 批次3四臂 = 28臂, 史上最大单家族。

同态可比性: 各家族臂共享同数据态+同span+同信号集(fp豁免或注入均为有意配置差);
跨家族绝对NAV不可比(池口径/数据态不同), 家族间不合并。
覆盖缺口: 0f-v3 bracket臂E/F曲线未归档(仅日志), 该bracket=池口径诚实修正
(受双态纪律约束), 非收益校准选择; 其缺席对PBO影响=保守方向(少计入一次
过拟合选择), 已在F7 adopted=当前生产内体现最终落地值。
执行: /mnt/d/quant/.venv/bin/python analysis/probe_p02_cscv_extension_20260920.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscv  # noqa: E402

A17 = '/mnt/d/quant/strategy/arms_20260917'
A19 = '/mnt/d/quant/strategy/arms_20260919'
A20 = '/mnt/d/quant/strategy/arms_20260920'

FAMILIES = [
    {
        'name': 'F5 批次1 (9/14区间态, 基线868,611, 全否决无采纳)',
        'arms': [('base(868k)', f'{A17}/0d_impact_on/pre_equity_curve.csv'),
                 ('0d_impact', f'{A17}/0d_impact_on/post_equity_curve.csv'),
                 ('C2_035', f'{A17}/C2_boost_035/post_equity_curve.csv'),
                 ('C2_055', f'{A17}/C2_boost_055/post_equity_curve.csv'),
                 ('C5buf_005', f'{A17}/C5_buffer_005/post_equity_curve.csv'),
                 ('C5buf_010', f'{A17}/C5_buffer_010/post_equity_curve.csv'),
                 ('C8_005', f'{A17}/C8_tbonus_005/post_equity_curve.csv'),
                 ('C8_015', f'{A17}/C8_tbonus_015/post_equity_curve.csv')],
        'adopted': None,  # 全否决; C5buf_005 3-1(MDD)败 → 变体C5c异地采纳
    },
    {
        'name': 'F6 9/15态高分支 (基线1,728,548, 采纳=0g OFF诚实修正)',
        'arms': [('dt_ON(基)', f'{A17}/0g_dt_off/pre_equity_curve.csv'),
                 ('dt_OFF*', f'{A17}/0g_dt_off/post_equity_curve.csv')],
        'adopted': 'dt_OFF*',  # 生产已关dragon_tiger (诚实修正, 豁免铁律)
    },
    {
        'name': 'F7 0f后校准期 (fp 00b4bdd7|0, 采纳=C5c=732,689)',
        'arms': [('A_reproG', f'{A17}/A_repro_0915state/post_equity_curve.csv'),
                 ('C5_ref', f'{A17}/C5_ref_005/post_equity_curve.csv'),
                 ('C5b_dd05', f'{A17}/C5b_dd05/post_equity_curve.csv'),
                 ('C5b_dd08', f'{A17}/C5b_dd08/post_equity_curve.csv'),
                 ('C5d_fast', f'{A17}/C5d_fastoff/post_equity_curve.csv'),
                 ('C5c*', f'{A17}/C5c_profitonly/post_equity_curve.csv'),
                 ('C1_030', f'{A19}/C1_mlblend_030/post_equity_curve.csv'),
                 ('C1_050', f'{A19}/C1_mlblend_050/post_equity_curve.csv'),
                 ('C1_055', f'{A19}/C1_mlblend_055/post_equity_curve.csv'),
                 ('C1_52', f'{A19}/C1_mlblend_52_0/post_equity_curve.csv'),
                 ('C1_57', f'{A19}/C1_mlblend_57_0/post_equity_curve.csv'),
                 ('C1_faith55', f'{A19}/C1_faithful_0_55/post_equity_curve.csv'),
                 ('C2_035b', f'{A19}/C2_035/post_equity_curve.csv'),
                 ('C2_055b', f'{A19}/C2_055/post_equity_curve.csv'),
                 ('C8_005b', f'{A19}/C8_005/post_equity_curve.csv'),
                 ('C8_015b', f'{A19}/C8_015/post_equity_curve.csv'),
                 ('C3_off', f'{A19}/C3_off/post_equity_curve.csv'),
                 ('C4_025', f'{A19}/C4_025/post_equity_curve.csv'),
                 ('C6_010', f'{A19}/C6_010/post_equity_curve.csv'),
                 ('C6_020', f'{A19}/C6_020/post_equity_curve.csv'),
                 ('C9_015', f'{A19}/C9_015/post_equity_curve.csv'),
                 ('C9_025', f'{A19}/C9_025/post_equity_curve.csv'),
                 ('C11_lb60', f'{A19}/C11_lb60/post_equity_curve.csv'),
                 ('C2_050', f'{A20}/C2_050/post_equity_curve.csv'),
                 ('C2_060', f'{A20}/C2_060/post_equity_curve.csv'),
                 ('C2_065', f'{A20}/C2_065/post_equity_curve.csv'),
                 ('C14_080', f'{A20}/C14_080/post_equity_curve.csv')],
        'adopted': 'C5c*',
    },
]


def load_nav(path):
    import pandas as pd
    df = pd.read_csv(path)
    nav_col = 'nav' if 'nav' in df.columns else df.columns[1]
    dates = pd.to_datetime(df.iloc[:, 0])
    nav = df[nav_col].values.astype(float)
    return dates, nav


def block_matrix(navs, S):
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
        r = cscv.cscv_pbo(M, n_combos=None, seed=42)
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
            ranks_cond = []
            from itertools import combinations
            from scipy.stats import rankdata
            for combo in combinations(range(S), S // 2):
                iset = np.zeros(S, dtype=bool)
                iset[list(combo)] = True
                if np.argmax(M[:, iset].sum(1)) != k:
                    continue
                oos = M[:, ~iset].sum(1)
                ranks_cond.append(int(rankdata(oos)[k]))
            if ranks_cond:
                print(f'  采纳臂"{adopted}": IS冠军份额={share*100:.1f}%, '
                      f'条件OOS秩 均值{np.mean(ranks_cond):.2f}/N={N} '
                      f'(≥N/2即差于中位)')
            else:
                print(f'  采纳臂"{adopted}": 从未是IS冠军 (IS选择始终指向他臂)')


def main():
    print('P0-2 全实验史CSCV聚合过拟合度量 — 2026-09-20扩展 (F5/F6/F7)')
    for fam in FAMILIES:
        run_family(fam)
    print('=' * 74)
    print('判定参考: PBO高于零带95分位 → 该家族选择过程过拟合; '
          'PBO≈0且低于零带5分位 → 冠军OOS稳定; 落在零带内 → 无异于噪声选择。')


if __name__ == '__main__':
    main()
