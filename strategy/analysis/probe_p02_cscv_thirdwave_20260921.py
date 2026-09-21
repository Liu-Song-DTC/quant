#!/usr/bin/env python3
"""P0-2 CSCV终检 F9 — census第三波全臂家族 (2026-09-21)

覆盖第三波39臂: T(调仓周期)×5 + U(无交易带)×8 + X(理想仓位倍数)×3 +
Y(exit_speed通电)×2 + AA(min_rank_pct)×6 + AB(行业权重上限)×4 +
AC(rank_weight_cap)×2 + AD(每股行业上限)×2 + AE(力竭权重上限)×2 +
AF(力竭缩减系数)×2 + S(日历敞口)×3 + 基线。同态: 数据态9/17,
fp 6f1cb6b1|0, span≈1385, 基线=C26_deviat0(逐位=生产732,688.93)。

采纳='base': 39臂零采纳(铁律/孤立齿/单调边界argmax/死态=最优态)。
CSCV问题: ①基线作为被选臂的IS冠军份额与条件OOS秩 — "生产是全局最优"
是稳健结论还是噪声近并列; ②被教义否决的4-0臂AA1与3-1臂AB1/AF2:
否决是否有事后统计背书(IS冠军份额高但条件OOS秩差=彩票确认)。
③X3=生产逐位复现臂 — 与基线应逐位并列(identity烟测)。
"""
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscv  # noqa: E402

A20 = '/mnt/d/quant/strategy/arms_20260920'
BASE_CURVE = f'{A20}/C26_deviat0/post_equity_curve.csv'  # 逐位=生产锚点

THIRD_WAVE_ARMS = [
    # T批次: rebalance_interval {7,14,20,8,9} (生产10d未在臂内, 见基线)
    ('T1_rb07', f'{A20}/T1_rb07/post_equity_curve.csv'),
    ('T2_rb14', f'{A20}/T2_rb14/post_equity_curve.csv'),
    ('T3_rb20', f'{A20}/T3_rb20/post_equity_curve.csv'),
    ('T4_rb08', f'{A20}/T4_rb08/post_equity_curve.csv'),
    ('T5_rb09', f'{A20}/T5_rb09/post_equity_curve.csv'),
    # U批次: 无交易带 {0.005,0.02,0.05,0.10,0.15,0.07,0.08,0.12}×equity
    ('U1_tol0005', f'{A20}/U1_tol0005/post_equity_curve.csv'),
    ('U2_tol002', f'{A20}/U2_tol002/post_equity_curve.csv'),
    ('U3_tol005', f'{A20}/U3_tol005/post_equity_curve.csv'),
    ('U4_tol010', f'{A20}/U4_tol010/post_equity_curve.csv'),
    ('U5_tol015', f'{A20}/U5_tol015/post_equity_curve.csv'),
    ('U6_tol007', f'{A20}/U6_tol007/post_equity_curve.csv'),
    ('U7_tol008', f'{A20}/U7_tol008/post_equity_curve.csv'),
    ('U8_tol012', f'{A20}/U8_tol012/post_equity_curve.csv'),
    # X批次: ideal_position_max_mult {1.0,1.5,3.0} (生产2.0未在臂内)
    ('X1_lot10', f'{A20}/X1_lot10/post_equity_curve.csv'),
    ('X2_lot15', f'{A20}/X2_lot15/post_equity_curve.csv'),
    ('X3_lot30', f'{A20}/X3_lot30/post_equity_curve.csv'),  # 生产逐位复现臂
    # Y批次: exit_speed通电 {0.8,0.5}
    ('Y1_exit08', f'{A20}/Y1_exit08/post_equity_curve.csv'),
    ('Y2_exit05', f'{A20}/Y2_exit05/post_equity_curve.csv'),
    # AA批次: min_rank_pct {0.2,0.5,0.05,0.10,0.15,0.25} (生产0.3)
    ('AA1_rank02', f'{A20}/AA1_rank02/post_equity_curve.csv'),
    ('AA2_rank05', f'{A20}/AA2_rank05/post_equity_curve.csv'),
    ('AA3_rank005', f'{A20}/AA3_rank005/post_equity_curve.csv'),
    ('AA4_rank010', f'{A20}/AA4_rank010/post_equity_curve.csv'),
    ('AA5_rank015', f'{A20}/AA5_rank015/post_equity_curve.csv'),
    ('AA6_rank025', f'{A20}/AA6_rank025/post_equity_curve.csv'),
    # AB批次: industry_max_weight {0.2,0.3,0.5,0.6} (生产0.4)
    ('AB1_indw02', f'{A20}/AB1_indw02/post_equity_curve.csv'),
    ('AB2_indw03', f'{A20}/AB2_indw03/post_equity_curve.csv'),
    ('AB3_indw05', f'{A20}/AB3_indw05/post_equity_curve.csv'),
    ('AB4_indw06', f'{A20}/AB4_indw06/post_equity_curve.csv'),
    # AC批次: rank_weight_cap {0.15,0.35} (生产0.25)
    ('AC1_rankw015', f'{A20}/AC1_rankw015/post_equity_curve.csv'),
    ('AC2_rankw035', f'{A20}/AC2_rankw035/post_equity_curve.csv'),
    # AD批次: max_per_industry {2,4} (生产0=无限)
    ('AD1_mpi2', f'{A20}/AD1_mpi2/post_equity_curve.csv'),
    ('AD2_mpi4', f'{A20}/AD2_mpi4/post_equity_curve.csv'),
    # AE批次: exhaustion_max_weight {0.04,0.12} (生产0.08)
    ('AE1_exh004', f'{A20}/AE1_exh004/post_equity_curve.csv'),
    ('AE2_exh012', f'{A20}/AE2_exh012/post_equity_curve.csv'),
    # AF批次: exhaustion_reduce_mult {0.25,0.75} (生产0.5)
    ('AF1_exhr025', f'{A20}/AF1_exhr025/post_equity_curve.csv'),
    ('AF2_exhr075', f'{A20}/AF2_exhr075/post_equity_curve.csv'),
    # S批次: 日历季节性敞口 (8/9月alpha系统性为负)
    ('S1_cal_AugSep05', f'{A20}/S1_cal_AugSep05/post_equity_curve.csv'),
    ('S2_cal_Sep05', f'{A20}/S2_cal_Sep05/post_equity_curve.csv'),
    ('S3_cal_Aug05', f'{A20}/S3_cal_Aug05/post_equity_curve.csv'),
]


def load_nav(path):
    import pandas as pd
    df = pd.read_csv(path)
    nav_col = 'nav' if 'nav' in df.columns else df.columns[1]
    return df[nav_col].values.astype(float)


def block_matrix(navs, S):
    N, T = navs.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    M = np.zeros((N, S))
    for j in range(S):
        seg = navs[:, edges[j]:edges[j + 1]]
        M[:, j] = np.log(seg[:, -1] / seg[:, 0])
    return M


def main():
    print('P0-2 CSCV终检 F9 — census第三波全臂家族 (39臂+基线, 采纳=base)')
    navs, labels = [], []
    for lab, path in [(('base(生产)', BASE_CURVE))] + THIRD_WAVE_ARMS:
        if not os.path.exists(path):
            print(f'  缺文件跳过: {lab} {path}')
            continue
        navs.append(load_nav(path))
        labels.append(lab)
    T_min = min(len(n) for n in navs)
    navs = np.array([n[:T_min] for n in navs])
    N = len(labels)
    print(f'  臂数: {N} | 公共窗口 {T_min}行 | '
          f'期末NAV范围 [{min(navs[:, -1]):,.0f}, {max(navs[:, -1]):,.0f}]')
    for S in (16, 14, 12):
        M = block_matrix(navs, S)
        r = cscv.cscv_pbo(M, n_combos=None, seed=42)
        null = cscv.cscv_null(N=N, S=S, K=30, n_combos=2000, seed=0)
        band = (np.percentile(null, 5), np.percentile(null, 95))
        print(f'  S={S}: PBO={r["pbo"]*100:.1f}%  零带5-95分位'
              f'[{band[0]*100:.1f},{band[1]*100:.1f}]%')
        order = np.argsort(-r['is_best_hist'])
        print('    IS冠军top6: ' + ', '.join(
            f'{labels[i]}({r["is_best_hist"][i]}/{r["n_combos"]})'
            for i in order[:6]))
        from scipy.stats import rankdata
        for k_name in ('base(生产)', 'T1_rb07', 'T5_rb09', 'T3_rb20',
                       'AA1_rank02', 'AB1_indw02', 'AF2_exhr075', 'X3_lot30'):
            if k_name not in labels:
                print(f'    "{k_name}": 无此臂, 跳过')
                continue
            k = labels.index(k_name)
            share = r['is_best_hist'][k] / r['n_combos']
            ranks_cond = []
            for combo in combinations(range(S), S // 2):
                iset = np.zeros(S, dtype=bool)
                iset[list(combo)] = True
                if np.argmax(M[:, iset].sum(1)) != k:
                    continue
                oos = M[:, ~iset].sum(1)
                ranks_cond.append(int(rankdata(oos)[k]))
            if ranks_cond:
                print(f'    "{k_name}": IS冠军份额={share*100:.1f}%, '
                      f'条件OOS秩均值{np.mean(ranks_cond):.2f}/N={N} '
                      f'(≥{(N+1)/2:.0f}即差于中位)')
            else:
                print(f'    "{k_name}": 从未是IS冠军')
    print('=' * 74)
    print('读法: PBO落在零带内=第三波选择过程无异于噪声(零采纳是保守正确); '
          '基线IS冠军份额高且条件OOS秩高=生产优越性稳健; AA1/AB1/AF2若'
          '高份额但OOS秩差=彩票否决获背书; X3与基线应逐位并列(identity)。')


if __name__ == '__main__':
    main()
