#!/usr/bin/env python3
"""P0-2 CSCV终检 — F8: 0f后态第二轮全臂家族 (2026-09-21)

覆盖9/20扩展(F5-F7)之后的全部新臂: C15×2 + C5cw×3 + C12×4 + 批4a-4g
(C16-C26, C27-C29c, C30-C31, C32-C37, C40-C42) = 36臂 + 基线。
同态: 数据态9/17, fp 6f1cb6b1|0信号消费级与旧fp零差异(2021-25零,
末月29/1385日±3-8码, Z-bracket已收口), span1385, 基线=c577a524
(逐位=生产732,688.93, 与C26/C40-C42无操作臂post逐位一致)。

采纳='base': 生产决策=铁律+4-0≠充分对36臂全否决(唯一4-0的C29×2被
教义否决)。CSCV问题: ①基线作为被选臂的IS冠军份额与条件OOS秩 —
基线优越性是稳健的还是噪声近并列; ②C29×2(4-0被否臂)的IS冠军份额:
教义否决是否有事后统计背书。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscv  # noqa: E402

A20 = '/mnt/d/quant/strategy/arms_20260920'
BASE_CURVE = f'{A20}/C26_deviat0/post_equity_curve.csv'  # 逐位=生产锚点

F8_ARMS = [
    ('base(生产)', BASE_CURVE),
    ('C15_mh3', f'{A20}/C15_mh3/post_equity_curve.csv'),
    ('C15_mh8', f'{A20}/C15_mh8/post_equity_curve.csv'),
    ('C5cw_003', f'{A20}/C5cw_003/post_equity_curve.csv'),
    ('C5cw_010', f'{A20}/C5cw_010/post_equity_curve.csv'),
    ('C5cw_ref0', f'{A20}/C5cw_ref0/post_equity_curve.csv'),
    ('C12_trg008', f'{A20}/C12_trg008/post_equity_curve.csv'),
    ('C12_trg012', f'{A20}/C12_trg012/post_equity_curve.csv'),
    ('C12_rec007', f'{A20}/C12_rec007/post_equity_curve.csv'),
    ('C12_rec014', f'{A20}/C12_rec014/post_equity_curve.csv'),
    ('C16_bonus0', f'{A20}/C16_bonus0/post_equity_curve.csv'),
    ('C17_ladderhi0', f'{A20}/C17_ladderhi0/post_equity_curve.csv'),
    ('C18_ladderlo0', f'{A20}/C18_ladderlo0/post_equity_curve.csv'),
    ('C19_mult0', f'{A20}/C19_mult0/post_equity_curve.csv'),
    ('C20_momadj0', f'{A20}/C20_momadj0/post_equity_curve.csv'),
    ('C21_topdiv0', f'{A20}/C21_topdiv0/post_equity_curve.csv'),
    ('C22_volbreak0', f'{A20}/C22_volbreak0/post_equity_curve.csv'),
    ('C23_limitup0', f'{A20}/C23_limitup0/post_equity_curve.csv'),
    ('C24_exhaust0', f'{A20}/C24_exhaust0/post_equity_curve.csv'),
    ('C25_momchase0', f'{A20}/C25_momchase0/post_equity_curve.csv'),
    ('C26_deviat0', f'{A20}/C26_deviat0/post_equity_curve.csv'),
    ('C27_rw_half', f'{A20}/C27_reward_half/post_equity_curve.csv'),
    ('C28_rw_x15', f'{A20}/C28_reward_x15/post_equity_curve.csv'),
    ('C29_rw_x2*', f'{A20}/C29_reward_x2/post_equity_curve.csv'),
    ('C29b_rw_x25', f'{A20}/C29b_reward_x25/post_equity_curve.csv'),
    ('C29c_rw_x3', f'{A20}/C29c_reward_x3/post_equity_curve.csv'),
    ('C30_damp50', f'{A20}/C30_damp_soft50/post_equity_curve.csv'),
    ('C31_momadj50', f'{A20}/C31_momadj_soft50/post_equity_curve.csv'),
    ('C32_speed06', f'{A20}/C32_speed06/post_equity_curve.csv'),
    ('C33_speed10', f'{A20}/C33_speed10/post_equity_curve.csv'),
    ('C34_tvol024', f'{A20}/C34_tvol024/post_equity_curve.csv'),
    ('C35_tvol032', f'{A20}/C35_tvol032/post_equity_curve.csv'),
    ('C36_emerg05', f'{A20}/C36_emerg05/post_equity_curve.csv'),
    ('C37_emerg08', f'{A20}/C37_emerg08/post_equity_curve.csv'),
    ('C40_mroff', f'{A20}/C40_mroff/post_equity_curve.csv'),
    ('C41_mrsoft', f'{A20}/C41_mrsoft/post_equity_curve.csv'),
    ('C42_mrhard', f'{A20}/C42_mrhard/post_equity_curve.csv'),
]


def load_nav(path):
    import pandas as pd
    df = pd.read_csv(path)
    nav_col = 'nav' if 'nav' in df.columns else df.columns[1]
    nav = df[nav_col].values.astype(float)
    return nav


def block_matrix(navs, S):
    N, T = navs.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    M = np.zeros((N, S))
    for j in range(S):
        seg = navs[:, edges[j]:edges[j + 1]]
        M[:, j] = np.log(seg[:, -1] / seg[:, 0])
    return M


def main():
    print('P0-2 CSCV终检 F8 — 0f后态第二轮全臂家族 (36臂+基线, 采纳=base)')
    navs, labels = [], []
    for lab, path in F8_ARMS:
        if not os.path.exists(path):
            print(f'  缺文件跳过: {lab} {path}')
            continue
        navs.append(load_nav(path))
        labels.append(lab)
    T_min = min(len(n) for n in navs)
    navs = np.array([n[:T_min] for n in navs])
    N = len(labels)
    print(f'  臂: {labels}')
    print(f'  公共窗口 {T_min}行 | 期末NAV {[f"{n[-1]:,.0f}" for n in navs]}')
    for S in (16, 14, 12):
        M = block_matrix(navs, S)
        r = cscv.cscv_pbo(M, n_combos=None, seed=42)
        null = cscv.cscv_null(N=N, S=S, K=30, n_combos=2000, seed=0)
        band = (np.percentile(null, 5), np.percentile(null, 95))
        print(f'  S={S}: PBO={r["pbo"]*100:.1f}%  零带5-95分位'
              f'[{band[0]*100:.1f},{band[1]*100:.1f}]%')
        # IS冠军分布 top6
        order = np.argsort(-r['is_best_hist'])
        print('    IS冠军top6: ' + ', '.join(
            f'{labels[i]}({r["is_best_hist"][i]}/{r["n_combos"]})'
            for i in order[:6]))
        # 基线 + C29×2 的条件OOS秩
        from itertools import combinations
        from scipy.stats import rankdata
        for k_name in ('base(生产)', 'C29_rw_x2*'):
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
    print('读法: PBO落在零带内=选择过程无异于噪声; 基线IS冠军份额高且条件OOS'
          '秩高=基线优越性稳健; C29×2若高份额但OOS秩差=教义否决有背书。')


if __name__ == '__main__':
    main()
