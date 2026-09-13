#!/usr/bin/env python3
"""cscv.py — CSCV回测过拟合概率框架 (Bailey, Borwein, López de Prado 2017)

输入: 性能矩阵 M (N个配置 × S个等长时块, S为偶数)。对每个IS/OOS组合
  (C(S,S/2)全枚举≤2000, 否则抽样): IS最优配置n*在OOS中的相对秩ω∈(0,1),
  logit λ=ln(ω/(1−ω))。PBO = λ<0的组合占比 = IS冠军在OOS低于中位的概率。
  秩约定(论文): rank 1 = OOS最差 (rankdata直出, 非反向)。

判定参考(论文): PBO>0.5拒绝。
  **单矩阵PBO零分布带(2026-09-13实证)**: "噪声→PBO≈0.5"只在(E_M, E_C)联合
  期望下成立; 固定矩阵M的PBO有天然大离散度, 驱动项=E_C[OOS_冠军|M]
  = R_胜者(被IS冠军筛选的行和) − E[max IS] ≠ 0 (恒等式 IS+OOS=行和)。
  故真实矩阵的PBO必须与同(N,S)零分布带(cscv_null, K个iid噪声矩阵的PBO)
  比对, 不得直接对0.5点值判读。典型零带宽≈±0.15~0.2。

自验证矩阵: 真信号(某行全域+2.0)→PBO=0, 须低于零带5分位;
  交替结构(偶数chunk好的行奇数chunk必差)→PBO须高于零带95分位;
  噪声→PBO落在零带内, 零带均值≈0.5(联合期望, 精确成立)。
用法: import cscv; r = cscv.cscv_pbo(M, n_combos=None, seed=42)
      null = cscv.cscv_null(K=20); print(r['pbo'], null)
独立文件, 无依赖, 可被探针复用。执行自验:
  .venv/bin/python strategy/analysis/cscv.py
"""
import itertools
import math
import numpy as np
from scipy.stats import rankdata

# 全枚举物化上限 (C(16,8)=12,870×~200B≈2.6MB可枚举; S=32时6e8个组合
# ×~200B≈120GB — 2026-09-13 OOM事故根因, 必须抽样不物化)
_ENUM_CAP = 100_000


def _sample_combos(S, n_combos, seed):
    """不物化全组合: 直接抽n_combos个互异组合 (replace=False语义保持)"""
    rng = np.random.default_rng(seed)
    seen = set()
    combos = []
    half = S // 2
    while len(combos) < n_combos:
        c = tuple(sorted(rng.permutation(S)[:half].tolist()))
        if c in seen:
            continue
        seen.add(c)
        combos.append(c)
    return combos


def cscv_pbo(M, n_combos=None, seed=42):
    """M: (N, S) ndarray, 值越大越好, 须全有限. 返回dict."""
    M = np.asarray(M, dtype=float)
    assert M.ndim == 2 and np.isfinite(M).all(), 'M须为全有限的(N,S)矩阵'
    N, S = M.shape
    assert S >= 4 and S % 2 == 0, f'S={S}须为≥4的偶数'
    total = math.comb(S, S // 2)
    if n_combos is None or n_combos >= total:
        assert total <= _ENUM_CAP, (
            f'全枚举需物化{total}个组合(约{total * 200 / 1e9:.0f}GB), S={S}过大; '
            f'必须显式传n_combos抽样')
        combos = list(itertools.combinations(range(S), S // 2))
    else:
        combos = _sample_combos(S, n_combos, seed)
    is_best_hist = np.zeros(N, dtype=int)
    oos_ranks = np.zeros(len(combos), dtype=int)
    logits = np.zeros(len(combos))
    for k, combo in enumerate(combos):
        iset = np.zeros(S, dtype=bool)
        iset[list(combo)] = True
        is_perf = M[:, iset].sum(axis=1)
        oos_perf = M[:, ~iset].sum(axis=1)
        n_star = int(np.argmax(is_perf))
        is_best_hist[n_star] += 1
        oos_rank = int(rankdata(oos_perf, method='average')[n_star])
        # 论文约定: rank 1 = OOS最差 (rankdata 1=最小值)。ω=rank/(N+1)
        oos_ranks[k] = oos_rank
        omega = oos_rank / (N + 1)
        logits[k] = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(logits < 0))
    return {
        'pbo': pbo,
        'n_combos': len(combos),
        'oos_ranks': oos_ranks,
        'logits': logits,
        'is_best_hist': is_best_hist,
        'N': N, 'S': S,
        'mean_omega': float(np.mean(oos_ranks / (N + 1))),
        'sd_lambda': float(np.std(logits)),
        # IS冠军OOS秩≤k的累计占比 (k=1..N), 与均匀分布对照
        'rank_cdf': [float(np.mean(oos_ranks <= k)) for k in range(1, N + 1)],
    }


def summarize(res, label=''):
    print(f'[{label}] N={res["N"]} S={res["S"]} 组合数={res["n_combos"]}')
    print(f'  PBO = {res["pbo"]*100:.1f}%  (λ均值'
          f'{np.mean(res["logits"]):+.2f}, λσ={res["sd_lambda"]:.2f}, '
          f'ω均值{res["mean_omega"]:.2f})')
    top3 = np.argsort(-res['is_best_hist'])[:3]
    print(f'  IS冠军分布top3: ' +
          ', '.join(f'行{i}({res["is_best_hist"][i]})' for i in top3))


def _synth_noise(N=50, S=32, seed=7):
    return np.random.default_rng(seed).standard_normal((N, S))


def _synth_signal(N=50, S=32, seed=7, mu=2.0):
    """行0全域+mu, 须压倒性(IS和=mu*S/2 >> 噪声σ*sqrt(S/2))"""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((N, S))
    M[0] += mu
    return M


def _synth_alternating(N=50, S=32, seed=7, b_scale=3.0):
    """经典高PBO结构: x_i[t] = b_i*(-1)^t + 噪声, b_i~N(0,b_scale)。
    偶数占多的IS子集里IS冠军=max b_i行, 其OOS(奇数占多)=−b_i → λ<0 → PBO高。
    S=32选因: 奇偶平衡IS子集(e=S/4)死区占比27.6% (S=16时38.1%→PBO上界仅81%,
    与零带95分位79.8%贴太近), S=32+b强→PBO≈0.86稳定越过零带95分位"""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(N) * b_scale
    return b[:, None] * ((-1.0) ** np.arange(S))[None, :] \
        + rng.standard_normal((N, S))


def cscv_null(N=50, S=32, K=30, n_combos=1000, seed=0):
    """零分布带: K个独立iid噪声矩阵的PBO数组。
    固定M下PBO有天然离散度(见模块docstring), 真实矩阵的PBO须与本带比对。"""
    pbos = []
    for k in range(K):
        M = np.random.default_rng(seed + k).standard_normal((N, S))
        pbos.append(cscv_pbo(M, n_combos=n_combos,
                             seed=seed + 10000 + k)['pbo'])
    return np.array(pbos)


def self_test():
    print('=== CSCV自验证 (零分布带K=30个iid噪声矩阵 × 各1000组合, S=32) ===')
    null = cscv_null(K=30)
    band = (np.percentile(null, 5), np.percentile(null, 95))
    print(f'零分布带: PBO均值{null.mean()*100:.1f}% '
          f'σ{null.std()*100:.1f}pp 范围[{null.min()*100:.1f}, '
          f'{null.max()*100:.1f}]% 5-95分位'
          f'[{band[0]*100:.1f}, {band[1]*100:.1f}]%')
    r1 = cscv_pbo(_synth_noise(), n_combos=1000)
    summarize(r1, '纯噪声(seed7)')
    print(f'  期望: 落在零带内 (零带内观测占比理论≈90%)')
    r2 = cscv_pbo(_synth_signal(), n_combos=1000)
    summarize(r2, '真信号(行0全域+2.0)')
    print(f'  期望: PBO=0, 须低于零带5分位{band[0]*100:.1f}%')
    r3 = cscv_pbo(_synth_alternating(), n_combos=1000)
    summarize(r3, '交替结构(偶数chunk好的行奇数chunk必差, b~N(0,3))')
    print(f'  期望: ≈86%, 须高于零带95分位{band[1]*100:.1f}%')
    checks = {
        '零带均值≈0.5(联合期望, ±0.08)': abs(null.mean() - 0.5) < 0.08,
        '真信号低于零带5分位': r2['pbo'] < band[0],
        '交替高于零带95分位': r3['pbo'] > band[1],
        '噪声落在零带内': band[0] <= r1['pbo'] <= band[1],
    }
    for name, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    ok = all(checks.values())
    print(f'自验: {"通过" if ok else "未过 — 检查实现"}')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if self_test() else 1)
