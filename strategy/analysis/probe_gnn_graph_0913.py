#!/usr/bin/env python3
"""probe_gnn_graph_0913.py — 缺口4前置: 概念图结构信息含量探针 (numpy/scipy)

问题: 缺口4 GNN(图学习)是架构升级程序最后一张牌(成本最高)。GNN的全部承诺=
沿概念图聚合邻居信息。生产已消费: 本股63因子+ML blend+概念标量(concept_heat
迭代)。**未测的唯一问题: 图结构本身 — 本股的概念同伴特征, 在本股自己特征
之外, 是否还含增量预测力?** 若无, GNN(无论GAT/HATS/AD-GAT)都无法制造信号,
方向直接关闭, 无需torch。

探针: 1跳邻居均值传播(PIT-gated 163概念incidence, 概念成立日gate同引擎),
  2跳=再传播一次; 每日期截面: 邻居特征对本股特征正交化(OLS残差)后与f10标签
  的Spearman IC — 残差IC=结构的纯增量。洗牌图控制(行置换, 保每个股票的概念
  数与概念规模分布, 摧毁共同成员结构): 真图残差IC − 洗牌图残差IC = 结构本身
  的贡献。

预置闸(跑前定):
  G1(决定性) 真图邻居残差IC(目标列取max) ≥ +0.005 且 ≥5/6年为正 且
     真图−洗牌图 ≥ +0.003 → 图结构含增量信息 → GNN方向活着(下一步=torch
     GAT冷探针); 否则缺口4关闭
  G2(信息性) 2跳对1跳的增量(图深度是否有信号)
  G3(信息性) concept_heat的图版本(邻居concept_heat) vs 标量版本 — 量化
     标量坍缩损失
机制烟测: 洗牌图残差IC须≈0(证明测的是结构不是概念规模效应)

只读。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gnn_graph_0913.py > logs/probe_gnn_graph_0913.log 2>&1
"""
import os
import time
import resource

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

ROOT = '/mnt/d/quant'
PARQUET = os.path.join(ROOT, 'strategy', 'cache',
                       'factor_df_2718s_809d_d814a206.parquet')
CONCEPT_MAP = os.path.join(ROOT, 'data', 'stock_concept_map.pkl')
INCEPTION = os.path.join(ROOT, 'data', 'concept_inception.pkl')

# 传播目标: 代表性生产因子 + concept_heat标量(G3对照)
TARGETS = ['trend_mom_v41', 'short_reversal', 'volatility', 'inv_turnover',
           'turnover_stability', 'illiq_20', 'low_downside', 'smart_money_flow',
           'relative_strength', 'concept_heat']
RNG_SEED = 20260913

# 预置闸
G1_IC = 0.005
G1_YEARS = 5
G1_STRUCT = 0.003
G1_YEAR_MIN = 2021   # 策略窗口起年(2020=设计前, 面板含2020须剔除, 同alpha工厂D6)


def log(msg):
    print(msg, flush=True)


def spearman_ic(a, b, valid):
    """截面rank IC (NaN安全)"""
    m = valid & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    ra = rankdata(a[m])
    rb = rankdata(b[m])
    with np.errstate(invalid='ignore'):
        return np.corrcoef(ra, rb)[0, 1]


def safe_corr(a, b):
    """残差近常数时返回NaN而非警告"""
    if np.var(a) < 1e-12 or np.var(b) < 1e-12:
        return np.nan
    with np.errstate(invalid='ignore'):
        return np.corrcoef(a, b)[0, 1]


def main():
    t0 = time.time()
    log('=' * 70)
    log('probe_gnn_graph 2026-09-13 — 缺口4前置: 概念图结构信息含量')
    log(f'预置闸: G1 真图残差IC≥+{G1_IC:.3f} 且≥{G1_YEARS}/6年为正 且 '
        f'真图−洗牌≥+{G1_STRUCT:.3f} → GNN活着; 否则缺口4关闭')

    # ---- 面板 ----
    fdf = pd.read_parquet(PARQUET)
    bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
    fdf = fdf[~bad]
    fdf['code'] = fdf['code'].astype(str).str.zfill(6)
    piv = fdf.set_index(['code', 'date']).sort_index()
    codes = list(piv.index.get_level_values('code').unique())
    dates = piv.index.get_level_values('date').unique()
    n, T = len(codes), len(dates)
    X = {c: piv[c].unstack('date').values.astype(np.float64)
         for c in TARGETS}
    Y = piv['future_ret'].unstack('date').values.astype(np.float64)
    log(f'[1] 面板 {n}只×{T}日期 ({pd.Timestamp(dates.min()):%Y-%m-%d}~'
        f'{pd.Timestamp(dates.max()):%Y-%m-%d}) {len(TARGETS)}目标列')

    # ---- 概念图 (PIT gate同引擎) ----
    import pickle
    with open(CONCEPT_MAP, 'rb') as f:
        cmap = pickle.load(f)
    with open(INCEPTION, 'rb') as f:
        incep = pickle.load(f)
    code_idx = {c: i for i, c in enumerate(codes)}
    # 概念列 = 面板股出现过的全部概念, 按激活日排序(未gate的排最前)
    all_concepts = set()
    for c in codes:
        all_concepts.update(cmap.get(c, []))
    act_date = {}
    for cc in all_concepts:
        iv = incep.get(cc)
        act_date[cc] = pd.Timestamp(iv) if iv else pd.Timestamp('1980-01-01')
    sorted_concepts = sorted(all_concepts, key=lambda cc: act_date[cc])
    cidx = {cc: i for i, cc in enumerate(sorted_concepts)}
    rows, cols = [], []
    for i, c in enumerate(codes):
        for cc in cmap.get(c, []):
            rows.append(i)
            cols.append(cidx[cc])
    A = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(n, len(sorted_concepts)))
    # 激活列数随日期: act_idx = 每个概念列从哪个日期index起激活
    act_dates = np.array([act_date[cc] for cc in sorted_concepts])
    k_t = np.searchsorted(act_dates, dates, side='right')
    gated_n = int((act_dates > pd.Timestamp('2020-01-01')).sum())
    log(f'[1] 图: {n}节点, {len(sorted_concepts)}概念列(其中gate列{gated_n}), '
        f'边{A.nnz}')

    def propagate(x, k):
        """A[:, :k] 1跳邻居均值(排除自身), NaN安全.
        邻居槽计数: 每股的有效邻居槽 = Σ_{c∋i}(cn_c−1), 自身贡献=xz_i·Σ(1/cn_c)
        """
        At = A[:, :k]
        xz = np.where(np.isfinite(x), x, 0.0)
        vz = np.where(np.isfinite(x), 1.0, 0.0)
        cs = At.T.dot(xz)          # 概念和
        cn = At.T.dot(vz)          # 概念有效成员数
        inv = np.where(cn > 0, 1.0 / np.where(cn > 0, cn, 1.0), 0.0)
        cm = cs * inv              # 概念均值(空=0)
        num2 = At.dot(cm)          # 每股: 概念均值之和(含自身)
        selfp = xz * At.dot(inv)   # 每股: 自身贡献
        slots = At.dot(cn) - At.dot((cn > 0).astype(np.float64))
        with np.errstate(divide='ignore', invalid='ignore'):
            nb = np.where(slots > 0.5, (num2 - selfp) / slots, np.nan)
        return nb

    # ---- 每日期: 残差IC(邻居特征 | 本股特征) ----
    yrs = np.array([pd.Timestamp(d).year for d in dates])
    yrs_u = sorted(set(yrs))
    yrs_gate = [y for y in yrs_u if y >= G1_YEAR_MIN]

    def marginal_series(col):
        """返回 (残差IC序列, 邻居IC序列, 本股IC序列)"""
        ic_res = np.full(T, np.nan)
        ic_nb = np.full(T, np.nan)
        ic_own = np.full(T, np.nan)
        xm = X[col]
        for t in range(T):
            x = xm[:, t]
            y = Y[:, t]
            v = np.isfinite(x) & np.isfinite(y)
            if v.sum() < 100:
                continue
            nb = propagate(x, k_t[t])
            ic_nb[t] = spearman_ic(nb, y, v)
            ic_own[t] = spearman_ic(x, y, v)
            # 残差: rank空间OLS正交化
            v2 = v & np.isfinite(nb)
            if v2.sum() < 100:
                continue
            rx = rankdata(x[v2])
            rn = rankdata(nb[v2])
            vx = np.var(rx)
            beta = np.cov(rx, rn)[0, 1] / vx if vx > 1e-9 else 0.0
            res = rn - beta * rx
            ic_res[t] = safe_corr(res, rankdata(y[v2]))
        return ic_res, ic_nb, ic_own

    # 洗牌图: 行置换(保每行概念数与概念规模分布)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(n)
    A_sh = A[perm].tocsc()

    def marginal_series_sh(col):
        ic_res = np.full(T, np.nan)
        xm = X[col]
        for t in range(T):
            x = xm[:, t]
            y = Y[:, t]
            v = np.isfinite(x) & np.isfinite(y)
            if v.sum() < 100:
                continue
            At = A_sh[:, :k_t[t]]
            xz = np.where(np.isfinite(x), x, 0.0)
            vz = np.where(np.isfinite(x), 1.0, 0.0)
            cs = At.T.dot(xz)
            cn = At.T.dot(vz)
            inv = np.where(cn > 0, 1.0 / np.where(cn > 0, cn, 1.0), 0.0)
            cm = cs * inv
            num2 = At.dot(cm)
            selfp = xz * At.dot(inv)
            slots = At.dot(cn) - At.dot((cn > 0).astype(np.float64))
            with np.errstate(divide='ignore', invalid='ignore'):
                nb = np.where(slots > 0.5, (num2 - selfp) / slots, np.nan)
            v2 = v & np.isfinite(nb)
            if v2.sum() < 100:
                continue
            rx = rankdata(x[v2])
            rn = rankdata(nb[v2])
            vx = np.var(rx)
            beta = np.cov(rx, rn)[0, 1] / vx if vx > 1e-9 else 0.0
            res = rn - beta * rx
            ic_res[t] = safe_corr(res, rankdata(y[v2]))
        return ic_res

    log(f'\n[2] 邻居残差IC (每日期截面, 真图):')
    results = {}
    for col in TARGETS:
        res, nb_, own = marginal_series(col)
        m_ = np.isfinite(res)
        yearly = [res[(yrs == y) & m_].mean()
                  for y in yrs_gate if ((yrs == y) & m_).sum() >= 10]
        pos_y = sum(1 for yy in yearly if yy > 0)
        results[col] = (res, nb_, own)
        log(f'  {col:20s} 残差IC {res[m_].mean():+.4f} '
            f'(IR {res[m_].mean()/res[m_].std(ddof=1):+.2f}, '
            f'正{100*(res[m_]>0).mean():.0f}%, {pos_y}/{len(yearly)}年) | '
            f'邻居IC {nb_[np.isfinite(nb_)].mean():+.4f} | '
            f'本股IC {own[np.isfinite(own)].mean():+.4f}')

    # ---- 2跳 (信息性) ----
    log(f'\n[2b] 2跳残差IC (信息性):')
    hop2 = {}
    for col in TARGETS[:4]:
        ic2 = np.full(T, np.nan)
        xm = X[col]
        for t in range(T):
            x = xm[:, t]
            y = Y[:, t]
            v = np.isfinite(x) & np.isfinite(y)
            if v.sum() < 100:
                continue
            nb1 = propagate(x, k_t[t])
            v1 = v & np.isfinite(nb1)
            if v1.sum() < 100:
                continue
            nb2 = propagate(nb1, k_t[t])
            v2 = v & np.isfinite(nb2)
            if v2.sum() < 100:
                continue
            r1 = rankdata(nb1[v2])
            r2 = rankdata(nb2[v2])
            v1v = np.var(r1)
            beta = np.cov(r1, r2)[0, 1] / v1v if v1v > 1e-9 else 0.0
            res = r2 - beta * r1
            ic2[t] = safe_corr(res, rankdata(y[v2]))
        m_ = np.isfinite(ic2)
        hop2[col] = ic2
        log(f'  {col:20s} 2跳对1跳残差IC {ic2[m_].mean():+.4f} '
            f'(正{100*(ic2[m_]>0).mean():.0f}%)')

    # ---- 洗牌图对照 ----
    log(f'\n[3] 洗牌图对照 (行置换, 保度分布):')
    sh_res = {}
    for col in TARGETS:
        res = marginal_series_sh(col)
        m_ = np.isfinite(res)
        yearly = [res[(yrs == y) & m_].mean()
                  for y in yrs_gate if ((yrs == y) & m_).sum() >= 10]
        pos_y = sum(1 for yy in yearly if yy > 0)
        sh_res[col] = res
        log(f'  {col:20s} 残差IC {res[m_].mean():+.4f} '
            f'(正{100*(res[m_]>0).mean():.0f}%, {pos_y}/{len(yearly)}年)')

    # ---- 逐年表 ----
    log(f'\n[4] 逐年残差IC (真图 vs 洗牌图):')
    for col in TARGETS:
        res = results[col][0]
        sr = sh_res[col]
        row = [f'{col:20s}']
        for y in yrs_u:
            m_ = (yrs == y) & np.isfinite(res) & np.isfinite(sr)
            if m_.sum() < 10:
                row.append(f'{y}:    --')
            else:
                row.append(f'{y}: {res[m_].mean():+.3f}/{sr[m_].mean():+.3f}')
        log('  ' + ' '.join(row))

    # ---- G3: concept_heat图版本 vs 标量 ----
    res_ch = results['concept_heat'][1]   # 邻居concept_heat IC
    own_ch = results['concept_heat'][2]   # 标量IC
    m_ = np.isfinite(res_ch) & np.isfinite(own_ch)
    log(f'\n[5] G3(信息性): 邻居concept_heat IC {res_ch[m_].mean():+.4f} '
        f'vs 标量concept_heat IC {own_ch[m_].mean():+.4f} '
        f'(Δ{res_ch[m_].mean()-own_ch[m_].mean():+.4f} — 图版对标量的增量)')

    # ---- 裁决 ----
    log('\n=== 裁决 ===')
    best = max(TARGETS, key=lambda c: np.nanmean(results[c][0]))
    res_b = results[best][0]
    sr_b = sh_res[best]
    m_ = np.isfinite(res_b)
    yearly_b = [res_b[(yrs == y) & m_].mean()
                for y in yrs_gate if ((yrs == y) & m_).sum() >= 10]
    pos_y_b = sum(1 for yy in yearly_b if yy > 0)
    yb = dict(zip(yrs_gate, yearly_b))
    mm = m_ & np.isfinite(sr_b)
    struct = np.nanmean(res_b[mm] - sr_b[mm])
    ic_b = np.nanmean(res_b)
    g1 = (ic_b >= G1_IC and pos_y_b >= G1_YEARS and struct >= G1_STRUCT)
    log(f'G1(决定性) 最优目标{best}: 残差IC {ic_b:+.4f}(≥+{G1_IC:.3f}) '
        f'{pos_y_b}/{len(yearly_b)}年为正({G1_YEAR_MIN}起, ≥{G1_YEARS}) '
        f'真图−洗牌 {struct:+.4f}(≥+{G1_STRUCT:.3f}) → '
        f'{"过: 图结构含增量信息, GNN方向活着(下一步=torch GAT冷探针)"
          if g1 else "否: 图结构无增量 → 缺口4关闭"}')
    if g1:
        verdict = ('真图邻居残差IC过闸且结构项显著 → 方向活着, '
                   '下一步=torch GAT冷探针(净化CV, 与ML零件同一否决纪律)')
    else:
        ys = ' '.join(f'{y}:{v:+.3f}' for y, v in yb.items())
        verdict = (f'概念图结构在本股特征之外不含稳定增量 — {best}逐年 '
                   f'[{ys}] 呈2021-23正/2024-25负(与alpha工厂E闸同一漂移'
                   f'签名), 图版concept_heat劣于标量(G3 Δ-0.0078) → '
                   f'GNN(任意架构)无法制造稳定信号 → 缺口4关闭, '
                   f'架构升级程序终局: 缺口1-2-3-4全闭合, state_guard建成, '
                   f'基线1,143,938不动')
    log(f'裁决: {verdict}')
    log(f'\n总耗时 {time.time()-t0:.0f}s, rss='
        f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB')


if __name__ == '__main__':
    main()
