#!/usr/bin/env python3
"""2026-09-13 差距1联合约束实证探针 (用户挑战: "联合约束呢")

背景: 差距1(组合层联合优化)此前降级的论证 = "max_per_industry=0+固定槽数下贪婪
  对线性目标最优" — 但那只覆盖无约束/拟阵选择。联合选择+风险权重(Qlib式联合约束
  优化: 行业上限/换手预算/风险分散目标下的选股+配权)从未实证。本探针在调仓日层面
  回答: 任一约束/联合变体能否用相同信息(纯score+收盘矩阵)显著超过无约束贪婪?

臂(每调仓日, 池=当日buy信号行且有完整f20, 槽数N=该日实际持仓数):
  actual_w / actual_e: 生产实际持仓, 生产权重 / 等权 (参考锚)
  top_score: 无约束score贪婪top-N (拟阵最优参考, 闸的比较基准)
  top_f20: 前视top-N (上界上下文)
  J1_cap1/J1_cap2: 行业上限 max_per_industry=1/2 的score贪婪 (拟阵可行集收紧)
  J2_K2/J2_K3: 硬换手预算K=2/3的score贪婪 (老持仓=上一调仓日实际持仓, 保留免费)
  J3_qp: 联合选股+配权 — top-50 score候选 → MV-QP(μ=候选内score z分,
         Σ=60日收益相关阵, λ=1, w∈[0,0.25]) → 按QP权重取top-N → 重解权重
  J3q_lam: J3灵敏度 λ∈{0.5,2,4} (仅报告, 不参与闸)
  J4_comb: 行业cap2+换手预算3贪婪选股, 再对N只QP配权 (联合约束capstone)

闸(跑前定): 任一臂 vs top_score 配对均值 ≥ +0.15pp 且 正比例 ≥ 55% 且
  ≥4/6年非负 → 候选冷跑四指标裁决; 若还超过 actual_w → 强候选(裸score+约束
  即胜过全套生产机制); 全臂不达标 → 差距1降级获得联合约束维度实证确认。
口径: 与probe_gap1_sel_upper_0912一致 (fwd20 close-to-close, 信号日=调仓日,
  北交所排除, 停牌ffill, 选股日<=2026-08-13)。J臂用裸score(与生产effective_score
  机制差+0.89pp, 已知), 故科学基准=top_score(同信息), 采纳基准=actual_w。
只读。串行。.venv。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap1_joint_0913.py > logs/probe_gap1_joint_0913.log 2>&1
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
PS = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'
IDX = os.path.join(BT, 'sh000001_qfq.csv')
FWD = 20
END_OK = '2026-08-13'
CORR_WIN = 60          # 相关阵回看交易日
CORR_MIN = 30          # 配对最少有效收益日
M_CAND = 50            # J3候选数
W_CAP = 0.25           # QP单票权重上限 (=rank_weight_cap)
GATE_PP = 0.15         # 闸: vs top_score 配对均值 ≥ +0.15pp
GATE_POS = 0.55        # 闸: 正比例 ≥ 55%
GATE_YR = 4            # 闸: ≥4/6年非负


def build_close():
    idx = pd.read_csv(IDX, usecols=['datetime'], parse_dates=['datetime'])
    idx = idx[(idx.datetime >= '2020-12-01') & (idx.datetime <= '2026-10-15')]
    D = idx['datetime'].values.astype('datetime64[ns]')
    T = len(D)
    codes = []
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        c = fn[:-len('_qfq.csv')]
        if c.startswith(('sh', 'sz')) or c.startswith(('4', '8', '92')):
            continue
        codes.append(c)
    colmap = {c: i for i, c in enumerate(codes)}
    print(f'[0] 日期 {T} 天 x 股票 {len(codes)} 只', flush=True)
    close = np.full((T, len(codes)), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, c in enumerate(codes):
        try:
            df = pd.read_csv(os.path.join(BT, f'{c}_qfq.csv'),
                             usecols=['datetime', 'close'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(D, dt)
        pos = pos[pos < T]
        if len(pos) == 0:
            continue
        close[pos, i] = df['close'].values[:len(pos)].astype(np.float32)
    print(f'[0] 矩阵加载完成 {time.time()-t0:.0f}s, ffill...', flush=True)
    return D, codes, colmap, pd.DataFrame(close).ffill(axis=0).values


def top_n_by(vals, n):
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.array([], dtype=int)
    order = np.argsort(-v[ok], kind='stable')
    idx_ok = np.where(ok)[0]
    return idx_ok[order[:min(n, len(order))]]


def greedy_pick(g, N, cap=0, incumbents=None, budget=999):
    """score降序贪婪. cap>0=行业上限; incumbents=免费保留集合(换手预算).
    返回 g 的行位置数组."""
    order = np.argsort(-g.score.values, kind='stable')
    ind_count = {}
    news = 0
    sel = []
    for p in order:
        if len(sel) >= N:
            break
        code = g.code.iloc[p]
        if incumbents is not None and code in incumbents:
            sel.append(p)
            continue
        if incumbents is not None and news >= budget:
            continue
        if cap > 0:
            ind = g.industry.iloc[p]
            ind = ind if isinstance(ind, str) else f'__NA__{code}'
            if ind_count.get(ind, 0) >= cap:
                continue
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if incumbents is not None:
            news += 1
        sel.append(p)
    return np.array(sel, dtype=int)


def qp_solve(mu, S, lam=1.0):
    """min w'S w - (1/λ)μ'w  s.t. Σw=1, 0<=w<=W_CAP. 失败返回None."""
    m = len(mu)
    if m == 1:
        return np.ones(1)
    def fun(w):
        return w @ S @ w - (1.0 / lam) * (mu @ w)
    def jac(w):
        return 2.0 * S @ w - mu / lam
    res = minimize(fun, np.full(m, 1.0 / m), jac=jac, method='SLSQP',
                   bounds=[(0.0, W_CAP)] * m,
                   constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1.0}),
                   options={'maxiter': 300, 'ftol': 1e-14})
    if not res.success or abs(res.x.sum() - 1.0) > 1e-4:
        return None
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def qp_joint(ti, g, N, lam=1.0):
    """top-M_CAND score候选 → QP → 按权重取top-N → 重解. 返回 (行位置, 权重)
    或 (None, None) 回退 (调用方走score贪婪等权)."""
    M = min(M_CAND, len(g))
    if M < 2:
        return None, None
    cand = top_n_by(g.score.values, M)
    cand = np.sort(cand)  # 按g原行序
    cols = g.ci.iloc[cand].values.astype(int)
    t_lo = max(0, ti - CORR_WIN)
    seg = close[t_lo:ti + 1, cols]          # 61行 -> 60个收益
    ret = seg[1:] / seg[:-1] - 1.0
    S = np.corrcoef(ret, rowvar=False)
    bad = ~np.isfinite(S)
    if bad.mean() > 0.5 or np.isnan(seg).mean() > 0.5:
        return None, None                   # 数据不足回退
    S = np.nan_to_num(S, nan=0.0)
    np.fill_diagonal(S, 1.0)
    mu_raw = g.score.iloc[cand].values.astype(float)
    mu = mu_raw - mu_raw.mean()
    sd = mu.std()
    mu = mu / sd if sd > 1e-12 else np.zeros(M)
    w = qp_solve(mu, S, lam=lam)
    if w is None:
        return None, None
    keep = top_n_by(w, N)
    if len(keep) < 2:
        return cand[keep], np.ones(len(keep))
    mu2 = mu[keep]
    mu2 = mu2 - mu2.mean()
    sd2 = mu2.std()
    mu2 = mu2 / sd2 if sd2 > 1e-12 else np.zeros(len(keep))
    w2 = qp_solve(mu2, S[np.ix_(keep, keep)], lam=lam)
    if w2 is None:
        w2 = np.ones(len(keep)) / len(keep)
    return cand[keep], w2


def main():
    global close
    t0 = time.time()
    D, codes, colmap, close = build_close()
    f20 = (pd.DataFrame(close).shift(-FWD) / pd.DataFrame(close) - 1).values
    print(f'[0] fwd20完成 ({time.time()-t0:.0f}s)', flush=True)

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'score', 'industry'],
                      dtype={'code': str})
    sig = sig[sig.buy == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[(sig.date >= '2021-04-01') & (sig.date <= END_OK)]
    ps = pd.read_csv(PS, dtype={'code': str})
    ps['code'] = ps['code'].str.zfill(6)
    ps['date'] = pd.to_datetime(ps['date'])
    ps = ps[ps.date <= END_OK]
    tmap = {d: i for i, d in enumerate(pd.to_datetime(D))}
    sig = sig[sig.date.isin(tmap)]
    sig['ti'] = sig['date'].map(tmap)
    sig = sig[sig.code.isin(colmap)]
    sig['ci'] = sig['code'].map(colmap)
    sig['f20'] = f20[sig['ti'].values, sig['ci'].values]
    print(f'[1] buy池行 {len(sig)}, 调仓日 {ps.date.nunique()}', flush=True)

    ps_dates = sorted(ps.date.unique())
    ps_by_date = {d: g for d, g in ps.groupby('date')}
    rows = []
    fallback = {k: 0 for k in ['J3_qp', 'J3q_lam', 'J4_comb']}
    n_arm_dates = 0
    for d, g in sig.groupby('date'):
        ti = tmap[d]
        pick = ps_by_date.get(d)
        if pick is None or len(pick) == 0:
            continue
        g_ok = g[np.isfinite(g.f20)]
        if len(g_ok) == 0:
            continue
        N = len(pick)
        n_arm_dates += 1

        # 生产实际 (等权 + 生产权重, 持仓可能不在今日池内 — 锁仓持仓)
        pick_c = pick.code.map(colmap).values
        pick_c = pick_c[~pd.isna(pick_c)].astype(int)
        pick_f = f20[ti, pick_c]
        fin = np.isfinite(pick_f)
        r = {'date': d, 'pool': len(g_ok), 'slots': N}
        if fin.sum():
            r['actual_e'] = float(np.nanmean(pick_f))
            w_p = pick.weight.values[fin].astype(float)
            r['actual_w'] = float((w_p * pick_f[fin]).sum() / w_p.sum())
        else:
            r['actual_e'] = np.nan
            r['actual_w'] = np.nan

        # 参考: 无约束贪婪 + 前视上界
        s_top = top_n_by(g_ok.score.values, N)
        r['top_score'] = float(np.nanmean(g_ok.f20.iloc[s_top].values))
        f_top = top_n_by(g_ok.f20.values, N)
        r['top_f20'] = float(np.nanmean(g_ok.f20.iloc[f_top].values))
        r['ov_top_j'] = {}  # 各J臂与top_score重叠率

        def _arm(name, pos):
            if len(pos) == 0:
                r[name] = np.nan
                return
            r[name] = float(np.nanmean(g_ok.f20.iloc[pos].values))
            r['ov_top_j'][name] = len(set(g_ok.code.iloc[pos]) &
                                     set(g_ok.code.iloc[s_top])) / N

        # J1 行业上限
        for cap in (1, 2):
            _arm(f'J1_cap{cap}', greedy_pick(g_ok, N, cap=cap))
        # J2 换手预算 (老持仓 = 上一调仓日实际持仓)
        prev = ps_by_date.get(ps_dates[ps_dates.index(d) - 1]
                              if ps_dates.index(d) > 0 else None)
        inc = set(prev.code) if prev is not None else None
        if inc is None:  # 首日无老持仓 → 预算无限
            for K in (2, 3):
                _arm(f'J2_K{K}', greedy_pick(g_ok, N))
        else:
            for K in (2, 3):
                _arm(f'J2_K{K}', greedy_pick(g_ok, N, incumbents=inc, budget=K))
        # J3 MV-QP联合 (λ=1主闸) + 灵敏度
        pos_j3, w_j3 = qp_joint(ti, g_ok, N, lam=1.0)
        if pos_j3 is None:
            fallback['J3_qp'] += 1
            pos_j3 = greedy_pick(g_ok, N)
            w_j3 = np.ones(len(pos_j3)) / max(len(pos_j3), 1)
        r['J3_qp'] = float((w_j3 * g_ok.f20.iloc[pos_j3].values).sum())
        r['ov_top_j']['J3_qp'] = len(set(g_ok.code.iloc[pos_j3]) &
                                     set(g_ok.code.iloc[s_top])) / N
        for lam in (0.5, 2.0, 4.0):
            pj, wj = qp_joint(ti, g_ok, N, lam=lam)
            if pj is None:
                fallback['J3q_lam'] += 1
                pj = greedy_pick(g_ok, N)
                wj = np.ones(len(pj)) / max(len(pj), 1)
            r[f'J3q_l{lam}'] = float((wj * g_ok.f20.iloc[pj].values).sum())
        # J4 联合capstone: cap2+budget3贪婪选股 → QP配权
        if inc is None:
            p4 = greedy_pick(g_ok, N, cap=2)
        else:
            p4 = greedy_pick(g_ok, N, cap=2, incumbents=inc, budget=3)
        if len(p4) >= 2:
            cols4 = g_ok.ci.iloc[p4].values.astype(int)
            seg4 = close[max(0, ti - CORR_WIN):ti + 1, cols4]
            ret4 = seg4[1:] / seg4[:-1] - 1.0
            S4 = np.nan_to_num(np.corrcoef(ret4, rowvar=False), nan=0.0)
            np.fill_diagonal(S4, 1.0)
            mu4 = g_ok.score.iloc[p4].values.astype(float)
            mu4 = mu4 - mu4.mean()
            sd4 = mu4.std()
            mu4 = mu4 / sd4 if sd4 > 1e-12 else np.zeros(len(p4))
            w4 = qp_solve(mu4, S4, lam=1.0)
            if w4 is None:
                fallback['J4_comb'] += 1
                w4 = np.ones(len(p4)) / len(p4)
            r['J4_comb'] = float((w4 * g_ok.f20.iloc[p4].values).sum())
            r['ov_top_j']['J4_comb'] = len(set(g_ok.code.iloc[p4]) &
                                           set(g_ok.code.iloc[s_top])) / N
        else:
            _arm('J4_comb', p4)
        rows.append(r)
    r = pd.DataFrame(rows)
    r['yr'] = r.date.dt.year
    print(f'[2] 臂构建完成 n={len(r)} 调仓日 ({time.time()-t0:.0f}s)')

    # === [3] 总体均值 ===
    cols = ['actual_e', 'actual_w', 'top_score', 'top_f20', 'J1_cap1',
            'J1_cap2', 'J2_K2', 'J2_K3', 'J3_qp', 'J4_comb']
    print(f'\n[3] 总体 mean f20 (n={n_arm_dates})')
    for c in cols:
        v = r[c].dropna()
        print(f'  {c:>10s}: {v.mean()*100:+7.2f}%  (中位{v.median()*100:+6.2f}%, '
              f'正比例{(v>0).mean()*100:.0f}%, n={len(v)})')
    lam_cols = [c for c in r.columns if c.startswith('J3q_l')]
    for c in lam_cols:
        v = r[c].dropna()
        print(f'  {c:>10s}: {v.mean()*100:+7.2f}%  (中位{v.median()*100:+6.2f}%, '
              f'n={len(v)})')

    # === [4] 配对差: J臂 vs top_score (科学基准) 和 vs actual_w (采纳基准) ===
    jarms = ['J1_cap1', 'J1_cap2', 'J2_K2', 'J2_K3', 'J3_qp', 'J4_comb']
    print(f'\n[4] 配对差 (逐日算差再平均)')
    print(f'  {"臂":>10s} {"vs_top_score":>14s} {"正%":>5s} '
          f'{"vs_actual_w":>14s} {"正%":>5s} {"重叠top_score":>12s}')
    res = {}
    for a in jarms:
        d1 = r[a] - r.top_score
        d2 = r[a] - r.actual_w
        d1v = d1.dropna()
        d2v = d2.dropna()
        ov = pd.Series([r['ov_top_j'].iloc[i].get(a, np.nan)
                        for i in range(len(r))])
        res[a] = (d1v.mean(), (d1v > 0).mean())
        print(f'  {a:>10s} {d1v.mean()*100:+11.2f}pp {(d1v>0).mean()*100:>4.0f}% '
              f'{d2v.mean()*100:+11.2f}pp {(d2v>0).mean()*100:>4.0f}% '
              f'{ov.mean()*100:>9.0f}%')

    # === [5] 逐年 (J臂 - top_score 配对差) ===
    print(f'\n[5] 逐年 (J臂 - top_score 配对差, pp)')
    print(f'  {"年":>5s} ' + ' '.join(f'{a:>9s}' for a in jarms))
    years = sorted(r.yr.unique())
    for y in years:
        m = r.yr == y
        line = f'  {y:>5d} '
        for a in jarms:
            d = (r[a] - r.top_score)[m].dropna()
            line += f'{d.mean()*100:+8.2f} ' if len(d) else f'{"":>9s} '
        print(line)
    # 逐年 actual/top_score 上下文
    print(f'  {"年":>5s} actual_w / top_score')
    for y in years:
        m = r.yr == y
        print(f'  {y:>5d} {r.actual_w[m].mean()*100:+6.2f}% / '
              f'{r.top_score[m].mean()*100:+6.2f}%')

    # === [6] 闸判定 ===
    print(f'\n[6] 闸判定 (闸: vs top_score ≥+{GATE_PP}pp, 正≥{GATE_POS*100:.0f}%, '
          f'≥{GATE_YR}/6年非负)')
    for a in jarms:
        d1 = r[a] - r.top_score
        d1v = d1.dropna()
        yr_neg = 0
        yr_n = 0
        for y in years:
            dv = d1[r.yr == y].dropna()
            if len(dv):
                yr_n += 1
                if dv.mean() < 0:
                    yr_neg += 1
        yr_ok = (yr_n - yr_neg) >= GATE_YR
        m_ok = d1v.mean() >= GATE_PP / 100
        p_ok = (d1v > 0).mean() >= GATE_POS
        vs_aw = (r[a] - r.actual_w).dropna().mean()
        verdict = '候选' if (m_ok and p_ok and yr_ok) else '否'
        if verdict == '候选' and vs_aw > 0:
            verdict = '强候选(超actual_w)'
        print(f'  {a:>10s}: {verdict:>18s} | Δ={d1v.mean()*100:+.2f}pp '
              f'正{(d1v>0).mean()*100:.0f}% 年{yr_n-yr_neg}/{yr_n} '
              f'vsAW={vs_aw*100:+.2f}pp')
    print(f'\n[7] QP回退计数: {fallback} | 总耗时 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
