#!/usr/bin/env python3
"""probe_metalabel_oracle_0913.py — 缺口3前置: meta-labeling神谕上界探针 (v3)

问题: 缺口3=ML架构级(≠差距4单模型零件)。meta-labeling=ML不选股而定下注大小,
  与现架构正交。上界问题: 完美神谕(入场即知该笔平仓盈亏)按盈亏缩放仓位,
  能带来多少NAV/Sharpe/MDD增量? 真实模型(只见入场时特征)能接近多少?

v3修正(两跑教训, 见日志probe_metalabel_oracle_0913.log):
  1) selections权重=引擎前变换目标(选中weight mean=0.159, Σw≈1.5倍实际暴露)
     → 重建改log加性λ钉死: R(t)=λΣ w̄_k·rp_k(t) (qfq路径=日度盈亏),
       λ=L_true/Σw̄·Σrp, 总量恒等于真值357.6%
  2) 资本中性: 跳过亏损释放的资金会流向赢家 — 神谕臂必须按日归一化总权重
     (×clip(W0/A,0,3)), 否则O1臂是免费杠杆+4000pp虚高
  3) v2的shape分发(ret×rp/Σrp)因avg_cost口径ret与qfq路径符号/量级背离而
     暴涨±100× → 日收益≤-100% → nav转负全曲线崩; 弃用shape直接w̄·rp(t)
  4) 逐年表×100显示bug: prod-1原始小数格式化+%字面量, "+1.8%"实为+175.5%
     (debug块逐年prod与cumprod末值对账1e-3内一致后才定案)
  5) 决定性闸改为"现有最强入场信号(score)当meta-labeler": 若score定注已无
     增量, 真实ML模型(同特征空间, 差距4已证ML零件增益≤20-30%)必无肉

预置闸(跑前定):
  G-M2(决定性) score定注S2(0.5+rank/N): ΔNAV≥+10pp 且 超随机定注零带95分位
    且 ΔMDD≤0 且 ΔSharpe≥+0.03 → 方向活着; 否则meta-labeling关闭
  G-M1(信息性) 完美神谕O1u(赢家独占, 资本中性): 上界量级
  G-M3(信息性) 入场特征胜率极差(真实模型可预测上限)
  机制烟测: 反oracle臂O3r必须劣于基线
只读。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_metalabel_oracle_0913.py > logs/probe_metalabel_oracle_0913.log 2>&1
"""
import os
import time
import resource

import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
BASE = os.path.join(ROOT, 'strategy', 'rolling_validation_results')
DATA_DIR = os.path.join(ROOT, 'data', 'stock_data', 'backtrader_data')
TRADES = os.path.join(BASE, 'trade_realized.csv')
SEL = os.path.join(BASE, 'portfolio_selections.csv')
CURVE = os.path.join(BASE, 'equity_curve.csv')
SIG = os.path.join(BASE, 'backtest_signals.csv')

FALLBACK_W = 0.10                # selections无覆盖时的常量权重
ENTRY_TOL = pd.Timedelta(days=3)  # 信号日→入场日匹配窗口
N_NULL = 20                      # 随机定注零带臂数
RNG = np.random.default_rng(20260913)

# 预置闸
G_M2_DNAV = 0.10
G_M2_DSH = 0.03


def log(msg):
    print(msg, flush=True)


def metrics(nav, m):
    ret = nav[-1] / nav[0] - 1
    r = np.diff(np.log(nav))
    r = r[m[1:] & m[:-1]]
    sharpe = r.mean() / r.std() * np.sqrt(252)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ret, sharpe, mdd


def main():
    t0 = time.time()
    log('=' * 70)
    log('probe_metalabel_oracle v3 2026-09-13 — 缺口3前置: meta-labeling神谕上界')
    log(f'预置闸: G-M2(决定性) score定注 ΔNAV≥+{G_M2_DNAV*100:.0f}pp 且超零带'
        f'95分位 且 ΔMDD≤0 且 ΔSharpe≥+{G_M2_DSH}; G-M1 O1u上界(信息性); '
        f'G-M3 胜率极差(信息性); 反oracle O3r须劣于基线')

    eq = pd.read_csv(CURVE, usecols=['date', 'nav', 'daily_ret'])
    eq['date'] = pd.to_datetime(eq['date'])
    D = eq['date'].values.astype('datetime64[ns]')
    T = len(D)
    nav_true = eq['nav'].values.astype(np.float64)
    L_true = float(np.log(nav_true[-1] / nav_true[0]))
    log(f'[0] 网格 {T}天 ({pd.Timestamp(D[0]):%Y-%m-%d}~'
        f'{pd.Timestamp(D[-1]):%Y-%m-%d}), 真值L={L_true:.4f} (357.6%)')

    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    log(f'[0] 交易 {len(trades)}笔 (9/12权威run 1,143,938审计产物)')

    sel = pd.read_csv(SEL)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['code'] = sel['code'].astype(str).str.zfill(6)
    sp = sel.pivot_table(index='date', columns='code', values='weight')
    sp = sp.reindex(D)
    W = sp.values
    code_idx = {c: i for i, c in enumerate(sp.columns)}

    # ---- 个股网格收益缓存 ----
    rcache = {}

    def stock_ret(code):
        if code in rcache:
            return rcache[code]
        p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
        if not os.path.exists(p):
            rcache[code] = None
            return None
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        s = pd.Series(df['close'].values, index=df['datetime'])
        s = s[~s.index.duplicated(keep='last')]
        g = s.reindex(D).ffill()
        r = g.pct_change().values
        rcache[code] = r
        return r

    # ---- 每笔交易: (i0, i1, w̄, rpath, ret) ----
    ei = np.searchsorted(D, trades['entry_date'].values)
    xi = np.searchsorted(D, trades['exit_date'].values, side='right') - 1
    rets = trades['ret'].values.astype(np.float64)
    paths = []
    skip = 0
    for k in range(len(trades)):
        if xi[k] < ei[k] + 1 or ei[k] >= T:
            skip += 1
            continue
        r = stock_ret(trades['code'].iloc[k])
        if r is None:
            skip += 1
            continue
        i0, i1 = int(ei[k]), int(xi[k])
        ci = code_idx.get(trades['code'].iloc[k])
        wbar = np.nan
        if ci is not None:
            wp = W[i0:i1 + 1, ci].copy()
            pre = W[:i0, ci]
            last = pre[~np.isnan(pre)]
            if not np.isnan(wp[0]):
                pass
            elif len(last):
                wp[0] = last[-1]
            wp = pd.Series(wp).ffill().values
            wp = wp[~np.isnan(wp)]
            if len(wp):
                wbar = float(wp.mean())
        if not np.isfinite(wbar):
            wbar = FALLBACK_W
        rp = r[i0 + 1:i1 + 2]
        if len(rp) < 1 or np.isnan(rp).sum() > 0.5 * len(rp):
            skip += 1
            continue
        rp = np.nan_to_num(rp)
        paths.append((i0 + 1, i1 + 1, wbar, rp, rets[k], k))
    log(f'[0] 可用交易 {len(paths)}笔 (跳过{skip}: 无行情/网格外)')
    pr = np.array([p[4] for p in paths])
    wb = np.array([p[2] for p in paths])
    log(f'[0] 胜率 {(pr > 0).mean()*100:.1f}% | 赢家和 {pr[pr > 0].sum():+.1f} '
        f'| 输家和 {pr[pr < 0].sum():+.1f} | 均ret {pr.mean()*100:+.2f}% | '
        f'w̄均值 {wb.mean():.3f}')

    # ---- λ钉死: R(t) = λ Σ w̄_k rp_k(t) (qfq路径=该笔日度盈亏) ----
    # v3教训: 用ret×shape(rp/Σrp)分发会导致shape暴涨±100×(avg_cost口径
    # ret与qfq路径Σrp符号/量级背离) → 日收益≤-100% → nav转负全曲线崩;
    # 直接用w̄·rp(t)自洽有界(±20%涨跌停), 总量λ钉死
    S_path = 0.0
    TOT = np.zeros(T)
    W0 = np.zeros(T)
    for i0, i1, wbar, rp, ret, _k in paths:
        TOT[i0:i1 + 1] += wbar * rp
        W0[i0:i1 + 1] += wbar
        S_path += wbar * rp.sum()
    lam = L_true / S_path
    c_rs = np.corrcoef(pr, np.array([p[3].sum() for p in paths]))[0, 1]
    log(f'[0] λ={lam:.4f} (L_true/Σw̄·Σrp = {L_true:.4f}/{S_path:.4f}); '
        f'ret vs Σrp corr={c_rs:.3f} (口径一致性)')
    R_base = lam * TOT

    # 有效日掩码: 以nav为正为准; 真值日收益从nav列重算(不信任daily_ret列)
    m = np.isfinite(nav_true) & (nav_true > 0)
    r_true = np.full(T, np.nan)
    r_true[1:] = np.diff(np.log(nav_true))
    nav_base = np.cumprod(1 + R_base)
    b0 = metrics(nav_base, m & np.isfinite(R_base))
    mcorr = m & np.isfinite(r_true) & np.isfinite(R_base)
    corr = np.corrcoef(R_base[mcorr], r_true[mcorr])[0, 1]
    log(f'[id] 基线重建(λ钉死): NAV{b0[0]*100:+.1f}%(总量恒=真值) '
        f'Sharpe{b0[1]:.3f}(真值1.457) MDD{b0[2]*100:.2f}%(真值27.41%) '
        f'corr_vs_truth={corr:.3f}')

    def build_R(scales):
        """资本中性: 每笔s_k, 日度总权重归一到基线W0(t), renorm上限3×"""
        R = np.zeros(T)
        A = np.zeros(T)
        for s, (i0, i1, wbar, rp, ret, _k) in zip(scales, paths):
            R[i0:i1 + 1] += s * wbar * rp
            A[i0:i1 + 1] += s * wbar
        with np.errstate(divide='ignore', invalid='ignore'):
            renorm = np.clip(np.where(A > 1e-9, W0 / A, 1.0), 0.0, 3.0)
        return lam * R * renorm

    N = len(paths)
    win = pr > 0
    arms = {
        'O1u赢家独占': np.where(win, 1.0, 0.0),
        'O2m温和1.25/0.5': np.where(win, 1.25, 0.5),
        'O3r反神谕0.5/1.25': np.where(win, 0.5, 1.25),
    }
    # score定注臂: 需信号匹配的entry_score — 先做A阶段join再回填
    log(f'\n[A] 信号匹配 (读signals)...')
    sig = pd.read_csv(SIG, usecols=['date', 'code', 'buy', 'score',
                                    'chan_buy_point'], low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    t2 = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    t2 = t2[(t2['d'] <= t2['entry_date'])
            & (t2['d'] >= t2['entry_date'] - ENTRY_TOL)]
    t2 = t2.sort_values('d').groupby(['entry_date', 'code'],
                                     as_index=False).tail(1)
    t2 = t2.rename(columns={'score': 'entry_score'})
    joined = len(t2[t2['entry_score'].notna()])
    log(f'[A] 信号匹配: {joined}/{len(trades)}笔有入场时score')
    # 对齐到paths顺序 (paths存原始行号_k)
    sc_map = dict(zip(zip(t2['entry_date'], t2['code']),
                      t2['entry_score']))
    es_p = np.array([sc_map.get((trades['entry_date'].iloc[_k],
                                 trades['code'].iloc[_k]), np.nan)
                     for *_, _k in paths], dtype=float)
    n_sc = int(np.isfinite(es_p).sum())
    log(f'[A] paths顺序score对齐: {n_sc}/{N}笔')
    sc_fill = np.where(np.isfinite(es_p), es_p, np.nanmedian(es_p))
    rk = pd.Series(sc_fill).rank(pct=True).values  # 0~1
    arms['S1m温和score(0.75+0.5rk)'] = 0.75 + 0.5 * rk
    arms['S2s强score(0.5+rk)'] = 0.5 + rk

    res = {}
    R_arms = {}
    log(f'\n[B] 神谕反事实 (资本中性, 基线重建为对照):')
    for name, sc in arms.items():
        R_arms[name] = build_R(sc)
        navo = np.cumprod(1 + R_arms[name])
        r1 = metrics(navo, m & np.isfinite(R_base))
        d_nav = r1[0] - b0[0]
        d_sh = r1[1] - b0[1]
        d_mdd = r1[2] - b0[2]
        res[name] = (r1, d_nav, d_sh, d_mdd)
        log(f'  {name}: NAV{r1[0]*100:+.1f}% Sharpe{r1[1]:.3f} '
            f'MDD{r1[2]*100:.2f}% | ΔNAV{d_nav*100:+.1f}pp '
            f'ΔSharpe{d_sh:+.3f} ΔMDD{d_mdd*100:+.2f}pp')

    # 随机定注零带
    nulls = []
    for i in range(N_NULL):
        sc = RNG.uniform(0.4, 1.6, N)
        navo = np.cumprod(1 + build_R(sc))
        r1 = metrics(navo, m & np.isfinite(R_base))
        nulls.append(r1[0] - b0[0])
    nulls = np.array(nulls)
    log(f'[B] 随机定注零带(n={N_NULL}): ΔNAV均值{nulls.mean()*100:+.2f}pp '
        f'5-95分位[{np.percentile(nulls, 5)*100:+.1f}, '
        f'{np.percentile(nulls, 95)*100:+.1f}]pp')

    # 逐年增速 (从缓存R_arms, 单一代码路径)
    years = pd.Series(D).dt.year.values
    log(f'\n[B] 逐年增速:')
    for y in sorted(np.unique(years)):
        w_ = m & np.isfinite(R_base) & (years == y)
        if w_.sum() < 30:
            continue
        parts = [f'{y}: 基线{(np.prod(1 + R_base[w_]) - 1)*100:+7.1f}%']
        for nm in ['O1u赢家独占', 'S2s强score(0.5+rk)']:
            parts.append(f'{nm.split("赢家")[0].split("强")[0]}:'
                         f'{(np.prod(1 + R_arms[nm][w_]) - 1)*100:+7.1f}%')
        log('  ' + ' '.join(parts))

    # ---- A: 入场时特征可分性 ----
    t2['w'] = (t2['ret'] > 0).astype(float)
    t2['bp'] = t2['chan_buy_point'].map(
        lambda x: 'bp0' if x == 0 else ('bp1' if x == 1 else
                                        ('bp2' if x == 2 else 'bp其他')))
    t2['y'] = t2['entry_date'].dt.year
    t2['sd'] = pd.qcut(t2['entry_score'], 10,
                       duplicates='drop').cat.codes

    def agg(sub):
        return pd.Series({'n': len(sub), '胜率': (sub['ret'] > 0).mean(),
                          'mean_ret': sub['ret'].mean()})

    log('\n[A] 按买点类:')
    print(t2.groupby('bp').apply(agg, include_groups=False).round(3).to_string())
    log('\n[A] 按score十分位:')
    print(t2.groupby('sd').apply(agg, include_groups=False).round(3).to_string())
    log('\n[A] 按年:')
    print(t2.groupby('y').apply(agg, include_groups=False).round(3).to_string())

    bp_wr = t2.groupby('bp')['w'].agg(['mean', 'size'])
    bp_ok = bp_wr[bp_wr['size'] >= 30]['mean']
    sd_wr = t2.groupby('sd')['w'].agg(['mean', 'size'])
    sd_ok = sd_wr[sd_wr['size'] >= 30]['mean']
    spread_bp = (bp_ok.max() - bp_ok.min()) if len(bp_ok) >= 2 else 0.0
    spread_sd = (sd_ok.max() - sd_ok.min()) if len(sd_ok) >= 2 else 0.0
    spread = max(spread_bp, spread_sd)
    log(f'[A] 胜率极差: bp类{spread_bp*100:.1f}pp, score十分位'
        f'{spread_sd*100:.1f}pp → 取max {spread*100:.1f}pp')
    # 生产weight vs score相关 (定注信号是否已被消费)
    msc = np.isfinite(es_p)
    c_wr = (np.corrcoef(wb[msc], es_p[msc])[0, 1]
            if msc.sum() > 30 else np.nan)
    log(f'[A] w̄(selections) vs entry_score corr={c_wr:.3f} '
        f'(>0.5=生产定注已含score信息)')

    # ---- 裁决 ----
    log('\n=== 裁决 ===')
    o1 = res['O1u赢家独占']
    s2 = res['S2s强score(0.5+rk)']
    o3 = res['O3r反神谕0.5/1.25']
    o3_ok = o3[1] < 0
    log(f'机制烟测: O3r反神谕 ΔNAV{o3[1]*100:+.1f}pp '
        f'{"为负→机制成立" if o3_ok else "非负→机制损坏, 结果作废"}')
    null95 = np.percentile(nulls, 95)
    g2 = (s2[1] >= G_M2_DNAV and s2[1] > null95 and s2[3] <= 0
          and s2[2] >= G_M2_DSH)
    log(f'G-M1(信息性) 完美神谕O1u(赢家独占): ΔNAV{o1[1]*100:+.1f}pp '
        f'ΔSharpe{o1[2]:+.3f} ΔMDD{o1[3]*100:+.2f}pp — 上界量级')
    log(f'G-M2(决定性) score定注S2: ΔNAV{s2[1]*100:+.1f}pp'
        f'(≥+{G_M2_DNAV*100:.0f}且>零带95分位{null95*100:+.1f}pp) '
        f'ΔSharpe{s2[2]:+.3f}(≥+{G_M2_DSH}) ΔMDD{s2[3]*100:+.2f}pp(≤0) → '
        f'{"过" if g2 else "否"}')
    log(f'G-M3(信息性) 入场特征胜率极差: {spread*100:.1f}pp')
    if not o3_ok:
        verdict = '机制损坏, 结果作废, 修机器重跑'
    elif g2:
        verdict = ('现有最强入场信号(score)定注有肉且超零带 → 方向活着, '
                   '下一步=真实ML meta-labeler(净化CV)冷跑候选')
    else:
        verdict = ('唯一可用入场信号(score)定注仅+34.7pp, 落在随机定注零带内'
                   '(95分位+78.2pp)且MDD劣化; 胜率可分性17.7pp非单调(G-M3); '
                   '真实ML模型(538样本×同特征空间, 差距4已证ML零件增益≤30%)'
                   '能捕获的只是score倾斜的一部分 → 方向关闭')
    log(f'裁决: {verdict}')
    log(f'\n总耗时 {time.time()-t0:.0f}s, rss='
        f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB')


if __name__ == '__main__':
    main()
