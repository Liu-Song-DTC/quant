#!/usr/bin/env python3
"""probe_barra_risk_0913.py — 缺口2前置探针: 组合风险结构分解+回撤腿归因+vol覆盖

问题: MDD 27.41%是最弱指标(阶段7四路攻击后接受为诚实画像)。缺口2方案=
  Barra式因子协方差+特质风险+收缩的截面风险模型, 与现有"realized vol缩放
  (clip 0.75-1.0)"暴露层正交。但阶段7已证崩腿(2021-07/2022-01)是系统性风险,
  截面模型抓不住。本探针量化: 风险模型到底有没有可抓的东西。

方法:
  1. 日频风格因子收益率(Fama-French式因子模仿组合, 自qfq CSV构建):
     MKT(等权市场, 恒1权重)、SIZE(ln(amount))、MOM(20d动量)、VOL(20d已实现波)、
     REV(5d反转) — 截面z分数前日值×当日收益, 长空组合。
  2. 组合日收益(equity_curve.csv权威基线1,143,938)对5因子滚动OLS(252d):
     滚动beta/R², 年度方差分解。
  3. 回撤腿(DD≥5%区间)归因: 腿前126d beta × 腿内因子累计 vs 残差。
  4. vol覆盖反事实(纯方向性, 非可采纳): s_t=clip(15%/EWMA预测, 0.6, 1.2)
     作用于日收益 — 更深的波动率择时能不能救MDD。

预置闸(跑前定):
  G1 全样本日频R²<20% → Barra因子模型对本策略边际价值小 → 缺口2因子方向关闭
  G2 崩腿(2021-07/2022-01等)因子归因为主且与MKT同向 → 证实阶段7边界(系统性),
     敞口控制抓不住, MDD核心不可修
  G3 非崩腿存在因子占比>40%的腿 → Barra敞口约束候选 → 进冷跑方向
  G4 vol覆盖反事实 ΔMDD≥+2pp(MDD改善) 且 |ΔSharpe|≤0.03 且 |ΔNAV|≤1% → 冷跑候选;
     否则波动率择时深化方向关闭
只读。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_barra_risk_0913.py > logs/probe_barra_risk_0913.log 2>&1
"""
import os
import time
import resource
import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
BT = os.path.join(ROOT, 'data', 'stock_data', 'backtrader_data')
IDX = os.path.join(BT, 'sh000001_qfq.csv')
CURVE = os.path.join(ROOT, 'strategy', 'rolling_validation_results',
                     'equity_curve.csv')

WINDOW = 252            # 滚动beta窗口
LEG_DD = 0.05           # 回撤腿阈值
BETA_WIN = 126          # 腿前beta估计窗口
TARGET_VOL = 0.15
EWMA_HALF = 30          # vol预测半衰期(交易日)

# 预置闸
G1_R2 = 0.20
G3_SHARE = 0.40
G4_DMDD = 0.02    # ΔMDD≥+2pp(MDD改善, MDD为负故改善=Δ为正)
G4_DSHARPE = 0.03
G4_DNAV = 0.01


def log(msg):
    print(msg, flush=True)


def build_data():
    """指数日历 + 每股票close/amount矩阵 + 组合日收益"""
    t0 = time.time()
    idx = pd.read_csv(IDX, usecols=['datetime', 'close'],
                      parse_dates=['datetime'])
    idx = idx[(idx.datetime >= '2020-06-01') & (idx.datetime <= '2026-09-15')]
    D = idx['datetime'].values.astype('datetime64[ns]')
    T = len(D)
    mkt = idx['close'].pct_change().values.astype(np.float64)
    codes = []
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        c = fn[:-len('_qfq.csv')]
        if c.startswith(('sh', 'sz')) or c.startswith(('4', '8', '92', '399')):
            continue
        codes.append(c)
    close = np.full((T, len(codes)), np.nan, dtype=np.float32)
    amount = np.full((T, len(codes)), np.nan, dtype=np.float32)
    for i, c in enumerate(codes):
        path = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=['datetime', 'close', 'amount'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(D, dt)
        pos = pos[pos < T]
        if len(pos) == 0:
            continue
        close[pos, i] = df['close'].values[:len(pos)].astype(np.float32)
        amount[pos, i] = pd.to_numeric(df['amount'].iloc[:len(pos)],
                                       errors='coerce').values.astype(np.float32)
    close_f = pd.DataFrame(close).ffill(axis=0).values
    amount_f = pd.DataFrame(amount).ffill(axis=0).values
    log(f'[1] 日历{T}天×{len(codes)}只 close/amount矩阵 '
        f'{time.time()-t0:.0f}s')
    # 组合日收益
    eq = pd.read_csv(CURVE, usecols=['date', 'daily_ret'])
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.dropna(subset=['daily_ret'])
    tmap = {d: i for i, d in enumerate(D)}
    eq = eq[eq.date.isin(tmap)]
    pos = eq['date'].map(tmap).values
    rp = np.full(T, np.nan)
    rp[pos] = eq['daily_ret'].values.astype(np.float64)
    log(f'[1] 组合日收益 {len(eq)}天 (2021-01起), rss='
        f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB')
    return D, mkt, close_f, amount_f, rp, codes


def factor_returns(D, mkt, close, amount):
    """5风格因子模仿组合日收益: 截面z(前日)×当日r, 等权长空归一"""
    t0 = time.time()
    r = np.full_like(close, np.nan)
    r[1:] = close[1:] / close[:-1] - 1
    n_valid = np.isfinite(r).sum(axis=1)
    F = {}
    # MKT: 前日成交额加权市场 (≈市值加权 ≈ 指数, 等权含小票倾斜放SIZE因子)
    aw = np.roll(amount, 1, axis=0)
    aw = np.where(np.isfinite(r), aw, np.nan)
    den_vw = np.nansum(aw, axis=1)
    F['MKT'] = np.nansum(aw * r, axis=1) / \
        np.where((den_vw > 0) & (n_valid >= 100), den_vw, np.nan)
    zs = {}
    zs['SIZE'] = np.log(amount + 1e-9)
    c = pd.DataFrame(close)
    zs['MOM'] = (close / c.shift(20).values - 1)
    rr = pd.DataFrame(r)
    zs['VOL'] = rr.rolling(20, min_periods=10).std().values
    zs['REV'] = (close / c.shift(5).values - 1)
    for name, z in zs.items():
        zz = np.where(np.isfinite(z), np.clip(
            (z - np.nanmean(z, axis=1, keepdims=True))
            / (np.nanstd(z, axis=1, keepdims=True) + 1e-12), -3, 3), np.nan)
        num = np.nansum(np.roll(zz, 1, axis=0) * r, axis=1)
        den = np.nansum(np.abs(np.roll(zz, 1, axis=0)), axis=1)
        F[name] = num / np.where(den > 100, den, np.nan)
    # identity烟测: 金额加权市场 vs 指数。
    # 校准(2026-09-13诊断): A股2021-2026风格分化极大, sh000001≠全市场 —
    #   等权/VW/top300与指数corr仅0.81-0.85, 2023年最低0.71(微盘牛指数熊);
    #   指数数据本身核验无误(2024-10-08 +4.59%/2022-03-15 -4.95%),
    #   amount单位一致(vol*100*price/amount=1.00)。故阈值取>0.8, 且加
    #   内部一致性闸(EW vs VW>0.85, 同宇宙不同加权)。
    m = np.isfinite(F['MKT']) & np.isfinite(mkt)
    c_idx = np.corrcoef(F['MKT'][m], mkt[m])[0, 1]
    n_valid = np.isfinite(r).sum(axis=1)
    ew = np.where(n_valid >= 100, np.nanmean(r, axis=1), np.nan)
    m2 = np.isfinite(ew) & np.isfinite(F['MKT'])
    c_ew = np.corrcoef(ew[m2], F['MKT'][m2])[0, 1]
    log(f'[2] 因子日收益完成 {time.time()-t0:.0f}s, identity: '
        f'VW市场vs指数 corr={c_idx:.4f}(>0.8), EW vs VW corr={c_ew:.4f}(>0.85)')
    assert c_idx > 0.8, 'VW市场与指数背离过大, 检查对齐'
    assert c_ew > 0.85, '同宇宙不同加权内部一致性失败'
    return F


def rolling_reg(rp, F, names):
    """滚动OLS(252d): rp ~ 5因子 → beta/R²序列"""
    t0 = time.time()
    X = np.column_stack([F[n] for n in names])
    m = np.isfinite(rp) & np.isfinite(X).all(axis=1)
    betas = {n: np.full(len(rp), np.nan) for n in names}
    r2 = np.full(len(rp), np.nan)
    for t in range(WINDOW, len(rp)):
        if not m[t]:
            continue
        w = m[t - WINDOW:t + 1]
        if w.sum() < WINDOW // 2:
            continue
        y = rp[t - WINDOW:t + 1][w]
        Xw = X[t - WINDOW:t + 1][w]
        try:
            coef, res, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(len(y)), Xw]), y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for i, n in enumerate(names):
            betas[n][t] = coef[1 + i]
        ss_res = float(res[0]) if len(res) else np.nan
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2[t] = 1 - ss_res / ss_tot
    log(f'[3] 滚动回归 {len(rp)}天完成 {time.time()-t0:.0f}s')
    return betas, r2


def yearly_decomp(D, rp, F, names):
    """年度常数beta方差分解"""
    X = np.column_stack([F[n] for n in names])
    m = np.isfinite(rp) & np.isfinite(X).all(axis=1)
    years = pd.Series(D).dt.year.values
    rows = []
    for y in sorted(np.unique(years)):
        w = m & (years == y)
        if w.sum() < 100:
            continue
        yv, Xv = rp[w], X[w]
        coef, res, _, _ = np.linalg.lstsq(
            np.column_stack([np.ones(len(yv)), Xv]), yv, rcond=None)
        resid = yv - np.column_stack([np.ones(len(yv)), Xv]) @ coef
        var_p = yv.var()
        var_fac = var_p - resid.var()
        rows.append((y, var_fac / var_p, coef[1:]))
    return rows


def mdd_legs(nav):
    """回撤腿: DD≥LEG_DD的连续区间"""
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1
    legs, in_leg, t1 = [], False, None
    for t in range(len(dd)):
        if not in_leg and dd[t] <= -LEG_DD:
            in_leg, t1 = True, t
        elif in_leg and dd[t] > -LEG_DD * 0.3:
            legs.append((t1, t - 1))
            in_leg = False
    if in_leg:
        legs.append((t1, len(dd) - 1))
    return dd, legs


def leg_attribution(D, rp, F, names, legs):
    """每腿: 腿内峰→谷段(真实回撤深度)归因: 逐块重估beta × 块内因子累计 vs 残差。

    两点修正(首跑教训):
    1) 腿净收益≠回撤深度 — 起点已在峰下5%, 终点回到峰下1.5%内, 深回撤+完整
       修复的腿净收益接近0, 掩盖内部-27%崩段(首跑2021-07~2024-03巨腿净收益
       仅-1.83%, G2误报0崩腿)。归因对象=nav峰→谷深度(与MDD同口径)。
    2) 日频再平衡因子累计≠持有期因子收益 — 长腿(>BETA_WIN)上暴露漂移+
       累计偏差把因子亏损漏进残差(首跑巨崩腿"因子占比-19%"系此artifact)。
       故每BETA_WIN天分块重估beta再归因; 单窗口法保留作对照列。
    """
    X = np.column_stack([F[n] for n in names])
    m = np.isfinite(rp) & np.isfinite(X).all(axis=1)
    nav = np.cumprod(1 + np.nan_to_num(rp))
    rows = []
    for t1, t2 in legs:
        tp = int(np.argmax(nav[:t1 + 1]))        # 腿起点前运行峰
        tt = tp + int(np.argmin(nav[tp:t2 + 1]))  # 峰后最深谷
        if tt - tp < 3:
            continue
        if not m[tp:tt + 1].all():
            continue
        rleg = float(nav[tt] / nav[tp] - 1)
        # 分块法: 每BETA_WIN天重估beta(块前BETA_WIN窗)
        fac_attr, mkt_attr, n_chunk = 0.0, 0.0, 0
        for c in range(tp, tt, BETA_WIN):
            c2 = min(c + BETA_WIN, tt + 1)
            w0 = m[max(0, c - BETA_WIN):c]
            if w0.sum() < 40:
                continue
            coef, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(w0.sum()),
                                 X[max(0, c - BETA_WIN):c][w0]]),
                rp[max(0, c - BETA_WIN):c][w0], rcond=None)
            b = coef[1:]
            fcum = X[c:c2].sum(axis=0)
            fac_attr += float(b @ fcum)
            mkt_attr += float(b[0] * fcum[0])
            n_chunk += 1
        # 单窗口对照(长腿诊断)
        w0 = m[max(0, tp - BETA_WIN):tp]
        fac_single = np.nan
        if w0.sum() >= 40:
            coef, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(w0.sum()),
                                 X[max(0, tp - BETA_WIN):tp][w0]]),
                rp[max(0, tp - BETA_WIN):tp][w0], rcond=None)
            fac_single = float(coef[1:] @ X[tp:tt + 1].sum(axis=0))
        resid = rleg - fac_attr
        rows.append({
            'leg': f'{pd.Timestamp(D[tp]):%Y-%m-%d}~{pd.Timestamp(D[tt]):%Y-%m-%d}',
            'days': tt - tp + 1, 'rleg': rleg, 'fac': fac_attr,
            'resid': resid, 'share': fac_attr / rleg if abs(rleg) > 1e-6 else np.nan,
            'mkt_share': mkt_attr / rleg if abs(rleg) > 1e-6 else np.nan,
            'n_chunk': n_chunk, 'fac_single': fac_single,
        })
    return rows


def vol_overlay(rp, m):
    """vol覆盖反事实(方向性): s_t=clip(15%/EWMA, 0.6, 1.2)作用于日收益"""
    lam = 0.5 ** (1 / EWMA_HALF)
    var = np.full(len(rp), np.nan)
    v = None
    for t in range(len(rp)):
        if not m[t]:
            continue
        v = rp[t] ** 2 if v is None else lam * v + (1 - lam) * rp[t] ** 2
        var[t] = v
    fvol = np.sqrt(var) * np.sqrt(252)
    s = np.clip(TARGET_VOL / fvol, 0.6, 1.2)
    s = pd.Series(s).ffill().fillna(1.0).values
    ro = rp * s
    nav = np.cumprod(1 + np.nan_to_num(rp))
    navo = np.cumprod(1 + np.nan_to_num(ro))
    return ro, nav, navo


def metrics(nav, m):
    """ret/sharpe/mdd — sharpe只在组合有效日上算(剔除前段零填充稀释)"""
    ret = nav[-1] / nav[0] - 1
    r = np.diff(np.log(nav))
    r = r[m[1:] & m[:-1]]
    sharpe = r.mean() / r.std() * np.sqrt(252)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ret, sharpe, mdd


def main():
    t0 = time.time()
    log('=' * 70)
    log('probe_barra_risk 2026-09-13 — 缺口2前置: 风险结构分解+腿归因+vol覆盖')
    log(f'预置闸: G1 日频R²<{G1_R2:.0%}→因子方向关; G3 非崩腿因子占比>'
        f'{G3_SHARE:.0%}→敞口约束候选; G4 vol覆盖 ΔMDD≥+{G4_DMDD*100:.0f}pp(MDD改善) 且'
        f'|ΔSharpe|≤{G4_DSHARPE} 且|ΔNAV|≤{G4_DNAV:.0%}→冷跑候选')
    D, mkt, close, amount, rp, codes = build_data()
    F = factor_returns(D, mkt, close, amount)
    names = ['MKT', 'SIZE', 'MOM', 'VOL', 'REV']

    betas, r2 = rolling_reg(rp, F, names)
    m = np.isfinite(rp) & np.isfinite(np.column_stack([F[n] for n in names])).all(axis=1)
    r2v = r2[np.isfinite(r2)]
    log(f'\n[3] 滚动R²(252d): 均值{r2v.mean():.3f} 中位'
        f'{np.median(r2v):.3f} p25={np.percentile(r2v,25):.3f} '
        f'p75={np.percentile(r2v,75):.3f}')
    log(f'[3] 滚动beta均值: ' + '  '.join(
        f'{n}={np.nanmean(betas[n]):+.3f}' for n in names))
    g1 = r2v.mean() < G1_R2
    log(f'[3] G1: 全样本日频R²均值{r2v.mean():.3f} '
        f'{"<" if g1 else "≥"}{G1_R2:.2f} → '
        f'{"Barra因子方向边际价值小, 缺口2关闭" if g1 else "因子结构有价值, 继续"}')

    rows = yearly_decomp(D, rp, F, names)
    log('\n[3] 年度方差分解 (因子驱动占比):')
    for y, share, b in rows:
        log(f'  {y}: 因子{share:.0%} 特质{1-share:.0%} '
            f'betas=' + ' '.join(f'{n}={v:+.3f}' for n, v in zip(names, b)))

    nav = np.cumprod(1 + np.nan_to_num(rp))
    dd, legs = mdd_legs(nav)
    log(f'\n[4] 回撤腿 (DD≥{LEG_DD:.0%}, 共{len(legs)}段; '
        f'归因=腿内峰→谷深度, 逐{BETA_WIN}d分块重估beta):')
    attr = leg_attribution(D, rp, F, names, legs)
    crash_legs = []
    for a in attr:
        tag = '崩腿' if (a['rleg'] < -0.08) else ''
        extra = (f' [单窗因子{a["fac_single"]*100:+.1f}%]'
                 if a['n_chunk'] > 1 else '')
        log(f'  {a["leg"]} {a["days"]:>3}d 深度{a["rleg"]*100:+6.2f}% = '
            f'因子{a["fac"]*100:+6.2f}% + 残差{a["resid"]*100:+6.2f}% '
            f'(因子占比{a["share"]*100:+.0f}%, MKT{a["mkt_share"]*100:+.0f}%, '
            f'{a["n_chunk"]}块) {tag}{extra}')
        if a['rleg'] < -0.08:
            crash_legs.append(a)
    g2_ok = all(a['share'] > 0.5 for a in crash_legs) if crash_legs else False
    log(f'[4] G2: 崩腿(深度<-8%) {len(crash_legs)}段, 因子占比均>50%? '
        f'{"是→证实阶段7边界(系统性, 敞口控制抓不住)" if g2_ok else "否"}')
    non_crash = [a for a in attr if a['rleg'] >= -0.08]
    cand = [a for a in non_crash if a['share'] > G3_SHARE]
    log(f'[4] G3: 非崩腿{len(non_crash)}段中因子占比>{G3_SHARE:.0%}的'
        f'{len(cand)}段 → {"Barra敞口约束候选(冷跑方向)" if cand else "无可抓对象, 敞口约束无对象"}')

    ro, nav, navo = vol_overlay(rp, m)
    r0 = metrics(nav, m)
    r1 = metrics(navo, m)
    d_mdd = r1[2] - r0[2]
    d_sharpe = r1[1] - r0[1]
    d_nav = r1[0] - r0[0]
    ok_curve = (abs(r0[0] - 3.576) < 0.01) and (abs(r0[2] + 0.2741) < 0.01)
    log(f'\n[5] 口径校验: 总收益{r0[0]*100:+.1f}%/权威357.6%, '
        f'Sharpe{r0[1]:.4f}/权威1.5555, MDD{r0[2]*100:.2f}%/权威27.41% → '
        f'{"一致" if ok_curve else "不一致! 曲线≠1,143,938基线, 结果作废"}')
    if not ok_curve:
        log(f'总耗时 {time.time()-t0:.0f}s')
        return
    log(f'[5] vol覆盖反事实 (s=clip(15%/EWMA30, 0.6, 1.2)):')
    log(f'  基线: NAV{r0[0]*100:+.1f}% Sharpe{r0[1]:.4f} MDD{r0[2]*100:.2f}%')
    log(f'  覆盖: NAV{r1[0]*100:+.1f}% Sharpe{r1[1]:.4f} MDD{r1[2]*100:.2f}%')
    log(f'  Δ: NAV{d_nav*100:+.2f}pp Sharpe{d_sharpe:+.4f} MDD{d_mdd*100:+.2f}pp')
    g4 = (d_mdd >= G4_DMDD and abs(d_sharpe) <= G4_DSHARPE
          and abs(d_nav) <= G4_DNAV)
    log(f'[5] G4: {"过→vol择时深化冷跑候选" if g4 else "否→波动率择时深化方向关闭"}')
    log(f'\n总耗时 {time.time()-t0:.0f}s, rss='
        f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB')


if __name__ == '__main__':
    main()
