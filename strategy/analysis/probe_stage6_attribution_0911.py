#!/usr/bin/env python3
"""2026-09-11 阶段6: 收益暴露归因 — "我们赚的是什么钱?"

三层分解, 全PIT(入场日已知信息):
A. 已实现成交层(506笔): 入场时点 ln_mv/avg_val20/mom20/vol20 截面分位五桶 → 实际ret;
   OLS ret ~ 四暴露 → 哪个敞口在解释收益
B. 信号层(buy行, ~69万): fwd20 分解 = 市场中位(剔除buy) + (同市值×流动性桶中位-市场中位)
   [=风格溢价] + 残差[=纯选股alpha]
C. 已实现成交层 20d口径: 同上分解 → 组合实际执行的alpha构成

口径: 市值proxy=amount/turnover_rate(换手率同源), 流动性=20日均amount(shift1, 与执行层
tradable矩阵同口径), 桶=ln_mv×avg_val20 各自截面五分位(逐日, PIT)。市场中位/桶中位
剔除全部buy行避免自污染。停牌=前值ffill。北交所(4/8/92)全程排除。
"""
import os
import time
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
TR = '/mnt/d/quant/strategy/rolling_validation_results/trade_realized.csv'
IDX = os.path.join(BT, 'sh000001_qfq.csv')


def build_matrix():
    idx = pd.read_csv(IDX, usecols=['datetime'], parse_dates=['datetime'])
    idx = idx[(idx.datetime >= '2020-12-01') & (idx.datetime <= '2026-09-10')]
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
    N = len(codes)
    colmap = {c: i for i, c in enumerate(codes)}
    print(f'[0] 日期 {T} 天 x 股票 {N} 只 (指数/北交所已排除)', flush=True)

    close = np.full((T, N), np.nan, dtype=np.float32)
    amount = np.full((T, N), np.nan, dtype=np.float32)
    tmv = np.full((T, N), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, c in enumerate(codes):
        try:
            df = pd.read_csv(os.path.join(BT, f'{c}_qfq.csv'),
                             usecols=['datetime', 'close', 'amount', 'turnover_rate'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(D, dt)
        pos = pos[pos < T]
        if len(pos) == 0:
            continue
        close[pos, i] = df['close'].values[:len(pos)].astype(np.float32)
        amount[pos, i] = df['amount'].values[:len(pos)].astype(np.float32)
        tr = df['turnover_rate'].values[:len(pos)]
        with np.errstate(divide='ignore', invalid='ignore'):
            mv = np.where(tr > 0, df['amount'].values[:len(pos)] / tr * 100.0, np.nan)
        tmv[pos, i] = mv.astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f'  加载 {i+1}/{N} ({time.time()-t0:.0f}s)', flush=True)
    print(f'[0] 矩阵加载完成 {time.time()-t0:.0f}s, ffill停牌...', flush=True)
    cdf = pd.DataFrame(close); cdf.ffill(axis=0, inplace=True); close = cdf.values
    adf = pd.DataFrame(amount); adf.ffill(axis=0, inplace=True); amount = adf.values
    tdf = pd.DataFrame(tmv); tdf.ffill(axis=0, inplace=True); tmv = tdf.values
    return D, codes, colmap, close, amount, tmv


def exposures(close, amount, tmv):
    """mom20/vol20/avg_val20/ln_mv — 全部PIT(只用当日及以前)"""
    c = pd.DataFrame(close)
    a = pd.DataFrame(amount)
    mom20 = (c / c.shift(20) - 1).values.astype(np.float32)
    r1 = np.log(c / c.shift(1))
    vol20 = r1.rolling(20).std().values.astype(np.float32)
    avg_val20 = a.rolling(20, min_periods=20).mean().shift(1).values.astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_mv = np.log(tmv).astype(np.float32)
    return mom20, vol20, avg_val20, ln_mv


def cross_bins(mat, q=5):
    """逐日截面五分位 → bin 0..q-1, 无效=255"""
    T, N = mat.shape
    bins = np.full((T, N), 255, dtype=np.uint8)
    for t in range(T):
        row = mat[t]
        valid = ~np.isnan(row)
        if valid.sum() < 100:
            continue
        th = np.nanpercentile(row, np.linspace(0, 100, q + 1)[1:-1])
        bins[t, valid] = np.searchsorted(th, row[valid])
    return bins


def cross_rank(mat):
    """逐日截面分位(0~1), NaN→NaN。全矩阵一次算好, 供成交层查表。"""
    T, N = mat.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T):
        row = mat[t]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < 100:
            continue
        order = np.argsort(row[valid])
        rank = np.empty(len(valid), dtype=np.float32)
        rank[order] = np.arange(1, len(valid) + 1) / len(valid)
        out[t, valid] = rank
    return out


def fwd20(close):
    c = pd.DataFrame(close)
    return (c.shift(-20) / c - 1).values.astype(np.float32)


def pos_of(df, D, colmap, date_col):
    """date→行, code→列; 未命中colmap的行剔除(长度对齐)"""
    t = np.searchsorted(D, df[date_col].values.astype('datetime64[ns]'))
    t = np.clip(t, 0, len(D) - 1)
    n = df['code'].map(colmap)
    keep = n.notna().values
    return t[keep].astype(np.int64), n[keep].values.astype(np.int64), keep


def load_trades():
    tr = pd.read_csv(TR, dtype={'code': str})
    tr['code'] = tr['code'].str.zfill(6)
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    return tr


def load_buys():
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig.buy == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[(sig.date >= '2021-01-01') & (sig.date <= '2026-08-13')]
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    print(f'[B] buy信号(2021+, 有20d前视期): {len(sig)} 条', flush=True)
    return sig


def decompose(name, tt, nn, bin_mv, bin_val, bin_med, mkt_med, D):
    """通用分解: 原始 - 市场 = 风格溢价 + 残差alpha"""
    raw = fwd20_all[tt, nn]
    bm = bin_mv[tt, nn]
    bv = bin_val[tt, nn]
    ok = ~np.isnan(raw) & (bm < 5) & (bv < 5)
    if not ok.any():
        print(f'[{name}] 无有效行')
        return
    b = (bm[ok].astype(int) * 5 + bv[ok].astype(int))
    raw_v = raw[ok]
    mm_v = mkt_med[tt[ok]]
    bin_v = bin_med[tt[ok], b]
    ok2 = ~np.isnan(bin_v) & ~np.isnan(mm_v)
    raw_v, mm_v, bin_v = raw_v[ok2], mm_v[ok2], bin_v[ok2]
    print(f'\n[{name}] n={len(raw_v)} 有效(20d口径, 市场/桶中位已剔除buy)')
    print(f'  fwd20原始:        {raw_v.mean()*100:+.2f}%  (中位{np.median(raw_v)*100:+.2f}%)')
    print(f'  同日市场中位:      {mm_v.mean()*100:+.2f}%')
    print(f'  同市值×流动桶中位:  {bin_v.mean()*100:+.2f}%')
    em = (raw_v - mm_v).mean() * 100
    sp = (bin_v - mm_v).mean() * 100
    ra = (raw_v - bin_v).mean() * 100
    print(f'  ── 分解 ──')
    print(f'  超额(对市场): {em:+.2f}pp = 风格溢价(桶-市场) {sp:+.2f}pp + 残差alpha(自身-桶) {ra:+.2f}pp')
    print(f'  残差alpha命中率(raw>桶中位): {100*(raw_v>bin_v).mean():.0f}%')
    print(f'  逐年 残差alpha(raw-桶中位) / 对市场超额:')
    yrs = pd.DatetimeIndex(D[tt[ok][ok2]]).year
    for y in sorted(yrs.unique()):
        m = yrs == y
        print(f'    {y}: n={m.sum():5d} 残差 {(raw_v[m]-bin_v[m]).mean()*100:+.2f}pp '
              f'对市场 {(raw_v[m]-mm_v[m]).mean()*100:+.2f}pp')


def main():
    t0 = time.time()
    D, codes, colmap, close, amount, tmv = build_matrix()
    mom20, vol20, avg_val20, ln_mv = exposures(close, amount, tmv)
    global fwd20_all
    fwd20_all = fwd20(close)
    print(f'[0] 暴露/fwd20计算完成 ({time.time()-t0:.0f}s)', flush=True)

    buys = load_buys()
    bt, bn, _ = pos_of(buys, D, colmap, 'date')
    buy_mask = np.zeros((len(D), len(codes)), dtype=bool)
    buy_mask[bt, bn] = True
    print(f'[0] buy掩码: {buy_mask.sum()} 格', flush=True)

    print('[0] 截面五分位分桶...', flush=True)
    bin_mv = cross_bins(ln_mv)
    bin_val = cross_bins(avg_val20)
    print(f'[0] ln_mv有效 {np.mean(bin_mv<5)*100:.0f}% avg_val20有效 {np.mean(bin_val<5)*100:.0f}%', flush=True)
    print('[0] 截面分位rank矩阵(四暴露)...', flush=True)
    rk_lnmv = cross_rank(ln_mv)
    rk_val = cross_rank(avg_val20)
    rk_mom = cross_rank(mom20)
    rk_vol = cross_rank(vol20)
    print(f'[0] rank完成 ({time.time()-t0:.0f}s)', flush=True)

    T, N = len(D), len(codes)
    mkt_med = np.full(T, np.nan, dtype=np.float32)
    bin_med = np.full((T, 25), np.nan, dtype=np.float32)
    print('[0] 计算市场中位+25桶中位(剔除buy)...', flush=True)
    for t in range(T):
        f = fwd20_all[t]
        okf = ~np.isnan(f) & ~buy_mask[t] & (bin_mv[t] < 5) & (bin_val[t] < 5)
        if okf.sum() < 50:
            continue
        mkt_med[t] = np.nanmedian(f[okf])
        b = (bin_mv[t].astype(int) * 5 + bin_val[t].astype(int))
        for k in range(25):
            m = okf & (b == k)
            if m.sum() >= 10:
                bin_med[t, k] = np.nanmedian(f[m])
    print(f'[0] 中位矩阵完成 ({time.time()-t0:.0f}s), 桶中位覆盖 {np.isfinite(bin_med).mean()*100:.0f}%', flush=True)

    # A. 已实现成交: 暴露分桶 + OLS
    tr = load_trades()
    tt, tn, tkeep = pos_of(tr, D, colmap, 'entry_date')
    ret = tr['ret'].values[tkeep].astype(float)
    hold = tr['hold_days'].values[tkeep].astype(float)
    ranks = {'ln_mv': rk_lnmv[tt, tn], 'avg_val20': rk_val[tt, tn],
             'mom20': rk_mom[tt, tn], 'vol20': rk_vol[tt, tn]}
    ok = np.all([np.isfinite(v) for v in ranks.values()], axis=0) & np.isfinite(ret)
    print(f'\n[A] 已实现成交暴露分桶 (n={ok.sum()}/{len(ret)}, 截面分位五桶, 实际ret口径)')
    for name in ('ln_mv', 'avg_val20', 'mom20', 'vol20'):
        r = ranks[name][ok]
        rr = ret[ok]
        print(f'  [{name}] 五桶 mean ret / 胜率 / n / 持有:')
        for q in range(5):
            m = (r >= q * 0.2) & (r < (q + 1) * 0.2)
            if m.sum() == 0:
                continue
            print(f'    Q{q+1} [{q*20:3d}-{(q+1)*20:3d})%: ret={rr[m].mean()*100:+6.2f}% '
                  f'胜率={100*(rr[m]>0).mean():.0f}% n={m.sum():3d} 持有={hold[ok][m].mean():.0f}d')
    X = np.column_stack([ranks[k][ok] for k in ('ln_mv', 'avg_val20', 'mom20', 'vol20')])
    Xz = (X - X.mean(0)) / X.std(0)
    A = np.column_stack([np.ones(len(Xz)), Xz])
    coef, *_ = np.linalg.lstsq(A, ret[ok], rcond=None)
    resid = ret[ok] - A @ coef
    r2 = 1 - (resid ** 2).sum() / ((ret[ok] - ret[ok].mean()) ** 2).sum()
    n = len(Xz)
    se = np.sqrt((resid ** 2).sum() / (n - 5)) * np.sqrt(np.linalg.inv(A.T @ A).diagonal())
    print(f'\n[A2] OLS ret ~ 四暴露(截面分位, 标准化): n={n} R²={r2:.3f}')
    for name, b, s in zip(('截距', 'ln_mv', 'avg_val20', 'mom20', 'vol20'),
                          coef, se):
        print(f'    {name:>9s}: β={b:+.4f} (se={s:.4f}, t={b/s:+.2f})')

    # B. 信号层分解
    decompose('B 信号层(buy行)', bt, bn, bin_mv, bin_val, bin_med, mkt_med, D)
    # C. 成交层 20d
    decompose('C 成交层(20d口径)', tt, tn, bin_mv, bin_val, bin_med, mkt_med, D)
    print(f'\n[总耗时 {time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
