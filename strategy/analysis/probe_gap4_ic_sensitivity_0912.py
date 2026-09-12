#!/usr/bin/env python3
"""2026-09-12 差距4 go/no-go: 池内IC敏感性模拟 — 池内排序力提升值多少钱

背景: gap1探针已证组合层贪婪机制无罪(净胜纯分数排序+0.89pp), 真瓶颈=池内IC
(score 0.036 / ml_score 0.065)。本探针回答: 若池内排名器IC提升到ρ, top-N选股
fwd20能多拿多少 — 决定标签工程(差距4)是否值得动。
方法: 每选股日, 池内f20转正态秩z, 加高斯噪声校准到目标IC ρ
(ρ=corr(z, z+ε), σ=sqrt(1/ρ²-1)), 按z+ε取top-N(slots), 测均值f20。
校准锚(上一条探针实测): ρ=0.036模拟应≈top_score +3.24%; ρ=0.065应≈top_ml +0.79%;
ρ=0=纯随机top-N(下界); ρ=1=前视上界+83.68%。若校准锚命中→模型可信。
只读。串行。.venv。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap4_ic_sensitivity_0912.py > logs/probe_gap4_ic_0912.log 2>&1
"""
import os
import time
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
PS = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'
IDX = os.path.join(BT, 'sh000001_qfq.csv')
FWD = 20
END_OK = '2026-08-13'
RHOS = [0.0, 0.036, 0.05, 0.065, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0]
REPS = 20


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
    print(f'[0] 加载 {time.time()-t0:.0f}s, ffill...', flush=True)
    return D, codes, colmap, pd.DataFrame(close).ffill(axis=0).values


def main():
    t0 = time.time()
    D, codes, colmap, close = build_close()
    f20 = (pd.DataFrame(close).shift(-FWD) / pd.DataFrame(close) - 1).values
    print(f'[0] fwd20完成 ({time.time()-t0:.0f}s)', flush=True)

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
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
    print(f'[1] pool行数 {len(sig)}', flush=True)

    # 预收集: 每选股日 pool f20 + slots
    days = []
    for d, g in sig.groupby('date'):
        ti = tmap[d]
        pick = ps[ps.date == d]
        if len(pick) == 0:
            continue
        g_ok = g[np.isfinite(g.f20)]
        if len(g_ok) < max(2, len(pick)):
            continue
        days.append((d, g_ok.f20.values, len(pick)))
    print(f'[2] 有效选股日 {len(days)}', flush=True)

    rng = np.random.default_rng(20260912)
    print(f'\n[3] IC敏感性表 (REPS={REPS}):')
    print(f'  {"目标IC":>7s} {"top-N fwd20":>12s} {"Δvs实际+4.13":>12s}')
    res = {}
    for rho in RHOS:
        if rho >= 1.0:
            sigma = 0.0
        elif rho <= 0.0:
            sigma = 1e9
        else:
            sigma = np.sqrt(1.0 / rho**2 - 1.0)
        vals = []
        for d, f, slots in days:
            n = len(f)
            if n < 2:
                continue
            rk = pd.Series(f).rank(pct=True).values
            rk = np.clip(rk, 1e-6, 1 - 1e-6)
            z = np.sqrt(2) * np.array([_erfinv(2 * p - 1) for p in rk])
            rep_means = []
            for _ in range(REPS):
                if sigma >= 1e8:
                    noise_rank = rng.permutation(n)  # 纯随机
                    top = noise_rank[:slots]
                else:
                    noisy = z + rng.normal(0.0, sigma, n)
                    top = np.argsort(-noisy, kind='stable')[:slots]
                rep_means.append(float(np.mean(f[top])))
            vals.append(np.mean(rep_means))
        m = float(np.mean(vals))
        res[rho] = m
        print(f'  {rho:>7.3f} {m*100:>+11.2f}% {(m-0.0413)*100:>+11.2f}pp')
    print(f'\n[4] 校准锚核对:')
    print(f'  ρ=0.036模拟={res[0.036]*100:+.2f}% vs top_score实测=+3.24%')
    print(f'  ρ=0.065模拟={res[0.065]*100:+.2f}% vs top_ml实测=+0.79%')
    print(f'  ρ=0.000模拟={res[0.0]*100:+.2f}% vs 池中位=-0.46%')
    print(f'  ρ=1.000模拟={res[1.0]*100:+.2f}% vs 前视实测=+83.68%')
    print(f'\n[总耗时 {time.time()-t0:.0f}s]')


def _erfinv(x):
    # numpy 2.x: 用 scipy 若可用, 否则数值近似
    try:
        from scipy.special import erfinv as _e
        return float(_e(x))
    except Exception:
        a = 0.147
        ln = np.log(1 - x * x)
        t = 2 / (np.pi * a) + ln / 2
        return np.sign(x) * np.sqrt(np.sqrt(t * t - ln / a) - t)


if __name__ == '__main__':
    main()
