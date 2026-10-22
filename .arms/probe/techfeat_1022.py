#!/usr/bin/env python3
"""用户问询指标的首轮扫描 (2026-10-22, 只读):
换手率 / 量比 / MACD / BOLL / DMI / ATR — 在 cap2 候选样本上的 w5/w20 胜率判别力.

样本: `.arms/pool_1022.csv` 非持仓 nr<=2 (cap2 会考虑的候选), 与 shortfeat_1022 同口径.
指标全部用信号日收盘及之前的数据计算 (因果); 前瞻 = 次日开盘→+H 收盘 (add_forward).
统计: 日级簇 bootstrap 5000 + BH; 三分位 hi vs lo; 另做 nr 内 z 标准化剔除名次代理.
判据同主计划: |Δw5|>=8pp 且 q<=0.10 且 w20 同向 才进候选.

量比 volume_ratio 已在池中 (shortfeat 已扫: Δw5 +4.9pp CI[-3.6,+13.1], 不显著);
本探针把它与其余指标并排复算, 并补 换手率/MACD/BOLL/DMI/ATR.

用法: python techfeat_1022.py
"""
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import (load, terciles, zwithin, ClusterBoot, bh, wilson)
import shortfeat_1022 as S

BT = '/mnt/d/quant/data/stock_data/backtrader_data'


def ema(x, n):
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wilder(x, n):
    """Wilder 平滑 (RMA)."""
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) <= n:
        return out
    out[n] = np.nanmean(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def ind_features(d):
    """d: qfq DataFrame (datetime, open, high, low, close, volume, turnover_rate)."""
    c = d['close'].values.astype(float)
    h = d['high'].values.astype(float)
    l = d['low'].values.astype(float)
    v = d['volume'].values.astype(float)
    n = len(d)
    out = pd.DataFrame(index=d.index)
    out['datetime'] = d['datetime'].values
    # --- 换手率 ---
    if 'turnover_rate' in d:
        tr = pd.to_numeric(d['turnover_rate'], errors='coerce').values.astype(float)
        out['turnover_rate'] = tr
        s = pd.Series(tr)
        out['turnover_ma20'] = (s / s.rolling(20, min_periods=10).mean()).values
        m = s.rolling(20, min_periods=10).mean()
        sd = s.rolling(20, min_periods=10).std()
        out['turnover_z20'] = ((s - m) / (sd + 1e-12)).values
    # --- 量比 (当日量/20日均量) ---
    vs = pd.Series(v)
    out['volratio_calc'] = (vs / vs.rolling(20, min_periods=10).mean()).values
    # --- MACD 12/26/9 ---
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    hist = 2 * (dif - dea)
    out['macd_dif'] = dif / (c + 1e-12)
    out['macd_dea'] = dea / (c + 1e-12)
    out['macd_hist'] = hist / (c + 1e-12)
    out['macd_above0'] = (dif > 0).astype(float)
    cross = (dif > dea).astype(int)
    days = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if i == 0 or cross[i] != cross[i - 1]:
            last = i
        if last >= 0:
            days[i] = i - last
    out['macd_days_since_cross'] = days
    hs = pd.Series(hist)
    out['macd_hist_slope3'] = ((hs - hs.shift(3)) / (pd.Series(c) + 1e-12)).values
    # --- BOLL 20,2 ---
    mid = pd.Series(c).rolling(20, min_periods=20).mean()
    sd20 = pd.Series(c).rolling(20, min_periods=20).std()
    up = mid + 2 * sd20
    lo = mid - 2 * sd20
    out['bb_pos'] = ((pd.Series(c) - lo) / (up - lo + 1e-12)).values
    out['bb_width'] = (4 * sd20 / (mid + 1e-12)).values
    # --- ATR 14 ---
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = wilder(tr, 14)
    out['atr_ratio'] = atr / (c + 1e-12)
    atr_s = pd.Series(out['atr_ratio'])
    out['atr_z20'] = ((atr_s - atr_s.rolling(20, min_periods=10).mean()) /
                      (atr_s.rolling(20, min_periods=10).std() + 1e-12)).values
    # --- DMI 14 (Wilder) ---
    up_move = np.r_[0.0, h[1:] - h[:-1]]
    dn_move = np.r_[0.0, l[:-1] - l[1:]]
    pdm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    ndm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_w = wilder(tr, 14)
    pdi = 100 * wilder(pdm, 14) / (atr_w + 1e-12)
    ndi = 100 * wilder(ndm, 14) / (atr_w + 1e-12)
    dx = 100 * np.abs(pdi - ndi) / (pdi + ndi + 1e-12)
    adx = wilder(dx, 14)
    out['dmi_pdi'] = pdi
    out['dmi_ndi'] = ndi
    out['dmi_spread'] = (pdi - ndi) / 100.0
    out['adx'] = adx
    out['dmi_bull'] = (pdi > ndi).astype(float)
    return out


def main():
    p = load()
    q = p[(p['nr'].notna()) & (p['nr'] <= 2)].copy()
    print(f'样本: nr<=2 n={len(q)} 日={q["date"].nunique()}')
    # 逐票算指标
    need = sorted(q['code'].unique())
    cache = {}
    miss = []
    for code in need:
        f = f'{BT}/{code}_qfq.csv'
        if not os.path.exists(f):
            miss.append(code)
            continue
        d = pd.read_csv(f, usecols=['datetime', 'open', 'high', 'low', 'close',
                                    'volume', 'turnover_rate'])
        d['datetime'] = pd.to_datetime(d['datetime'])
        d = d.sort_values('datetime').reset_index(drop=True)
        cache[code] = ind_features(d)
    if miss:
        print(f'⚠ 缺 qfq {len(miss)} 只')
    feats = ['turnover_rate', 'turnover_ma20', 'turnover_z20', 'volratio_calc',
             'macd_dif', 'macd_dea', 'macd_hist', 'macd_above0',
             'macd_days_since_cross', 'macd_hist_slope3', 'bb_pos', 'bb_width',
             'atr_ratio', 'atr_z20', 'dmi_pdi', 'dmi_ndi', 'dmi_spread', 'adx',
             'dmi_bull']
    recs = []
    for r in q.itertuples():
        c = cache.get(r.code)
        if c is None:
            recs.append({k: np.nan for k in feats})
            continue
        j = np.searchsorted(c['datetime'].values, np.datetime64(r.date))
        if j >= len(c):
            recs.append({k: np.nan for k in feats})
            continue
        recs.append({k: c[k].iloc[j] for k in feats})
    ff = pd.DataFrame(recs, index=q.index)
    q = pd.concat([q, ff], axis=1)
    # 覆盖度
    print('\n覆盖度 (非空/全样本):')
    for k in feats:
        print(f'  {k:22s} {q[k].notna().mean()*100:5.1f}%')
    cb = ClusterBoot(sorted(q['date'].unique()))
    rows = []
    print('\n' + '=' * 118)
    print(f'{"feature":22s} {"n":>4s} {"lo%":>6s} {"hi%":>6s} {"Δw5pp":>7s} '
          f'{"CI":>16s} {"p":>7s} {"q":>7s} {"Δw20pp":>8s} {"p20":>7s} {"zΔw5":>7s} {"zp":>7s}')
    for col in feats:
        lo_m, hi_m = terciles(q, col)
        if lo_m is None:
            continue
        n = int((lo_m.fillna(False) | hi_m.fillna(False)).sum())
        if n < 40:
            continue
        d5, l5, h5, p5 = cb.diff(q, hi_m, lo_m, 'w5')
        d20, _, _, p20 = cb.diff(q, hi_m, lo_m, 'w20')
        w5lo = q.loc[lo_m.fillna(False), 'w5'].mean() * 100
        w5hi = q.loc[hi_m.fillna(False), 'w5'].mean() * 100
        rho = q[[col, 'w5']].dropna().corr(method='spearman').iloc[0, 1]
        z = zwithin(q, col, 'nr')
        qz = q.assign(_z=z)
        lo_z, hi_z = terciles(qz, '_z')
        zd, zl, zh, zp = cb.diff(qz, hi_z, lo_z, 'w5') if lo_z is not None else (np.nan,) * 4
        rows.append(dict(feature=col, n=n, rho=rho, lo=w5lo, hi=w5hi,
                         d5=d5, lo5=l5, hi5=h5, p5=p5, d20=d20, p20=p20,
                         zd=zd, zp=zp))
        print(f'{col:22s} {n:4d} {w5lo:6.1f} {w5hi:6.1f} {d5:+7.1f} '
              f'[{l5:+.1f},{h5:+.1f}] {p5:7.4f} {"":>7s} {d20:+8.1f} {p20:7.4f} '
              f'{zd:+7.1f} {zp:7.4f}')
    r = pd.DataFrame(rows)
    r['q5'] = bh(r['p5'].values)
    print('\nBH (按 q5 排序):')
    print(r.sort_values('q5')[['feature', 'n', 'rho', 'lo', 'hi', 'd5', 'lo5',
                               'hi5', 'p5', 'q5', 'd20', 'p20', 'zd', 'zp']]
          .round(2).to_string(index=False))
    short = r[(r['d5'].abs() >= 8) & (r['q5'] <= 0.10) &
              (np.sign(r['d20']) == np.sign(r['d5']))]
    print('\n候选 (|Δw5|>=8 & q5<=.10 & w20同向):')
    print(short.round(2).to_string(index=False) if len(short) else '  (空)')
    r.to_csv('/mnt/d/quant/.arms/armout/techfeat_1022.csv', index=False)
    q.to_csv('/mnt/d/quant/.arms/armout/techfeat_rows_1022.csv', index=False)
    print('\n落盘: .arms/armout/techfeat_1022.csv / techfeat_rows_1022.csv')


if __name__ == '__main__':
    main()
