#!/usr/bin/env python3
"""真·筹码峰 + 业绩预告 + 解禁 + 资金流代理 — w5/w20 胜率判别力 (2026-10-22, 只读).

用户问询: 筹码结构(筹码峰)/主力资金流入 + 未测的 yjyg/解禁.
- 筹码: factors.py 的 price_concentration/cost_basis 是 20日VWAP偏离(伪筹码);
  本探针第一次算真模型 — 全史点质量成本分布, 换手率衰减:
    A_t = (1-tr_t)·A_{t-1} + tr_t·δ(当日典型价); softmax 权重无下溢.
  特征: chip_profit(获利比例)/chip_cost_dist/chip_conc(90%区间)/chip_peak_dist/
        chip_profit_chg20.
- yjyg: as-of notice_date 的最近预告 (类型/幅度/新鲜度). PIT=公告日驱动.
- unlock: 下一解禁天数/比例 + 近20日已解禁比例. 解禁日程事前可知=零前视.
- 资金流: 真实大单数据 8b 已闭(剥离价量代理增量≈0); 此处放可算代理
  mf5/mf20 = 成交额加权 (收盘位置-0.5) 累计.
样本: nr<=2 (cap2 候选, 304行) 同 techfeat 口径; 日簇 bootstrap 5000 + BH;
判据 |Δw5|>=8pp & q<=0.10 & w20 同向; 事件桶另报锚点交集数(臂化门槛>=30).

用法: python altfeat_1022.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
from shortfeat_1022 import load, terciles, zwithin, ClusterBoot, bh

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
AD = '/mnt/d/quant/data/alternative_data'
R = '/mnt/d/quant/strategy/rolling_validation_results'

YJYG_SCORE = {'预增': 2, '扭亏': 2, '略增': 1, '续盈': 1, '预盈': 1,
              '预减': -2, '首亏': -2, '略减': -1, '续亏': -1, '预亏': -2,
              '预平': 0, '不确定': 0}


def chip_feats_at(cost, logP, tr, t, close_t, t20):
    """t(含)为止的筹码分布特征. cost=典型价, logP=cumsum(log(1-tr)), tr[0]=0占位."""
    if t < 30:
        return None
    j = np.arange(t + 1)
    # logw' = [0(init@day0)] + [log(tr_j) - logP_j] (j>=1); softmax 常数抵消
    lw = np.empty(t + 1)
    lw[0] = 0.0
    with np.errstate(divide='ignore'):
        lw[1:] = np.log(np.maximum(tr[1:t + 1], 1e-12)) - logP[1:t + 1]
    m = lw.max()
    w = np.exp(lw - m)
    w /= w.sum()
    c = cost[:t + 1]
    pr = float(w[c <= close_t].sum())
    avg = float((w * c).sum())
    o = np.argsort(c)
    cs = np.cumsum(w[o])
    p05 = float(c[o][np.searchsorted(cs, 0.05)])
    p95 = float(c[o][np.searchsorted(cs, 0.95)])
    conc = (p95 - p05) / (p95 + p05 + 1e-12)
    hist, edges = np.histogram(c, bins=40, weights=w)
    peak = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
    # t-20 的获利比例
    pr20 = np.nan
    if t20 >= 30:
        jn = np.arange(t20 + 1)
        lw2 = np.empty(t20 + 1)
        lw2[0] = 0.0
        with np.errstate(divide='ignore'):
            lw2[1:] = np.log(np.maximum(tr[1:t20 + 1], 1e-12)) - logP[1:t20 + 1]
        m2 = lw2.max()
        w2 = np.exp(lw2 - m2)
        w2 /= w2.sum()
        pr20 = float(w2[c[:t20 + 1] <= cost[t20]].sum())
    return dict(chip_profit=pr, chip_cost_dist=close_t / (avg + 1e-12) - 1,
                chip_conc=conc, chip_peak_dist=close_t / (peak + 1e-12) - 1,
                chip_profit_chg20=pr - pr20)


def main():
    p = load()
    q = p[(p['nr'].notna()) & (p['nr'] <= 2)].copy()
    print(f'样本: nr<=2 n={len(q)} 日={q["date"].nunique()}')
    # ---- 逐票筹码/资金流代理 (qfq) ----
    need = sorted(q['code'].unique())
    cache = {}
    miss = 0
    for code in need:
        f = f'{BT}/{code}_qfq.csv'
        if not os.path.exists(f):
            miss += 1
            continue
        d = pd.read_csv(f, usecols=['datetime', 'high', 'low', 'close', 'volume',
                                    'amount', 'turnover_rate'])
        d['datetime'] = pd.to_datetime(d['datetime'])
        d = d.sort_values('datetime').reset_index(drop=True)
        h = d['high'].values.astype(float)
        l = d['low'].values.astype(float)
        c = d['close'].values.astype(float)
        trr = pd.to_numeric(d['turnover_rate'], errors='coerce').fillna(0.0).values
        tr = np.minimum(trr / 100.0, 0.99)
        tr[0] = 0.0
        cost = (h + l + c) / 3.0
        logP = np.cumsum(np.log1p(-tr))
        amt = pd.to_numeric(d['amount'], errors='coerce').fillna(0.0).values
        cp = (c - l) / (np.maximum(h - l, 1e-12))  # 收盘位置
        s5 = pd.Series((cp - 0.5) * amt).rolling(5, min_periods=3).sum()
        a5 = pd.Series(amt).rolling(5, min_periods=3).sum()
        s20 = pd.Series((cp - 0.5) * amt).rolling(20, min_periods=10).sum()
        a20 = pd.Series(amt).rolling(20, min_periods=10).sum()
        cache[code] = (d['datetime'].values, cost, logP, tr, c,
                       (s5 / (a5 + 1e-12)).values, (s20 / (a20 + 1e-12)).values)
    if miss:
        print(f'⚠ 缺 qfq {miss} 只')
    # ---- yjyg / unlock as-of 索引 ----
    yj = pd.read_pickle(f'{AD}/yjyg_records.pkl')
    yj['notice_date'] = pd.to_datetime(yj['notice_date'])
    yj['code'] = yj['code'].astype(str).str.zfill(6)
    yj = yj.sort_values('notice_date')
    yj_map = {c: g[['notice_date', 'forecast_type', 'change_pct']].values
              for c, g in yj.groupby('code')}
    uk = pd.read_pickle(f'{AD}/unlock_schedule.pkl')
    uk['unlock_date'] = pd.to_datetime(uk['unlock_date'])
    uk['code'] = uk['code'].astype(str).str.zfill(6)
    uk = uk.sort_values('unlock_date')
    uk_map = {c: g[['unlock_date', 'ratio']].values for c, g in uk.groupby('code')}

    rows = []
    for r in q.itertuples():
        rec = {}
        cc = cache.get(r.code)
        if cc is not None:
            dts, cost, logP, tr, c, mf5, mf20 = cc
            t = int(np.searchsorted(dts, np.datetime64(r.date), side='right') - 1)
            if t >= 30:
                cf = chip_feats_at(cost, logP, tr, t, c[t], t - 20)
                if cf:
                    rec.update(cf)
                rec['mf5'] = float(mf5[t])
                rec['mf20'] = float(mf20[t])
        arr = yj_map.get(r.code)
        if arr is not None:
            idx = np.searchsorted(arr[:, 0], np.datetime64(r.date), side='right') - 1
            if idx >= 0:
                nd = pd.Timestamp(arr[idx, 0])
                dd = (pd.Timestamp(r.date) - nd).days
                if dd <= 400:  # 只算新鲜的
                    rec['yjyg_days'] = dd
                    rec['yjyg_score'] = YJYG_SCORE.get(arr[idx, 1], 0)
                    rec['yjyg_chg'] = float(arr[idx, 2]) if arr[idx, 2] == arr[idx, 2] else np.nan
                    rec['yjyg_type'] = arr[idx, 1]
        arr = uk_map.get(r.code)
        if arr is not None:
            nxt = np.searchsorted(arr[:, 0], np.datetime64(r.date), side='right')
            if nxt < len(arr):
                rec['unlock_days_next'] = float((pd.Timestamp(arr[nxt, 0]) - pd.Timestamp(r.date)).days)
                rec['unlock_ratio_next'] = float(arr[nxt, 1]) * 100
            past = np.searchsorted(arr[:, 0], np.datetime64(pd.Timestamp(r.date) - pd.Timedelta(days=20)), side='right')
            rec['unlock_recent20'] = float(arr[past:nxt, 1].astype(float).sum()) * 100 if nxt > past else 0.0
        rows.append(rec)
    ff = pd.DataFrame(rows, index=q.index)
    q = pd.concat([q, ff], axis=1)

    # ---- 锚点交集 (臂化门槛) ----
    sel = pd.read_csv(f'{R}/portfolio_selections.csv.anchor_1022', dtype={'code': str},
                      parse_dates=['date'])
    sel['code'] = sel['code'].str.zfill(6)
    akey = set(zip(sel.date, sel.code))
    q['is_ent'] = [(d, c) in akey for d, c in zip(q.date, q.code)]

    feats = ['chip_profit', 'chip_cost_dist', 'chip_conc', 'chip_peak_dist',
             'chip_profit_chg20', 'mf5', 'mf20', 'yjyg_days', 'yjyg_score',
             'yjyg_chg', 'unlock_days_next', 'unlock_ratio_next', 'unlock_recent20']
    print('\n覆盖度:')
    for k in feats:
        print(f'  {k:20s} {q[k].notna().mean()*100:5.1f}%')
    cb = ClusterBoot(sorted(q['date'].unique()))
    rows = []
    print('\n== 连续特征 tercile (hi vs lo) ==')
    print(f'{"feature":20s} {"n":>4s} {"lo%":>6s} {"hi%":>6s} {"Δw5":>7s} '
          f'{"CI":>16s} {"p":>7s} {"q":>7s} {"Δw20":>7s} {"p20":>7s} {"zΔ":>6s} {"zp":>6s} {"锚∩":>4s}')
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
        z = zwithin(q, col, 'nr')
        qz = q.assign(_z=z)
        lo_z, hi_z = terciles(qz, '_z')
        zd, zl, zh, zp = cb.diff(qz, hi_z, lo_z, 'w5') if lo_z is not None else (np.nan,) * 4
        nent = int(q[hi_m.fillna(False)].is_ent.sum() + q[lo_m.fillna(False)].is_ent.sum())
        rows.append(dict(feature=col, n=n, lo=w5lo, hi=w5hi, d5=d5, lo5=l5, hi5=h5,
                         p5=p5, d20=d20, p20=p20, zd=zd, zp=zp, nent=nent))
        print(f'{col:20s} {n:4d} {w5lo:6.1f} {w5hi:6.1f} {d5:+7.1f} '
              f'[{l5:+.1f},{h5:+.1f}] {p5:7.4f} {"":>7s} {d20:+7.1f} {p20:7.4f} '
              f'{zd:+6.1f} {zp:6.4f} {nent:4d}')
    rt = pd.DataFrame(rows)
    if len(rt):
        rt['q5'] = bh(rt['p5'].values)
        print('\nBH q5:', rt.sort_values('q5')[['feature', 'd5', 'p5', 'q5', 'd20', 'zd']]
              .round(3).to_string(index=False))
        short = rt[(rt['d5'].abs() >= 8) & (rt['q5'] <= 0.10) &
                   (np.sign(rt['d20']) == np.sign(rt['d5']))]
        print('过闸 (|Δw5|>=8 & q<=.10 & w20同向):',
              short[['feature', 'd5', 'q5']].round(3).to_dict('records') if len(short) else '(空)')

    # ---- 事件桶 ----
    print('\n== 事件桶 (bucket vs 其余) ==')
    ev = [
        ('预增/扭亏 ≤60d', (q.yjyg_days <= 60) & (q.yjyg_score >= 2)),
        ('预增/扭亏 ≤120d', (q.yjyg_days <= 120) & (q.yjyg_score >= 2)),
        ('预亏/首亏 ≤120d', (q.yjyg_days <= 120) & (q.yjyg_score <= -2)),
        ('解禁≤10d 且 比例≥1%', (q.unlock_days_next <= 10) & (q.unlock_ratio_next >= 1)),
        ('解禁≤20d (任意)', q.unlock_days_next <= 20),
        ('近20d已解禁>1%', q.unlock_recent20 > 1),
        ('获利盘<20% (深套)', q.chip_profit < 0.2),
        ('获利盘>80% (全利)', q.chip_profit > 0.8),
        ('mf5>0 vs <0 (流向)', None),
    ]
    for name, mask in ev:
        if name.startswith('mf5'):
            m1 = q.mf5 > 0
            m0 = q.mf5 <= 0
            d5, l5, h5, p5 = cb.diff(q, m1, m0, 'w5')
            d20, _, _, p20 = cb.diff(q, m1, m0, 'w20')
            a, b = q[m1], q[m0]
        else:
            mask = mask.fillna(False) if mask is not None else None
            if mask is None or mask.sum() < 10:
                print(f'  {name:24s} n<10 跳过')
                continue
            d5, l5, h5, p5 = cb.diff(q, mask, ~mask, 'w5')
            d20, _, _, p20 = cb.diff(q, mask, ~mask, 'w20')
            a, b = q[mask], q[~mask]
        print(f'  {name:24s} n={int(mask.sum() if mask is not None else m1.sum()):3d}(锚∩{int(a.is_ent.sum()):2d}) '
              f'桶w5={a.w5.mean()*100:5.1f}/w20={a.w20.mean()*100:5.1f}/r20={a.r20.mean()*100:+.2f} '
              f'vs 其余 {b.w5.mean()*100:5.1f}/{b.w20.mean()*100:5.1f}/{b.r20.mean()*100:+.2f} || '
              f'Δw5={d5:+.1f}[{l5:+.1f},{h5:+.1f}]p={p5:.3f} Δw20={d20:+.1f}p={p20:.3f}')
    q.to_csv('/mnt/d/quant/.arms/armout/altfeat_rows_1022.csv', index=False)
    rt.to_csv('/mnt/d/quant/.arms/armout/altfeat_1022.csv', index=False)
    print('\n落盘: .arms/armout/altfeat_1022.csv / altfeat_rows_1022.csv')


if __name__ == '__main__':
    main()
