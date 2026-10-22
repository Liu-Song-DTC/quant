#!/usr/bin/env python3
"""持有期上限 / 快速调仓离线反事实 (2026-10-22, 只读).

用户: "胜率高, 调仓快, 拉升高 ... 资金利用率才会高".
本探针: 对基线 246 笔已实现交易, 用 qfq 价格计算"第 N 个交易日收盘退出"
的反事实收益/胜率 (不计再投资; 只回答"拿太久是否在回吐").
注意: 这是因果规则 (N 天到点退出) 的离线评估, 非前视策略.
"""
import os
import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
TRD = f'{R}/trade_realized.csv.base_tr1022'
KS = [3, 5, 10, 15, 20, 30, 40, 60]


def wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def main():
    trd = pd.read_csv(TRD, dtype={'code': str}, parse_dates=['entry_date', 'exit_date'])
    trd['code'] = trd['code'].str.zfill(6)
    recs = []
    for _, r in trd.iterrows():
        f = f'{BT}/{r["code"]}_qfq.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=['datetime', 'open', 'high', 'low', 'close'])
        d['datetime'] = pd.to_datetime(d['datetime'])
        d = d.sort_values('datetime').reset_index(drop=True)
        dt = d['datetime'].values
        ei = int(np.searchsorted(dt, np.datetime64(r['entry_date'])))
        xi = int(np.searchsorted(dt, np.datetime64(r['exit_date'])))
        if ei >= len(d):
            continue
        entry = float(d['open'].iloc[ei])
        if entry <= 0:
            continue
        out = {'code': r['code'], 'entry': r['entry_date'], 'exit': r['exit_date'],
               'hold': int(r['hold_days']), 'ret': float(r['ret']), 'entry_px': entry,
               'exit_idx': xi, 'entry_idx': ei}
        for k in KS:
            j = ei + k
            if j < len(d):
                out[f'r{k}'] = float(d['close'].iloc[j]) / entry - 1
        # 20日内 MFE 与到达日
        j20 = min(ei + 20, len(d) - 1)
        seg = d['high'].iloc[ei:j20 + 1]
        if len(seg):
            out['mfe20'] = float(seg.max()) / entry - 1
            out['mfe20_day'] = int(seg.values.argmax())
        # 实际持有期内 MFE (是否回吐)
        jx = min(xi, len(d) - 1)
        seg_x = d['high'].iloc[ei:jx + 1]
        if len(seg_x):
            out['mfe_hold'] = float(seg_x.max()) / entry - 1
        recs.append(out)
    x = pd.DataFrame(recs)
    print(f'样本 {len(x)} / {len(trd)} 笔 (qfq 可用)')
    print(f'实际: tW={(x.ret>0).mean()*100:.2f}% mean={x.ret.mean()*100:+.2f}% '
          f'meanWin={x.ret[x.ret>0].mean()*100:+.2f}% meanLoss={x.ret[x.ret<=0].mean()*100:+.2f}%')
    print(f'实际持有: mean={x.hold.mean():.1f}d med={x.hold.median():.0f}d; '
          f'持有期MFE={x.mfe_hold.mean()*100:.1f}% vs 终收益 {x.ret.mean()*100:+.1f}% '
          f'⇒ 回吐 {((x.mfe_hold - x.ret).mean())*100:.1f}pp')
    print('\n持有期上限反事实 (第N交易日收盘退出; 只算有N日价格的样本):')
    print(f'{"N":>3s} {"n":>4s} {"tW%":>6s} {"CI":>8s} {"mean%":>7s} {"med%":>7s} '
          f'{"均值差":>7s} {"胜率差":>7s}')
    for k in KS:
        c = f'r{k}'
        v = x[c].dropna()
        if len(v) < 20:
            continue
        kk = int((v > 0).sum())
        lo, hi = wilson(kk, len(v))
        sub = x.loc[v.index]
        # 同一子样本的实际收益做配对
        dmean = (v - sub.ret).mean()
        dwin = ((v > 0).mean() - (sub.ret > 0).mean())
        print(f'{k:3d} {len(v):4d} {v.gt(0).mean()*100:6.1f} {lo:3.0f}-{hi:3.0f} '
              f'{v.mean()*100:+7.2f} {v.median()*100:+7.2f} {dmean*100:+7.2f} {dwin*100:+7.2f}')
    print('\n按实际持有分桶: 若在第20/30/40日退出 (配对子样本):')
    xb = x.copy()
    xb['hold_b'] = pd.cut(xb['hold'], [0, 10, 20, 40, 80, 999],
                          labels=['≤10', '11-20', '21-40', '41-80', '>80'])
    for hb, g in xb.groupby('hold_b', observed=True):
        line = f'  {hb:6s} n={len(g):3d} 实际tW={(g.ret>0).mean()*100:5.1f}% 实际mean={g.ret.mean()*100:+6.2f}%'
        for k in (20, 30, 40):
            c = f'r{k}'
            v = g[c].dropna()
            if len(v) >= 5:
                line += f' | {k}d tW={(v>0).mean()*100:5.1f}% mean={v.mean()*100:+6.2f}%'
        print(line)
    print('\n到达 MFE 的交易日分布 (mfe20_day):')
    print(x['mfe20_day'].value_counts().sort_index().to_string())
    print(f'\nMFE20 均值 {x.mfe20.mean()*100:.1f}%; 20日内最大回撤(MAE)未计; '
          f'实际退出的平均持有 {x.hold.mean():.1f}d')

    x.to_csv('/mnt/d/quant/.arms/armout/holdcap_panel_base1022.csv', index=False)


if __name__ == '__main__':
    main()
