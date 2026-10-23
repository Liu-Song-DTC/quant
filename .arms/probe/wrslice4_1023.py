#!/usr/bin/env python3
"""★score衰减出场反事实 (2026-10-23夜, §23, 只读).

机理链 (已证): 出场主引擎=信号sell旗标+score<0(因子转负);
688063 型灾难 = 基本面因子季度滞后 ⇒ score 衰减慢 ⇒ 崩盘腿数月无出场信号
(1.27→0.50 在−30%价格后仍是正分, 直到07-27 score转负才出, −47%).

候选机制 (信号原生, 非价格): score 衰减出场 — 持仓 score 较入场时衰减≥D% 即出.
反事实: 每笔每日 score (backtest_signals.csv, 166 codes 全史),
  t* = 首个 score <= entry_score×(1−D) 的交易日 (须≥entry日);
  saved = close[t*]/cost − 1 − ret_actual.
验证 (探针可信度): 基线规则 "首个 score<0 日" vs 实际出场日 的分布 —
  若 sig_sell 主导, 两者应接近 (相关性/中位差).
判读: D∈{0.3,0.5,0.6}; Q5/非Q5; 救输家/杀赢家; Σsaved; 单笔集中度;
  tW效应 (被改判盈亏符号的笔数).

用法: .venv/bin/python .arms/probe/wrslice4_1023.py
"""
import numpy as np
import pandas as pd

PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
R = '/mnt/d/quant/strategy/rolling_validation_results'
AR = '/mnt/d/quant/.arms/armout'
DECAYS = [0.3, 0.5, 0.6]


def main():
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['exit_date'] = pd.to_datetime(mm['exit_date'])
    mm['sig_date'] = pd.to_datetime(mm['sig_date'])
    thr = mm['score'].quantile(0.8)
    mm['Q5'] = mm['score'] >= thr

    codes = set(mm['code'])
    # 信号全史 (166 codes 所有日期): score + sell
    rows = []
    for ch in pd.read_csv(f'{R}/backtest_signals.csv',
                          usecols=['code', 'date', 'score', 'sell'],
                          dtype={'code': str}, chunksize=2_000_000):
        ch = ch[ch['code'].str.zfill(6).isin(codes)]
        if len(ch):
            rows.append(ch)
    sig = pd.concat(rows, ignore_index=True)
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig.sort_values(['code', 'date']).drop_duplicates(['code', 'date'], keep='last')
    sg = {c: g.set_index('date')[['score', 'sell']] for c, g in sig.groupby('code')}

    px = pd.read_parquet(PQ, columns=['code', 'date', 'close'])
    px = px[px['code'].isin(codes)].copy()
    px['date'] = pd.to_datetime(px['date'])
    px = px.sort_values(['code', 'date'])
    pg = {c: g.set_index('date')['close'] for c, g in px.groupby('code')}

    recs = []
    for r in mm.itertuples():
        s = sg.get(r.code)
        p = pg.get(r.code)
        if s is None or p is None:
            continue
        win = s[(s.index >= r.entry_date) & (s.index <= r.exit_date)]
        cost = p.get(r.entry_date, np.nan)
        if len(win) < 2 or not (cost == cost):
            continue
        rec = {'code': r.code, 'ret': r.ret, 'hold': r.hold_days, 'Q5': r.Q5,
               'escore': r.score}
        # 基线验证: 首个 score<0 日 / 首个 sell=1 日 vs 实际出日
        neg = win[win['score'] < 0]
        sel = win[win['sell'].astype(bool)]
        rec['d_neg'] = (neg.index[0] - r.exit_date).days if len(neg) else np.nan
        rec['d_sell'] = (sel.index[0] - r.exit_date).days if len(sel) else np.nan
        # 反事实
        for D in DECAYS:
            k = f'd{int(D * 100)}'
            tgt = r.score * (1 - D)
            fired = win['score'] <= tgt
            if fired.any():
                i = int(np.argmax(fired.values))
                t = win.index[i]
                rec['fire' + k] = True
                rec['cf' + k] = p.get(t, np.nan) / cost - 1
                rec['saved' + k] = rec['cf' + k] - r.ret
            else:
                rec['fire' + k] = False
                rec['cf' + k] = np.nan
                rec['saved' + k] = 0.0
        recs.append(rec)
    dd = pd.DataFrame(recs)
    print(f'重建 {len(dd)}/{len(mm)} 笔 (Q5 {int(dd.Q5.sum())})')

    # ── 探针自验证 ──
    print('\n══ 自验证: 基线规则日 vs 实际出场日 (负=提前) ══')
    for c, lab in [('d_neg', '首个score<0日'), ('d_sell', '首个sell=1日')]:
        v = dd[c].dropna()
        if len(v):
            print(f'  {lab}: 覆盖 {len(v)}/{len(dd)} | 中位偏差 {v.median():+.0f}天 | '
                  f'|偏差|≤3天 {(v.abs() <= 3).mean() * 100:.0f}% | '
                  f'提前>20天 {(v < -20).mean() * 100:.0f}%')

    # ── 反事实 ──
    for D in DECAYS:
        k = f'd{int(D * 100)}'
        print(f'\n══ score衰减出场 D={D:.0%} (score≤入场分×{1 - D:.1f}) ══')
        for lab, g in [('全簿', dd), ('Q5', dd[dd.Q5]), ('非Q5', dd[~dd.Q5])]:
            nf = int(g['fire' + k].sum())
            saved = g['saved' + k].sum()
            n_wk = int(((g.ret > 0) & g['fire' + k]).sum())
            wk_saved = g[(g.ret > 0) & g['fire' + k]]['saved' + k].sum()
            flip2win = int(((g.ret <= 0) & g['fire' + k] &
                            (g['cf' + k] > 0)).sum())
            flip2lose = int(((g.ret > 0) & g['fire' + k] &
                             (g['cf' + k] <= 0)).sum())
            print(f'  {lab:4s} n={len(g):3d} fire={nf:3d} ({nf / len(g) * 100:4.1f}%) '
                  f'Σsaved={saved * 100:+7.1f}pp | 杀赢家 {n_wk:2d}笔 {wk_saved * 100:+6.1f}pp '
                  f'| 盈亏改判: 输→赢 {flip2win} 赢→输 {flip2lose}')
        g = dd[dd.Q5 & dd['fire' + k]].sort_values('saved' + k, ascending=False)
        if len(g):
            top = g.iloc[0]
            print(f'  Q5单笔最大saved: {top.code} {top["saved" + k] * 100:+.1f}pp '
                  f'(Σ的 {top["saved" + k] / max(g["saved" + k].sum(), 1e-9) * 100:.0f}%集中度)')

    dd.to_csv(f'{AR}/wrslice_decay_1023.csv', index=False)
    print(f'\n落盘 {AR}/wrslice_decay_1023.csv')


if __name__ == '__main__':
    main()
