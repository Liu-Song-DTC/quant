#!/usr/bin/env python3
"""★高信念仓出场失效: 止损档位路径反事实 (2026-10-23晚, 只读).

链条 (wrslice→wrslice2→本探针):
  1. realized score 五分位倒U: Q5(>0.99) tW 57.6%/均ret −0.08%, 唯一非正桶;
  2. 池级前瞻 Q5 最好 (w5 50.8%/w20 61.0%, 5/5年) ⇒ 入场没错;
  3. 出场救援差: 其余桶 exits +8.3pp, Q5 只 +0.64pp; Q5输家 f20 +5.28% 但实际 −9.75%;
  4. 灾难单 688063(−47%/140d)/688258(−29%)/603590(−13%/157d) = 出场信号永不触发.

假设: 成本价止损档可单独修复高信念仓的灾难腿。
反事实 (逐笔路径, 不跑回测): 对每笔从入场到实际出日的每日收盘,
  fire = 首个 close <= cost×(1+stop) 的交易日; 若存在:
    ret_stop = 该日 close/cost − 1 (按收盘出场近似, 保守不含滑点)
    saved = ret_stop − ret_actual (正=止损改善该笔)
统计口径:
  A. 全簿逐笔: 各档 fire率/救输家/杀赢家 / Σsaved
  B. Q5-only vs 全簿 — 判"Q5条件化是否有增量" (防 in-sample 适配)
  C. 跌破但最终盈利的笔 (止损机会成本)
成本近似: 入场日收盘 (实际为次日开盘; 探针口径统一故相对比较有效).

用法: .venv/bin/python .arms/probe/wrslice3_1023.py
"""
import numpy as np
import pandas as pd

PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
AR = '/mnt/d/quant/.arms/armout'
STOPS = [-0.10, -0.15, -0.20]


def main():
    mm = pd.read_csv(f'{AR}/wrslice_realized_1023.csv', dtype={'code': str})
    mm['entry_date'] = pd.to_datetime(mm['entry_date'])
    mm['exit_date'] = pd.to_datetime(mm['exit_date'])
    thr = mm['score'].quantile(0.8)
    mm['Q5'] = mm['score'] >= thr

    codes = set(mm['code'])
    px = pd.read_parquet(PQ, columns=['code', 'date', 'close'])
    px = px[px['code'].isin(codes)].copy()
    px['date'] = pd.to_datetime(px['date'])
    px = px.sort_values(['code', 'date'])
    pxg = {c: g.set_index('date')['close'] for c, g in px.groupby('code')}

    rows = []
    for r in mm.itertuples():
        s = pxg.get(r.code)
        if s is None:
            continue
        path = s[(s.index >= r.entry_date) & (s.index <= r.exit_date)]
        if len(path) < 2:
            continue
        cost = path.iloc[0]
        rel = path / cost - 1.0
        rec = {'code': r.code, 'ret': r.ret, 'hold': r.hold_days, 'Q5': r.Q5,
               'score': r.score, 'minrel': rel.min()}
        for st in STOPS:
            k = f's{abs(int(st * 100))}'
            fired = rel <= st
            if fired.any():
                i = int(np.argmax(fired.values))
                rec['fire' + k] = True
                rec['stop' + k] = rel.iloc[i]
                rec['saved' + k] = rel.iloc[i] - r.ret
            else:
                rec['fire' + k] = False
                rec['stop' + k] = np.nan
                rec['saved' + k] = 0.0
        rows.append(rec)
    dd = pd.DataFrame(rows)
    print(f'路径重建 {len(dd)}/{len(mm)} 笔 (Q5 {int(dd.Q5.sum())} 笔)')

    for st in STOPS:
        k = f's{abs(int(st * 100))}'
        print(f'\n══ 止损档 {st:+.0%} ══')
        for lab, g in [('全簿', dd), ('Q5', dd[dd.Q5]), ('非Q5', dd[~dd.Q5])]:
            nf = int(g['fire' + k].sum())
            saved = g['saved' + k].sum()
            losers_saved = g[g.ret < 0]['saved' + k].sum()
            n_wk = int(((g.ret > 0) & g['fire' + k]).sum())
            winners_killed = g[(g.ret > 0) & g['fire' + k]]['saved' + k].sum()
            print(f'  {lab:4s} n={len(g):3d} fire={nf:3d} ({nf / len(g) * 100:4.1f}%) '
                  f'Σsaved={saved * 100:+7.1f}pp | 救输家 {losers_saved * 100:+6.1f}pp '
                  f'| 杀赢家 {n_wk:2d}笔 {winners_killed * 100:+6.1f}pp')
        dip_win = dd[dd['fire' + k] & (dd.ret > 0)]
        if len(dip_win):
            print(f'  跌破但最终盈利: {len(dip_win)} 笔 (这些笔止损损失 '
                  f'{dip_win["saved" + k].sum() * 100:+.1f}pp)')

    print(f'\n══ −15%档 Q5 fire 笔明细 ══')
    g = dd[dd.Q5 & dd.fires15].sort_values('saveds15', ascending=False)
    for r in g.itertuples():
        print(f'  {r.code} score={r.score:.2f} 实际{r.ret * 100:+6.1f}% '
              f'(hold {r.hold}d) 止损位{r.stops15 * 100:+6.1f}% '
              f'⇒ saved {r.saveds15 * 100:+6.1f}pp')
    print(f'\n══ −15%档 全簿 saved 最差10笔 (止损最伤) ══')
    g = dd[dd.fires15].nsmallest(10, 'saveds15')
    for r in g.itertuples():
        print(f'  {r.code} {"Q5" if r.Q5 else "--"} score={r.score:.2f} '
              f'实际{r.ret * 100:+6.1f}% ⇒ saved {r.saveds15 * 100:+6.1f}pp '
              f'(hold {r.hold}d)')

    dd.to_csv(f'{AR}/wrslice_stop_1023.csv', index=False)
    print(f'\n落盘 {AR}/wrslice_stop_1023.csv')


if __name__ == '__main__':
    main()
